# Qwen3.5-27B-Dense 在 vLLM 中的执行链路与 Shape 分析

> 配置来源: `qwen3_5_27_dense.json`
> 场景: **prefill token = 8192, decode token = 1, batch size = 1**，TP=1，dtype=bfloat16（权重为 NVFP4 量化）

---

## 1. 模型关键配置（text_config）

| 参数 | 值 | 说明 |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | 入口类（非 MoE） |
| `model_type` | `qwen3_5` | |
| `hidden_size` | **5120** | 模型主干 hidden 维度 |
| `intermediate_size` | **17408** | Dense FFN 中间维度 |
| `num_hidden_layers` | **64** | Decoder 层数 |
| `head_dim` | **256** | 每个 attention head 维度 |
| `num_attention_heads` | 24 | Full attention 的 Q head 数 |
| `num_key_value_heads` | 4 | Full attention 的 KV head 数（GQA 比例 6:1） |
| `partial_rotary_factor` | 0.25 | RoPE 只对前 `256*0.25=64` 维应用 |
| `attn_output_gate` | true | Full attention 有 sigmoid output gate |
| `layer_types` | `[linear, linear, linear, full] × 16` | 共 48 linear + 16 full |
| `full_attention_interval` | 4 | |
| `linear_num_key_heads` | 16 | |
| `linear_num_value_heads` | **48** | |
| `linear_key_head_dim` | 128 | |
| `linear_value_head_dim` | 128 | |
| `linear_conv_kernel_dim` | 4 | Causal Conv1D 核大小 |
| `vocab_size` | 248320 | |
| `mtp_num_hidden_layers` | 1 | 投机解码用的 MTP 层 |

**派生尺寸：**
- **Linear Attention**: `key_dim = 16 × 128 = 2048`，`value_dim = 48 × 128 = 6144`
- **Full Attention**: `q_size = 24 × 256 = 6144`（with output gate 时 proj 出 `q_size*2`），`kv_size = 4 × 256 = 1024`
- **Dense FFN**: `intermediate = 17408`，`gate_up = 2 × 17408 = 34816`
- **RoPE 实际旋转维度**: `256 × 0.25 = 64`
- **激活量化**: NVFP4 group_size=16（线性层输入输出 FP4）

> 与 MoE 版（35B-A3B）的核心差异：**hidden_size 从 2048 翻到 5120，FFN 从 MoE(256专家×512) 变成 Dense(1 × 17408)，linear_num_value_heads 从 32 升到 48**，层数从 40 升到 64。

---

## 2. vLLM 入口模块链路概览

```
HTTP / OpenAI Server
    │
    ▼
LLMEngine  ──▶  EngineCore  ──▶  Scheduler (PagedAttention-style block manager)
                                          │
                                          ▼
                                   ModelRunner (v1/worker/gpu_model_runner.py)
                                          │
       ┌──────────────────────────────────┤
       │ build attn_metadata              │
       │   - full attention: FlashAttn     │
       │   - linear attention: GDN state   │
       │ build mamba_metadata              │
       │ input_ids / positions             │
       ▼
Qwen3_5ForConditionalGeneration.forward()
  └─ language_model = Qwen3_5ForCausalLM
       └─ language_model.model = Qwen3_5Model.forward()
            ├─ VocabParallelEmbedding
            ├─ Qwen3_5DecoderLayer × 64   (layer_type 分发)
            │    ├─ GatedDeltaNetAttention  (linear_attention, 48 层)
            │    │   └─ vllm/model_executor/layers/mamba/gdn_linear_attn.py
            │    ├─ Qwen3NextAttention      (full_attention, 16 层)
            │    │   └─ vllm/model_executor/models/qwen3_next.py
            │    └─ Qwen3NextMLP            (Dense FFN，所有 64 层)
            │         └─ 基类 Qwen2MoeMLP，expert_gate=None
            │         └─ vllm/model_executor/models/qwen2_moe.py
            └─ GemmaRMSNorm (final norm)

Qwen3_5ForConditionalGeneration.compute_logits()
  └─ Qwen3_5ForCausalLM.compute_logits()
       └─ ParallelLMHead  (not tied) + LogitsProcessor ──▶ logits
       │
       ▼
Sampler ──▶ token
```

核心模块对应文件：
- **入口 / Decoder**: `vllm/model_executor/models/qwen3_5.py`
- **Full Attention**: `vllm/model_executor/models/qwen3_next.py`
- **Dense MLP 基类**: `vllm/model_executor/models/qwen2_moe.py`（`Qwen2MoeMLP`，别名 `Qwen3NextMLP`）
- **Linear Attention (GDN)**: `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
- **Causal Conv1D**: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
- **RoPE**: `vllm/model_executor/layers/rotary_embedding/__init__.py`
- **RMSNorm**: `vllm/model_executor/layers/layernorm.py`（`GemmaRMSNorm`：`x * (1 + w)`）

> Dense 版和 MoE 版共享 **完全相同的 Attention 代码路径**（`GatedDeltaNetAttention` / `Qwen3NextAttention`），
> 区别仅在于 FFN 分支：Dense 用 `Qwen3NextMLP`（单条 SwiGLU），MoE 用 `Qwen3NextSparseMoeBlock`（256 expert + shared）。
> `Qwen3_5DecoderLayer` 通过 `model_type` 字段（`"qwen3_5_text"` vs `"qwen3_5_moe_text"`）决定走哪条路径，
> 完全绕过了父类 `Qwen3NextDecoderLayer` 中基于 `mlp_only_layers` / `decoder_sparse_step` 的判定逻辑。

### 2.1 全局数据流与量化边界

下面把一次 forward 的主数据流和量化边界按原始执行顺序展开。`FP4Q` 表示 activation 被 `scaled_fp4_quant` 或 fused quant kernel 转成 NVFP4，随后进入 `flashinfer_mm_fp4` / CUTLASS FP4 GEMM。权重侧的 Linear 权重已离线量化为 NVFP4；运行时主要是量化 activation 和读取权重 scale。

```
input_ids / positions
  │
  ▼
VocabParallelEmbedding
  input:  input_ids [T]
  output: hidden_states [T, 5120] bf16
  quant:  不量化
  │
  ▼
64 × Qwen3_5DecoderLayer
  │
  ├─ input GemmaRMSNorm(+residual)
  │    input/output: [T, 5120] bf16
  │    eager trace:  aten::add / mean / pow / mul / copy_
  │    compiled:     trace 中已融合成 triton_*rms_norm* 或 triton_*rms_norm_scaled_fp4_quant*
  │
  ├─ FP4Q for attention Linear input
  │    op:           _C::scaled_fp4_quant
  │    kernel:       void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>
  │    output:       FP4 activation + block scale
  │
  ├─ attention branch by layer_type
  │    ├─ linear_attention (48 层)
  │    │    in_proj_qkvz / in_proj_ba / out_proj:
  │    │      NVFP4 Linear -> vllm::flashinfer_mm_fp4 -> CUTLASS FP4 GEMM
  │    │    Conv1D + GDN chunk/recurrent core:
  │    │      bf16/fp32 state，不是 NVFP4 Linear
  │    │
  │    └─ full_attention (16 层)
  │         qkv_proj / o_proj:
  │           NVFP4 Linear -> vllm::flashinfer_mm_fp4 -> CUTLASS FP4 GEMM
  │         Q/K RMSNorm + RoPE + FlashAttention/PagedAttention:
  │           bf16 / KV cache，不是 NVFP4 Linear
  │
  ├─ post-attention GemmaRMSNorm(+residual)
  │    input/output: [T, 5120] bf16
  │    compiled:     trace 中已观察到与后续 MLP gate_up 输入 FP4Q 的融合形态
  │
  ├─ Dense MLP (64 层都有)
  │    ├─ FP4Q for gate_up_proj input
  │    ├─ gate_up_proj: [T,5120] -> [T,34816]
  │    │    NVFP4 Linear -> CUTLASS FP4 GEMM
  │    ├─ SiluAndMul: [T,34816] -> [T,17408]
  │    │    eager:    act_and_mul_kernel
  │    │    compiled: silu_mul_cvt_fp16_to_fp4，已把 SiLU*Mul + down 输入量化合并
  │    ├─ FP4Q for down_proj input
  │    └─ down_proj: [T,17408] -> [T,5120]
  │         NVFP4 Linear -> CUTLASS FP4 GEMM
  │
  ▼
final GemmaRMSNorm
  output: [T,5120] bf16
  │
  ▼
ParallelLMHead
  input:  [N_logits,5120]
  output: [N_logits,248320]
  quant:  ignore 中包含 lm_head，所以这里是 bf16 GEMM，不走 NVFP4
  │
  ▼
Sampler / top-k top-p
```

### 2.2 DecoderLayer 内部数据流（含 eager/compiled 对照）

```
hidden_states [T,5120] bf16
  │
  ▼
① input GemmaRMSNorm(+residual)
  │   eager trace:
  │     aten::add / aten::mean / aten::pow / aten::mul / aten::copy_
  │   compiled trace:
  │     triton_*rms_norm*
  │     triton_*rms_norm_scaled_fp4_quant*  （当后面紧接 FP4Q 时）
  │
  ▼
② FP4Q before attention Linear
  │   op/kernel:
  │     _C::scaled_fp4_quant
  │     void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>
  │
  ▼
③ Attention branch
  │   linear_attn: in_proj_qkvz / in_proj_ba / out_proj 使用 FP4 GEMM
  │   full_attn:   qkv_proj / o_proj 使用 FP4 GEMM
  │   trace:
  │     _ZN7cutlass...GemmUniversal
  │     vllm::flashinfer_mm_fp4
  │
  ▼
④ post-attention GemmaRMSNorm(+residual)
  │   eager/compiled 对照同 ①
  │
  ▼
⑤ FP4Q before MLP gate_up_proj
  │
  ▼
⑥ gate_up_proj
  │   [T,5120] -> [T,34816]
  │   NVFP4 Linear / CUTLASS FP4 GEMM
  │
  ▼
⑦ SiluAndMul
  │   [T,34816] -> [T,17408]
  │   eager trace:
  │     act_and_mul_kernel / _C::silu_and_mul
  │   compiled trace:
  │     silu_mul_cvt_fp16_to_fp4 / _C::silu_and_mul_nvfp4_quant
  │     即 SiLU*Mul 后直接产生 down_proj 所需 FP4 activation
  │
  ▼
⑧ down_proj
  │   [T,17408] -> [T,5120]
  │   NVFP4 Linear / CUTLASS FP4 GEMM
  │
  ▼
MLP output [T,5120]
```

---

## 3. Shape 约定（bs=1, prefill=8192, decode=1）

vLLM V1 在 forward 时 **flatten batch × seq**，进入模型的 token 维度是 `num_tokens = Σ seq_i`：

| 阶段 | `num_tokens` | `input_ids` shape |
|---|---|---|
| Prefill | **8192** | `[8192]` |
| Decode  | **1**    | `[1]` |

以下凡涉及 `T` 即表示 `num_tokens`，Prefill 时 `T=8192`，Decode 时 `T=1`。

---

## 4. 逐模块 Shape 变换

### 4.1 Embedding

`VocabParallelEmbedding(vocab=248320, hidden=5120)`

| 张量 | Prefill (T=8192) | Decode (T=1) |
|---|---|---|
| input_ids | `[8192]` int64 | `[1]` int64 |
| positions | `[8192]` int64 | `[1]` int64 |
| hidden_states | `[8192, 5120]` bf16 | `[1, 5120]` bf16 |

---

### 4.2 DecoderLayer 内部（每层都是如此）

vLLM 中 `hidden_states` 和 `residual` 分开携带，残差加法由 `GemmaRMSNorm(..., residual)` 融合完成：
```
if residual is None:
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
else:
    hidden_states, residual = input_layernorm(hidden_states, residual)

attention_out = attention_branch(hidden_states)  # linear or full
hidden_states, residual = post_attention_layernorm(attention_out, residual)
hidden_states = Qwen3NextMLP(hidden_states)      # Dense FFN（每层都有）

# MLP 分支的残差加法在下一层 input_layernorm(hidden_states, residual)
# 或最后的 final norm 中完成。
```

形状始终是 `[T, 5120]`。

---

### 4.3 Linear Attention 分支（48 层，layer 0/1/2/4/5/6/...）

源文件 `gdn_linear_attn.py`，类 `GatedDeltaNetAttention`

```
num_k_heads  = 16,  num_v_heads  = 48
head_k_dim   = 128, head_v_dim   = 128
key_dim      = 16 × 128 = 2048
value_dim    = 48 × 128 = 6144
conv_dim     = key_dim + key_dim + value_dim = 10240
conv_kernel  = 4
```

#### 4.3.1 逐步计算流

##### 投影阶段（逻辑拆分为 QKV / BA / Z）

非 LoRA 路径中，代码实际使用 `in_proj_qkvz` 一次 fused projection 得到 `[q, k, v, z]`，
再用 `in_proj_ba` 得到 `[b, a]`。下面按逻辑张量拆开看 shape：

```
x [T, 5120]
  │
  ├──────────────────────────────────────────┐
  │                                          │
  ▼                                          ▼
① in_proj_qkvz                            ② in_proj_ba
  MergedColumnParallelLinear(5120 → 16384)  MergedColumnParallelLinear(5120 → 96)
  逻辑 W: [16384, 5120] bf16                逻辑 W: [96, 5120] bf16
  output_sizes: [2048, 2048, 6144, 6144]    output_sizes: [48, 48]
  Prefill: [8192, 5120] → [8192, 16384]     Prefill: [8192, 5120] → [8192, 96]
  Decode:  [1, 5120]    → [1, 16384]        Decode:  [1, 5120]    → [1, 96]
  │                                          │
  │ split 为 qkv [T,10240] + z [T,6144]      │ split 为 b [T,48] + a [T,48]
  ▼                                          ▼
```

> 权重加载映射：checkpoint 中 `in_proj_qkv` → fused 的 shard (0,1,2)，`in_proj_z` → shard 3。

##### Conv + Split 阶段

```
qkv [T, 10240]       (来自 ① 的前 10240 维；z [T,6144] 后面 ⑨ 用)
  │
  ▼
④ Causal Conv1D (kernel=4, channels=10240)
  Prefill: causal_conv1d_fn       → [8192, 10240]
  Decode:  causal_conv1d_update   → [1, 10240]
           + 读写 conv_state；默认 cache 布局 [batch_slot, 3, 10240] bf16
             kernel 中转置成 [batch_slot, 10240, 3] 视图
  │
  ▼
⑤ split 为 Q / K / V，按 [2048, 2048, 6144] 切分
  Prefill: Q [8192, 2048]  K [8192, 2048]  V [8192, 6144]
  Decode:  Q [1, 2048]     K [1, 2048]     V [1, 6144]
  │
  ▼
⑥ reshape to heads
  Q: [T, 16, 128]   K: [T, 16, 128]   V: [T, 48, 128]
```

##### GDN 核心计算

```
ba [T, 96]           (来自 ②)
  │
  ▼
⑦ b, a = chunk(ba, 2)    各 [T, 48]
  │
  │
  ▼
⑧ gdn_attention_core
  输入: Q [T,16,128], K [T,16,128], V [T,48,128], b [T,48], a [T,48]
  + 读写 recurrent_state [48, 128, 128] fp32（每请求持有，大小不随序列长度变）
  │
  │  Prefill: chunk-wise causal scan，并行处理 T 个位置
  │  Decode:  单 token 递归更新 state，O(1)
  │
  ▼
  core_attn_out
  Prefill: [8192, 48, 128]
  Decode:  [1, 48, 128]
```

##### 输出门控 + 投影

```
core_attn_out [T, 48, 128]       z [T, 6144] (来自 ①)
  │                                │
  │                                ▼
  │                           ⑨ z reshape → [T, 48, 128]
  │                                │
  └──────────────┬─────────────────┘
                 │
                 ▼
⑩ RMSNormGated: norm(core_attn_out) * silu(z)
  Prefill: [8192, 48, 128]
  Decode:  [1, 48, 128]
  │
  ▼
⑪ flatten heads → [T, 6144]
  │
  ▼
⑫ out_proj: RowParallelLinear(6144 → 5120)     W: [5120, 6144] bf16
  Prefill: [8192, 6144] → [8192, 5120]
  Decode:  [1, 6144]    → [1, 5120]
  │
  ▼
linear_attn_out [T, 5120]
```

#### 4.3.2 State（非传统 KV Cache）

Linear Attention 的 "KV cache" 是**固定大小**的状态，不随序列长度线性增长：
- `conv_state`: 默认 `[batch_cache_slot, 3, 10240]` bf16（SD 布局；DS 布局时为 `[batch_cache_slot, 10240, 3]`）
- `recurrent_state`: `[batch_cache_slot, 48, 128, 128]` fp32

Prefill/Decode 差异：
- **Prefill**: chunk-wise causal scan，token 并行处理 8192 个位置
- **Decode**: 单 token 递归更新 state，O(1)（相对序列长度）的显存读写

---

### 4.4 Full Attention 分支（16 层，layer 3/7/11/.../63）

源文件 `qwen3_next.py`，类 `Qwen3NextAttention`，`attn_output_gate=True`

```
num_heads = 24, num_kv_heads = 4, head_dim = 256
q_size  = 24 × 256 = 6144
kv_size = 4  × 256 = 1024
qkv_proj 输出维度 = q_size×2 + kv_size×2 = 12288 + 2048 = 14336   (output gate 使 Q 通道翻倍)
```

#### 4.4.1 逐步计算流

##### QKV 投影 + Split

```
x [T, 5120]
  │
  ▼
① qkv_proj: QKVParallelLinear(5120 → 14336)     W: [14336, 5120] bf16
  Prefill: [8192, 5120] → [8192, 14336]
  Decode:  [1, 5120]    → [1, 14336]
  │
  ▼
② split 为 [q_gate=12288, k=1024, v=1024]
  Prefill: q_gate [8192, 12288]   K [8192, 1024]   V [8192, 1024]
  Decode:  q_gate [1, 12288]      K [1, 1024]      V [1, 1024]
  │
  ▼
③ q, gate = chunk(q_gate, 2)
  Prefill: Q [8192, 6144]   gate [8192, 6144]
  Decode:  Q [1, 6144]      gate [1, 6144]
  │
  ▼
④ reshape to heads
  Q: [T, 24, 256]   K: [T, 4, 256]   V: [T, 4, 256]
  (gate 暂存，后面 ⑧ 用)
```

##### Norm + RoPE + Attention

```
Q [T, 24, 256]            K [T, 4, 256]
  │                         │
  ▼                         ▼
⑤ q_norm (per-head)       k_norm (per-head)
  RMSNorm on 256-dim       RMSNorm on 256-dim
  shape 不变                shape 不变
  │                         │
  ▼                         ▼
⑥ RoPE (partial, rot_dim=64)
  对每个 head 前 64 维旋转，后 192 维保留
  shape 不变                shape 不变
  │                         │
  └────────────┬────────────┘
               │                     V [T, 4, 256]
               │                         │
               ▼                         ▼
⑦ Attention kernel
  Prefill: FlashAttention (causal MHA)
  Decode:  PagedAttention（读取 KV cache）
  │
  ▼
  attn_out
  Prefill: [8192, 24, 256]
  Decode:  [1, 24, 256]
```

##### Output Gate + 投影

```
attn_out [T, 24, 256]                gate [T, 6144] (来自 ③)
  │                                      │
  ▼                                      │
⑧ flatten → [T, 6144]                   │
  │                                      │
  └──────────────┬───────────────────────┘
                 │
                 ▼
⑨ output gate: sigmoid(gate) * attn_out
  Prefill: [8192, 6144]
  Decode:  [1, 6144]
  │
  ▼
⑩ o_proj: RowParallelLinear(6144 → 5120)     W: [5120, 6144] bf16
  Prefill: [8192, 6144] → [8192, 5120]
  Decode:  [1, 6144]    → [1, 5120]
  │
  ▼
full_attn_out [T, 5120]
```

#### 4.4.2 KV Cache（per full-attention layer）

- 16 个 full-attention 层 × 每层 K/V
- 每 token 每层 KV：`2 × num_kv_heads × head_dim × bf16 = 2 × 4 × 256 × 2B = 4096 B = 4 KB`
- Prefill 8192 token 写入一次；Decode 每步追加 1 个 token 的 KV
- vLLM 以 PagedAttention block 为单位管理，通常 `block_size=16`

---

### 4.5 Dense FFN（`Qwen3NextMLP`，64 层，每层都有）

#### 4.5.0 架构算法数据流（与框架无关）

Dense FFN 是标准的 SwiGLU MLP，没有 expert routing，也没有 shared expert gate。
`Qwen3NextMLP` 实际上是 `Qwen2MoeMLP`（别名），构造时 `expert_gate=None`。

```
hidden_size = 5120, intermediate_size = 17408, hidden_act = silu
```

```
hidden_states x [T, 5120]
         │
         ▼
┌─── Dense SwiGLU MLP ──────────────────────────────────────────────┐
│                                                                    │
│  ① gate_up_proj                                                    │
│     MergedColumnParallelLinear(5120 → 2×17408 = 34816)            │
│     逻辑上 split 为 gate [T,17408] + up [T,17408]                  │
│         │                                                          │
│         ▼                                                          │
│  ② SiluAndMul (SwiGLU)                                            │
│     silu(gate) * up                                                │
│     → [T, 17408]                                                  │
│         │                                                          │
│         ▼                                                          │
│  ③ down_proj                                                       │
│     RowParallelLinear(17408 → 5120)                                │
│     → mlp_out [T, 5120]                                           │
│                                                                    │
│  （无 expert_gate，无 sigmoid 门控）                                 │
│                                                                    │
└──────────────────────────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                                    FFN 输出 [T, 5120]
```

#### 4.5.1 逐步计算流（含 Shape 变换）

```
x [T, 5120]
  │
  ▼
① gate_up_proj: MergedColumnParallelLinear(5120 → 34816)     W: [34816, 5120] bf16/NVFP4
  │     Prefill: [8192, 5120] → [8192, 34816]
  │     Decode:  [1, 5120]    → [1, 34816]
  │
  ▼
② SiluAndMul (SwiGLU): split 为 gate[...,17408] 和 up[...,17408]，silu(gate) * up
  │     Prefill: [8192, 34816] → [8192, 17408]
  │     Decode:  [1, 34816]    → [1, 17408]
  │
  ▼
③ down_proj: RowParallelLinear(17408 → 5120)                  W: [5120, 17408] bf16/NVFP4
  │     Prefill: [8192, 17408] → [8192, 5120]
  │     Decode:  [1, 17408]    → [1, 5120]
  │
  ▼
mlp_out [T, 5120]
```

#### 4.5.2 vLLM 实际执行流

##### 相关源文件

- `vllm/model_executor/models/qwen3_5.py` — `Qwen3_5DecoderLayer`（通过 `model_type` 决定 FFN 类型）
- `vllm/model_executor/models/qwen2_moe.py` — `Qwen2MoeMLP`（即 `Qwen3NextMLP`）

##### 调用链

```python
# Qwen3_5DecoderLayer.__init__:
# 通过 model_type 判断，绕过父类 Qwen3NextDecoderLayer 的 mlp_only_layers 逻辑
if config.model_type == "qwen3_5_text":
    self.mlp = Qwen3NextMLP(
        hidden_size=5120,
        intermediate_size=17408,
        hidden_act="silu",
        quant_config=quant_config,
        # expert_gate 不传 → 默认 None
    )

# Qwen3_5DecoderLayer.forward:
hidden_states = self.mlp(hidden_states)
```

##### `Qwen2MoeMLP.forward` 的实际执行顺序

```python
# vllm/model_executor/models/qwen2_moe.py:114
def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)                           # ①
    out = self.act_fn(gate_up)                                  # ②
    out, _ = self.down_proj(out)                                # ③
    if self.expert_gate is not None:                            # Dense 版: False
        out = F.sigmoid(self.expert_gate(x)[0]) * out           # 不执行
    return out
```

> Dense 版的 `expert_gate=None`，所以 forward 就是纯粹的 gate_up → SiluAndMul → down，
> 没有 MoE 版 shared expert 的 sigmoid 门控。

#### 4.5.3 每层计算量（bs=1）

Dense FFN 每层都要完整计算，没有 MoE 的 "只激活部分专家" 机制：
- **Prefill**: 8192 token × 每 token 2 次大 GEMM（gate_up + down），全量通过 17408 中间维度
- **Decode**: 1 token 仍需加载全部权重，是 decode 阶段的核心带宽瓶颈

> 对比 MoE 版：MoE decode 每步只激活 top-8 个 512 维小专家 + 1 个 shared expert；
> Dense 版每步必须过完整的 17408 维 FFN —— 权重搬运量大约 5-6 倍。

---

### 4.6 Final RMSNorm + LM Head

| 操作 | shape |
|---|---|
| `GemmaRMSNorm(5120)` | `[T, 5120]` |
| `ParallelLMHead(5120 → 248320)` (tie_word_embeddings=False，**不共享**) | `[N_logits, 248320]` |
| Prefill 普通采样 | V1 `ModelRunner` 先用 `input_batch.logits_indices` 选采样位置 hidden states；bs=1 通常 `N_logits=1`，若请求 prompt logprobs 可能更多 |
| Decode | `[1, 248320]` |

Sampler 输出下一个 token id。

---

## 5. 整图 Shape 汇总

### 5.1 Prefill (T=8192)
```
input_ids [8192]
  → embed → [8192, 5120]
  → 64 × DecoderLayer
      linear_attn 层 (48):
         qkvz_proj: [8192,5120] → [8192,16384]，split qkv=[8192,10240], z=[8192,6144]
         conv1d:   [8192,10240] → [8192,10240]
         ba_proj:  [8192,5120] → [8192,96]
         gdn_core:                [8192,48,128]
         out_proj: [8192,6144] → [8192,5120]
      full_attn 层 (16):
         qkv_proj: [8192,5120] → [8192,14336]
         Q:[8192,24,256] K:[8192,4,256] V:[8192,4,256]
         RoPE(partial dim=64)
         flash_attn out: [8192,24,256]
         o_proj:  [8192,6144] → [8192,5120]
      Dense MLP (64):
         gate_up: [8192,5120] → [8192,34816]
         silu*up → [8192, 17408]
         down:    [8192,17408] → [8192,5120]
  → RMSNorm → [8192, 5120]
  → ModelRunner 取 logits_indices → lm_head → 普通采样 [1, 248320]
```

### 5.2 Decode (T=1)
```
input_ids [1]
  → embed → [1, 5120]
  → 64 × DecoderLayer
      linear_attn 层:
         qkvz_proj → [1,16384]，split qkv=[1,10240], z=[1,6144] ; ba→[1,96]
         causal_conv1d_update(state 默认 [3,10240])  → [1,10240]
         gdn_recurrent_update(state[48,128,128] fp32) → [1,48,128]
         out_proj → [1,5120]
      full_attn 层:
         qkv_proj → [1,14336]
         Q:[1,24,256] K:[1,4,256] V:[1,4,256]
         PagedAttention(读 KV cache)
         o_proj → [1,5120]
      Dense MLP:
         gate_up → [1, 34816]
         silu*up → [1, 17408]
         down → [1, 5120]
  → RMSNorm → [1, 5120]
  → lm_head → [1, 248320]
```

---

## 6. 显存 / 计算要点（bs=1）

**权重（NVFP4，group_size=16，每个元素 4 bit + scale 开销）**
- 48 个 linear-attn 层：`in_proj_qkvz 5120×16384 + in_proj_ba 5120×96 + conv1d 10240×4 + out_proj 6144×5120 ≈ 115.5 M params / 层`
- 16 层 full_attn：`qkv_proj 5120×14336 + o_proj 6144×5120 ≈ 104.8 M params / 层`
- 64 层 Dense MLP：`gate_up 5120×34816 + down 17408×5120 ≈ 267.4 M params / 层`
- 合计约 **27B 总参数，27B 激活参数**（Dense 没有稀疏激活）

**权重显存估算**
- FP4 近似按 0.5 B/param 估算，~13 GB 权重（还需加上 scale）
- Embedding + lm_head: `2 × 248320 × 5120 × 2B ≈ 5 GB`（bf16，不量化）

**KV Cache（仅 full attention 层消耗，16 层）**
- 每 token 每层：`2 (K+V) × 4 heads × 256 dim × 2B (bf16) = 4096 B = 4 KB`
- 16 层 → 每 token 64 KB
- Prefill 8192 token：≈ **512 MB** 的 KV cache（单请求）
- Decode 每步新增 64 KB

**Mamba-like State（linear attention 48 层，每请求固定大小）**
- `conv_state`: 48 × 3 × 10240 × 2B ≈ **2.8 MB**（默认 SD 布局）
- `recurrent_state`: 48 × 48 × 128 × 128 × 4B (fp32) ≈ **150 MB**
- 合计约 **153 MB / request**，与序列长度无关

**Decode 阶段计算瓶颈**
- **Dense FFN 是核心带宽消耗**：
  - 每层两次大 GEMM：`[1,5120] @ [5120,34816]` + `[1,17408] @ [17408,5120]`
  - 每层约 **133 MB 权重搬运**（FP4 打包 + scale），64 层 → 每 decode step ~**8.5 GB** 的权重读取
- Full attention 16 层：受 KV cache 带宽限制
- Linear attention 48 层：state 加载 + 少量 GEMM，几乎没有长序列的开销
- 对比 MoE 27B（A3B）每 step 只读 ~1.5 GB 的 activated 权重，**Dense 的 decode 带宽压力显著更高（~5.7×）**

---

## 7. 与 vLLM Runtime 的交互要点

- **Attention Backend**:
  - Full attention → 根据运行配置选择 `FlashAttn`/`FlashInfer` 等 backend + PagedKV
  - Linear attention → `GatedDeltaNet` custom op（在 `vllm/model_executor/layers/mamba/`），有独立的 `MambaStateManager` 管理 `conv_state` / `recurrent_state`
- **Scheduler**: 同一序列的 full-attn KV block 和 linear-attn state slot 一起调度；`HybridKVCacheManager` 同时管两套。
- **CUDA Graph**: Decode 阶段对这两类 attention 的 kernel 都会捕获，batch=1 常走 graph-replay 路径。
- **FFN 特征差异（vs MoE）**: Dense FFN 没有 expert routing / permute / unpermute 的开销，单条 SwiGLU 的 GEMM 对 cuBLAS 更友好（大矩阵、高 GPU 利用率），但 decode 时每层都要加载全量权重，bandwidth-bound 严重。

---

## 8. Trace 对照：kernel 到数据流的映射

本节基于两份 trace：

- compiled / graph trace: `rank0.1778566047660882823.pt.trace.json`
- eager trace: `eager.1778572740212119173.pt.trace.json`

eager trace 更适合定位源码，因为它保留了直接 Python location；compiled trace 更接近实际启用 torch.compile / CUDA graph 后的融合形态。

### 8.1 Prefill 主要 kernel 对照

| 数据流位置 | eager trace 证据 | compiled trace 证据 | 说明 |
|---|---:|---:|---|
| FP4 Linear GEMM | `_ZN7cutlass...GemmUniversal`，`407.60 ms` prefill top rows，位置 `vllm/utils/flashinfer.py:656 flashinfer_scaled_fp4_mm` | `_ZN7cutlass...GemmUniversal`，`322.44 + 97.56 ms` | 所有 NVFP4 Linear 的主计算；Dense MLP 的 `gate_up/down` 占最大份额，也包含 attention 投影 |
| Full attention prefill | `flash_fwd_splitkv_kernel`，`72.89 ms`，位置 `vllm_flash_attn/flash_attn_interface.py:176` | `flash_fwd_splitkv_kernel`，`73.42 ms` | 16 个 full-attn 层的 FlashAttention prefill |
| MLP SwiGLU | `act_and_mul_kernel`，`46.15 ms`，位置 `activation.py:143 forward_cuda` | `silu_mul_cvt_fp16_to_fp4`，`35.18 ms` | eager 中只是 `SiLU*Mul`；compiled trace 中已把 `SiLU*Mul + down_proj 输入 FP4Q` 融到一起 |
| FP4 activation quant | `cvt_fp16_to_fp4`，`40.09 ms`，位置 `_custom_ops.py:1711 scaled_fp4_quant` | `cvt_fp16_to_fp4`，`21.35 ms`，另有 fused quant kernel | 每个 FP4 Linear 前的 activation quant；compiled 会吸收一部分到 fused Triton 或 fused SiLU kernel |
| RMSNorm / residual | eager 中拆成 `aten::mul 86.44 ms`、`pow 44.57 ms`、`copy_ 34.44 ms`、`mean 17.42 ms`，位置 `ir/ops/layernorm.py:9 rms_norm` / `layernorm.py:375 forward_native` | `triton_red_fused__to_copy_add_copy__rms_norm_4 12.53 ms`、`triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3 8.03 ms`、`triton_poi_fused__...rms_norm...fp4_quant... 11.92 ms` | compiled 把 eager 的 RMSNorm 原子链融合；部分路径还把后续 FP4Q 一起融合 |
| GDN chunk scan | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64 23.93 ms`、`chunk_fwd_kernel_o 20.55 ms`、`merge_16x16 18.93 ms`、`recompute_w_u 17.98 ms` | 同类 kernel 共约 `81 ms` | 48 个 linear-attn 层的 chunk-wise GDN core；不是 NVFP4 Linear |
| Conv1D / post-conv | `_causal_conv1d_fwd_kernel 15.94 ms`、`_fused_post_conv_kernel 14.97 ms`，位置 `gdn_linear_attn.py:519 forward_cuda` | `_causal_conv1d_fwd_kernel 14.38 ms`、`_fused_post_conv_kernel 14.97 ms` | `linear_attn.conv1d` 在 quant config 的 `ignore` 里，不走 NVFP4 |

### 8.2 Decode 主要 kernel 对照

| 数据流位置 | eager trace 证据 | compiled trace 证据 | 说明 |
|---|---:|---:|---|
| FP4 Linear GEMM | CUTLASS FP4 GEMM `7.49 + 5.17 + 0.37 ms`，位置 `flashinfer_scaled_fp4_mm` | CUTLASS FP4 GEMM `5.71 + 3.97 + 3.09 + 0.37 ms` | decode 的核心带宽瓶颈；batch=1 时 M 很小，主要受权重读取影响 |
| LM Head | cublas GEMV `2.06 ms`，位置 `model_executor/layers/utils.py:92 default_unquantized_gemm` | 同为 `2.06 ms` | `lm_head` 在 `ignore` 中，不量化 |
| Full attention decode | `flash_fwd_splitkv_kernel 0.53 ms` | `0.53 ms` | 16 个 full-attn 层读 KV cache；不是主瓶颈 |
| GDN recurrent decode | `fused_recurrent_gated_delta_rule_packed_decode_kernel 0.32 ms` | `0.32 ms` | 48 个 linear-attn 层 O(1) recurrent state update |
| FP4 activation quant | `cvt_fp16_to_fp4 0.45 ms` | `cvt_fp16_to_fp4 0.31 ms` | decode 里 quant 本身不是主要耗时，主耗时仍是 FP4 GEMM 权重带宽 |

### 8.3 Linear Attention 分支数据流（量化点）

```
x [T,5120] bf16
  │
  ├─ FP4Q
  │    op:     _C::scaled_fp4_quant
  │    kernel: void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>
  │
  ├─ in_proj_qkvz
  │    Linear: 5120 -> 16384
  │    quant:  NVFP4 weight + FP4 activation
  │    kernel: vllm::flashinfer_mm_fp4 / CUTLASS FP4 GEMM
  │    output: [T,16384] = q [T,2048] + k [T,2048] + v [T,6144] + z [T,6144]
  │
  └─ in_proj_ba
       Linear: 5120 -> 96
       quant:  NVFP4 weight + FP4 activation
       kernel: vllm::flashinfer_mm_fp4 / CUTLASS FP4 GEMM
       output: b [T,48] + a [T,48]

qkv [T,10240]
  │
  ▼
Causal Conv1D
  input/output: [T,10240] bf16
  quant:        不走 NVFP4；`linear_attn.conv1d` 在 ignore 中
  trace:        _causal_conv1d_fwd_kernel / _causal_conv1d_update_kernel
  │
  ▼
split + reshape
  Q [T,16,128], K [T,16,128], V [T,48,128]
  │
  ▼
GDN chunk/recurrent core + b/a gate
  prefill trace: chunk_gated_delta_rule_fwd_kernel_h_blockdim64
                 chunk_fwd_kernel_o
                 merge_16x16_to_64x64_inverse_kernel
                 recompute_w_u_fwd_kernel
  decode trace:  fused_recurrent_gated_delta_rule_packed_decode_kernel
  state:         recurrent_state [48,128,128] fp32
  quant:         不是 NVFP4 Linear
  │
  ▼
RMSNormGated / z gate
  input:  core_attn_out [T,48,128] + z [T,6144]
  output: [T,6144] bf16
  │
  ▼
FP4Q
  │
  ▼
out_proj
  Linear: 6144 -> 5120
  quant:  NVFP4 weight + FP4 activation
  kernel: vllm::flashinfer_mm_fp4 / CUTLASS FP4 GEMM
```

量化点只有 `in_proj_qkvz`、`in_proj_ba`、`out_proj` 这些 Linear 的输入和权重。`conv1d` 被 `ignore`，GDN core / recurrent state 不是 NVFP4 Linear。

### 8.4 Full Attention 分支数据流（量化点）

```
x [T,5120] bf16
  │
  ▼
FP4Q
  op/kernel: _C::scaled_fp4_quant / void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>
  │
  ▼
qkv_proj
  Linear: 5120 -> 14336
  quant:  NVFP4 weight + FP4 activation
  kernel: vllm::flashinfer_mm_fp4 / CUTLASS FP4 GEMM
  output: q_gate [T,12288] + K [T,1024] + V [T,1024]
  │
  ▼
split q_gate
  Q [T,6144] + gate [T,6144]
  │
  ├─ Q/K RMSNorm
  │    input/output: Q [T,24,256], K [T,4,256]
  │    quant:        不走 NVFP4 Linear
  │
  ├─ RoPE
  │    partial rotary dim = 64
  │    quant: 不走 NVFP4
  │
  ├─ FlashAttention / PagedAttention
  │    prefill trace: flash_fwd_splitkv_kernel
  │    decode trace:  flash_fwd_splitkv_kernel + KV cache read
  │    KV cache:      bf16
  │
  └─ sigmoid output gate
       gate [T,6144] * attn_out [T,6144]
       quant: 不是 NVFP4 Linear
       │
       ▼
     FP4Q
       │
       ▼
     o_proj
       Linear: 6144 -> 5120
       quant:  NVFP4 weight + FP4 activation
       kernel: vllm::flashinfer_mm_fp4 / CUTLASS FP4 GEMM
```

Full attention 的 `qkv_proj` 和 `o_proj` 走 NVFP4 Linear。`q_norm/k_norm`、RoPE、FlashAttention、KV cache 写入/读取都不属于 NVFP4 Linear quant。

### 8.5 Dense MLP 分支数据流（量化点）

```
x [T,5120] bf16
  │
  ▼
FP4Q for gate_up_proj
  op/kernel: _C::scaled_fp4_quant / void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>
  │
  ▼
gate_up_proj
  Linear: [T,5120] -> [T,34816]
  quant:  NVFP4 weight + FP4 activation
  kernel: vllm::flashinfer_mm_fp4 / CUTLASS FP4 GEMM
  │
  ▼
split gate/up
  gate [T,17408] + up [T,17408]
  │
  ▼
SiluAndMul
  output: [T,17408] bf16 before quant
  eager trace:
    act_and_mul_kernel / _C::silu_and_mul
  compiled trace:
    silu_mul_cvt_fp16_to_fp4 / _C::silu_and_mul_nvfp4_quant
    即 SiLU*Mul 后直接输出 down_proj 所需 FP4 activation + scale
  │
  ▼
FP4Q for down_proj
  eager:    独立 cvt_fp16_to_fp4
  compiled: trace 中已与 SiluAndMul 融合
  │
  ▼
down_proj
  Linear: [T,17408] -> [T,5120]
  quant:  NVFP4 weight + FP4 activation
  kernel: vllm::flashinfer_mm_fp4 / CUTLASS FP4 GEMM
```

Dense MLP 是 trace 中最大的 NVFP4 数据流：每层 `gate_up_proj` 和 `down_proj` 都是大 GEMM。compiled trace 中 `silu_mul_cvt_fp16_to_fp4` 表示 `SiluAndMul` 的输出直接量化为 FP4，供 `down_proj` 使用；eager trace 中它拆成 `act_and_mul_kernel` 和后续 `cvt_fp16_to_fp4`。

### 8.6 Kernel 名称速查

| Kernel / CPU op | 对应数据流 | 是否量化相关 |
|---|---|---|
| `_ZN7cutlass...GemmUniversal...` / `vllm::flashinfer_mm_fp4` | NVFP4 Linear GEMM | 是，FP4 activation × FP4 weight |
| `void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>` / `_C::scaled_fp4_quant` | Linear 前 activation quant | 是 |
| `void vllm::silu_mul_cvt_fp16_to_fp4<__nv_bfloat16, false>` / `_C::silu_and_mul_nvfp4_quant` | MLP `SiluAndMul` + `down_proj` 输入量化 | 是，compiled fused path |
| `act_and_mul_kernel` / `_C::silu_and_mul` | MLP `SiluAndMul` | eager 中本身未量化，后面接 `cvt_fp16_to_fp4` |
| `triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3` | RMSNorm/residual + FP4Q | 是，compiled fused path |
| `triton_poi_fused__...rms_norm...scaled_fp4_quant...silu...` | RMSNorm / gate / view / FP4Q 组合 | 是，compiled fused path |
| `aten::mean/pow/mul/add/copy_` under `rms_norm` | eager RMSNorm 原子链 | 不是量化本身，但常紧接 FP4Q |
| `flash_fwd_splitkv_kernel` / `_vllm_fa2_C::varlen_fwd` | Full attention | 不是 NVFP4 Linear quant |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` / `chunk_fwd_kernel_o` / `merge_16x16` / `recompute_w_u` | GDN prefill core | 不是 NVFP4 Linear quant |
| `_causal_conv1d_fwd_kernel` / `_fused_post_conv_kernel` | Linear-attn Conv1D 和后处理 | 不是 NVFP4 Linear quant，conv1d 被 `ignore` |
| cublas `gemvx` / `default_unquantized_gemm` | `lm_head` | 不量化，`lm_head` 被 `ignore` |

---

## 9. 与 MoE 版（35B-A3B）的关键差异对比

| 维度 | Dense 27B | MoE 35B-A3B |
|---|---|---|
| hidden_size | **5120** | 2048 |
| 层数 | **64** | 40 |
| attention heads | 24 (Q) / 4 (KV) | 16 (Q) / 2 (KV) |
| head_dim | 256 | 256 |
| Q/KV 尺寸 | 6144 / 1024 | 4096 / 512 |
| linear_num_value_heads | **48** | 32 |
| linear value_dim | 6144 | 4096 |
| linear conv_dim | 10240 | 8192 |
| in_proj_qkvz 输出 | 16384 | 12288 |
| FFN 类型 | Dense SwiGLU | 256 Expert MoE (top-8) + 1 Shared |
| FFN intermediate | **17408** | 512 (per expert) |
| FFN expert_gate | None | sigmoid(Linear(2048→1)) |
| Full attention 层数 | 16 | 10 |
| 总参数 / 激活参数 | 27B / **27B** | 35B / ~3B |
| 每 token KV 总大小 | 64 KB | 20 KB |
| Prefill 8192 KV 占用 | 512 MB | 160 MB |
| Linear state per request | 153 MB | 61 MB |
| Decode 每步权重带宽 | ~8.5 GB | ~1.5 GB（top-8 experts + shared + attention） |
| Decode 延迟特征 | 严重 memory-bound | experts 读取虽少但 scatter-gather 有 latency |

**结论**：
- **Prefill 8192**: Dense 版计算量远大（FFN 每层 178M+89M 权重×8192 tokens），但 Dense 的大矩阵 GEMM 对 GPU 利用率更高，不存在 MoE 的 permute/unpermute + grouped-gemm 开销。
- **Decode 1**: Dense 是典型的 bandwidth-bound；MoE 因为只激活 top-8 专家，理论 FLOPs 和权重搬运都小 5-6 倍，非常适合低并发场景。
- 两模型都用 **同一套 Attention 代码路径**（hybrid linear + full），**只有 FFN 分支不同**，因此在 vLLM 中的 scheduler / kv-cache-manager / mamba-state-manager 都是复用的。
