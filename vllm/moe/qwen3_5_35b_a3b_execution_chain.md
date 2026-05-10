# Qwen3.5-35B-A3B (MoE) 在 vLLM 中的执行链路与 Shape 分析

> 配置来源: `qwen3_5_35b_a3b_nvfp4.json`
> 场景: **prefill token = 4096, decode token = 1, batch size = 1**，TP=1，EP=1，dtype=bfloat16（权重为 NVFP4 量化）

---

## 1. 模型关键配置（text_config）

| 参数 | 值 | 说明 |
|---|---|---|
| `architectures` | `Qwen3_5MoeForConditionalGeneration` | 入口类 |
| `model_type` | `qwen3_5_moe` | |
| `hidden_size` | **2048** | 模型主干 hidden 维度 |
| `num_hidden_layers` | **40** | Decoder 层数 |
| `head_dim` | **256** | 每个 attention head 维度 |
| `num_attention_heads` | 16 | Full attention 的 Q head 数 |
| `num_key_value_heads` | 2 | Full attention 的 KV head 数（GQA 比例 8:1） |
| `partial_rotary_factor` | 0.25 | RoPE 只对前 `256*0.25=64` 维应用 |
| `attn_output_gate` | true | Full attention 有 sigmoid output gate |
| `layer_types` | `[linear, linear, linear, full] × 10` | 每 4 层插 1 层 Full Attention，共 30 linear + 10 full |
| `full_attention_interval` | 4 | |
| `linear_num_key_heads` | 16 | |
| `linear_num_value_heads` | 32 | |
| `linear_key_head_dim` | 128 | |
| `linear_value_head_dim` | 128 | |
| `linear_conv_kernel_dim` | 4 | Causal Conv1D 核大小 |
| `num_experts` | **256** | 路由专家总数 |
| `num_experts_per_tok` | **8** | Top-K |
| `moe_intermediate_size` | **512** | 单个路由专家的 FFN 中间维度 |
| `shared_expert_intermediate_size` | 512 | 共享专家 FFN 中间维度 |
| `vocab_size` | 248320 | |
| `mtp_num_hidden_layers` | 1 | 投机解码用的 MTP 层 |

**派生尺寸：**
- **Linear Attention**: `key_dim = 16 × 128 = 2048`，`value_dim = 32 × 128 = 4096`
- **Full Attention**: `q_size = 16 × 256 = 4096`（with output gate 时 proj 出 `q_size*2`），`kv_size = 2 × 256 = 512`
- **RoPE 实际旋转维度**: `256 × 0.25 = 64`
- **激活量化**: NVFP4 group_size=16（线性层输入输出 FP4）

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
Qwen3_5MoeForConditionalGeneration.forward()
  └─ language_model = Qwen3_5MoeForCausalLM
       └─ language_model.model = Qwen3_5Model.forward()
            ├─ VocabParallelEmbedding
            ├─ Qwen3_5DecoderLayer × 40   (layer_type 分发)
            │    ├─ GatedDeltaNetAttention  (linear_attention, 30 层)
            │    │   └─ vllm/model_executor/layers/mamba/gdn_linear_attn.py
            │    ├─ Qwen3NextAttention      (full_attention, 10 层)
            │    │   └─ vllm/model_executor/models/qwen3_next.py
            │    └─ Qwen3NextSparseMoeBlock (所有层, mlp_only_layers=[])
            │         ├─ gate (router)
            │         ├─ shared_expert_gate
            │         ├─ Qwen3NextMLP (shared_expert)
            │         └─ FusedMoE + MoERunner/SharedExperts (256 experts, top-8)
            └─ GemmaRMSNorm (final norm)

Qwen3_5MoeForConditionalGeneration.compute_logits()
  └─ Qwen3_5MoeForCausalLM.compute_logits()
       └─ ParallelLMHead  (not tied) + LogitsProcessor ──▶ logits
       │
       ▼
Sampler ──▶ token
```

核心模块对应文件：
- **入口 / Decoder**: `vllm/model_executor/models/qwen3_5.py`
- **Full Attention / SparseMoeBlock / 共享 MLP**: `vllm/model_executor/models/qwen3_next.py`
- **Dense MLP 基类**: `vllm/model_executor/models/qwen2_moe.py`（`Qwen2MoeMLP`）
- **Linear Attention (GDN)**: `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
- **Causal Conv1D**: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
- **FusedMoE / Shared Expert 调度**: `vllm/model_executor/layers/fused_moe/{layer.py, runner/moe_runner.py, runner/shared_experts.py}`
- **RoPE**: `vllm/model_executor/layers/rotary_embedding/__init__.py`
- **RMSNorm**: `vllm/model_executor/layers/layernorm.py`（`GemmaRMSNorm`：`x * (1 + w)`）

---

## 3. Shape 约定（bs=1, prefill=4096, decode=1）

vLLM V1 在 forward 时 **flatten batch × seq**，进入模型的 token 维度是 `num_tokens = Σ seq_i`：

| 阶段 | `num_tokens` | `input_ids` shape |
|---|---|---|
| Prefill | **4096** | `[4096]` |
| Decode  | **1**    | `[1]` |

以下凡涉及 `T` 即表示 `num_tokens`，Prefill 时 `T=4096`，Decode 时 `T=1`。

---

## 4. 逐模块 Shape 变换

### 4.1 Embedding

`VocabParallelEmbedding(vocab=248320, hidden=2048)`

| 张量 | Prefill (T=4096) | Decode (T=1) |
|---|---|---|
| input_ids | `[4096]` int64 | `[1]` int64 |
| positions | `[4096]` int64 | `[1]` int64 |
| hidden_states | `[4096, 2048]` bf16 | `[1, 2048]` bf16 |

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
hidden_states = Qwen3NextSparseMoeBlock(hidden_states)

# MLP 分支的残差加法在下一层 input_layernorm(hidden_states, residual)
# 或最后的 final norm 中完成。
```

形状始终是 `[T, 2048]`。

---

### 4.3 Linear Attention 分支（30 层，layer 0/1/2/4/5/6/...）

源文件 `gdn_linear_attn.py`，类 `GatedDeltaNetAttention`

```
num_k_heads  = 16,  num_v_heads  = 32
head_k_dim   = 128, head_v_dim   = 128
key_dim      = 16 × 128 = 2048
value_dim    = 32 × 128 = 4096
conv_dim     = key_dim + key_dim + value_dim = 8192
conv_kernel  = 4
```

#### 4.3.1 逐步计算流

##### 投影阶段（逻辑拆分为 QKV / BA / Z）

非 LoRA 路径中，代码实际使用 `in_proj_qkvz` 一次 fused projection 得到 `[q, k, v, z]`，
再用 `in_proj_ba` 得到 `[b, a]`。下面按逻辑张量拆开看 shape：

```
x [T, 2048]
  │
  ├──────────────────────────────────────────┐
  │                                          │
  ▼                                          ▼
① in_proj_qkvz                            ② in_proj_ba
  MergedColumnParallelLinear(2048 → 12288)  MergedColumnParallelLinear(2048 → 64)
  逻辑 W: [12288, 2048] bf16                逻辑 W: [64, 2048] bf16
  Prefill: [4096, 2048] → [4096, 12288]     Prefill: [4096, 2048] → [4096, 64]
  Decode:  [1, 2048]    → [1, 12288]        Decode:  [1, 2048]    → [1, 64]
  │                                          │
  │ split 为 qkv [T,8192] + z [T,4096]       │ split 为 b [T,32] + a [T,32]
  ▼                                          ▼
```

##### Conv + Split 阶段

```
qkv [T, 8192]       (来自 ① 的前 8192 维；z [T,4096] 后面 ⑨ 用)
  │
  ▼
④ Causal Conv1D (kernel=4, channels=8192)
  Prefill: causal_conv1d_fn       → [4096, 8192]
  Decode:  causal_conv1d_update   → [1, 8192]
           + 读写 conv_state；默认 cache 布局 [batch_slot, 3, 8192] bf16
             kernel 中转置成 [batch_slot, 8192, 3] 视图
  │
  ▼
⑤ split 为 Q / K / V，按 [2048, 2048, 4096] 切分
  Prefill: Q [4096, 2048]  K [4096, 2048]  V [4096, 4096]
  Decode:  Q [1, 2048]     K [1, 2048]     V [1, 4096]
  │
  ▼
⑥ reshape to heads
  Q: [T, 16, 128]   K: [T, 16, 128]   V: [T, 32, 128]
```

##### GDN 核心计算

```
ba [T, 64]           (来自 ②)
  │
  ▼
⑦ b, a = chunk(ba, 2)    各 [T, 32]
  │
  │
  ▼
⑧ gdn_attention_core
  输入: Q [T,16,128], K [T,16,128], V [T,32,128], b [T,32], a [T,32]
  + 读写 recurrent_state [32, 128, 128] fp32（每请求持有，大小不随序列长度变）
  │
  │  Prefill: chunk-wise causal scan，并行处理 T 个位置
  │  Decode:  单 token 递归更新 state，O(1)
  │
  ▼
  core_attn_out
  Prefill: [4096, 32, 128]
  Decode:  [1, 32, 128]
```

##### 输出门控 + 投影

```
core_attn_out [T, 32, 128]       z [T, 4096] (来自 ①)
  │                                │
  │                                ▼
  │                           ⑨ z reshape → [T, 32, 128]
  │                                │
  └──────────────┬─────────────────┘
                 │
                 ▼
⑩ RMSNormGated: norm(core_attn_out) * silu(z)
  Prefill: [4096, 32, 128]
  Decode:  [1, 32, 128]
  │
  ▼
⑪ flatten heads → [T, 4096]
  │
  ▼
⑫ out_proj: RowParallelLinear(4096 → 2048)     W: [2048, 4096] bf16
  Prefill: [4096, 4096] → [4096, 2048]
  Decode:  [1, 4096]    → [1, 2048]
  │
  ▼
linear_attn_out [T, 2048]
```

#### 4.3.2 State（非传统 KV Cache）

Linear Attention 的 "KV cache" 是**固定大小**的状态，不随序列长度线性增长：
- `conv_state`: 默认 `[batch_cache_slot, 3, 8192]` bf16（SD 布局；DS 布局时为 `[batch_cache_slot, 8192, 3]`）
- `recurrent_state`: `[batch_cache_slot, 32, 128, 128]` fp32

Prefill/Decode 差异：
- **Prefill**: chunk-wise causal scan，token 并行处理 4096 个位置
- **Decode**: 单 token 递归更新 state，O(1)（相对序列长度）的显存读写

---

### 4.4 Full Attention 分支（10 层，layer 3/7/11/.../39）

源文件 `qwen3_next.py`，类 `Qwen3NextAttention`，`attn_output_gate=True`

```
num_heads = 16, num_kv_heads = 2, head_dim = 256
q_size  = 16 × 256 = 4096
kv_size = 2  × 256 = 512
qkv_proj 输出维度 = q_size×2 + kv_size×2 = 8192 + 1024 = 9216   (output gate 使 Q 通道翻倍)
```

#### 4.4.1 逐步计算流

##### QKV 投影 + Split

```
x [T, 2048]
  │
  ▼
① qkv_proj: QKVParallelLinear(2048 → 9216)     W: [9216, 2048] bf16
  Prefill: [4096, 2048] → [4096, 9216]
  Decode:  [1, 2048]    → [1, 9216]
  │
  ▼
② split 为 [q_gate=8192, k=512, v=512]
  Prefill: q_gate [4096, 8192]   K [4096, 512]   V [4096, 512]
  Decode:  q_gate [1, 8192]      K [1, 512]      V [1, 512]
  │
  ▼
③ q, gate = chunk(q_gate, 2)
  Prefill: Q [4096, 4096]   gate [4096, 4096]
  Decode:  Q [1, 4096]      gate [1, 4096]
  │
  ▼
④ reshape to heads
  Q: [T, 16, 256]   K: [T, 2, 256]   V: [T, 2, 256]
  (gate 暂存，后面 ⑧ 用)
```

##### Norm + RoPE + Attention

```
Q [T, 16, 256]            K [T, 2, 256]
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
               │                     V [T, 2, 256]
               │                         │
               ▼                         ▼
⑦ Attention kernel
  Prefill: FlashAttention (causal MHA)
  Decode:  PagedAttention（读取 KV cache）
  │
  ▼
  attn_out
  Prefill: [4096, 16, 256]
  Decode:  [1, 16, 256]
```

##### Output Gate + 投影

```
attn_out [T, 16, 256]                gate [T, 4096] (来自 ③)
  │                                      │
  ▼                                      │
⑧ flatten → [T, 4096]                   │
  │                                      │
  └──────────────┬───────────────────────┘
                 │
                 ▼
⑨ output gate: sigmoid(gate) * attn_out
  Prefill: [4096, 4096]
  Decode:  [1, 4096]
  │
  ▼
⑩ o_proj: RowParallelLinear(4096 → 2048)     W: [2048, 4096] bf16
  Prefill: [4096, 4096] → [4096, 2048]
  Decode:  [1, 4096]    → [1, 2048]
  │
  ▼
full_attn_out [T, 2048]
```

#### 4.4.2 KV Cache（per full-attention layer）

- 10 个 full-attention 层 × 每层 K/V
- 每 token 每层 KV：`2 × num_kv_heads × head_dim × bf16 = 2 × 2 × 256 × 2B = 2048 B = 2 KB`
- Prefill 4096 token 写入一次；Decode 每步追加 1 个 token 的 KV
- vLLM 以 PagedAttention block 为单位管理，通常 `block_size=16`

---

### 4.5 Qwen3NextSparseMoeBlock（每层都有，40 层）

```
hidden_size = 2048, num_experts = 256, top_k = 8
moe_intermediate_size = 512 (路由专家)
shared_expert_intermediate_size = 512 (共享专家)
```

#### 4.5.0 架构算法数据流（与框架无关）

从模型架构角度，SparseMoeBlock 的数学计算如下。
gate (router) 和 shared expert 两条路径在数据依赖上互不依赖，均只读取输入 x：

```
hidden_states x [T, 2048]
         │
         │  x 同时送入两条路径
         │
  ┌──────┴─────────────────────────────────────┐
  │                                            │
  ▼                                            ▼
┌─── 路径 A: Routed Experts ──────┐  ┌─── 路径 B: Shared Expert ────────────┐
│                                 │  │                                      │
│  ① gate (router)                │  │  ④ shared_expert MLP                 │
│     Linear(2048→256)            │  │     gate_up_proj(x)                  │
│     W: [256, 2048] bf16         │  │     W: [1024, 2048] bf16             │
│     → router_logits [T, 256]   │  │     → [T, 1024]                     │
│         │                       │  │         │                            │
│         ▼                       │  │         ▼                            │
│  ② softmax + top-8 + renorm     │  │     SiluAndMul (SwiGLU)             │
│     topk_weights [T, 8]        │  │     → [T, 512]                      │
│     topk_ids     [T, 8]        │  │         │                            │
│         │                       │  │         ▼                            │
│         ▼                       │  │     down_proj                        │
│  ③ 256 个 SwiGLU MLP            │  │     W: [2048, 512] bf16              │
│     每 token 只经过 top-8 个     │  │     → shared_mlp_out [T, 2048]      │
│                                 │  │         │                            │
│     w13: [256, 1024, 2048] bf16 │  │  ⑤ shared_expert_gate(x)            │
│     w2:  [256, 2048,  512] bf16 │  │     Linear(2048→1)                  │
│                                 │  │     W: [1, 2048] bf16                │
│     permute → w13@x → SwiGLU   │  │     → gate_val [T, 1]              │
│     → w2@x → un-permute        │  │         │                            │
│     → weighted sum              │  │         ▼                            │
│                                 │  │  ⑥ sigmoid(gate_val) * mlp_out      │
│     → routed_out [T, 2048]      │  │     → shared_out [T, 2048]          │
└────────────────┬────────────────┘  └───────────────────┬──────────────────┘
                 │                                       │
                 └──────────────┬─────────────────────────┘
                                │
                                ▼
                  ⑦ routed_out + shared_out
                        │
                        ▼
                  MoE 输出 [T, 2048]
```

#### 4.5.1 逐步计算流（含 Shape 变换）

##### 路径 A：Routed Experts

```
x [T, 2048]
  │
  ▼
① gate: x @ W_gate^T                        W_gate [256, 2048] bf16
  │     Prefill: [4096, 2048] @ [2048, 256] → [4096, 256]
  │     Decode:  [1, 2048]    @ [2048, 256] → [1, 256]
  │
  ▼
② softmax + top-8 + renormalize
  │     Prefill: topk_weights [4096, 8] fp32,  topk_ids [4096, 8] int32
  │     Decode:  topk_weights [1, 8]    fp32,  topk_ids [1, 8]    int32
  │     默认 renormalize=True，top-8 权重会再归一化到每个 token 内 sum=1
  │
  ▼
③ 256 个 SwiGLU MLP，每 token 只经过 top-8 个
  │
  │  ③a. permute：按 topk_ids 将 token 重排为 expert-major 顺序
  │     Prefill: [4096, 2048] → [T×top_k=32768, 2048]   (每 token 复制 8 份)
  │     Decode:  [1, 2048]    → [8, 2048]
  │
  │  ③b. gate_up：w13 @ x                    w13 [256, 1024, 2048] bf16
  │     Prefill: [32768, 2048] → [32768, 1024]
  │     Decode:  [8, 2048]     → [8, 1024]
  │
  │  ③c. SwiGLU：split 为 gate[...,512] 和 up[...,512]，silu(gate) * up
  │     Prefill: [32768, 1024] → [32768, 512]
  │     Decode:  [8, 1024]     → [8, 512]
  │
  │  ③d. down：w2 @ x                        w2 [256, 2048, 512] bf16
  │     Prefill: [32768, 512] → [32768, 2048]
  │     Decode:  [8, 512]     → [8, 2048]
  │
  │  ③e. un-permute + topk_weights 加权求和，每 token 聚合 8 个专家
  │     Prefill: [32768, 2048] → [4096, 2048]
  │     Decode:  [8, 2048]     → [1, 2048]
  │
  ▼
routed_out [T, 2048]
```

##### 路径 B：Shared Expert

```
x [T, 2048]
  │
  ▼
④a gate_up_proj: Linear(2048 → 1024)                    W: [1024, 2048] bf16
  │  Prefill: [4096, 2048] → [4096, 1024]
  │  Decode:  [1, 2048]    → [1, 1024]
  │
  ▼
④b SiluAndMul (SwiGLU): split → silu(gate) * up
  │  Prefill: [4096, 1024] → [4096, 512]
  │  Decode:  [1, 1024]    → [1, 512]
  │
  ▼
④c down_proj: Linear(512 → 2048)                        W: [2048, 512] bf16
  │  Prefill: [4096, 512] → [4096, 2048]
  │  Decode:  [1, 512]    → [1, 2048]
  │
  ▼
shared_mlp_out [T, 2048]
  │
  │                                     x [T, 2048] (原始输入)
  │                                       │
  │                                       ▼
  │                                     ⑤ shared_expert_gate(x)
  │                                       Linear(2048 → 1)
  │                                       W: [1, 2048] bf16
  │                                       Prefill: → [4096, 1]
  │                                       Decode:  → [1, 1]
  │                                       │
  └──────────────┬────────────────────────┘
                 │
                 ▼
⑥ sigmoid(gate_val) * shared_mlp_out
  Prefill: [4096, 1] * [4096, 2048] → [4096, 2048]   (broadcast)
  Decode:  [1, 1]    * [1, 2048]    → [1, 2048]
  │
  ▼
shared_out [T, 2048]
```

##### 合并

```
⑦ moe_output = routed_out + shared_out
  Prefill: [4096, 2048] + [4096, 2048] → [4096, 2048]
  Decode:  [1, 2048]    + [1, 2048]    → [1, 2048]
```

#### 4.5.2 vLLM 实际执行流

上面 4.5.0/4.5.1 描述的是架构层面的数据依赖关系。
vLLM 中的实际执行顺序与架构图有以下差异：

##### 相关源文件

- `vllm/model_executor/models/qwen3_next.py` — `Qwen3NextSparseMoeBlock`
- `vllm/model_executor/models/qwen2_moe.py` — `Qwen2MoeMLP`（即 `Qwen3NextMLP`）
- `vllm/model_executor/layers/fused_moe/layer.py` — `FusedMoE`
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` — `MoERunner`
- `vllm/model_executor/layers/fused_moe/runner/shared_experts.py` — `SharedExperts`

##### 调用链

```python
# Qwen3NextSparseMoeBlock.__init__:
self.gate = ReplicatedLinear(2048 → 256)
self.shared_expert_gate = ReplicatedLinear(2048 → 1)
self.shared_expert = Qwen3NextMLP(..., expert_gate=self.shared_expert_gate)
self.experts = FusedMoE(gate=self.gate, shared_experts=self.shared_expert, ...)

# Qwen3NextSparseMoeBlock.forward:
#   gate 传给了 FusedMoE → MoERunner.gate，is_internal_router=True
#   所以 forward 中不单独调用 gate，而是把原始 hidden_states 作为 router_logits 传入
final_hidden_states = self.experts(
    hidden_states=hidden_states, router_logits=hidden_states
)
```

##### Shared Expert 在 `Qwen2MoeMLP.forward` 中的执行顺序

架构图中 `shared_expert_gate(x)` 与 `gate_up_proj(x)` 互不依赖，
但 Python 代码的实际执行顺序是 **gate_up → SiluAndMul → down → shared_expert_gate → sigmoid×out**：

```python
# vllm/model_executor/models/qwen2_moe.py:114
def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)                           # ④a
    out = self.act_fn(gate_up)                                  # ④b
    out, _ = self.down_proj(out)                                # ④c
    if self.expert_gate is not None:
        out = F.sigmoid(self.expert_gate(x)[0]) * out           # ⑤⑥
    return out
```

> `shared_expert_gate(x)` 在 `down_proj` **之后**执行，传入的是**原始输入 x**（非中间结果）。

##### Routed / Shared 两路的调度

`MoERunner._forward_impl` 中的执行流：

```
x [T, 2048]
  │
  ▼
gate(x) → router_logits [T, 256]          ← 始终在 main stream 最先执行
  │
  │  运行时根据 SharedExpertsOrder 选择 shared expert 执行策略：
  │
  ├─── 策略 A: NO_OVERLAP ────────────────────────────────────────────────┐
  │    shared expert 先于 routed experts 执行（串行，main stream）        │
  │                                                                       │
  │    shared_expert(x) → shared_out                                      │
  │    ↓                                                                  │
  │    select_experts → fused_moe_kernel → routed_out                     │
  │    ↓                                                                  │
  │    shared_out + routed_out → output                                   │
  │                                                                       │
  ├─── 策略 B: MULTI_STREAM_OVERLAPPED ──────────────────────────────────┐
  │    shared expert 与 routed experts 在不同 CUDA stream 上并行执行      │
  │                                                                       │
  │    main stream:                   aux stream:                         │
  │      select_experts                 shared_expert(x)                  │
  │      fused_moe_kernel               → shared_out                     │
  │      → routed_out                                                     │
  │              │                           │                            │
  │              └─── stream sync ───────────┘                            │
  │                         │                                             │
  │                         ▼                                             │
  │              shared_out + routed_out → output                         │
  └───────────────────────────────────────────────────────────────────────┘
  │
  ▼
MoE 输出 [T, 2048]
```

> **策略选择条件**（`SharedExperts._determine_shared_experts_order`）：
> - 默认 `NO_OVERLAP`（串行）
> - 满足以下全部条件时启用 `MULTI_STREAM_OVERLAPPED`（并行）：
>   CUDA 平台 + aux stream 可用 + token 数 ≤ `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`（默认 256）
> - 另有 `MK_INTERNAL_OVERLAPPED` 模式：由 Modular Kernel 内部管理 shared expert

#### 4.5.3 每层激活专家数（bs=1）

- **Prefill**: 4096 token × 8 → 32768 个 token-expert assignment；不同专家数最多 256，是否覆盖全部专家取决于 router logits
- **Decode**: 1 token × 8 → 只激活 8 个专家

> 这是 MoE decode 阶段的关键优势：不管总专家多少，每步只算 `top_k=8` 个小 MLP（`512×2048` 量级）。

---

### 4.6 Final RMSNorm + LM Head

| 操作 | shape |
|---|---|
| `GemmaRMSNorm(2048)` | `[T, 2048]` |
| `ParallelLMHead(2048 → 248320)` (tie_word_embeddings=False，**不共享**) | `[N_logits, 248320]` |
| Prefill 普通采样 | V1 `ModelRunner` 先用 `input_batch.logits_indices` 选采样位置 hidden states；bs=1 通常 `N_logits=1`，若请求 prompt logprobs 可能更多 |
| Decode | `[1, 248320]` |

Sampler 输出下一个 token id。

---

## 5. 整图 Shape 汇总

### 5.1 Prefill (T=4096)
```
input_ids [4096]
  → embed → [4096, 2048]
  → 40 × DecoderLayer
      linear_attn 层 (30):
         qkvz_proj: [4096,2048] → [4096,12288]，split qkv=[4096,8192], z=[4096,4096]
         conv1d:   [4096,8192] → [4096,8192]
         ba_proj:  [4096,2048] → [4096,64]
         gdn_core:                [4096,32,128]
         out_proj: [4096,4096] → [4096,2048]
      full_attn 层 (10):
         qkv_proj: [4096,2048] → [4096,9216]
         Q:[4096,16,256] K:[4096,2,256] V:[4096,2,256]
         RoPE(partial dim=64)
         flash_attn out: [4096,16,256]
         o_proj:  [4096,4096] → [4096,2048]
      SparseMoE (40):
         gate:         [4096,2048] → [4096,256]
         shared_mlp:   2048→1024→512→2048      out=[4096,2048]
         routed (top8):token 派发 32768 → w13(256,1024,2048) / w2(256,2048,512) → un-permute → [4096,2048]
  → RMSNorm → [4096, 2048]
  → ModelRunner 取 logits_indices → lm_head → 普通采样 [1, 248320]
```

### 5.2 Decode (T=1)
```
input_ids [1]
  → embed → [1, 2048]
  → 40 × DecoderLayer
      linear_attn 层:
         qkvz_proj → [1,12288]，split qkv=[1,8192], z=[1,4096] ; ba→[1,64]
         causal_conv1d_update(state 默认 [3,8192])  → [1,8192]
         gdn_recurrent_update(state[32,128,128] fp32) → [1,32,128]
         out_proj → [1,2048]
      full_attn 层:
         qkv_proj → [1,9216]
         Q:[1,16,256] K:[1,2,256] V:[1,2,256]
         PagedAttention(读 KV cache)
         o_proj → [1,2048]
      SparseMoE:
         gate → [1,256]  top-8
         shared_mlp → [1,2048]
         routed: 只算 8 个小 MLP → [1,2048]
  → RMSNorm → [1, 2048]
  → lm_head → [1, 248320]
```

---

## 6. 显存 / 计算要点（bs=1）

**权重（NVFP4，group_size=16，每个元素 4 bit + scale 开销）**
- 30 个 linear-attn 层：`in_proj_qkvz 2048×12288 + in_proj_ba 2048×64 + conv1d 8192×4 + out_proj 4096×2048 ≈ 33.7 M params / 层`
- 10 层 full_attn ≈ 2048×9216 + 4096×2048 ≈ 27.3 M params / 层
- 40 层 SparseMoE: routed experts `256 × (2048×1024 + 512×2048) ≈ 805 M params / 层`，再加 shared expert / router / shared gate 约 3.7 M params / 层
- 合计约 **35B 总参数，~3B 激活参数**（A3B 即 activated 3B，精确值取决于是否计入 embedding、视觉塔、量化 scale 等）

**KV Cache（仅 full attention 层消耗，10 层）**
- 每 token 每层：`2 (K+V) × 2 heads × 256 dim × 2B (bf16) = 2048 B`
- 10 层 → 每 token 20 KB
- Prefill 4096 token：≈ **80 MB** 的 KV cache（单请求）
- Decode 每步新增 20 KB

**Mamba-like State（linear attention 30 层，每请求固定大小）**
- `conv_state`: 30 × 3 × 8192 × 2B ≈ **1.4 MB**（默认 SD 布局）
- `recurrent_state`: 30 × 32 × 128 × 128 × 4B (fp32) ≈ **60 MB**
- 合计约 **61 MB / request**，与序列长度无关

**Decode 阶段计算瓶颈**
- Full attention 层：受 KV cache 带宽限制
- Linear attention 层：state 加载 + 少量 GEMM，几乎没有长序列的开销
- MoE：仅激活 top-8 × 2 个矩阵 `[1,2048] @ [2048,1024]` + `[1,512] @ [512,2048]`，bs=1 时是严重的内存带宽 bound（每个 expert 权重只被一个 token 使用）

---

## 7. 与 vLLM Runtime 的交互要点

- **Attention Backend**:
  - Full attention → 根据运行配置选择 `FlashAttn`/`FlashInfer` 等 backend + PagedKV
  - Linear attention → `GatedDeltaNet` custom op（在 `vllm/model_executor/layers/mamba/`），有独立的 `MambaStateManager` 管理 `conv_state` / `recurrent_state`
- **Scheduler**: 同一序列的 full-attn KV block 和 linear-attn state slot 一起调度；`HybridKVCacheManager` 同时管两套。
- **CUDA Graph**: Decode 阶段对这两类 attention 的 kernel 都会捕获，batch=1 常走 graph-replay 路径。
- **Sequence Parallel / EP**: `Qwen3NextSparseMoeBlock` 支持 `use_sequence_parallel_moe`；`FusedMoE` 支持 EP（专家切到不同 rank），本文 bs=1、TP=1 未启用。
