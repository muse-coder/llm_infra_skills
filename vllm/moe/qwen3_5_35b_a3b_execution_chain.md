# Qwen3.5-35B-A3B (MoE) 在 vLLM 中的执行链路与 Shape 分析

> 配置来源: `qwen3_5_35b_a3b.json`
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
  └─ Qwen3_5ForCausalLM.forward()
       └─ Qwen3_5Model.forward()          (vllm/model_executor/models/qwen3_5.py)
            ├─ VocabParallelEmbedding
            ├─ Qwen3_5DecoderLayer × 40   (layer_type 分发)
            │    ├─ GatedDeltaNetAttention  (linear_attention, 30 层)
            │    │   └─ vllm/model_executor/layers/mamba/gdn_linear_attn.py
            │    └─ Qwen3NextAttention     (full_attention, 10 层)
            │        └─ vllm/model_executor/models/qwen3_next.py
            └─ Qwen3NextSparseMoeBlock     (所有层, mlp_only_layers=[])
                 ├─ gate (router)
                 ├─ shared_expert_gate
                 ├─ Qwen3NextMLP (shared_expert)
                 └─ SharedFusedMoE (256 experts, top-8)
       └─ GemmaRMSNorm (final norm)
  └─ ParallelLMHead  (not tied)  ──▶ logits
       │
       ▼
Sampler / LogitsProcessor ──▶ token
```

核心模块对应文件：
- **入口 / Decoder**: `vllm/model_executor/models/qwen3_5.py`
- **Full Attention / SparseMoeBlock / 共享 MLP**: `vllm/model_executor/models/qwen3_next.py`
- **Dense MLP 基类**: `vllm/model_executor/models/qwen2_moe.py`（`Qwen2MoeMLP`）
- **Linear Attention (GDN)**: `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
- **Causal Conv1D**: `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`
- **FusedMoE**: `vllm/model_executor/layers/fused_moe/{layer.py, shared_fused_moe.py}`
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

每层在做：
```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)   # GemmaRMSNorm, shape 不变
# ——— Attention 分支：linear or full ———
hidden_states = residual + attention_branch_out
residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = residual + Qwen3NextSparseMoeBlock(hidden_states)
```

形状始终是 `[T, 2048]`。

---

### 4.3 Linear Attention 分支（30 层，layer 0/1/2/4/5/6/...）

源文件 `gdn_linear_attn.py`，类 `GatedDeltaNetAttention`，关键派生量：

```
num_k_heads  = 16
num_v_heads  = 32
head_k_dim   = 128      head_v_dim = 128
key_dim      = 2048     value_dim  = 4096
conv_dim     = key_dim*2 + value_dim = 8192
conv_kernel  = 4
```

| 子步骤 | 权重 / 操作 | Prefill (T=4096) 输出 shape | Decode (T=1) 输出 shape |
|---|---|---|---|
| 0. 输入 | — | `[4096, 2048]` | `[1, 2048]` |
| 1. `in_proj_qkv` | `Linear(2048 → 2048+2048+4096=8192)` NVFP4 | `[4096, 8192]` | `[1, 8192]` |
| 2. `in_proj_ba`  | `Linear(2048 → 2*num_v_heads=64)`（b 和 a 各 32） | `[4096, 64]` | `[1, 64]` |
| 3. `in_proj_z`   | `Linear(2048 → value_dim=4096)` NVFP4 | `[4096, 4096]` | `[1, 4096]` |
| 4. Causal Conv1D | `conv1d(kernel=4, channels=8192)`；prefill 用 `causal_conv1d_fn`，decode 用 `causal_conv1d_update` + conv state | `[4096, 8192]` | `[1, 8192]` |
| 5. split → Q/K/V | 按 `[2048, 2048, 4096]` 切 | Q:`[4096,2048]` K:`[4096,2048]` V:`[4096,4096]` | Q:`[1,2048]` K:`[1,2048]` V:`[1,4096]` |
| 6. reshape to heads | Q:`[T,16,128]` K:`[T,16,128]` V:`[T,32,128]` | 同前 | 同前 |
| 7. `b, a = chunk(ba,2)` | `[T, num_v_heads=32]` | `[4096, 32]` x2 | `[1, 32]` x2 |
| 8. GDN recurrent state (每请求持有) | `[num_v_heads=32, head_v_dim=128, head_k_dim=128]` = `[32,128,128]` fp32 | per-request 不随 T 变 | 同前 |
| 9. `gdn_attention_core` (chunk-wise prefill / recurrent decode) | 输出 `core_attn_out` | `[4096, 32, 128]` | `[1, 32, 128]` |
| 10. z reshape | `z.view(T, num_v_heads, head_v_dim)` | `[4096, 32, 128]` | `[1, 32, 128]` |
| 11. `norm(core, z)` (RMSNormGated, gate = silu(z)) | shape 不变 | `[4096, 32, 128]` | `[1, 32, 128]` |
| 12. flatten heads | `.view(T, value_dim=4096)` | `[4096, 4096]` | `[1, 4096]` |
| 13. `out_proj` | `Linear(4096 → 2048)` NVFP4 (RowParallel) | `[4096, 2048]` | `[1, 2048]` |

> Linear Attention 的 KV "cache" 就是 **GDN recurrent state + Conv state**，大小 **不随序列长度线性增长**：
> - `conv_state`: `[batch_cache_slot, conv_dim=8192, conv_kernel-1=3]` bf16
> - `recurrent_state`: `[batch_cache_slot, 32, 128, 128]` fp32

Prefill/Decode 差异：
- **Prefill**: 做 chunk-wise causal scan，token 并行处理 4096 个位置。
- **Decode**: 单 token 递归更新 state，是 O(1)（相对序列长度）的显存读写。

---

### 4.4 Full Attention 分支（10 层，layer 3/7/11/.../39）

源文件 `qwen3_next.py`，类 `Qwen3NextAttention`，`attn_output_gate=True`：

```
num_heads = 16, num_kv_heads = 2, head_dim = 256
q_size  = 16 * 256 = 4096
kv_size = 2  * 256 = 512
qkv_proj 输出维度 = q_size*2 + kv_size*2 = 8192 + 1024 = 9216   (因为有 output gate，Q 通道翻倍)
```

| 子步骤 | 权重 / 操作 | Prefill (T=4096) | Decode (T=1) |
|---|---|---|---|
| 0. 输入 | — | `[4096, 2048]` | `[1, 2048]` |
| 1. `qkv_proj` | `QKVParallelLinear(2048 → 9216)` NVFP4 | `[4096, 9216]` | `[1, 9216]` |
| 2. split | `[q_gate=8192, k=512, v=512]` | `[4096,8192]` `[4096,512]` `[4096,512]` | `[1,8192]` `[1,512]` `[1,512]` |
| 3. `q, gate = chunk(q_gate, 2)` | 各 `[T, 4096]` | `[4096, 4096]` x2 | `[1, 4096]` x2 |
| 4. reshape heads | Q:`[T,16,256]` K:`[T,2,256]` V:`[T,2,256]` | 同前 | 同前 |
| 5. `q_norm` / `k_norm` | Per-head RMSNorm on 256-dim | shape 不变 | shape 不变 |
| 6. RoPE (partial, rot_dim=64) | 对 Q/K 每个 head 前 64 维旋转，后 192 维保留 | shape 不变 | shape 不变 |
| 7. **Attention kernel**（FlashAttention / FlexAttention） | Prefill: causal MHA；Decode: PagedAttention 读取 KV cache | out:`[4096,16,256]` | out:`[1,16,256]` |
| 8. flatten | `.view(T, 4096)` | `[4096, 4096]` | `[1, 4096]` |
| 9. output gate | `out = sigmoid(gate) * out` | `[4096, 4096]` | `[1, 4096]` |
| 10. `o_proj` | `Linear(4096 → 2048)` NVFP4 RowParallel | `[4096, 2048]` | `[1, 2048]` |

**KV Cache (per full-attention layer)**：
- 10 个 full-attention 层 × 每层 K/V
- 每 token 每层 KV: `2 * num_kv_heads * head_dim * bf16 = 2 * 2 * 256 * 2B = 2048 B = 2 KB`
- Prefill 4096 token 写入一次；Decode 每步追加 1 个 token 的 KV。
- vLLM 以 PagedAttention block 为单位管理，通常 `block_size=16`。

---

### 4.5 Qwen3NextSparseMoeBlock（每层都有，40 层）

```
hidden_size = 2048, num_experts = 256, top_k = 8
moe_intermediate_size = 512 (路由专家)
shared_expert_intermediate_size = 512 (共享专家)
```

#### (a) Router（`gate`）
| 操作 | 权重 | Prefill | Decode |
|---|---|---|---|
| `gate = ReplicatedLinear(2048 → 256)` | `[256, 2048]` bf16（router 不量化） | router_logits `[4096, 256]` | `[1, 256]` |

#### (b) Top-K 选择（`select_experts`）
| 输出 | Prefill | Decode |
|---|---|---|
| `topk_weights` | `[4096, 8]` fp32（renormalize） | `[1, 8]` |
| `topk_ids`     | `[4096, 8]` int32 | `[1, 8]` |

#### (c) Shared Expert（并行于 routed experts）
`Qwen3NextMLP(hidden=2048, intermediate=512, silu)` + `shared_expert_gate: Linear(2048→1)`

| 操作 | 权重 | Prefill | Decode |
|---|---|---|---|
| `gate_up_proj` | `Linear(2048 → 2×512=1024)` NVFP4 | `[4096, 1024]` | `[1, 1024]` |
| `SiluAndMul` (SwiGLU) | split 后 silu*up | `[4096, 512]` | `[1, 512]` |
| `down_proj` | `Linear(512 → 2048)` NVFP4 | `[4096, 2048]` | `[1, 2048]` |
| `shared_expert_gate(x)` → sigmoid 乘 | `[T, 1]` | `[4096, 1]` | `[1, 1]` |
| 共享专家输出 | `sigmoid(gate)*down_out` | `[4096, 2048]` | `[1, 2048]` |

#### (d) Routed Experts (`SharedFusedMoE`)
每个专家是个 `SwiGLU MLP`，权重打包为分组张量：
- `w13` (gate_up 合并)：`[num_experts=256, 2*intermediate=1024, hidden=2048]` NVFP4
- `w2`  (down)：`[num_experts=256, hidden=2048, intermediate=512]` NVFP4

Prefill/Decode 逻辑一致，按 token 派发到 top-8 专家：

| 步骤 | 操作 | Prefill (T=4096) | Decode (T=1) |
|---|---|---|---|
| 1. 派发 | 根据 `topk_ids` permute 得到 expert-major 顺序 | `[T*top_k=32768, 2048]` | `[8, 2048]` |
| 2. 按 expert 分段计算 `w13 @ x` | 每个 token 实际只经过 8 个专家 | 聚合 `[32768, 1024]` | `[8, 1024]` |
| 3. `SiluAndMul` | SwiGLU | `[32768, 512]` | `[8, 512]` |
| 4. `w2 @ x` | down proj | `[32768, 2048]` | `[8, 2048]` |
| 5. un-permute + weighted sum by `topk_weights` | 每 token 聚合 8 个专家 | `[4096, 2048]` | `[1, 2048]` |

最终 SparseMoeBlock 输出：`shared_out + routed_out`，shape `[T, 2048]`。

**每层 MoE 激活的专家数（bs=1）**：
- Prefill: 4096 token × 8 → 最多 32768 个 token-expert pair（实际 256 个专家都会被激活，但负载不均匀）
- Decode: 1 token × 8 → 只激活 8 个专家

> 这是 MoE decode 阶段的关键优势：不管总专家多少，每步只算 `top_k=8` 个小 MLP（`512×2048` 量级）。

---

### 4.6 Final RMSNorm + LM Head

| 操作 | shape |
|---|---|
| `GemmaRMSNorm(2048)` | `[T, 2048]` |
| `ParallelLMHead(2048 → 248320)` (tie_word_embeddings=False，**不共享**) | `[T, 248320]` |
| Prefill 只对最后一个 token 取 logits（vLLM 的 `LogitsProcessor` 只 slice 采样位置） | `[1, 248320]` (sampling 位置数 = bs = 1) |
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
         qkv_proj: [4096,2048] → [4096,8192]
         conv1d:   [4096,8192] → [4096,8192]
         z_proj:   [4096,2048] → [4096,4096]
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
  → lm_head → 取最后一个 token → [1, 248320]
```

### 5.2 Decode (T=1)
```
input_ids [1]
  → embed → [1, 2048]
  → 40 × DecoderLayer
      linear_attn 层:
         qkv_proj → [1,8192] ; z→[1,4096] ; ba→[1,64]
         causal_conv1d_update(state[8192,3])  → [1,8192]
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
- 40 层 × (linear_attn 权重 ≈ 2048×8192 + 2048×4096 + 2048×64 + 8192×4 + 4096×2048 ≈ 37.7 M params / 层)
- 10 层 full_attn ≈ 2048×9216 + 4096×2048 ≈ 27.3 M params / 层
- 40 层 SparseMoE: 256 × 2×(2048×512)×2 + shared 2×(2048×512)×2 + gate + shared_gate ≈ ~1.08 B params / 层 的 routed 权重
- 合计约 **35B 总参数，~3B 激活参数**（A3B 即 activated 3B）

**KV Cache（仅 full attention 层消耗，10 层）**
- 每 token 每层：`2 (K+V) × 2 heads × 256 dim × 2B (bf16) = 2048 B`
- 10 层 → 每 token 20 KB
- Prefill 4096 token：≈ **80 MB** 的 KV cache（单请求）
- Decode 每步新增 20 KB

**Mamba-like State（linear attention 30 层，每请求固定大小）**
- `conv_state`: 30 × 8192 × 3 × 2B ≈ **1.4 MB**
- `recurrent_state`: 30 × 32 × 128 × 128 × 4B (fp32) ≈ **60 MB**
- 合计约 **61 MB / request**，与序列长度无关

**Decode 阶段计算瓶颈**
- Full attention 层：受 KV cache 带宽限制
- Linear attention 层：state 加载 + 少量 GEMM，几乎没有长序列的开销
- MoE：仅激活 top-8 × 2 个矩阵 `[1,2048] @ [2048,1024]` + `[1,512] @ [512,2048]`，bs=1 时是严重的内存带宽 bound（每个 expert 权重只被一个 token 使用）

---

## 7. 与 vLLM Runtime 的交互要点

- **Attention Backend**:
  - Full attention → `FlashAttn`/`FlashInfer` + PagedKV
  - Linear attention → `GatedDeltaNet` custom op（在 `vllm/model_executor/layers/mamba/`），有独立的 `MambaStateManager` 管理 `conv_state` / `recurrent_state`
- **Scheduler**: 同一序列的 full-attn KV block 和 linear-attn state slot 一起调度；`HybridKVCacheManager` 同时管两套。
- **CUDA Graph**: Decode 阶段对这两类 attention 的 kernel 都会捕获，batch=1 常走 graph-replay 路径。
- **Sequence Parallel / EP**: `Qwen3NextSparseMoeBlock` 支持 `use_sequence_parallel_moe`；`SharedFusedMoE` 支持 EP（专家切到不同 rank），本文 bs=1、TP=1 未启用。
