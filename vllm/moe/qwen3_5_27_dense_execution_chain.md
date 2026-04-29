# Qwen3.5-27B-Dense 在 vLLM 中的执行链路与 Shape 分析

> 配置来源: `qwen3_5_27_dense.json`
> 场景: **prefill token = 4096, decode token = 1, batch size = 1**，TP=1，dtype=bfloat16（权重为 NVFP4 量化）

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
| `linear_num_value_heads` | **48** | （MoE 版是 32，这里更多） |
| `linear_key_head_dim` | 128 | |
| `linear_value_head_dim` | 128 | |
| `linear_conv_kernel_dim` | 4 | Causal Conv1D 核大小 |
| `mlp_only_layers` | `[]` | 所有层都走 Dense FFN |
| `vocab_size` | 248320 | |
| `mtp_num_hidden_layers` | 1 | |

**派生尺寸：**
- **Linear Attention**: `key_dim = 16 × 128 = 2048`，`value_dim = 48 × 128 = 6144`
- **Full Attention**: `q_size = 24 × 256 = 6144`（with output gate 时 Q 通道翻倍），`kv_size = 4 × 256 = 1024`
- **Dense FFN**: `intermediate = 17408`
- **RoPE 实际旋转维度**: `256 × 0.25 = 64`
- **激活量化**: NVFP4 group_size=16

> 与 MoE 版最大的不同：**hidden_size 从 2048 翻到 5120，FFN 从 MoE(256专家×512) 变成 Dense(1 × 17408)，linear_num_value_heads 从 32 升到 48**，层数从 40 升到 64。

---

## 2. vLLM 入口模块链路概览

```
LLMEngine → ModelRunner → Qwen3_5ForConditionalGeneration.forward
  └─ Qwen3_5ForCausalLM.forward
       └─ Qwen3_5Model.forward                 (vllm/model_executor/models/qwen3_5.py)
            ├─ VocabParallelEmbedding
            ├─ Qwen3_5DecoderLayer × 64   (layer_type 分发)
            │    ├─ GatedDeltaNetAttention  (linear_attention, 48 层)
            │    │   └─ vllm/model_executor/layers/mamba/gdn_linear_attn.py
            │    ├─ Qwen3NextAttention     (full_attention, 16 层)
            │    │   └─ vllm/model_executor/models/qwen3_next.py
            │    └─ Qwen3NextMLP            (Dense FFN，所有层都走)
            │        └─ 基类 Qwen2MoeMLP in vllm/model_executor/models/qwen2_moe.py
            └─ GemmaRMSNorm
       └─ ParallelLMHead  (not tied) → logits
```

> Dense 版和 MoE 版共享 **完全相同的 Attention 代码路径**（`GatedDeltaNetAttention` / `Qwen3NextAttention`），区别仅在于 FFN 分支：Dense 用 `Qwen3NextMLP`（单条 SwiGLU），MoE 用 `Qwen3NextSparseMoeBlock`（256 expert + shared）。

---

## 3. Shape 约定

| 阶段 | `num_tokens` | `input_ids` shape |
|---|---|---|
| Prefill | **4096** | `[4096]` |
| Decode  | **1**    | `[1]` |

以下 `T = num_tokens`。

---

## 4. 逐模块 Shape 变换

### 4.1 Embedding

`VocabParallelEmbedding(vocab=248320, hidden=5120)`

| 张量 | Prefill (T=4096) | Decode (T=1) |
|---|---|---|
| input_ids | `[4096]` | `[1]` |
| positions | `[4096]` | `[1]` |
| hidden_states | `[4096, 5120]` bf16 | `[1, 5120]` bf16 |

---

### 4.2 DecoderLayer 主干（每层都是）

```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)   # shape 不变
hidden_states = residual + attention_branch(hidden_states)   # linear or full
residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = residual + mlp(hidden_states)     # Qwen3NextMLP
```

形状始终是 `[T, 5120]`。

---

### 4.3 Linear Attention 分支（48 层）

```
num_k_heads = 16    num_v_heads = 48
head_k_dim  = 128   head_v_dim  = 128
key_dim     = 2048  value_dim   = 6144
conv_dim    = key_dim*2 + value_dim = 10240
conv_kernel = 4
```

| 子步骤 | 权重 / 操作 | Prefill (T=4096) | Decode (T=1) |
|---|---|---|---|
| 0. 输入 | — | `[4096, 5120]` | `[1, 5120]` |
| 1. `in_proj_qkv` | `Linear(5120 → 2048+2048+6144=10240)` NVFP4 | `[4096, 10240]` | `[1, 10240]` |
| 2. `in_proj_ba`  | `Linear(5120 → 2*num_v_heads=96)` | `[4096, 96]` | `[1, 96]` |
| 3. `in_proj_z`   | `Linear(5120 → value_dim=6144)` NVFP4 | `[4096, 6144]` | `[1, 6144]` |
| 4. Causal Conv1D | `conv1d(kernel=4, channels=10240)` | `[4096, 10240]` | `[1, 10240]` |
| 5. split → Q/K/V | 按 `[2048, 2048, 6144]` 切 | Q/K `[4096,2048]`, V `[4096,6144]` | Q/K `[1,2048]`, V `[1,6144]` |
| 6. reshape heads | Q:`[T,16,128]` K:`[T,16,128]` V:`[T,48,128]` | 同前 | 同前 |
| 7. `b, a = chunk(ba,2)` | `[T, 48]` x2 | `[4096,48]` x2 | `[1,48]` x2 |
| 8. GDN state | `[num_v_heads=48, head_v_dim=128, head_k_dim=128]` = `[48,128,128]` fp32 per-request | 不随 T 变 | 不随 T 变 |
| 9. `gdn_attention_core` | chunk prefill / recurrent decode | `[4096, 48, 128]` | `[1, 48, 128]` |
| 10. z reshape | `[T, 48, 128]` | `[4096, 48, 128]` | `[1, 48, 128]` |
| 11. `norm(core, z)` (RMSNormGated) | shape 不变 | `[4096, 48, 128]` | `[1, 48, 128]` |
| 12. flatten heads | `[T, 6144]` | `[4096, 6144]` | `[1, 6144]` |
| 13. `out_proj` | `Linear(6144 → 5120)` NVFP4 RowParallel | `[4096, 5120]` | `[1, 5120]` |

**State 存储（每请求，恒定大小）**：
- `conv_state`: `[conv_dim=10240, conv_kernel-1=3]` bf16
- `recurrent_state`: `[48, 128, 128]` fp32

---

### 4.4 Full Attention 分支（16 层）

```
num_heads=24, num_kv_heads=4, head_dim=256
q_size  = 24 × 256 = 6144
kv_size = 4  × 256 = 1024
qkv_proj 输出 = q_size*2 + kv_size*2 = 12288 + 2048 = 14336
```

| 子步骤 | 权重 / 操作 | Prefill (T=4096) | Decode (T=1) |
|---|---|---|---|
| 0. 输入 | — | `[4096, 5120]` | `[1, 5120]` |
| 1. `qkv_proj` | `QKVParallelLinear(5120 → 14336)` NVFP4 | `[4096, 14336]` | `[1, 14336]` |
| 2. split | `[q_gate=12288, k=1024, v=1024]` | `[4096,12288]` `[4096,1024]` `[4096,1024]` | `[1,12288]` `[1,1024]` `[1,1024]` |
| 3. `q, gate = chunk(q_gate, 2)` | 各 `[T, 6144]` | `[4096,6144]` x2 | `[1,6144]` x2 |
| 4. reshape heads | Q:`[T,24,256]` K:`[T,4,256]` V:`[T,4,256]` | 同前 | 同前 |
| 5. `q_norm` / `k_norm` | Per-head RMSNorm on 256-dim | shape 不变 | shape 不变 |
| 6. RoPE（rot_dim=64） | 前 64 维旋转 | shape 不变 | shape 不变 |
| 7. **Attention kernel** | FlashAttn (prefill) / PagedAttention (decode) | out:`[4096,24,256]` | out:`[1,24,256]` |
| 8. flatten | `[T, 6144]` | `[4096, 6144]` | `[1, 6144]` |
| 9. output gate | `sigmoid(gate) * out` | `[4096, 6144]` | `[1, 6144]` |
| 10. `o_proj` | `Linear(6144 → 5120)` NVFP4 RowParallel | `[4096, 5120]` | `[1, 5120]` |

**KV Cache (per full-attention layer)**：
- 每 token 每层 KV: `2 × num_kv_heads × head_dim × bf16 = 2 × 4 × 256 × 2B = 4096 B = 4 KB`
- 16 层合计：每 token **64 KB**
- Prefill 4096 token：≈ **256 MB** KV cache（单请求）
- Decode 每步新增 64 KB

---

### 4.5 Dense FFN (`Qwen3NextMLP`，64 层，每层都算)

`Qwen3NextMLP` 在 Dense 版继承自 `Qwen2MoeMLP`（`vllm/model_executor/models/qwen2_moe.py`），`expert_gate=None`：

```
hidden_size = 5120, intermediate_size = 17408, hidden_act = silu
```

| 操作 | 权重 | Prefill (T=4096) | Decode (T=1) |
|---|---|---|---|
| `gate_up_proj` | `MergedColumnParallelLinear(5120 → 2×17408=34816)` NVFP4 | `[4096, 34816]` | `[1, 34816]` |
| `SiluAndMul` (SwiGLU) | split `(gate, up)` 各 17408，返回 `silu(gate)*up` | `[4096, 17408]` | `[1, 17408]` |
| `down_proj` | `RowParallelLinear(17408 → 5120)` NVFP4 | `[4096, 5120]` | `[1, 5120]` |

Dense FFN 总权重（单层）：
- `gate_up_proj`: `34816 × 5120 ≈ 178 M params`
- `down_proj`:    `5120 × 17408 ≈ 89 M params`
- 单层 FFN ≈ **267 M params**，64 层 ≈ **17 B params** —— 这是 Dense 27B 模型参数的主体。

---

### 4.6 Final RMSNorm + LM Head

| 操作 | shape |
|---|---|
| `GemmaRMSNorm(5120)` | `[T, 5120]` |
| `ParallelLMHead(5120 → 248320)`（not tied） | `[T, 248320]` |
| Prefill 仅取最后 token | `[1, 248320]` |
| Decode | `[1, 248320]` |

---

## 5. 整图 Shape 汇总

### 5.1 Prefill (T=4096)
```
input_ids [4096]
  → embed → [4096, 5120]
  → 64 × DecoderLayer
      linear_attn 层 (48):
         qkv_proj: [4096,5120] → [4096,10240]
         conv1d:   [4096,10240] → [4096,10240]
         z_proj:   [4096,5120] → [4096,6144]
         ba_proj:  [4096,5120] → [4096,96]
         gdn_core:                [4096,48,128]
         out_proj: [4096,6144] → [4096,5120]
      full_attn 层 (16):
         qkv_proj: [4096,5120] → [4096,14336]
         Q:[4096,24,256] K:[4096,4,256] V:[4096,4,256]
         RoPE(partial dim=64)
         flash_attn out: [4096,24,256]
         o_proj:  [4096,6144] → [4096,5120]
      Dense MLP (64):
         gate_up: [4096,5120] → [4096,34816]
         silu*up → [4096, 17408]
         down:    [4096,17408] → [4096,5120]
  → RMSNorm → [4096, 5120]
  → lm_head → 取最后 token → [1, 248320]
```

### 5.2 Decode (T=1)
```
input_ids [1]
  → embed → [1, 5120]
  → 64 × DecoderLayer
      linear_attn 层:
         qkv_proj → [1,10240] ; z→[1,6144] ; ba→[1,96]
         causal_conv1d_update(state[10240,3])  → [1,10240]
         gdn_recurrent_update(state[48,128,128] fp32) → [1,48,128]
         out_proj → [1,5120]
      full_attn 层:
         qkv_proj → [1,14336]
         Q:[1,24,256] K:[1,4,256] V:[1,4,256]
         PagedAttention (读 KV cache)
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

**权重（NVFP4）**
- 总参数 ~27B（Dense）
- Dense FFN 主导：64 × 267 M ≈ **17 B** params
- Attention（linear 48 + full 16）约 **10 B** params
- FP4 近似按 0.5 B/param 估算，~13 GB 权重（还需加上 scale、embedding、lm_head）
- Embedding + lm_head: `2 × 248320 × 5120 × 2B ≈ 5 GB`（bf16，不量化）

**KV Cache（仅 full attention 16 层）**
- 每 token 每层：`2 × 4 × 256 × 2B = 4 KB`
- 16 层 → 每 token 64 KB
- Prefill 4096 token：**256 MB**（单请求）
- Decode 每步 +64 KB

**Mamba-like State（linear attention 48 层，每请求固定）**
- `conv_state`: 48 × 10240 × 3 × 2B ≈ **2.8 MB**
- `recurrent_state`: 48 × 48 × 128 × 128 × 4B ≈ **150 MB**
- 合计 ~**153 MB / request**，与序列长度无关

**Decode 阶段计算特征（bs=1）**
- Dense FFN 每层两次大 GEMM，是 decode 的核心带宽消耗来源：
  - `[1,5120] @ [5120,34816]` → 载入 ~178 M NVFP4 权重
  - `[1,17408] @ [17408,5120]` → 载入 ~89 M NVFP4 权重
  - 每层约 **133 MB 权重搬运**（FP4 打包 + scale），64 层 → 每 decode step ~**8.5 GB** 的权重读取
- 对比 MoE 27B（A3B）每 step 只读 ~3 B 的 activated 权重，**Dense 的 decode 带宽压力显著更高**。
- Full attention 16 层：`[1,24,256]` vs 当前 KV cache 读出 → 受 KV cache 增长影响
- Linear attention 48 层：state 小且恒定，decode 开销可预测

---

## 7. 与 MoE 版（35B-A3B）的关键差异对比

| 维度 | Dense 27B | MoE 35B-A3B |
|---|---|---|
| hidden_size | **5120** | 2048 |
| 层数 | **64** | 40 |
| attention heads | 24 (Q) / 4 (KV) | 16 (Q) / 2 (KV) |
| head_dim | 256 | 256 |
| Q/KV 尺寸 | 6144 / 1024 | 4096 / 512 |
| linear_num_value_heads | **48** | 32 |
| linear value_dim | 6144 | 4096 |
| FFN 类型 | Dense SwiGLU | 256 Expert MoE (top-8) + 1 Shared |
| FFN intermediate | **17408** | 512 (per expert) |
| Full attention 层数 | 16 | 10 |
| 总参数 / 激活参数 | 27B / 27B | 35B / ~3B |
| 每 token KV 总大小 | 64 KB | 20 KB |
| Prefill 4096 KV 占用 | 256 MB | 80 MB |
| Linear state per request | 153 MB | 61 MB |
| Decode 每步权重带宽 | ~8.5 GB | ~1.5 GB（top-8 experts + shared + attention） |
| Decode 延迟特征 | 严重 memory-bound | experts 读取虽少但 scatter-gather 有 latency |

**结论**：
- **Prefill 4096**: Dense 版计算量远大（FFN 每层 178M+89M 权重×4096 tokens），但 MoE 版需要把 4096 token 分派到 256 个专家，`permute/unpermute + grouped-gemm` 的实现效率决定吞吐。
- **Decode 1**: Dense 是典型的 bandwidth-bound；MoE 因为只激活 top-8 专家，理论 FLOPs 和权重搬运都小 5-6 倍，非常适合低并发场景。
- 两模型都用 **同一套 Attention 代码路径**（hybrid linear + full），**只有 FFN 分支不同**，因此在 vLLM 中的 scheduler / kv-cache-manager / mamba-state-manager 都是复用的。
