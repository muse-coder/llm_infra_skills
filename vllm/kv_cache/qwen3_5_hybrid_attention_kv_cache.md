# Qwen3.5 Hybrid Attention 架构与 KV Cache 管理知识库

> 基于 vLLM 真实源码分析，覆盖模型架构实现、Self Attention / Linear Attention KV Cache 管理、Hybrid Block 对齐机制、KV Cache 调度器、FP8 量化适配等完整主题。

---

## 目录

**Part 1：架构概览**
1. [整体架构概览](#1-整体架构概览)
2. [Hybrid Attention 实现](#2-hybrid-attention-实现)
3. [GatedDeltaNet 状态管理](#3-gateddeltanet-状态管理)

**Part 2：Self Attention KV Cache 管理**
4. [Self Attention KV Cache：FullAttentionSpec](#4-self-attention-kv-cache-fullattentionspec)

**Part 3：Linear Attention KV Cache 管理**
5. [Linear Attention KV Cache：MambaSpec](#5-linear-attention-kv-cache-mambaspec)

**Part 4：Hybrid Block 对齐机制**
6. [统一 Block 管理系统](#6-统一-block-管理系统)
7. [Block Size 自动对齐：_align_hybrid_block_size](#7-block-size-自动对齐-_align_hybrid_block_size)

**Part 5：KV Cache 调度器**
8. [KV Cache 调度器架构](#8-kv-cache-调度器架构)
9. [Self Attention 的 Prefix Cache 策略](#9-self-attention-的-prefix-cache-策略)
10. [Linear Attention 的 Prefix Cache 策略](#10-linear-attention-的-prefix-cache-策略)
11. [Hybrid 协调器：迭代固定点算法](#11-hybrid-协调器迭代固定点算法)
12. [Mamba Cache Mode](#12-mamba-cache-mode)

**Part 6：KV Cache 量化适配**
13. [FP8 量化如何适配 Hybrid Attention](#13-fp8-量化如何适配-hybrid-attention)
14. [量化对 Block Size 和 Prefix Cache 的影响](#14-量化对-block-size-和-prefix-cache-的影响)

**Part 7：附录**
15. [完整数据流图](#15-完整数据流图)
16. [参考文件索引](#16-参考文件索引)

---

# Part 1：架构概览

## 1. 整体架构概览

Qwen3.5 采用 **Linear Attention（GatedDeltaNet）+ Full Attention（标准 GQA）** 混合架构：

- **Linear Attention 层**：类 Mamba 的状态空间模型，KV cache 大小固定，不随序列长度增长
- **Full Attention 层**：标准 Transformer 注意力，KV cache 随序列长度线性增长

```
Layer 0,1,2 → linear_attention（GatedDeltaNet）
Layer 3     → full_attention（Qwen3NextAttention）
Layer 4,5,6 → linear_attention
Layer 7     → full_attention
...（每 4 层一个 full_attention，共 10 个 full_attention，30 个 linear_attention）
```

**两种 KV Cache 的本质差异**：

| 维度 | Self Attention（Full） | Linear Attention（GDN） |
|---|---|---|
| **缓存内容** | 每个 token 的 K/V 向量 | 固定大小的状态矩阵（conv_state + ssm_state） |
| **内存增长** | O(序列长度) | O(1)，与序列长度无关 |
| **Spec 类型** | `FullAttentionSpec` | `MambaSpec` |
| **Block 含义** | block_size 个 token 的 KV | 整个请求的一个状态快照 |
| **Prefix Cache** | 必须从头连续命中 | 只需最新状态命中 |

---

## 2. Hybrid Attention 实现

### 2.1 DecoderLayer 的分支逻辑

**文件**：`vllm/model_executor/models/qwen3_5.py`

```python
class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(self, vllm_config, layer_type, prefix):
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(...)  # Linear Attention
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(...)        # Self Attention
```

**Forward 方法**（父类 `Qwen3NextDecoderLayer`）：

```python
def forward(self, hidden_states, residual, positions=None, **kwargs):
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if self.layer_type == "linear_attention":
        self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
    elif self.layer_type == "full_attention":
        self.self_attn(hidden_states=hidden_states, output=self_attention_output, positions=positions)

    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

### 2.2 Full Attention 层：Qwen3NextAttention

**文件**：`vllm/model_executor/models/qwen3_next.py`

标准 GQA 注意力，带 QK-Norm 和 Attention Gate：

```python
class Qwen3NextAttention(nn.Module):
    def forward(self, positions, output, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)

        # QK-Norm（per-head RMSNorm）
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))

        # RoPE 位置编码
        q, k = self.rotary_emb(positions, q, k)

        # 标准注意力计算（同时读写 KV cache）
        attn_output = self.attn(q, k, v)

        # Attention Gate
        gate = torch.sigmoid(gate)
        attn_output = attn_output * gate
        output[:], _ = self.o_proj(attn_output)
```

### 2.3 Linear Attention 层：GatedDeltaNetAttention

**文件**：`vllm/model_executor/layers/mamba/gdn_linear_attn.py`

```python
def forward_cuda(self, hidden_states, output):
    # 1. 输入投影
    mixed_qkv, z = self.in_proj_qkvz(hidden_states).split([qkv_size, z_size], dim=-1)
    b, a = self.in_proj_ba(hidden_states).chunk(2, dim=-1)

    # 2. 核心计算（CUDA op，内部更新 conv_state 和 ssm_state）
    torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, self.prefix)

    # 3. 输出投影
    core_attn_out = self.norm(core_attn_out, z)  # RMSNorm * sigmoid(z)
    output[:], _ = self.out_proj(core_attn_out)
```

---

## 3. GatedDeltaNet 状态管理

### 3.1 状态定义与形状

**文件**：`vllm/model_executor/layers/mamba/mamba_utils.py`

```python
@classmethod
def gated_delta_net_state_shape(cls, tp_world_size, num_k_heads, num_v_heads,
                                 head_k_dim, head_v_dim, conv_kernel_size, num_spec=0):
    conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads

    conv_state_shape = (conv_dim // tp_world_size, conv_kernel_size - 1 + num_spec)
    temporal_state_shape = (num_v_heads // tp_world_size, head_v_dim, head_k_dim)
    return conv_state_shape, temporal_state_shape
```

| 状态 | 形状 | 含义 |
|---|---|---|
| `conv_state` | `(conv_dim, conv_kernel_size-1)` | 短卷积历史缓冲，保存最近几个 token 的 Q/K/V 拼接 |
| `ssm_state` | `(num_v_heads, head_v_dim, head_k_dim)` | Delta Rule 核心状态矩阵 S，压缩了所有历史 KV 信息 |

### 3.2 Delta Rule 数学原理

```
Gated Delta Rule：
  g_t = -exp(A_log) · softplus(a_t + dt_bias)   # 遗忘因子
  β_t = sigmoid(b_t)                              # 学习率
  S_t = exp(g_t) · S_{t-1} + β_t · (v_t - exp(g_t) · S_{t-1} · k_t) · k_t^T

输出：output_t = Q_t · S_t

其中 S_t 就是 ssm_state，压缩了所有历史 KV 信息
```

### 3.3 Forward 核心计算

| 阶段 | 卷积步骤 | 递归步骤 |
|---|---|---|
| Prefill | `causal_conv1d_fn`（并行） | `chunk_gated_delta_rule`（分块并行） |
| Decode | `causal_conv1d_update`（逐 token） | `fused_sigmoid_gating_delta_rule_update`（逐 token） |

---

# Part 2：Self Attention KV Cache 管理

## 4. Self Attention KV Cache：FullAttentionSpec

### 4.1 FullAttentionSpec 定义

**文件**：`vllm/v1/kv_cache_interface.py`

```python
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    block_size: int          # 每个 block 包含的 token 数
    num_kv_heads: int        # KV head 数量（GQA 后）
    head_size: int           # head 维度
    dtype: torch.dtype       # KV cache 存储精度（bf16 或 fp8）
    kv_quant_mode: KVQuantMode  # 量化模式

    @property
    def page_size_bytes(self) -> int:
        # 一个 block 的字节数 = block_size × 每 token 字节数
        return self.block_size * 2 * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)
```

**生成时机**：每个 full_attention 层的 `Attention.get_kv_cache_spec()` 调用时生成：

```python
def get_kv_cache_spec(self, vllm_config):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=self.num_kv_heads,
        head_size=self.head_size,
        dtype=self.kv_cache_torch_dtype,  # 由 cache_config.cache_dtype 决定
        kv_quant_mode=quant_mode,
    )
```

### 4.2 block_size 的含义与管理

**block_size 的含义**：每个 block 存储多少个 token 的 KV 向量。

```
block_size = N 时，一个 block 的物理布局：
┌──────────────────────────────────────────────────────┐
│ token_0_K │ token_0_V │ token_1_K │ token_1_V │ ... │  ← N 个 token
└──────────────────────────────────────────────────────┘
  共 N × 2 × num_kv_heads × head_size × dtype_size bytes
```

**block_size 的来源**：
- 默认值：用户配置（`--block-size 16`，默认 16）
- **Hybrid 模型中会被自动覆盖**：由 `_align_hybrid_block_size()` 根据 mamba state 大小重新计算（详见 Part 4）

### 4.3 block_id 的管理

Self Attention 的 block_id 由 **BlockPool** 统一分配，通过 **BlockTable** 管理每个请求的 block 序列：

```
请求 A 的 BlockTable（attention group）：
  [block_id=3, block_id=7, block_id=12]
  ↑ 表示该请求的 KV cache 存储在这 3 个 block 中

block_id=3 的物理内存：
  GPU 显存偏移 = 3 × page_size_bytes
  存储内容：token_0 到 token_{block_size-1} 的 K/V 向量
```

**BlockTable 数据结构**（`vllm/v1/worker/block_table.py`）：

```python
class BlockTable:
    # 二维张量：[max_num_reqs, max_num_blocks_per_req]
    # block_table[req_idx, block_idx] = block_id
    block_table: torch.Tensor
```

**block_id 分配流程**：
1. 请求到来时，`FullAttentionManager.allocate_new_blocks()` 向 BlockPool 申请 block
2. BlockPool 从 `free_block_queue` 中取出空闲 block，返回 block_id
3. block_id 写入该请求的 BlockTable 对应行
4. 请求结束时，block 的 ref_cnt 减 1，降为 0 时归还 free_block_queue

### 4.4 page_size 的计算

```python
# FullAttentionSpec.page_size_bytes
page_size = block_size × 2 × num_kv_heads × head_size × dtype_size

# bf16，block_size=16，num_kv_heads=2，head_size=256：
page_size = 16 × 2 × 2 × 256 × 2 = 32,768 bytes = 32 KB

# fp8，block_size=16，num_kv_heads=2，head_size=256：
page_size = 16 × 2 × 2 × 256 × 1 = 16,384 bytes = 16 KB
```

**page_size 的作用**：
- 决定 GPU 显存中每个 block 的物理大小
- 决定总 block 数量：`num_blocks = total_gpu_memory / page_size`
- **Hybrid 模型中**：attention page_size 必须 ≥ mamba page_size（详见 Part 4）

### 4.5 KV Cache 的读写时机

```
Prefill 阶段：
  Qwen3NextAttention.forward()
    → self.attn(q, k, v)
      → TritonAttentionBackend.reshape_and_cache()  ← 写入 KV cache
      → TritonAttentionBackend.unified_attention()  ← 读取 KV cache 计算注意力

Decode 阶段：
  同上，但每次只写入 1 个 token 的 KV，读取所有历史 token 的 KV
```

---

# Part 3：Linear Attention KV Cache 管理

## 5. Linear Attention KV Cache：MambaSpec

### 5.1 MambaSpec 定义

**文件**：`vllm/v1/kv_cache_interface.py`

```python
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]  # (conv_state_shape, ssm_state_shape)
    dtypes: tuple[torch.dtype]           # 各状态的 dtype
    block_size: int                      # 通常为 1（每个 block 是一个完整状态）
    mamba_type: str                      # "gdn_attention"
    mamba_cache_mode: str                # "none" / "align" / "all"
    num_speculative_blocks: int          # 投机解码额外 block 数

    @property
    def page_size_bytes(self) -> int:
        # 所有状态的字节数之和
        return sum(
            math.prod(shape) * get_dtype_size(dtype)
            for shape, dtype in zip(self.shapes, self.dtypes)
        )
```

**生成时机**：每个 linear_attention 层的 `MambaBase.get_kv_cache_spec()` 调用时生成：

```python
def get_kv_cache_spec(self, vllm_config):
    return MambaSpec(
        shapes=self.get_state_shape(),    # (conv_state_shape, ssm_state_shape)
        dtypes=self.get_state_dtype(),    # 由模型配置决定
        block_size=mamba_block_size,
        mamba_type=self.mamba_type,
        mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
    )
```

### 5.2 block_size 的含义与管理

**Linear Attention 的 block_size 含义与 Self Attention 完全不同**：

```
Self Attention：block_size = 每个 block 包含的 token 数（动态 KV）
Linear Attention：block_size = 每个 block 是一个完整的状态快照（固定大小）

mamba_cache_mode="none"（默认）：
  每个请求只有 1 个 block（当前状态）
  block_size 的值不影响实际存储，只影响 prefix cache 的 hash 粒度

mamba_cache_mode="align"：
  每个请求有 2 个 block（当前状态 + 上一步状态，用于投机解码）

mamba_cache_mode="all"（Qwen3.5 不支持）：
  缓存所有 token 的状态，block_size 才真正有意义
```

### 5.3 block_id 的管理

Linear Attention 的 block_id 同样由 **BlockPool** 统一分配，但管理方式不同：

```
请求 A 的 BlockTable（mamba group）：
  [block_id=1]  ← 只有 1 个 block，存储整个请求的当前 mamba state

block_id=1 的物理内存：
  GPU 显存偏移 = 1 × mamba_page_size_padded
  存储内容：conv_state (conv_dim, conv_kernel_size-1) + ssm_state (num_v_heads, head_v_dim, head_k_dim)
```

**关键差异**：Self Attention 的 block 数量随序列增长，Linear Attention 的 block 数量固定（通常为 1）。

**block_id 分配流程**：
1. 请求到来时，`MambaManager.allocate_new_blocks()` 向 BlockPool 申请 1 个 block
2. 该 block 在整个请求生命周期内持续被原地更新（不追加新 block）
3. 请求结束时归还 BlockPool

### 5.4 page_size 的计算

```python
# MambaSpec.page_size_bytes（真实状态大小）
page_size = conv_state_bytes + ssm_state_bytes
          = prod(conv_state_shape) × conv_dtype_size
          + prod(ssm_state_shape) × ssm_dtype_size
```

**注意**：mamba 的 page_size 是**固定的**，不随序列长度变化，也不受 KV cache 量化影响（mamba state 始终使用模型原始 dtype）。

### 5.4.1 mamba state 大小为什么固定？

mamba state 的形状完全由**模型架构参数**决定，与序列长度（token 数）无关：

```python
# vllm/model_executor/layers/mamba/mamba_utils.py
@classmethod
def gated_delta_net_state_shape(cls, tp_world_size, num_k_heads, num_v_heads,
                                 head_k_dim, head_v_dim, conv_kernel_size, num_spec=0):
    # conv_state：短卷积历史缓冲，保存最近 conv_kernel_size-1 个 token 的 Q/K/V 拼接
    conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    conv_state_shape = (conv_dim // tp_world_size, conv_kernel_size - 1 + num_spec)
    #                                               ↑ 固定为 3（conv_kernel_size=4 时）

    # ssm_state：Delta Rule 核心状态矩阵，压缩了所有历史 KV 信息
    temporal_state_shape = (num_v_heads // tp_world_size, head_v_dim, head_k_dim)
    #                        ↑ 所有参数都来自模型 config，没有一个是序列长度！
    return conv_state_shape, temporal_state_shape
```

**关键理解**：ssm_state 是一个**原地覆盖更新**的矩阵，每个新 token 来了就更新这个矩阵，而不是追加新的行。无论序列有多长，矩阵形状永远是 `(num_v_heads, head_v_dim, head_k_dim)`。

**Qwen3.5 数值估算**（以 7B 为例，`num_k_heads=num_v_heads=16, head_dim=256, conv_kernel=4`）：

```
conv_state_shape    = (12288, 3)      → 12288×3×2 ≈ 72 KB（bf16）
temporal_state_shape = (16, 256, 256) → 16×256×256×2 ≈ 2 MB（bf16）
总 mamba_page_size  ≈ 2.1 MB（每个请求，每一层）
```

这个 2.1 MB 的固定大小，正是 `_align_hybrid_block_size` 计算 attention block_size 的输入（详见 Part 4）。

### 5.5 mamba_cache_mode 详解

三种模式的**本质区别**在于：mamba state 是**原地覆盖**（只保留最新状态），还是**快照存档**（每隔 block_size 个 token 存一次历史状态）。

#### mode="none"（默认）：原地覆盖，只保留当前状态

```
token 0 → S_0 → 写入 block[0]
token 1 → S_1 → 覆盖写入 block[0]  ← S_0 丢失
token 2 → S_2 → 覆盖写入 block[0]  ← S_1 丢失
...
最终 block[0] 只有 S_N（最新状态）
```

每个请求固定只有 **1 个 block**，block_size 的值对实际存储没有意义（只有 1 个快照，无论 block_size 是多少都一样）。

Prefix Cache 行为：请求 B 命中请求 A 的 block[0]，直接加载 S_N，从第 N+1 个 token 继续推理。只能全命中或全不命中，没有中间粒度。

#### mode="align"：滑动窗口快照，始终保留当前 + 上一步状态（投机解码用）

`align` 模式的 block 数量**不是固定 2 个**，而是随序列增长：

```python
# vllm/v1/core/single_type_kv_cache_manager.py
num_required_blocks = cdiv(num_tokens, block_size) + num_speculative_blocks
```

但其中**绝大多数是 null block（占位符，不占显存）**，真正分配显存的只有最后 `1 + num_speculative_blocks` 个：

```
序列有 3×block_size 个 token 时，req_blocks 的结构：

  [null, null, S_prev, S_current]
   ↑      ↑     ↑        ↑
   占位   占位  上一步   当前状态
   （不占显存）  （回滚用）（推理用）

总 block 列表长度 = ceil(seq_len / block_size) + num_speculative_blocks
真实占显存的 block = 1 + num_speculative_blocks（始终固定）
```

**每一步的滑动过程**：

```
Step 1（prefill 完成）：req_blocks = [S_1]
Step 2（新增一个 block）：req_blocks = [S_1, S_2]
Step 3（再新增）：释放 S_1（两步前，不再需要），变成 null
                  req_blocks = [null, S_2, S_3]
Step 4：          req_blocks = [null, null, S_3, S_4]
```

**实际显存占用**：始终只有 `(1 + num_speculative_blocks) × page_size`，与 `none` 模式相近。null block 只是 block 列表里的占位符，不消耗真实显存。

`align` 模式主要为投机解码（MTP/EAGLE）服务：当投机 token 被拒绝时，可以从 `S_prev` 恢复上一步的状态重新推理。

#### mode="all"：快照存档，每隔 block_size 个 token 存一次状态（Qwen3.5 不支持）

这是与前两种模式**本质不同**的模式。mamba state 不再原地覆盖，而是像 Self Attention 的 KV Cache 一样，**每隔 `block_size` 个 token 存一次状态快照**：

```
token 0..15  → S_15  → 写入 block[0]  ← 第 1 个快照
token 16..31 → S_31  → 写入 block[1]  ← 第 2 个快照
token 32..47 → S_47  → 写入 block[2]  ← 第 3 个快照
...（block_size=16 时，每 16 个 token 存一次）
```

每个请求需要 `ceil(seq_len / block_size)` 个 block，**block_size 在这里才真正有意义**：决定快照粒度（越小越细，内存越大）。

源码印证（`vllm/v1/attention/backends/mamba_attn.py`）：

```python
if self.vllm_config.cache_config.mamba_cache_mode == "all":
    max_num_blocks = cdiv(
        self.vllm_config.model_config.max_model_len,
        self.kv_cache_spec.block_size,   # ← block_size 决定快照粒度
    )
    # 每个请求需要 max_num_blocks 个 block（不是 1 个！）
    self.state_indices_tensor_d = torch.empty(
        (self.decode_cudagraph_max_bs, max_num_blocks),
        dtype=torch.int32, device=device,
    )
else:
    # none/align 模式：每个请求只有 1 个（或 1+spec 个）block
    self.state_indices_tensor_d = torch.empty(
        (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),
        dtype=torch.int32, device=device,
    )
```

Prefix Cache 行为：请求 B 命中了 block[0..30]（480 token 的快照），只需重算最后 32 个 token，比 `none` 模式更细粒度的复用。

#### 三种模式对比

| 模式 | block 数量 | block_size 含义 | Prefix Cache 粒度 | 内存占用 |
|---|---|---|---|---|
| `none`（默认） | 固定 1 个 | 无实际意义 | 全命中或全不命中 | `(1 + num_spec_blocks) × page_size` |
| `align` | 固定 2 个 | 无实际意义 | 同 none，额外支持投机解码回滚 | `(2 + num_spec_blocks) × page_size` |
| `all` | `ceil(seq_len / block_size)` 个 | **每隔多少 token 存一次快照** | 可命中任意 block 边界 | `ceil(max_model_len / block_size) × page_size` |

> **Qwen3.5 只支持 `none` 和 `align`**，`all` 模式会直接抛出 `NotImplementedError`。

### 5.5.1 mamba_cache_mode 由什么决定？

**不是由是否开启投机采样决定，而是由是否开启 prefix caching 决定**。

**文件**：`vllm/model_executor/models/config.py`

```python
if cache_config.enable_prefix_caching:
    if cache_config.mamba_cache_mode == "none":
        # 自动切换：优先 "all"，不支持则降级 "align"
        cache_config.mamba_cache_mode = (
            "all" if model_config.supports_mamba_prefix_caching else "align"
        )
else:
    # prefix caching 关闭 → 强制 "none"，无论用户设了什么
    cache_config.mamba_cache_mode = "none"
```

**Qwen3.5 的实际情况**：

| 启动参数 | mamba_cache_mode 最终值 | 原因 |
|---|---|---|
| 默认（不加任何参数） | `none` | prefix caching 默认关闭 |
| `--enable-prefix-caching` | `align` | Qwen3.5 不支持 `all`，自动降级 |
| 开启投机采样但不开 prefix caching | `none` | prefix caching 未开，强制 none |
| 开启投机采样 + prefix caching | `align` | prefix caching 开了，自动切 align |

**`align` 模式的额外约束**：强制要求开启 chunked prefill（`enable_chunked_prefill=True`），否则启动报错。

**`none` vs `align` 核心差异**：

| 维度 | `none` | `align` |
|---|---|---|
| **block 列表长度** | 固定 1 个 | `ceil(seq_len / block_size) + num_spec_blocks`（随序列增长） |
| **真实占显存 block** | `1 + num_spec_blocks` | `1 + num_spec_blocks`（与 none 相同！） |
| **null block 数** | 0 | `ceil(seq_len / block_size) - 1`（占位符，不占显存） |
| **mamba 实际显存** | `(1 + num_spec_blocks) × page_size` | `(1 + num_spec_blocks) × page_size`（与 none 相近） |
| **mamba_block_size** | `max_model_len`（极粗粒度） | `= block_size`（与 attention 对齐） |
| **prefix cache** | 不支持 | 支持（实验性） |
| **投机解码回滚** | 不支持 | 支持（靠 S_prev 回滚） |
| **chunked prefill** | 不要求 | **强制要求** |

---

### 5.6 Linear Attention 的计算原理

> **核心问题**：Linear Attention 的 Q/K/V 是不是上下文所有 token 的？

**不是**。Linear Attention 的 Q/K/V 只是**当前 chunk（或当前 token）**的投影，历史信息被压缩进了固定大小的 `ssm_state`（状态矩阵 S）。

#### 与 Self Attention 的本质对比

```
Self Attention（Decode 阶段）：
  历史信息载体：KV cache = [K_0, K_1, ..., K_{n-1}]  ← O(n) 大小，随序列增长
  计算：o_t = softmax(Q_t · [K_0..K_{n-1}]^T) · [V_0..V_{n-1}]
  需要读取：所有历史 K、V

Linear Attention（Decode 阶段）：
  历史信息载体：ssm_state S_{n-1}  ← O(1) 大小，固定不变
  计算：
    S_n = exp(g_n) · S_{n-1} + β_n · (v_n - exp(g_n) · S_{n-1} · k_n) · k_n^T
    o_n = Q_n · S_n
  需要读取：只有 S_{n-1}（一个固定大小的矩阵）
```

#### Gated Delta Rule 数学公式

```
每个 token t 的更新：
  遗忘因子：g_t = -exp(A_log) · softplus(a_t + dt_bias)
  学习率：  β_t = sigmoid(b_t)
  状态更新：S_t = exp(g_t) · S_{t-1}
                 + β_t · (v_t - exp(g_t) · S_{t-1} · k_t) · k_t^T
  输出：    o_t = Q_t · S_t
```

`S_t` 是一个 `(num_v_heads, head_v_dim, head_k_dim)` 的矩阵，**大小固定**，每次更新是原地覆盖，不是追加。

#### 直觉理解

- **Self Attention** 像一个**完整的笔记本**，每个 token 都有一页，查询时翻遍所有页 → O(n) 内存
- **Linear Attention** 像一个**不断更新的摘要**，每个新 token 来了就更新摘要，查询时只看摘要 → O(1) 内存

代价是：摘要是**有损压缩**的，越长的序列，早期信息被遗忘得越多（`exp(g_t)` 的遗忘因子）。

| 维度 | Self Attention | Linear Attention |
|---|---|---|
| **历史信息载体** | KV cache（每个 token 的 K/V） | ssm_state（固定大小矩阵 S） |
| **历史信息大小** | O(n)，随序列增长 | O(1)，固定不变 |
| **Decode 计算复杂度** | O(n) | O(1) |
| **信息损失** | 无损，精确保留所有历史 | 有损，历史信息被压缩进 S |

### 5.7 KV Cache 的读写时机

```
Prefill 阶段：
  GatedDeltaNetAttention.forward_cuda()
    → torch.ops.vllm.gdn_attention_core()
      → causal_conv1d_fn()        ← 读写 conv_state（并行处理整个序列）
      → chunk_gated_delta_rule()  ← 读写 ssm_state（分块并行）

Decode 阶段：
  → causal_conv1d_update()                    ← 逐 token 更新 conv_state
  → fused_sigmoid_gating_delta_rule_update()  ← 逐 token 更新 ssm_state
```

**与 Self Attention 的关键区别**：Linear Attention 的 KV cache 是**原地更新**的状态矩阵，不是追加写入的 token KV 向量。

---

# Part 4：Hybrid Block 对齐机制

## 6. 统一 Block 管理系统

### 6.1 BlockPool：统一的物理内存池

**文件**：`vllm/v1/core/block_pool.py`

所有 KV cache（Self Attention 和 Linear Attention）都从**同一个 BlockPool** 分配物理内存：

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int          # 全局唯一 ID，范围 [0, num_gpu_blocks-1]
    ref_cnt: int = 0       # 引用计数，多请求共享同一 block 时 > 1
    _block_hash: BlockHashWithGroupId | None = None  # 仅满 block 时有值（prefix cache 用）
    prev_free_block: "KVCacheBlock | None" = None    # 双向链表，用于 free queue
    next_free_block: "KVCacheBlock | None" = None
    is_null: bool = False  # null block 永不缓存

class BlockPool:
    blocks: list[KVCacheBlock]               # 所有 block 的元数据
    free_block_queue: FreeKVCacheBlockQueue  # 空闲 block 双向链表，O(1) 分配/回收
    cached_block_hash_to_block: BlockHashToBlockMap  # hash → block 查找表（prefix cache）
```

### 6.2 MultiGroupBlockTable：每个 group 独立的 BlockTable

**文件**：`vllm/v1/worker/block_table.py`

虽然物理内存统一，但 Self Attention 和 Linear Attention 各自有**独立的 BlockTable**：

```python
class MultiGroupBlockTable:
    def __init__(self, block_sizes, kernel_block_sizes, ...):
        # 为每个 KV cache group 创建独立的 BlockTable
        self.block_tables = [
            BlockTable(block_size, ...)
            for block_size, kernel_block_size in zip(block_sizes, kernel_block_sizes)
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        # block_ids 是 tuple，每个元素对应一个 group 的 block_id 列表
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)
```

```
物理 GPU 显存（统一 BlockPool，block_id 全局唯一）
┌──────────────────────────────────────────────────────┐
│  Block 0  │  Block 1  │  Block 2  │  Block 3  │ ... │
└──────────────────────────────────────────────────────┘

Self Attention group BlockTable:   [block_id=3, block_id=7, block_id=12, ...]
Linear Attention group BlockTable: [block_id=1, block_id=5, block_id=9,  ...]
  ↑ 各自独立分配，但都来自同一个 BlockPool
```

### 6.3 为什么 block_size 必须对齐？

**Prefix Cache 的 Hash 机制要求两种 cache 的 block 覆盖相同的 token 范围**。

**文件**：`vllm/v1/core/block_pool.py`

```python
# hash key = BlockHash + group_id（区分不同 group 的同一段 token）
def make_block_hash_with_group_id(block_hash, group_id):
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))

# 查找时：同一个 block_hash 必须在所有 group 都命中，才算真正命中
def get_cached_block(self, block_hash, kv_cache_group_ids):
    for group_id in kv_cache_group_ids:
        block = self.cached_block_hash_to_block.get(block_hash + group_id)
        if not block:
            return None  # 任意一个 group miss → 整体 miss
    return cached_blocks
```

**如果 block_size 不对齐会怎样**：

```
假设 attention block_size=16，mamba block_size=32：
  attention 的第 i 个 block 覆盖 token [i×16, (i+1)×16)
  mamba 的第 i 个 block 覆盖 token [i×32, (i+1)×32)
  → 两者覆盖的 token 范围不同
  → 同一个 block_hash 在两个 group 中对应不同的 token 段
  → prefix cache 的 hash 逻辑崩溃
```

**结论**：block_size 必须统一，才能保证第 i 个 block_hash 在所有 group 中覆盖完全相同的 token 范围。

---

## 7. Block Size 自动对齐：_align_hybrid_block_size

### 7.1 对齐目标

**文件**：`vllm/platforms/interface.py`

对齐的核心约束：

```
attn_block_size × attn_page_size_1_token ≥ mamba_page_size
```

即 **attention 一个 block 的总字节数，必须能"装下"一个 mamba state**。这样两者的 block_size 才能统一，prefix cache hash 才能正确对应。

### 7.2 计算公式（mamba_cache_mode="none" 或 "align"，Qwen3.5 走这条路）

```python
# vllm/platforms/interface.py
attn_block_size = kernel_block_alignment_size * cdiv(
    mamba_page_size,
    kernel_block_alignment_size * attn_page_size_1_token,
)
# cdiv(a, b) = ceil(a / b)
# 含义：找到最小的 kernel_block_alignment_size 的整数倍，
#       使得 attn_block_size × attn_page_size_1_token ≥ mamba_page_size

# 强制覆盖用户配置的 block_size
if cache_config.block_size < attn_block_size:
    cache_config.block_size = attn_block_size

# align 模式：mamba_block_size 跟 attention block_size 保持一致
if cache_config.mamba_cache_mode == "align":
    cache_config.mamba_block_size = cache_config.block_size

# pad mamba page_size 到与 attention 完全相等（允许 mamba 内部有少量 padding）
attn_page_size = cache_config.block_size * attn_page_size_1_token
cache_config.mamba_page_size_padded = attn_page_size
```

**三个输入量**：

| 输入 | 来源 | 说明 |
|---|---|---|
| `mamba_page_size` | `MambaSpec.page_size_bytes` | mamba 层的固定状态大小（conv_state + ssm_state 之和） |
| `attn_page_size_1_token` | `FullAttentionSpec(block_size=1).page_size_bytes` | attention 层每个 token 占用的字节数，受 dtype 影响 |
| `kernel_block_alignment_size` | `backend_cls.get_supported_kernel_block_sizes()` | attention kernel 要求的最小 block 对齐单位（通常 16） |

### 7.3 mamba_cache_mode="all" 时的特殊路径（Qwen3.5 不支持）

```python
base_chunk_size = mamba_block_size or model_config.get_mamba_chunk_size()
attn_tokens_per_mamba_state = cdiv(mamba_page_size, attn_page_size_1_token)
chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)
attn_block_size = chunk_size * cdiv(attn_tokens_per_mamba_state, chunk_size)
# 用 LCM 同时对齐 mamba chunk size 和 kernel block size
```

### 7.4 对齐后的结果

| 量 | 对齐前 | 对齐后 |
|---|---|---|
| `cache_config.block_size` | 用户配置（默认 16） | 由 mamba_page_size 决定的更大值 |
| `mamba_page_size_padded` | 不存在 | = block_size × attn_page_size_1_token |
| 两种 cache 的 block_size | 不同 | 统一为同一个值 |

**关键推论**：
- `mamba_page_size` 越大（ssm_state 越大、dtype 精度越高），`block_size` 越大
- `attn_page_size_1_token` 越小（量化精度越低，如 fp8），`block_size` 越大
- 最终 `block_size = max(用户配置, 计算值)`，用户配置的默认 16 通常会被覆盖

### 7.5 对 Qwen3.5 block_size 的实际影响

**Qwen3.5 的 block_size 几乎必然被强制覆盖**，用户配置的默认值 16 会变得远更大。

**触发条件**：Qwen3.5 是 Hybrid 模型，同时有 `FullAttentionSpec` 和 `MambaSpec`，`mamba_page_size > 0` 必然满足，`_align_hybrid_block_size` 在启动时**必然触发**。

**Qwen3.5-7B 数值推导**（`num_k_heads=num_v_heads=16, head_dim=256, conv_kernel=4`）：

```
mamba_page_size ≈ 12288×3×2 + 16×256×256×2 ≈ 2.1 MB

bf16 时：
  attn_page_size_1_token = 2 × 2 × 256 × 2 = 2048 bytes（2 KB/token）
  attn_block_size = 16 × ceil(2,097,152 / (16 × 2048))
                 = 16 × 64 = 1024 tokens

fp8 时：
  attn_page_size_1_token = 2 × 2 × 256 × 1 = 1024 bytes（1 KB/token，减半）
  attn_block_size = 16 × ceil(2,097,152 / (16 × 1024))
                 = 16 × 128 = 2048 tokens（翻倍！）
```

**用户配置的 block_size=16 直接被覆盖成 1024（bf16）或 2048（fp8）**。

**直觉理解**：mamba state 是固定大小的"容器"（约 2 MB），attention token 是"砖块"。量化后每块砖变薄了（每个 token 占字节少了），需要更多块砖才能填满容器，block_size 就变大了。

**对 prefix cache 命中率的影响**：block_size 越大，公共前缀必须越长才能命中一个 block，命中门槛越高，命中率越低。FP8 量化进一步放大了这个问题（block_size 翻倍）。

---

# Part 5：KV Cache 调度器

## 8. KV Cache 调度器架构

### 8.1 整体架构

**文件**：`vllm/v1/core/kv_cache_coordinator.py`、`vllm/v1/core/single_type_kv_cache_manager.py`

```
HybridKVCacheCoordinator（Hybrid 模型使用）
├── FullAttentionManager（管理 Self Attention 的 KV cache）
│     ├── find_longest_cache_hit()  ← 从左到右扫描
│     ├── allocate_new_blocks()
│     └── cache_full_blocks()
└── MambaManager（管理 Linear Attention 的 KV cache）
      ├── find_longest_cache_hit()  ← 从右到左扫描
      ├── allocate_new_blocks()
      └── get_num_skipped_tokens()
```

### 8.2 KV Cache 分组

**文件**：`vllm/v1/core/kv_cache_utils.py`

按 spec 类型分组，Qwen3.5 分组结果：

```
Group 0（mamba group）：30 个 linear_attention 层 → MambaSpec
Group 1（attention group）：10 个 full_attention 层 → FullAttentionSpec
```

每个 group 有独立的 `SingleTypeKVCacheManager`，通过 `HybridKVCacheCoordinator` 协调。

---

## 9. Self Attention 的 Prefix Cache 策略

### 9.1 Block Hash 计算

**文件**：`vllm/v1/core/kv_cache_utils.py`

```python
# 滚动计算 block hash，依赖 parent_hash（链式依赖）
def hash_block_tokens(hash_fn, prev_hash, block_tokens, extra_keys):
    return hash_fn(prev_hash + block_tokens + extra_keys)

# 只对完整 block 计算 hash，不满 block_size 的尾部 token 不参与
while start_token_idx + block_size <= num_tokens:
    block_hash = hash_block_tokens(hash_fn, prev_hash, block_tokens, extra_keys)
    start_token_idx += block_size
```

### 9.2 FullAttentionManager：从左到右扫描

**文件**：`vllm/v1/core/single_type_kv_cache_manager.py`

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, ...):
    max_num_blocks = max_length // block_size

    # 从左到右扫描，遇到 cache miss 立即停止（必须连续）
    for block_hash in itertools.islice(block_hashes, max_num_blocks):
        if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
            computed_blocks.append(cached_block)
        else:
            break  # 遇到 miss 立即停止，不能跳过

    # alignment 对齐：确保命中长度是 lcm_block_size 的整数倍
    while len(computed_blocks) * block_size % alignment_tokens != 0:
        computed_blocks.pop()

    return computed_blocks
```

**为什么必须从左到右连续命中**：Self Attention 的注意力计算需要访问所有历史 token 的 KV，如果中间有 miss，后面的 block 即使命中也无法使用（缺少中间的 KV 数据）。

### 9.3 Block 分配与缓存

```python
# 分配新 block
new_blocks = block_pool.get_new_blocks(num_new_blocks)

# 写入 KV cache 后，缓存完整 block（用于 prefix cache）
block_pool.cache_full_blocks(
    request, blocks, num_cached_blocks, num_full_blocks,
    block_size, kv_cache_group_id
)
# 内部：计算 block_hash，存入 cached_block_hash_to_block
```

---

## 10. Linear Attention 的 Prefix Cache 策略

### 10.1 MambaManager：从右到左扫描

**文件**：`vllm/v1/core/single_type_kv_cache_manager.py`

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, ...):
    max_num_blocks = max_length // block_size

    # 从右到左扫描，找到最后一个匹配即停止
    for i in range(max_num_blocks - 1, -1, -1):
        if cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids):
            # alignment 对齐检查
            if (i + 1) * block_size % alignment_tokens != 0:
                continue
            computed_blocks.extend([block_pool.null_block] * i)  # 前面填充 null block
            computed_blocks.append(cached_block)
            break  # 只需最后一个匹配

    return computed_blocks
```

**为什么从右到左只需最新状态**：Linear Attention 的 ssm_state 已经压缩了所有历史 KV 信息，只需恢复最新的状态矩阵 S，就能继续推理，不需要重放历史 token。

### 10.2 get_num_skipped_tokens

```python
def get_num_skipped_tokens(self, num_computed_tokens):
    """Mamba 只需保留最后一个 token 的状态，前面所有 token 都可以跳过"""
    return num_computed_tokens - 1
```

### 10.3 跨请求缓存隔离

```python
def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks, ...):
    if (len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step):
        # Mamba 不能依赖当前 step 中其他请求生成的 blocks
        # 返回 num_gpu_blocks + 1 让调度器跳过本次调度
        return self.block_pool.num_gpu_blocks + 1
```

### 10.4 两种策略对比

| 维度 | Self Attention（FullAttentionManager） | Linear Attention（MambaManager） |
|---|---|---|
| **扫描方向** | 从左到右 | 从右到左 |
| **命中条件** | 必须从头连续命中 | 只需最新状态命中 |
| **miss 处理** | 遇 miss 立即停止 | 跳过，继续往左找 |
| **前置 block** | 必须全部命中 | 填充 null block（跳过计算） |
| **原因** | 注意力需要所有历史 KV | 状态矩阵已压缩所有历史 |

---

## 11. Hybrid 协调器：迭代固定点算法

### 11.1 HybridKVCacheCoordinator

**文件**：`vllm/v1/core/kv_cache_coordinator.py`

当模型有多种 KV cache group 时，使用 `HybridKVCacheCoordinator` 协调：

```python
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    hit_length = max_cache_hit_length
    hit_blocks_by_group = [None] * num_groups

    while True:
        curr_hit_length = hit_length

        for spec, group_ids, manager_cls in self.attention_groups:
            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                # Full Attention：直接截断到 block 边界
                num_blocks = curr_hit_length // spec.block_size
                curr_hit_length = num_blocks * spec.block_size
            else:
                # 调用对应类型的 manager 查找
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=..., max_length=curr_hit_length, ...
                )
                curr_hit_length = len(hit_blocks[0]) * spec.block_size

        if curr_hit_length >= hit_length:
            break   # 收敛
        hit_length = curr_hit_length
        if is_simple_hybrid:
            break   # 简单 hybrid 只需一次
```

### 11.2 迭代固定点算法示例

```
初始：hit_length = 1000 tokens

Round 1:
  FullAttentionManager：从左到右扫描，命中到 800 tokens → curr = 800
  MambaManager：在 800 范围内从右到左扫描，命中到 750 tokens → curr = 750

curr(750) < hit_length(1000)，继续迭代
hit_length = 750

Round 2:
  FullAttentionManager：cached_blocks 已存在，截断到 block 边界 → curr = 750
  MambaManager：在 750 范围内，命中到 750 → curr = 750

curr(750) >= hit_length(750)，收敛！最终 hit_length = 750
```

**为什么需要迭代**：两种 manager 的命中长度互相约束，FullAttention 的命中长度限制了 Mamba 的查找范围，Mamba 的命中长度又可能进一步限制 FullAttention 的有效 block 数。

---

## 12. Mamba Cache Mode

**文件**：`vllm/model_executor/models/qwen3_5.py`

Qwen3.5 只支持 `none` 和 `align` 模式：

```python
if cache_config.mamba_cache_mode == "all":
    raise NotImplementedError(
        "Qwen3.5 currently does not support 'all' prefix caching, "
        "please use '--mamba-cache-mode=align' instead"
    )
```

| 模式 | 内存占用 | 适用场景 |
|---|---|---|
| `none`（默认） | `(1 + num_spec_blocks) × page_size` | 普通推理 |
| `align` | `(2 + num_spec_blocks) × page_size` | 投机解码（MTP/EAGLE） |
| `all` | `ceil(max_model_len / block_size) × page_size` | **Qwen3.5 不支持** |

---

# Part 6：KV Cache 量化适配

## 13. FP8 量化如何适配 Hybrid Attention

### 13.1 量化只作用于 Self Attention 层

FP8 KV Cache **只对 full_attention 层生效**，linear_attention 层的 mamba state 始终使用模型原始 dtype：

| 层类型 | Spec | KV Cache dtype | 受量化影响 |
|---|---|---|---|
| `full_attention` | `FullAttentionSpec` | `fp8`（由 `--kv-cache-dtype fp8` 控制） | ✅ 是 |
| `linear_attention` | `MambaSpec` | 模型 dtype（通常 bf16 或 float32） | ❌ 否 |

**原因**：
- `FullAttentionSpec.dtype` 从 `cache_config.cache_dtype` 读取，受 `--kv-cache-dtype` 控制
- `MambaSpec.dtypes` 由模型自身通过 `get_state_dtype()` 直接给出，不经过 `kv_cache_dtype` 配置

### 13.2 量化对 Attention 层完全透明

FP8 的量化/反量化完全在 attention backend 内部自动完成，模型层代码不感知：

```
Qwen3NextAttention（模型层，不感知 FP8）
  └── self.attn = Attention(...)
        └── TritonAttentionBackend（FP8 在这里处理）
              ├── reshape_and_cache()  ← 写入时：BF16 → FP8 量化
              └── unified_attention()  ← 计算时：FP8 → BF16 反量化
```

### 13.3 量化模式

**文件**：`vllm/v1/kv_cache_interface.py`

```python
class KVQuantMode(IntEnum):
    NONE = 0                # 不量化
    FP8_PER_TENSOR = 1      # --kv-cache-dtype fp8（单个 tensor-wide scale）
    INT8_PER_TOKEN_HEAD = 2 # INT8 per-token-head
    FP8_PER_TOKEN_HEAD = 3  # FP8 per-token-head（每个 token 每个 head 独立 scale）
```

### 13.4 量化/反量化实现

**写入时量化**（`reshape_and_cache`，`vllm/v1/attention/backends/triton_attn.py`）：

```python
def reshape_and_cache(self, layer, key, value, kv_cache, slot_mapping):
    # FP8 per-tensor 量化路径
    triton_reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        slot_mapping, self.kv_cache_dtype,
        layer._k_scale, layer._v_scale     # per-tensor scale
    )
```

**计算时反量化**（`unified_attention` kernel，`vllm/v1/attention/ops/triton_unified_attention.py`）：

```python
@triton.jit
def _prepare_kv_tile(data, Q, tensor_scale, ..., KV_QUANT_MODE):
    if KV_QUANT_MODE == 1:  # FP8_PER_TENSOR
        # 反量化：FP8 → float32 → Q.dtype
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
```

### 13.5 Scale 因子管理

**文件**：`vllm/model_executor/layers/quantization/kv_cache.py`

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    def create_weights(self, layer):
        layer.k_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)

    def process_weights_after_loading(self, layer):
        if layer.k_scale > 0.0:
            # checkpoint 中有 scale → 直接加载（离线校准过）
            layer._k_scale.copy_(layer.k_scale.item())
        else:
            # checkpoint 中没有 scale → 使用默认值 1.0
            layer._k_scale.copy_(1.0)
```

**三种校准方式**：

| 方式 | 精度 | 适用场景 |
|---|---|---|
| 无校准（scale=1.0） | ★★☆ | 快速实验 |
| 随机 Token 动态校准（`calculate_kv_scales=True`） | ★★★☆ | 无校准数据集时的折中 |
| 数据集离线校准（llm-compressor） | ★★★★★ | **生产环境推荐** |

---

## 14. 量化对 Block Size 和 Prefix Cache 的影响

### 14.1 量化如何影响 block_size

FP8 量化让 `attn_page_size_1_token` 减半（1 byte vs 2 bytes），但 `mamba_page_size` 不变，导致 `_align_hybrid_block_size` 计算出更大的 `attn_block_size`：

```
约束：attn_block_size × attn_page_size_1_token ≥ mamba_page_size

bf16：attn_page_size_1_token 较大 → attn_block_size 较小
fp8： attn_page_size_1_token 减半 → attn_block_size 翻倍
```

**直觉理解**：mamba state 是固定大小的"容器"，attention token 是"砖块"。量化后每块砖变薄了（每个 token 占字节少了），需要更多块砖才能填满容器，block_size 就变大了。

| 量化精度 | attn_page_size_1_token | 相对 block_size | prefix cache 命中门槛 |
|---|---|---|---|
| bf16（无量化） | 较大（基准） | 基准 | 较低 |
| fp8（1 byte） | 减半 | 翻倍 | 翻倍 |

### 14.2 block_size 变大对 Prefix Cache 命中率的影响

Prefix cache 的命中单位是**完整的 block**，只有凑满一个 block 的 token 才能被缓存和命中：

```python
# vllm/v1/core/kv_cache_utils.py
while start_token_idx + block_size <= num_tokens:
    block_hash = hash_block_tokens(...)  # 只对完整 block 计算 hash
    start_token_idx += block_size
# 尾部 num_tokens % block_size 个 token 永远不会被缓存
```

**三个维度的影响**：

| 影响 | 说明 |
|---|---|
| **命中粒度变粗** | 公共前缀必须 ≥ block_size 才能命中一个 block |
| **尾部浪费增加** | 尾部最多浪费 block_size-1 个 token（永远不被缓存） |
| **全命中重算增加** | 即使全命中，也需重算最后一个 block（最多 block_size 个 token） |

### 14.3 实际工程影响

| 场景 | prefix cache 是否有效 | 说明 |
|---|---|---|
| 公共前缀 < block_size | ❌ 无效 | 凑不满一个 block，无法命中 |
| 公共前缀 ≥ block_size | ✅ 有效 | 可以命中前几个完整 block |
| 超长 system prompt / RAG 文档 | ✅ 有效 | 文档部分可以命中多个 block |
| 多轮对话（每轮追加较少 token） | ⚠️ 部分有效 | 历史轮次满 block 后可命中 |
| FP8 量化 | ❌ 命中率进一步降低 | block_size 翻倍，命中门槛更高 |

**根本原因**：Hybrid 模型的 `block_size` 由 **mamba state 的物理大小**决定，而非 attention 层的配置。mamba state 越大（更多 heads、更高精度的 ssm_dtype），block_size 越大，prefix cache 命中率越低。FP8 量化进一步放大了这个问题。

---

# Part 7：附录

## 15. 完整数据流图

```
输入 Token IDs
      ↓
Embedding Layer
      ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Qwen3_5Model（40 层，每 4 层一个 full_attention）        │
│                                                                     │
│  Layer 0,1,2（linear_attention）：                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ GatedDeltaNetAttention.forward_cuda:                          │  │
│  │   in_proj_qkvz → mixed_qkv, z                                │  │
│  │   in_proj_ba   → b (beta), a (alpha)                          │  │
│  │   causal_conv1d → 更新 conv_state                             │  │
│  │   gated_delta_rule → 更新 ssm_state                           │  │
│  │   RMSNormGated(output, z) → out_proj                          │  │
│  │ KV Cache：MambaSpec（固定大小，不随序列增长）                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Layer 3（full_attention）：                                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Qwen3NextAttention.forward:                                   │  │
│  │   qkv_proj → Q (×2 for gate), K, V                           │  │
│  │   QK-Norm → RoPE → Attention(Q,K,V) → Gate → o_proj          │  │
│  │ KV Cache：FullAttentionSpec（O(seq_len)，可 FP8 量化）          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Layer 4,5,6（linear）... Layer 7（full）... 以此类推               │
└─────────────────────────────────────────────────────────────────────┘
      ↓
Final Norm → LM Head → Logits

KV Cache 管理层：
┌─────────────────────────────────────────────────────────────────────┐
│  HybridKVCacheCoordinator                                           │
│  ├── FullAttentionManager（Self Attention）                         │
│  │     从左到右扫描，必须连续命中                                    │
│  │     block_size 由 mamba_page_size 决定（通常远大于 16）           │
│  ├── MambaManager（Linear Attention）                               │
│  │     从右到左扫描，只需最新状态命中                                │
│  │     block_size 与 attention 统一（prefix cache hash 对齐）        │
│  └── 迭代固定点算法协调两种类型                                      │
│                                                                     │
│  统一 BlockPool：attention 和 mamba 共享物理内存池                   │
│  独立 BlockTable：每个 group 独立管理 block_id 序列                  │
│  FP8：只对 full_attention 层生效，mamba 层保持原始 dtype             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 16. 参考文件索引

| 组件 | 文件路径 |
|---|---|
| Qwen3.5 模型实现 | `vllm/model_executor/models/qwen3_5.py` |
| Qwen3Next Attention | `vllm/model_executor/models/qwen3_next.py` |
| GatedDeltaNetAttention | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` |
| Mamba 状态工具 | `vllm/model_executor/layers/mamba/mamba_utils.py` |
| MambaBase 抽象类 | `vllm/model_executor/layers/mamba/abstract.py` |
| KV Cache Spec 定义 | `vllm/v1/kv_cache_interface.py` |
| KV Cache 分组与 Hash | `vllm/v1/core/kv_cache_utils.py` |
| Hybrid KV 协调器 | `vllm/v1/core/kv_cache_coordinator.py` |
| 单类型 KV Cache 管理器 | `vllm/v1/core/single_type_kv_cache_manager.py` |
| Block Pool | `vllm/v1/core/block_pool.py` |
| MultiGroup Block Table | `vllm/v1/worker/block_table.py` |
| Page Size 对齐逻辑 | `vllm/platforms/interface.py` |
| TritonAttentionBackend | `vllm/v1/attention/backends/triton_attn.py` |
| Triton Attention Kernel | `vllm/v1/attention/ops/triton_unified_attention.py` |
| KV Cache 量化 Scale 管理 | `vllm/model_executor/layers/quantization/kv_cache.py` |
