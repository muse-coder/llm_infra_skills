# vLLM v1 Chunked Prefill 知识库

> 基于 vLLM v1 调度器，结合 Qwen3.5 hybrid 架构（Attention + GDN 混合层）。
> 本文档针对 `vllm/v1/` 代码路径；v0 的实现与参数语义已不同。

---

## 1. 什么是 Chunked Prefill

当一个请求的 prompt 长度超过单步的 token 预算（`max_num_batched_tokens`），调度器不会等到预算够大才处理，而是把 prefill 拆成多个 chunk，每步处理一部分。更准确地说，v1 调度器里根本不区分 "prefill phase" 和 "decode phase"——每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec` 两个状态字段，调度器的目标就是尽量让前者追上后者。Chunked prefill 只是这套通用机制下"长 prompt 被预算切断"的自然结果。

收益：

- **decode 请求不被长 prompt 饿死**：长 prefill 切块后，decode 请求可以和 prefill chunk 混合调度
- **GPU 利用率更稳定**：每步的 token 数受 budget 控制，避免一步处理上万 token、下一步只处理几十个的波动
- **显存可控**：activation 内存按 chunk 大小分配，避免超长 prompt 一次性占满

---

## 2. 配置参数

所有参数定义在 `vllm/config/scheduler.py` 的 `SchedulerConfig` 类中。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_chunked_prefill` | `True` | 总开关。关闭后长 prompt 必须完整放入 budget 才能调度 |
| `max_num_batched_tokens` | 硬件相关 | 每步 token 预算。H100/MI300x: `16384`(LLM_CLASS) / `8192`(OPENAI_API_SERVER)；其他 GPU: `8192` / `2048`。设备显存 ≥ 70 GiB 且名字不含 "a100" 触发高档位（`vllm/engine/arg_utils.py::get_batch_defaults`） |
| `max_num_scheduled_tokens` | `= max_num_batched_tokens` | 调度器实际使用的预算，一般等同上一行；Spec decode 场景可能更小 |
| `long_prefill_token_threshold` | `0` | **单请求每步最多处理多少 token**。`>0` 时生效。注意：它对 **RUNNING（中间 chunk）和 WAITING（新请求）两个阶段都生效** |
| `max_num_seqs` | H100/MI300x: `1024`，其他: `256` | 一轮同时在 `self.running` 里的请求数上限 |
| `scheduler_reserve_full_isl` | `True` | 准入新请求前检查完整 ISL 是否放得下 KV cache，防止过度准入 |
| `max_num_partial_prefills` | `1` | **v1 中不支持 > 1**。`EngineArgs._check_feature_supported()` 会抛 "Concurrent Partial Prefill" unsupported error（v0 遗留） |
| `max_long_partial_prefills` | `1` | 同上，v1 中不支持 > 1 |

硬件相关默认值由 `vllm/engine/arg_utils.py::EngineArgs.get_batch_defaults()` 设定。

---

## 3. 调度器核心逻辑

文件：`vllm/v1/core/sched/scheduler.py`

### 3.1 核心设计理念（L348-358 NOTE 注释）

> _There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`. ... This is general enough to cover chunked prefills, prefix caching, speculative decoding, and the "jump decoding" optimization in the future._

调度器在每一步尝试分配 token 让 `num_computed_tokens` 追上 `num_tokens_with_spec`。

### 3.2 `schedule()` 方法（L348 起）

```python
token_budget = self.max_num_scheduled_tokens   # L367

# ═══════════════════════════════════════════════
# 阶段 A：RUNNING 请求（decode + 中间 chunk 的 prefill）L385-518
# ═══════════════════════════════════════════════
for request in self.running:
    if token_budget <= 0: break
    # 还需要处理多少 token（对 decode = 1+spec，对 prefill 中间 chunk = 剩余 prompt）
    num_new_tokens = (request.num_tokens_with_spec
                    + request.num_output_placeholders
                    - request.num_computed_tokens)                 # L404-408

    # 单请求上限（对长 prefill 的 chunk size 控制）
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold              # L409-410

    # 按全局剩余预算截断
    num_new_tokens = min(num_new_tokens, token_budget)             # L411

    # Mamba/GDN 混合模型：对齐 block_size（见 3.4）
    if need_mamba_block_aligned_split:
        num_new_tokens = _mamba_block_aligned_split(request, num_new_tokens)

    token_budget -= num_new_tokens                                 # L517

# ═══════════════════════════════════════════════
# 阶段 B：WAITING 请求（新 prefill）L563-851
# ═══════════════════════════════════════════════
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs: break        # L568

    # 先扣除 prefix cache 命中数（local + KVConnector external）
    num_computed_tokens = prefix_cache_hits_for(request)            # L610-650

    num_new_tokens = request.num_tokens - num_computed_tokens       # L665

    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold               # L666-668

    # 不开 chunked prefill → 塞不下就整体 break
    if not enable_chunked_prefill and num_new_tokens > token_budget:
        break                                                       # L672-678

    num_new_tokens = min(num_new_tokens, token_budget)              # L680

    if need_mamba_block_aligned_split:
        num_new_tokens = _mamba_block_aligned_split(request, num_new_tokens)

    token_budget -= num_new_tokens
```

**核心规则汇总（一轮里 prefill / decode 各能分到多少 token）**：

- `max_num_batched_tokens` 是 **prefill + decode 共享的同一个 token 预算**，没有单独划额度
- 调度顺序是**先 running，再 waiting**，所以 decode 请求（几乎总在 running 里）**天然优先**于新 prefill 吃预算
- 每个请求一步能拿的 token = `min(需要量, long_prefill_token_threshold, 剩余预算)`
- 纯 decode 请求的"需要量" = `1`（若有 spec = `1 + len(spec_tokens)`），所以 decode 只会吃掉 1~几 token
- 长 prefill 的"需要量" = `num_tokens - num_computed_tokens`，会被后两项截断成块

### 3.3 `_update_after_schedule()`（L978-997）

```python
num_scheduled_tokens = scheduler_output.num_scheduled_tokens
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token                # L991
    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders)         # L992-994
```

- `num_computed_tokens` 在 schedule 阶段**立即推进**，不等 GPU forward 完，使得下一步 `schedule()` 可以继续切下一 chunk（async scheduling 能和 chunked prefill 组合的关键）
- `is_prefill_chunk = True` → 本请求还没吃完整个 prompt

**`is_prefill_chunk` 的真实作用范围**：只在调度器内部使用，用途是：
1. 跳过 structured output 推进（L995-996、L1282）
2. 跳过 spec tokens 处理（L1680）
3. async scheduler 跳过 sampled-token 处理（`async_scheduler.py` L23）

**真正让 GPU 侧"不 sample 中间 chunk"的不是 `is_prefill_chunk`**，而是 worker 侧的 `discard_request_mask`（见 §5.4）。

### 3.4 Mamba/GDN Block 对齐（`_mamba_block_aligned_split()`，L298-346）

Hybrid 模型（Qwen3.5）的 GDN 层使用类似 Mamba 的状态缓存。为支持 block-aligned 的状态缓存，`num_new_tokens` 需要对齐到 `block_size`：

```python
block_size = cache_config.block_size
last_cache_position = request.num_tokens - request.num_tokens % block_size

if num_computed_tokens_after_sched < last_cache_position:
    # 中间 chunk：对齐到 block_size 倍数
    num_new_tokens = num_new_tokens // block_size * block_size
elif num_computed_tokens < last_cache_position < num_computed_tokens_after_sched:
    # 强制在 last_cache_position 处切分，确保最后一个 cache 块正确写入
    num_new_tokens = last_cache_position - num_computed_tokens
else:
    # 最后几个 token，不需要对齐
    pass
```

这是 GDN 与纯 Attention 模型在 chunked prefill 调度上的**第一个区别**。

---

## 4. 请求状态跟踪

文件：`vllm/v1/request.py`

| 字段 | 初始值 | 说明 |
|------|--------|------|
| `num_computed_tokens` | `0` | 已处理的 token 数。每步在 `_update_after_schedule()` 增加 `num_scheduled_tokens`，preempt 时重置为 `0` |
| `is_prefill_chunk` | `False` | **调度器侧**标记；True = 非最终 chunk。影响 spec decode / structured output / async schedule |
| `num_cached_tokens` | `-1` | prefix cache 命中数（统计用） |
| `num_output_placeholders` | `0` | async spec decode 优化中用于占位的 draft tokens 数 |

---

## 5. GPU Model Runner：输入准备与 sample 决策

文件：`vllm/v1/worker/gpu_model_runner.py`

### 5.1 `_prepare_inputs()`（L1786 起）

```python
positions_np = (
    self.input_batch.num_computed_tokens_cpu[req_indices]
    + self.query_pos.np[: cu_num_tokens[-1]]
)  # L1819-1822
```

例：prompt 4096 token，chunk_size 2048
- Chunk 1：`positions = [0..2047]`
- Chunk 2：`positions = [2048..4095]`

### 5.2 seq_lens 计算（L2002-2003）

```python
self.seq_lens[:num_reqs] = (
    self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
)
```

这是 attention 层看到的 KV 总长度：
- Chunk 1：`seq_lens = 0 + 2048 = 2048`
- Chunk 2：`seq_lens = 2048 + 2048 = 4096`

### 5.3 Batch 重排（`reorder_batch_to_split_decodes_and_prefills`）

文件：`vllm/v1/attention/backends/utils.py` L589

把 batch 按以下 **4 个区域的固定顺序**排好（便于后端对 decode / prefill 分别走不同 kernel 路径）：

```
┌────────┬──────────────┬─────────────┬──────────────┐
│ decode │ short_extend │ long_extend │ pure_prefill │
└────────┴──────────────┴─────────────┴──────────────┘
```

分类条件（基于 L615-627 的代码）：

```python
has_context       = num_computed_tokens > 0
is_below_threshold = num_scheduled_tokens <= decode_threshold
done_prefilling   = num_computed_tokens >= num_prompt_tokens

is_pure_prefill   = ~has_context
is_long_extend    = has_context & ~is_below_threshold
is_short_extend   = has_context & is_below_threshold & ~done_prefilling
is_decode         = has_context & is_below_threshold & done_prefilling
```

| 区域 | 条件 |
|------|------|
| `decode` | 有 context，本步 query_len ≤ threshold，已完成 prefill |
| `short_extend` | 有 context，本步 query_len ≤ threshold，**仍在 prefill**（少见但存在：prompt 剩余 < threshold 的最后几个 chunk，或被预算切成小 chunk） |
| `long_extend` | 有 context，本步 query_len > threshold（不管是否完成 prefill） |
| `pure_prefill` | `num_computed_tokens == 0`（第一个 chunk） |

**对于 Qwen3.5 (hybrid 模型)**：`short_extend` 区域特别关键——当一个 GDN 请求已经跑过至少一个 chunk（`has_context`），但本步因为预算或 block 对齐只分到很少 token，会被归入 `short_extend`。GDN 后端必须把这类请求当 prefill 处理（走 `chunk_gated_delta_rule` 且 `has_initial_state=True`），不能按 decode 路径（单 token 更新）。

### 5.4 Sample 决策：`discard_request_mask`（真正的 "中间 chunk 不出 token" 机制）

```python
# L1927-1932
self.discard_request_mask.np[:num_reqs] = (
    self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
)
self.discard_request_mask.copy_to_gpu(num_reqs)
```

即：若 `seq_lens < num_tokens`（仍在 chunk 中间），打标丢弃。后续 sampler 输出时：

```python
# L3374-3401
discard_sampled_tokens_req_indices = np.nonzero(
    self.discard_request_mask.np[:num_reqs]
)[0]
...
for i in discard_sampled_tokens_req_indices:
    valid_sampled_token_ids[int(i)].clear()   # 丢弃中间 chunk 的 sample
```

**注意**：logits 照常算（因为 kernel 是统一的混合 batch），只是把 sampled token **丢掉不返回**。直到最后一个 chunk（`seq_lens >= num_tokens`）才真正交付第一个输出 token，request 从此进入 decode。

---

## 6. Attention Metadata

文件：`vllm/v1/attention/backend.py`

### 6.1 `CommonAttentionMetadata`（L344）

所有 attention 后端共享的元数据：

| 字段 | 说明 |
|------|------|
| `query_start_loc` | `(batch+1,)` 累积 query token 数。`query_len = loc[i+1] - loc[i]` |
| `seq_lens` | `(batch,)` 每个请求的总 KV 长度 = `num_computed_tokens + num_scheduled_tokens` |
| `is_prefilling` | `(batch,)` bool tensor：`num_computed_tokens < num_prompt_tokens`，供后端区分真正的 decode 与 short_extend（L386-389） |
| `max_query_len` | 批内最长 query。`> 1` 说明有 prefill 请求 |
| `max_seq_len` | 批内最长 KV 长度（可能是上界） |

### 6.2 `compute_num_computed_tokens()`（L437）

```python
def compute_num_computed_tokens(self) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    return seq_lens - query_lens   # 已缓存的 token 数
```

GDN 后端用这个值设定 `has_initial_state`：

```python
# gdn_attn.py
has_initial_state = context_lens_tensor > 0
# context_lens_tensor = compute_num_computed_tokens()
```

---

## 7. GDN 层的特殊处理（与标准 Attention 的区别）

文件：`vllm/v1/attention/backends/gdn_attn.py` 和 `vllm/model_executor/layers/mamba/gdn_linear_attn.py`

### 7.1 核心区别：有状态 vs 无状态

| | 标准 Attention 层 | GDN 层 |
|---|---|---|
| 状态类型 | KV Cache（append-only，自动衔接） | `conv_state` + `ssm_state`（循环更新，需显式加载/保存） |
| Chunked prefill 衔接 | 透明，Attention 自动读取之前的 KV | 需要 `has_initial_state` 标记 + state 加载/保存 |
| 内部 chunking | 无 | `FLA_CHUNK_SIZE = 64`（Triton kernel 内部再分段） |
| Metadata 类 | `FlashAttentionMetadata` 等 | `GDNAttentionMetadata`（有额外字段） |

### 7.2 `GDNAttentionMetadata` 特有字段

```python
@dataclass
class GDNAttentionMetadata:
    # 基础计数
    num_prefills: int
    num_decodes: int
    num_actual_tokens: int

    # 跨 chunk 衔接的关键标记
    has_initial_state: torch.Tensor | None        # (num_prefills,) bool
    # True = 从 ssm_state cache 恢复循环状态（非首 chunk）
    # False = 从零开始（首 chunk 或新请求）

    # FLA chunk 元数据（避免 GPU→CPU sync）
    chunk_indices: torch.Tensor | None            # 预计算的 FLA chunk 索引
    chunk_offsets: torch.Tensor | None            # 预计算的 FLA chunk 偏移

    # Spec decode 相关
    spec_query_start_loc: torch.Tensor | None
    non_spec_query_start_loc: torch.Tensor | None
    spec_state_indices_tensor: torch.Tensor | None
    non_spec_state_indices_tensor: torch.Tensor | None

    # Conv1d 的 triton 元数据
    nums_dict: dict | None
    batch_ptr: torch.Tensor | None
    token_chunk_offset_ptr: torch.Tensor | None
```

### 7.3 GDN 前向流程（`_forward_core()`，L779 起）

```
┌─────────────────────────────────────────────────────────────┐
│                    _forward_core()                          │
├─────────────────────────────────────────────────────────────┤
│  1. Conv1d 序列变换                                         │
│     ├─ prefill: causal_conv1d_fn()                         │
│     │   • conv_states=conv_state   ← 读取 conv 缓存        │
│     │   • has_initial_state        ← 标记是否恢复           │
│     │   • cache_indices            ← 写回位置               │
│     └─ decode:  causal_conv1d_update()                     │
│                                                             │
│  2. Post-Conv 准备（仅 prefill）                             │
│     └─ fused_post_conv_prep()                              │
│        split + l2norm + gating 融合到单个 Triton kernel      │
│        输出: q, k, v, g, beta                              │
│                                                             │
│  3. 循环注意力（Recurrent Attention）                        │
│     ├─ prefill: chunk_gated_delta_rule()                   │
│     │   • initial_state = ssm_state[indices]               │
│     │   • initial_state[~has_initial_state] = 0  ← 首 chunk│
│     │   • output_final_state=True                          │
│     │   • ssm_state[indices] = final_state  ← 写回         │
│     └─ decode:  fused_sigmoid_gating_delta_rule_update()   │
│                                                             │
│  4. 合并输出                                                │
└─────────────────────────────────────────────────────────────┘
```

关键代码（L972-994）：

```python
if attn_metadata.num_prefills > 0:
    initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
    initial_state[~has_initial_state, ...] = 0    # 首 chunk 清零

    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        q=query, k=key, v=value,
        g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=non_spec_query_start_loc,
        chunk_indices=attn_metadata.chunk_indices,
        chunk_offsets=attn_metadata.chunk_offsets,
    )

    ssm_state[non_spec_state_indices_tensor] = last_recurrent_state
```

### 7.4 FLA 内部 Chunk 机制

GDN 的 `chunk_gated_delta_rule` 内部有自己的 chunk 机制，与调度器的 chunked prefill 是**两层不同的 chunking**：

| 层级 | 粒度 | 控制参数 | 作用 |
|------|------|----------|------|
| 调度器 chunked prefill | 数千 token（~2048-16384） | `max_num_batched_tokens`, `long_prefill_token_threshold` | 控制每步处理量，允许混合 prefill+decode |
| FLA 内部 chunk | 64 token | `FLA_CHUNK_SIZE = 64` | 循环注意力计算的分块大小，用于 Triton kernel 优化 |

`chunk_indices` 和 `chunk_offsets` 在 `GDNAttentionMetadataBuilder.build()` 中预计算，避免 FLA kernel 内部产生 GPU→CPU 同步：

```python
from vllm.model_executor.layers.fla.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
chunk_indices = prepare_chunk_indices(query_start_loc_cpu, FLA_CHUNK_SIZE)
chunk_offsets = prepare_chunk_offsets(query_start_loc_cpu, FLA_CHUNK_SIZE)
```

### 7.5 GDN Prefill 内核后端选择

`ChunkGatedDeltaRule`（`gdn_linear_attn.py:121`）的后端选择逻辑（L124-159）：

```python
backend_cfg = additional_config.get("gdn_prefill_backend", "auto")
supports_flashinfer = current_platform.is_cuda() and current_platform.is_device_capability(90)

if backend == "flashinfer":
    use_flashinfer = supports_flashinfer   # 强制，硬件不支持时 warn + 降级
elif backend == "triton":
    use_flashinfer = False                  # 强制 Triton
else:                                       # "auto" 默认
    use_flashinfer = supports_flashinfer    # SM90 → FlashInfer，否则 Triton
```

| 后端 | 选择条件 | 特点 |
|------|---------|------|
| FlashInfer (`forward_cuda` → `fi_chunk_gated_delta_rule`) | `auto` + SM90（H100/H200） 或显式 `flashinfer` 且 SM90 | 性能最佳，但 JIT 编译首次较慢；**不使用** `chunk_indices`/`chunk_offsets` |
| Triton/FLA (`forward_native` → `fla_chunk_gated_delta_rule`) | 非 SM90 或显式 `--gdn-prefill-backend triton` | 兼容性好，需要预计算的 `chunk_indices`/`chunk_offsets` |

---

## 8. Qwen3.5 Hybrid 架构

文件：`vllm/model_executor/models/qwen3_5.py` + `vllm/transformers_utils/configs/qwen3_5.py`

### 8.1 层类型模式

默认 `full_attention_interval = 4`，即每 4 层有 1 层是标准 Attention：

```python
# configs/qwen3_5.py L89-95
interval_pattern = kwargs.get("full_attention_interval", 4)
self.layer_types = [
    "linear_attention"
    if bool((i + 1) % interval_pattern)
    else "full_attention"
    for i in range(self.num_hidden_layers)
]
```

以 36 层模型为例（索引从 0 起）：

```
Layer  0: linear_attention  (GDN)   # (0+1)%4=1
Layer  1: linear_attention  (GDN)
Layer  2: linear_attention  (GDN)
Layer  3: full_attention    (标准 Attention + KV Cache)  # (3+1)%4=0
Layer  4: linear_attention  (GDN)
Layer  5: linear_attention  (GDN)
Layer  6: linear_attention  (GDN)
Layer  7: full_attention    (标准 Attention + KV Cache)
...
```

**比例：75% GDN + 25% Attention**

### 8.2 Hybrid 接口

`Qwen3_5ForConditionalGeneration` 实现了 `IsHybrid` protocol：

```python
# models/qwen3_5.py L574
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    ...
    @classmethod
    def get_mamba_state_shape_from_config(cls, ...):      # L720
        return MambaStateShapeCalculator.gated_delta_net_state_shape(...)
```

> `IsHybrid` 是 `runtime_checkable` Protocol（`models/interfaces.py` L787），通过继承关系生效；具体的判别靠 `is_hybrid(model)`（L837-840）查询 `getattr(model, "is_hybrid", False)` 或 Protocol 成员检查。

### 8.3 GDN 状态形状

每个 GDN 层维护两种状态：

| 状态 | 形状 | 说明 |
|------|------|------|
| `conv_state` | `(num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim, conv_kernel_size - 1)` | 卷积滑动窗口 |
| `ssm_state` | `(num_v_heads, head_v_dim, head_k_dim)` | 循环注意力的 hidden state 矩阵 |

Qwen3.5-35B-A3B 典型参数：`num_k_heads=8, head_k_dim=256, num_v_heads=64, head_v_dim=64, conv_kernel_size=4`

---

## 9. 端到端示例

### 场景：Qwen3.5-35B，prompt 4096 token，`max_num_batched_tokens = 2048`

```
Step 1: 调度 chunk [0, 2048)
  ├─ 调度器：token_budget=2048, num_new_tokens=min(4096, 2048)=2048
  ├─ Attention 层（Layer 3,7,11,...）：
  │   ├─ positions = [0..2047]
  │   ├─ seq_lens = 2048
  │   └─ 写入 KV Cache 的前 2048 个 slot
  ├─ GDN 层（Layer 0,1,2,4,5,...）：
  │   ├─ has_initial_state = False（首 chunk）
  │   ├─ initial_state = zeros
  │   ├─ causal_conv1d_fn → 更新 conv_state
  │   ├─ chunk_gated_delta_rule (内部按 FLA_CHUNK_SIZE=64 分段)
  │   └─ ssm_state[idx] = final_state  ← 保存
  ├─ _update_after_schedule：
  │   ├─ num_computed_tokens: 0 → 2048
  │   └─ is_prefill_chunk = True (2048 < 4096)
  └─ discard_request_mask = True → 丢弃 sample

Step 2: 调度 chunk [2048, 4096)
  ├─ 调度器：token_budget=2048, num_new_tokens=min(2048, 2048)=2048
  ├─ Attention 层：
  │   ├─ positions = [2048..4095]
  │   ├─ seq_lens = 4096
  │   └─ KV Cache 已有前 2048，追加后 2048
  ├─ GDN 层：
  │   ├─ has_initial_state = True（非首 chunk）
  │   ├─ initial_state = ssm_state[idx]  ← 从 Step 1 恢复
  │   ├─ causal_conv1d_fn → 从 conv_state 恢复滑动窗口
  │   ├─ chunk_gated_delta_rule → 以 loaded state 为起点
  │   └─ ssm_state[idx] = new_final_state
  ├─ _update_after_schedule：
  │   ├─ num_computed_tokens: 2048 → 4096
  │   └─ is_prefill_chunk = False (4096 == 4096)
  └─ discard_request_mask = False → 采样第一个 output token

Step 3+: 正常 decode
  ├─ Attention 层：query_len=1, 读取完整 KV Cache
  └─ GDN 层：fused_sigmoid_gating_delta_rule_update (单 token 循环更新)
```

### 混合调度示例

如果 Step 2 执行时有一个正在 decode 的请求 B：

```
Step 2 的调度过程（token_budget=2048）：
  1. 先处理 self.running：
     - B (decode): 吃 1 个 token，budget=2047
     - A (chunk 续块): 吃 min(2048, 2047)=2047 个 token，budget=0
  2. 跳过 waiting 阶段（预算耗尽）

Step 2 batch composition (after reorder)：
  ┌───────────┬──────────────────────┐
  │ Request B │     Request A        │
  │ (decode)  │   (long_extend)      │
  │ 1 token   │   2047 tokens        │
  └───────────┴──────────────────────┘
       ↑              ↑
   query_len=1    query_len=2047
   seq_lens=500   seq_lens=4095

Attention 层：
  - B: decode attention (1 query, 500 KV)
  - A: prefill attention (2047 query, 4095 KV)

GDN 层：
  - B: fused_sigmoid_gating_delta_rule_update (decode path)
  - A: chunk_gated_delta_rule + has_initial_state=True (prefill path)
```

---

## 10. Profiler Annotation 与 Chunked Prefill

Worker 侧每步 forward 会打 profiler annotation（`vllm/v1/worker/gpu_worker.py` L724-739）：

```
execute_context_{A}({B})_generation_{C}({D})
```

| 字段 | 含义 |
|------|------|
| `A` | context phase 请求数（`num_output_tokens == 0` 或本步新来的请求，即处于 prefill / chunk 中） |
| `B` | context phase 的 token 数（这些请求本步一共处理多少 token） |
| `C` | generation phase 请求数（已产出过 token 的 decode 请求） |
| `D` | generation phase 的 token 数（通常 ≈ `C`，每 decode 请求 1 token，有 spec decode 时更多） |

判定逻辑在 `vllm/v1/utils.py::compute_iteration_details()`。

**这只是外部观察者对一次 forward 的事后分类**，调度器内部并不区分；但对分析 trace、判断 prefill/decode 混合比例很有用。例如：

- `execute_context_3(1264)_generation_1(1)` = 3 个请求在做 prefill（共 1264 token），1 个 decode（1 token）。典型的 chunked prefill + decode 混批
- `execute_context_0(0)_generation_128(128)` = 纯 decode 步，128 个请求各吃 1 token
- `execute_context_1(8192)_generation_0(0)` = 单个长 prefill 吃满整个预算

---

## 11. 关键文件索引

| 文件 | 核心作用 |
|------|----------|
| `vllm/config/scheduler.py` | chunked prefill 配置参数定义 |
| `vllm/engine/arg_utils.py` | 硬件相关默认值（`get_batch_defaults`）；v1 feature gate（`_check_feature_supported`） |
| `vllm/v1/core/sched/scheduler.py` | 调度核心：token_budget 分配、chunk 切分、block 对齐、`_update_after_schedule` |
| `vllm/v1/core/sched/async_scheduler.py` | async 调度器：对 `is_prefill_chunk` 的额外处理 |
| `vllm/v1/core/sched/output.py` | `CachedRequestData.is_context_phase`（判断 context / generation） |
| `vllm/v1/utils.py` | `compute_iteration_details`（profiler annotation 数据源） |
| `vllm/v1/request.py` | `num_computed_tokens`、`is_prefill_chunk` 状态 |
| `vllm/v1/worker/gpu_worker.py` | `execute_context_N(M)_generation_X(Y)` annotation 注入 |
| `vllm/v1/worker/gpu_model_runner.py` | positions/seq_lens/query_start_loc 准备；`discard_request_mask` 决定采样 |
| `vllm/v1/attention/backend.py` | `CommonAttentionMetadata` 定义；`compute_num_computed_tokens` |
| `vllm/v1/attention/backends/utils.py` | `split_decodes_and_prefills` / `reorder_batch_to_split_decodes_and_prefills` |
| `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadata` 构建，`has_initial_state` 计算 |
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | GDN 层前向：conv_state/ssm_state 加载保存；后端选择 |
| `vllm/model_executor/layers/fla/ops/index.py` | `prepare_chunk_indices/offsets`（FLA 内部 chunk 元数据） |
| `vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py` | 融合 post-conv1d Triton kernel |
| `vllm/model_executor/layers/fla/ops/utils.py` | `FLA_CHUNK_SIZE = 64` |
| `vllm/model_executor/models/qwen3_5.py` | Qwen3.5 模型定义（hybrid: Attention + GDN） |
| `vllm/model_executor/models/interfaces.py` | `IsHybrid` protocol 与 `is_hybrid()` 查询 |
| `vllm/transformers_utils/configs/qwen3_5.py` | `layer_types` 配置（`full_attention_interval=4`） |
| `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` | 标准 Attention 的 chunked prefill Triton kernel |
