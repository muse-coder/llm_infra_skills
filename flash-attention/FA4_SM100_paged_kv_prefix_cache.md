# FA4 SM100 推理：Prefill、Decode、Varlen Prefix Cache、Paged KV 与 PackGQA

> 范围：当前仓库中的 FlashAttention-4 CuTeDSL 前向实现，重点讨论 Blackwell
> SM100/SM110、标准 attention、head_dim=128。这里区分 runtime 的缓存管理、
> wrapper 接口看到的 tensor layout，以及 FA kernel 内部消费的 tile layout。

---

## 0. 先记住五个结论

1. **Prefill/Decode 不是两种 KV layout。** 它们描述 Q 的工作阶段；KV 可以是连续存储，也可以是 Paged KV。
2. **Paged KV 是 KV cache 的物理存储和寻址方案。** 常用于 decode，也用于 prefix cache 命中后的 chunked/varlen prefill。
3. **Prefix cache 是 runtime 语义。** runtime 负责 page 分配、共享、引用计数和写入；FA4 只按 `page_table` 读取。
4. **PackGQA 与 Paged KV 正交。** 前者重排 Q-head 到 M 维，后者映射逻辑 KV token 到物理 page。
5. **FA 内部单个 KV tile layout 不按 prefill/decode 切换。** hd128 默认仍消费 `N=128, D=128` 的 K/V tile；差别主要发生在 global-memory 寻址和调度。

```
阶段语义             KV 物理存储             Q-head 计算组织
prefill/decode   ×   contiguous/paged   ×   MHA/PackGQA
     独立                    独立                    独立
```

---

## 1. 三层 layout：不要混在一起

### 1.1 Runtime/wrapper 输入 layout

连续 KV：

```text
K: [batch, seqlen_k, num_kv_heads, head_dim]
V: [batch, seqlen_k, num_kv_heads, head_dim_v]
```

Packed varlen KV：

```text
K: [total_k, num_kv_heads, head_dim]
V: [total_k, num_kv_heads, head_dim_v]
cu_seqlens_k: [batch + 1]
```

Paged KV：

```text
K: [num_pages, page_size, num_kv_heads, head_dim]
V: [num_pages, page_size, num_kv_heads, head_dim_v]
page_table: [batch, max_num_pages_per_seq]
seqused_k: [batch]                 # 每条序列的有效 KV token 数
```

三种形式都要求最后的 head-dim 连续。Paged KV 的前两维不再是
`[batch, sequence]`，而是 `[physical_page, offset_in_page]`。

### 1.2 FA wrapper 内的逻辑 view

SM100 wrapper 会把输入 view 调整成适合 kernel 的访问顺序：

```text
连续/paged K: (sequence-or-page_size, D, Hkv, batch-or-num_pages)
连续/paged V: 对 SM100 使用适合 PV 的转置 view
```

这只是 CuTe tensor view/layout 变换，不会搬运整份 K/V。是否存在
`page_table` 决定最后一维应解释为 batch 还是 physical page。

### 1.3 FA kernel 内部 tile layout

hd128 默认 `tile_n=128`。无论当前调用属于 prefill 还是 decode，加载完成后，
K/V 都进入统一的 SMEM layout，供后续 QK、online softmax 和 PV 使用：

```text
global K/V
  ├─ contiguous：按 batch + logical n_block 直接定位
  └─ paged：logical n_block → page_table → physical page
                     │
                     ▼
              SMEM K/V tile
                 N=128, D=128
                     │
                     ▼
                 UMMA QK/PV
```

所以“Paged KV layout 不同”主要指 global memory 的物理布局与寻址；它不会创建一套
独立的 decode MMA layout。

---

## 2. 普通 Prefill

普通 prefill 一般没有可复用 prefix，Q/K/V 都覆盖完整 prompt：

```text
Q/K/V: [batch, prompt_len, heads, D]
causal=True
```

多请求长度不同但 K/V 连续 packed 时：

```text
Q: [total_q, Hq, D]       K/V: [total_k, Hkv, D]
cu_seqlens_q              cu_seqlens_k
```

这是“packed varlen prefill”，不是 Paged KV。它通过 `cu_seqlens_k` 找每条 K/V
序列的连续区间。

Paged KV 也可以用于完整 prefill，但需要 runtime 先把本次 prompt 的 K/V 写入 cache。
是否值得这么做取决于系统是否希望 prefill 后直接保留同一份 page 供 decode 使用。

---

## 3. Decode + Paged KV

Decode 每条请求通常只有一个新 Q token，但需要读取从第一个 token 到当前位置的全部 KV：

```text
Q: [batch, 1, Hq, D]，或 packed 为 [total_q, Hq, D]
K/V cache: [num_pages, page_size, Hkv, D]
page_table[b, logical_page] -> physical_page
seqused_k[b] = 当前有效 KV 长度
```

寻址过程：

```text
logical token position p
        │
        ├─ logical_page = p // page_size
        ├─ page_offset  = p %  page_size
        └─ physical_page = page_table[batch, logical_page]
                              │
                              ▼
              K/V[physical_page, page_offset, kv_head, :]
```

对于 hd128 默认 `tile_n=128`：

| page size | KV load 路径 |
|---|---|
| `page_size == 128` | 一个逻辑 KV tile 对应一个 page，可按 `page_table` 选择 page 后走 TMA |
| `page_size != 128` | 一个 tile 可能跨 page，走 `PagedKVManager` + async copy 路径 |

Decode 并没有专用的 KV SMEM/MMA layout。它的主要差异是 Q 很短、`q_stage` 通常为 1，
并可能使用 PackGQA 或 SplitKV 补充并行度。

### SplitKV 注意事项

SplitKV 和 Paged KV 也是正交概念：

```text
Paged KV：解决“KV 在哪里”
SplitKV：解决“一条很长的 KV 由多少 CTA 并行扫描”
```

FA4 当前 `num_splits` 默认是 1，不会自动拆分。传 `num_splits < 1` 才运行 heuristic，
或显式传 `num_splits > 1`；拆分后需要 combine kernel 合并 partial O/LSE。

---

## 4. Prefix cache + varlen/chunked prefill

### 4.1 语义

命中 prefix cache 后，不应该重新对已缓存 token 做完整 prefill。Q 只包含未命中的 suffix，
但每个 suffix Q 仍需要看到“共享 prefix + 当前 suffix”的全部 K/V。

```text
请求 A：prefix=1000，本轮 suffix=200，attention KV length=1200
请求 B：prefix=2000，本轮 suffix=100，attention KV length=2100

packed Q:       [300, Hq, D]
cu_seqlens_q:   [0, 200, 300]
seqused_k:      [1200, 2100]
```

K/V 不再通过 `cu_seqlens_k` 表达，而是位于全局 page pool：

```text
K/V:        [num_pages, page_size, Hkv, D]
page_table: [2, max_pages_per_seq]
```

### 4.2 page 共享

```text
请求 A page_table: [P3, P8, P12, A_private0, ...]
请求 B page_table: [P3, P8, P12, B_private0, ...]
                       └─ shared prefix ─┘
```

runtime 负责：

- 查找 prefix-cache hit；
- 让不同请求的 page table 引用相同 prefix pages；
- 为 suffix 分配私有 pages；
- 在 FA 调用前把 suffix K/V 写入正确 page；
- 管理引用计数、释放和必要的 copy-on-write。

FA4 kernel 不负责上述生命周期，只消费 `page_table` 和有效长度。

### 4.3 一次调用的数据流

```text
1. Runtime 找到每条请求的 cached prefix length
2. 为本轮 suffix 分配 page，并写入新 K/V
3. 把所有 suffix Q pack 成 total_q
4. 传 cu_seqlens_q，描述每条请求本轮 Q chunk
5. 传 page_table，描述每条请求完整逻辑 KV 的物理 page
6. 传 seqused_k = prefix_len + suffix_len
7. FA4 causal attention 只输出 suffix Q 对应的 O
```

典型接口组合：

```python
out, lse = flash_attn_varlen_func(
    q_suffix,                    # [total_new_q, Hq, D]
    k_cache_paged,               # [num_pages, page_size, Hkv, D]
    v_cache_paged,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=None,           # Paged KV 禁止同时传 packed-K 边界
    max_seqlen_q=max_chunk_q,
    max_seqlen_k=max_total_k,
    seqused_k=total_k_per_request,
    page_table=page_table,
    causal=True,
)
```

### 4.4 causal 对齐

这里 Q 是完整序列的尾部 chunk，而 K/V 是 prefix 加当前 chunk。Causal mask 必须按右下角对齐：

```text
Q 的全局起点 = seqlen_k - seqlen_q
```

因此 suffix 中第一个 Q 可以看到全部 prefix 和它自己，后续 Q 再逐个看到此前的 suffix。
调用方必须保证 `seqused_k`、Q chunk 长度以及已写入 cache 的 K/V 一致。

### 4.5 为什么不能同时传 `cu_seqlens_k`

两者表达的是互斥的 K/V 存储模型：

```text
cu_seqlens_k：每条 K/V 在一块 packed contiguous tensor 中的起止 offset
page_table： 每条逻辑 K/V 序列由哪些离散 physical pages 组成
```

当前 wrapper 明确要求：

```python
if page_table is not None:
    assert cu_seqlens_k is None
```

Paged varlen K 的真实长度使用 `seqused_k`，物理映射使用 `page_table`。

### 4.6 prefix 尾部不是整页

共享完整 page 最简单。若 prefix 在 page 中间结束，本轮 suffix 还要继续写这个 page，runtime
必须保证不会改坏其他请求共享的数据。常见策略是：

- 只共享完整 page；或
- 对最后一个部分页执行 copy-on-write；或
- 使用能安全管理部分页所有权的 block manager。

这属于 serving runtime 的职责，不是 FA4 kernel 自动处理的行为。

---

## 5. PackGQA：与 Paged KV/Prefix Cache 正交

设：

```text
qhead_per_kvhead = num_q_heads / num_kv_heads
```

当比值大于 1 时，FA4 默认启用 `pack_gqa`。它把共享同一个 KV head 的多个 Q head
打包到 tile 的 M 维：

```text
不 PackGQA：每个 Q head 分别调度，重复读取同一 KV head

PackGQA：
M rows = (token, q_head_within_group)
一个 tile 处理多个共享同一 KV head 的 Q heads
K/V 只按该 KV head 加载
```

PackGQA 不改变：

- K/V cache 的 `[num_pages, page_size, Hkv, D]` 物理 layout；
- `page_table` 的含义；
- prefix page 是否共享；
- logical token 到 physical page 的映射。

它会改变：

- Q 的有效 M 维长度：`seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead`；
- Q tile 数和 scheduler 看到的 head 数；
- `q_stage` 选择；
- 某些输出 epilogue/TMA-O 路径。

因此这些组合都合理：

| Q-head 组织 | KV 存储 | 场景示例 |
|---|---|---|
| MHA | contiguous | 普通 prefill |
| PackGQA | contiguous | GQA prefill |
| MHA | paged | MHA decode/prefix prefill |
| PackGQA | paged | GQA/MQA decode 或 prefix-cache varlen prefill |

---

## 6. 场景对照表

| 场景 | Q 表示 | K/V 表示 | 长度元数据 | 常用优化 |
|---|---|---|---|---|
| dense prefill | `[B, Sq, Hq, D]` | `[B, Sk, Hkv, D]` | tensor shape | q_stage=2、可 PackGQA |
| packed varlen prefill | `[total_q, Hq, D]` | `[total_k, Hkv, D]` | `cu_seqlens_q/k` | varlen scheduler、可 PackGQA |
| paged decode | 每请求通常 1 个 Q | page pool | `page_table + seqused_k` | PackGQA、可 SplitKV |
| prefix-cache varlen prefill | packed suffix Q | prefix+suffix page pool | `cu_seqlens_q + page_table + seqused_k` | varlen scheduler、可 PackGQA/SplitKV |

判断逻辑：

```text
“Q 是 full prompt、suffix chunk 还是单 token？” → prefill/chunked prefill/decode
“K/V 连续还是物理 page pool？”             → contiguous/paged
“多个 Q head 是否共享 KV head？”            → MHA/GQA；是否 PackGQA
“KV 太长而 Q tile 太少吗？”                 → 是否 SplitKV
```

---

## 7. SM100 当前实现限制与容易踩坑的点

- Paged KV 的实际读取实现位于 SM100/SM110 通用前向 kernel；SM80、SM120 明确拒绝。
- 当前 SM90 通用前向签名虽有 `mPageTable`，但没有实际按它完成 paged 寻址，不应视为可用支持。
- 本文讨论 inference forward；不要默认 Paged KV backward 可用。
- Paged KV 与 `cu_seqlens_k` 不能同时使用。
- `page_table` 必须是 int32，最后一维连续，shape 为 `[batch, max_pages_per_seq]`。
- `seqused_k` 必须准确给出每个请求的有效 KV 长度，不能把 page pool 容量当作有效长度。
- `page_size=tile_n=128` 是 hd128 的高效 TMA 路径；其他 page size 会进入更通用的 async-copy 路径。
- Prefix page 的共享、部分页 copy-on-write 和 suffix K/V 写入顺序由 runtime 保证。
- `num_splits=1` 是默认值；想让 FA4 自动选择 SplitKV，需传小于 1 的值。
- `pack_gqa` 默认由 `num_q_heads > num_kv_heads` 触发，但仍是独立优化，不是启用 Paged KV 的前提。

---

## 8. 源码入口

| 主题 | 文件/位置 |
|---|---|
| wrapper shape 与 `page_table` 校验 | `flash_attn/cute/interface.py::_flash_attn_fwd` |
| q_stage、SplitKV、2CTA 选择 | `flash_attn/cute/interface.py::_flash_attn_fwd` |
| SM100 tensor view 与 TMA 构造 | `flash_attn/cute/flash_fwd_sm100.py::FlashAttentionForwardSm100.__call__` |
| paged/contiguous 分支 | `flash_attn/cute/flash_fwd_sm100.py::load_Q` 所在主循环附近 |
| 非 TMA 跨 page 加载 | `flash_attn/cute/paged_kv.py::PagedKVManager` |
| SplitKV partial 合并 | `flash_attn/cute/flash_fwd_combine.py` |
| PackGQA 行映射与 store | `flash_attn/cute/pack_gqa.py` |

---

## 9. 一句话心智模型

```text
Prefix cache 决定“哪些历史 KV 可以复用”；
Paged KV 决定“这些 KV 存在哪些物理 page”；
varlen metadata 决定“每条请求本轮 Q 和有效 K 有多长”；
PackGQA 决定“共享一个 KV head 的多个 Q head 如何塞进 M tile”；
FA4 最终把不同 global-memory 来源统一加载成相同的内部 K/V tile 做 attention。
```

---

## 变更记录

- 2026-07-16：首次创建；覆盖 prefill、decode、prefix-cache varlen、Paged KV、PackGQA、SplitKV 及 layout 分层。
