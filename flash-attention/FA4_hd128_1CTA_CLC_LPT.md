# FA4 Blackwell hd128 1CTA：varlen、causal、LPT 与 CLC 调度

> 整理日期：2026-07-14  
> 代码基线：FlashAttention 仓库 `muse` branch，commit `7cca125`  
> 上游背景：PR [#2218](https://github.com/Dao-AILab/flash-attention/pull/2218) 与 [#2346](https://github.com/Dao-AILab/flash-attention/pull/2346)  
> 范围：FA4 CuTeDSL，Blackwell SM100/SM110 forward，重点是 `head_dim=128 + 1CTA + varlen + causal + GQA + LPT + CLC`。

## 1. 先给结论

hd128 的 FA4 通用 SM100 kernel 支持 1CTA 下的 LPT 与 CLC。对于当前 `muse` branch：

```text
varlen + causal + MHA (Hq == Hkv)
    → 1CTA + SingleTileVarlenScheduler + LPT
    → host 因性能回退关闭 CLC

varlen + causal + GQA/MQA (Hq != Hkv)
    → 1CTA + SingleTileVarlenScheduler + LPT
    → 设置 FA_CLC=1 后允许 CLC
```

所以当前真正形成 `hd128 + 1CTA + varlen + causal + LPT + CLC` 的典型路径是 GQA/MQA，而不是 MHA。

CLC 和 LPT 是两个独立层次：

```text
CLC：运行时决定哪个已启动 CTA 接管哪个尚未启动的 work-ID
LPT：把 work-ID 映射成“较重的 causal Q block 优先”
```

它们都不会减少 attention 的数学 FLOPs。它们优化的是 SM 之间的任务分配和 kernel 尾部时间。

## 2. 为什么这是 1CTA

通用 SM100 forward 只有在固定长度、非 causal、非 local、非 SplitKV 等条件下才考虑 2CTA。host 端的关键条件包含：

```python
use_2cta_instrs = (
    arch // 10 in [10, 11]
    and not causal
    and not local
    and not is_split_kv
    and cu_seqlens_q is None
    and seqused_q is None
    ...
)
```

因此只要是 `varlen + causal`：

```text
use_2cta_instrs = False
cta_group_size  = 1
cluster_shape   = (1, 1)
```

一次 CLC work 对应一个 CTA，不涉及两个 CTA 协同领取同一个任务。

相关代码：

```text
flash_attn/cute/interface.py
flash_attn/cute/flash_fwd_sm100.py
```

## 3. 一个 attention work tile 是什么

FA4 scheduler 最终产生：

```text
WorkTileInfo(m_block, head_idx, batch_idx, split_idx)
```

各坐标含义：

- `m_block`：Q 序列方向上的 tile。
- `head_idx`：scheduler 看到的 head；pack GQA 时通常是 KV head。
- `batch_idx`：varlen batch 中的序列编号。
- `split_idx`：SplitKV 分片编号；非 SplitKV 为 0。

这个 work tile 不是一个物理内存对象，而是一个逻辑计算任务。它确定：

```text
读取哪段 Q
读取哪个 KV head 的 K/V
扫描哪些 n_block
结果写到哪段 O/LSE
```

## 4. flat tile 是什么

CLC 和 CUDA grid 更容易处理简单的一维或三维 grid 坐标，而不是直接理解 FA 的 `(batch, head, m_block, split)`。因此 scheduler 先把所有任务编号为：

```text
flat tile 0
flat tile 1
...
flat tile N-1
```

随后在 GPU kernel 内把 flat ID 解码成真实 attention 坐标。

固定长度时，每个 batch 的 block 数相同，可以用除法和取模解码。varlen 时，每条 sequence 的 block 数不同，例如：

```text
sequence A：3 blocks
sequence B：1 block
sequence C：2 blocks

prefix range：
A → [0, 3)
B → [3, 4)
C → [4, 6)
```

`flat=4` 落在 `[4,6)`，因此属于 C 的 local block 0。

FA4 不在 host 上为每个请求建立完整任务表，而是在 device scheduler 中读取 `cu_seqlens_q`，用 warp prefix-sum、shuffle 和 ballot 定位 batch。

核心位置：

```text
flash_attn/cute/tile_scheduler.py
SingleTileVarlenScheduler._get_num_m_blocks()
SingleTileVarlenScheduler._varlen_coord_map()
```

## 5. LPT 的原理与实现

LPT 是 Longest Processing Time First。经典目标是先启动预计最耗时的任务，再让短任务填补执行尾部，从而缩短 makespan。

### 5.1 为什么 causal tile 有轻重差异

对方形 causal self-attention，越靠后的 Q block 能看到的 KV 前缀越长：

```text
Q block 0 → KV block 0             ：约 1 份工作
Q block 1 → KV block 0..1          ：约 2 份工作
Q block 2 → KV block 0..2          ：约 3 份工作
Q block 3 → KV block 0..3          ：约 4 份工作
```

所以可以用 `m_block` 位置近似预测耗时：

```text
m_block 越大 → 扫描的 KV blocks 越多 → tile 越重
```

### 5.2 FA4 没有执行通用 sort

FA4 的 LPT 不是在 kernel 内构建任务数组并排序，而是用 O(1) 坐标反转：

```python
if const_expr(params.lpt):
    block = num_m_blocks - 1 - block
```

例如：

```text
自然映射：0, 1, 2, 3
LPT 映射：3, 2, 1, 0
```

`lpt` 由 host 根据 `causal or local` 设置，但它是 `cutlass.Constexpr[bool]`，因此具体反转逻辑会被 JIT 编译进 device scheduler；非 LPT 版本会在编译时消掉该分支。

这里的 `local` 指 **local/sliding-window attention（局部或滑动窗口注意力）**，不是 Python 的局部变量，也不是“只在单条 sequence 内处理”的意思。

公开接口通过窗口大小表达 local attention：

```python
flash_attn_func(
    q,
    k,
    v,
    window_size=(window_size_left, window_size_right),
)
```

对于位置 `i` 的 Q token，可见 K/V 范围近似为：

```text
[i - window_size_left, i + window_size_right]
```

并在 sequence 边界处截断。例如：

```text
window_size=(128, 64)
```

表示 Q 位置 `i` 最多关注左侧 128 个 token 和右侧 64 个 token。常见模式为：

```text
causal=True             → 当前位置及其左侧前缀
window_size=(128, 128)  → 双向 sliding window
window_size=(128, 0)    → 只向左看的局部窗口
未设置窗口              → dense/global attention
```

local attention 也可能存在 tile 负载差异：

```text
sequence 开头：窗口被左边界截断 → 扫描的 KV blocks 较少
sequence 中间：窗口完整         → 扫描的 KV blocks 较多
sequence 末尾：窗口被右边界截断 → 扫描的 KV blocks 较少
```

因此 host 对 causal 或 local 都设置：

```python
lpt = self.is_causal or self.is_local
```

这句话的准确含义是：host 根据 attention mask 类型选择是否编译带 LPT 坐标映射的 scheduler。host 只决定 `lpt=True/False`，真正的 block 映射仍在 GPU device scheduler 内执行。

还要注意算法精度：

- causal 的 tile 工作量通常随 `m_block` 单调增加，简单反转与 LPT 先验吻合。
- 双向 sliding window 的工作量更接近“轻 → 重 → 重 → 轻”。
- 当前 `num_m_blocks - 1 - block` 只是低成本启发式，并不是为双向 local tile 做了精确耗时估计和全局排序。

### 5.3 varlen 中是每条 sequence 内的近似 LPT

假设：

```text
Seq A：4 blocks → A3, A2, A1, A0
Seq B：2 blocks → B1, B0
Seq C：3 blocks → C2, C1, C0
```

当前实现不是把全 batch 所有 tile 按预计时间做严格全局排序，而是：

1. flat ID 通过 prefix-sum 定位 sequence。
2. 在该 sequence 内解出 `block/head`。
3. 根据 L2 section 做 head swizzle。
4. 对 local block 执行 `num_m_blocks - 1 - block`。

因此它是低成本、结构化的近似 LPT。

## 6. GQA 与 L2 swizzle

这一节先区分三个概念：

```text
GQA          ：多个 Q heads 在数学上共享一个 KV head
Pack-GQA     ：把这些 Q heads 的行折叠进同一个 MMA 的 M 维
L2 swizzle   ：改变 work-ID → (head, block) 的顺序，控制同时活跃的 KV 工作集
```

三者有关联，但不是一回事。GQA 是模型语义；Pack-GQA 是 kernel 数据布局和计算粒度；L2 swizzle 是 CTA 任务顺序。

### 6.1 GQA 到底共享了什么

设：

```text
Hq  = 32
Hkv = 8
r   = Hq / Hkv = 4
```

Q/K/V 的逻辑关系是：

```text
Q heads  0, 1, 2, 3  ──→ KV head 0
Q heads  4, 5, 6, 7  ──→ KV head 1
...
Q heads 28,29,30,31  ──→ KV head 7
```

数学上，每个 Q head 仍有独立的输出：

```text
O[q_head] = softmax(Q[q_head] @ K[kv_head]^T) @ V[kv_head]
```

只是同组的 `r` 个 Q heads 读取相同的 K 和 V。kernel 的优化机会来自：不要把同一份 K/V 当成完全无关的数据反复从较低存储层取回。

### 6.2 `pack_gqa=None/True/False` 的含义

接口首先计算：

```python
qhead_per_kvhead = num_head // num_head_kv
```

当前 forward 默认策略是：

```python
if pack_gqa is None:
    pack_gqa = qhead_per_kvhead > 1
```

因此：

| 参数 | 含义 |
| --- | --- |
| `pack_gqa=None` | 自动选择；GQA/MQA 默认打开，MHA 等价于不需要 pack |
| `pack_gqa=True` | 强制使用 packed GQA 布局，前提是该 kernel/feature 组合支持 |
| `pack_gqa=False` | 每个 Q head 作为独立 scheduler head/tile 处理 |

这里讨论的是 forward。当前 backward 路径会把 `pack_gqa` 强制设为 `False`，不能把 forward 的结论直接外推到 backward。

### 6.3 `pack_gqa=False`：Q head 是调度单位

未 pack 时，scheduler 的 `head_idx` 是 Q-head index：

```text
work tile = (batch, q_head, q_block, split)
```

kernel 再映射到 KV head：

```python
head_idx_kv = head_idx // qhead_per_kvhead
```

例如 `r=4`：

```text
CTA for Q head 0 ─┐
CTA for Q head 1 ─┼─→ 都读取 KV head 0
CTA for Q head 2 ─┤
CTA for Q head 3 ─┘
```

影响如下：

- 每个 CTA 的 M 维只包含一个 Q head 的 query rows，寻址简单。
- 同组的多个 CTA 会分别请求同一个 K/V tile。
- 如果 K/V 仍在 L2，后续 CTA 可以命中 L2；但 CTA 内没有把多个 Q heads 合并为一次 K/V 使用。
- scheduler 看到的是 `Hq` 个 heads，而不是 `Hkv` 个不同 K/V 工作集。L2 模型可能把共享同一 KV head 的 Q heads 视为独立 head，不能完整表达 GQA 共享关系。
- Q/O 通常更容易使用规则的 TMA tile；但具体仍取决于 varlen、SplitKV 和布局条件。

优点是并行粒度直接：Q heads 多时有大量独立 tiles。缺点是 KV load 指令、scheduler bookkeeping 以及跨 CTA 的 K/V 复用更依赖 L2。

### 6.4 `pack_gqa=True`：把 Q-head 维折进 M 维

Pack-GQA 不会复制或重新排列整个 Q/O tensor。`pack_gqa_layout()` 创建的是一个新的 CuTe layout view：

```text
Q/O 原布局：
(seqlen_q, headdim, Hq, batch)

packed view：
((r, seqlen_q), headdim, Hkv, batch)
```

LSE 类似地从：

```text
(seqlen_q, Hq, batch)
```

变成：

```text
((r, seqlen_q), Hkv, batch)
```

packed M 维中，Q-head offset 是快变化维。以 `r=4` 为例：

```text
packed row 0 → token 0, Q head offset 0
packed row 1 → token 0, Q head offset 1
packed row 2 → token 0, Q head offset 2
packed row 3 → token 0, Q head offset 3
packed row 4 → token 1, Q head offset 0
...
```

此时 scheduler 的 head 是 KV head：

```text
work tile = (batch, kv_head, packed_m_block, split)

scheduler head 0 → Q heads 0..3，共享 KV head 0
scheduler head 1 → Q heads 4..7，共享 KV head 1
```

一个 CTA 的 QK/PV MMA 的多行可以属于多个 Q heads，但这些行使用同一 KV head，因此同一个 CTA 载入的 K/V tile 可以服务多个 Q heads。

输出阶段必须把 packed row 反解回真实 `(token, q_head)`：

```text
token        = packed_row // r
q_head_offset= packed_row % r
q_head       = kv_head * r + q_head_offset
```

对应代码在 `PackGQA.compute_ptr()`、`load_Q()`、`store_O()` 和 `store_LSE()`。

### 6.5 Pack-GQA 改变了哪些东西

| 项目 | `pack_gqa=False` | `pack_gqa=True` |
| --- | --- | --- |
| scheduler head | Q head，数量 `Hq` | KV head，数量 `Hkv` |
| M 维语义 | token rows | `(q_head_in_group, token)` packed rows |
| KV head 映射 | `q_head // r` | scheduler head 本身 |
| CTA 内 K/V 复用 | 单个 Q head 使用 | 多个同组 Q heads 可共同使用 |
| Q/O/LSE 寻址 | 普通布局 | packed view + 输出反解/scatter |
| L2 模型看到的 head | 可能把共享 KV 的 Q heads 分开计算 | 与不同 KV 工作集更一致 |
| 并行粒度 | Q-head tiles 多 | head 轴缩小，但 packed M 轴变长 |
| 额外成本 | 跨 CTA 复用依赖 L2 | packed index、边界判断、特殊 load/store |

固定 M tile 的一个重要后果是：pack 后一个 tile 虽包含 `r` 个 Q heads，但每个 Q head 覆盖的 token 数大约缩小为原来的 `1/r`。因此不能简单推断：

```text
pack_gqa=True → 总 HBM K/V 流量必然下降 r 倍
```

更准确的说法是：

> Pack-GQA 把同组 Q heads 的 K/V 复用从“依赖多个 CTA 之间的 L2 命中”，提升为“一个 CTA 内显式共享一次载入的 K/V tile”；但端到端收益还取决于 packed tile 数、token 粒度、序列长度、可用并行度、TMA 路径和索引开销。

这也是为什么 Pack-GQA 必须实测，而不能只根据复用次数判断一定更快。

### 6.6 Pack-GQA 对 tile 数与 TMA 的影响

varlen scheduler 计算每条序列的 packed M blocks 时，会先把有效 Q 长度乘以 `r`：

```python
effective_q_rows = seqlen_q * r
num_m_blocks = ceil(effective_q_rows / tile_m_effective)
```

同时 scheduler head 数从 `Hq` 变为 `Hkv`。所以总 tile 数大致从：

```text
False：Hq  * ceil(seqlen_q     / tile_m_effective)
True ：Hkv * ceil(seqlen_q * r / tile_m_effective)
```

由于 `Hq = Hkv * r`，长序列时两者数量通常同阶，并不意味着 pack 后 CTA 数直接缩小 `r` 倍；差别主要在 tile 的内部组成和边界取整。

TMA 方面：

- packed Q 使用 TMA 的基本条件之一是 `m_block_size % r == 0`。
- 若不整除，`use_tma_Q=False`，改走普通异步 copy 和逐行 packed 指针计算。
- varlen 本身会使 `use_tma_O=False`，输出由 correction/epilogue warps 按 packed 指针写回。
- SplitKV、某些 block-sparse head 布局和专用 hd256 kernel 也可能关闭或限制 Pack-GQA。

因此 `pack_gqa=True` 不只改变 scheduler，还可能改变 load/store warp 的工作和同步路径。

### 6.7 L2 swizzle 要解决什么问题

先说明启用边界：**Pack-GQA 和 L2 swizzle 并不是绑定开关**。`pack_gqa=True` 负责改变 Q/O layout、scheduler head 的含义以及 CTA 内的 K/V 复用；本节这段 varlen forward 坐标 swizzle 则位于：

```python
if params.lpt or params.head_swizzle:
    # 估算 nheads_in_l2，并按 section 重排 (head, block)
```

在本文的 `causal/local + varlen` forward 场景里，host 设置 `lpt=True`，所以该分支会同时执行 L2 sectioning 和 block 反转。仅仅打开 `pack_gqa`，若既没有 `lpt` 也没有其他 `head_swizzle` 条件，并不能据此断言一定执行这段 L2 swizzle。

对同一个 `(batch, KV head)`，不同 causal Q blocks 会反复读取大量重叠 K/V：

```text
Q block 0 → K/V block 0
Q block 1 → K/V blocks 0..1
Q block 2 → K/V blocks 0..2
Q block 3 → K/V blocks 0..3
```

如果这些工作在时间上靠近，先前载入的 K/V cache lines 可能仍在共享 L2，后续 CTA 就不必再次从 HBM 读取。若调度顺序在大量 batch/heads 之间跳跃，复用距离变长，K/V 可能在再次使用前被挤出。

L2 swizzle 的目标是同时满足两个相反需求：

```text
locality：同时活跃的不同 K/V 工作集不要太多
parallelism：也不能只发一个 head，导致可并行 tiles 不够
```

所以它不是简单“按 head 顺序执行”，而是选择一个能放进 L2 的 head section 大小。

### 6.8 L2 容量启发式

当前 varlen scheduler 使用约 50 MiB 作为 K/V 可用 L2 预算：

```python
size_l2 = 50 * 1024 * 1024

kv_block_size = (
    (headdim + headdim_v)
    * element_size
    * tile_n
)

max_kvblock_in_l2 = size_l2 // kv_block_size
```

对于 `hdim=hdim_v=128`、BF16/FP16、`tile_n=128`：

```text
一个 K+V block
= (128 + 128) * 2 bytes * 128 tokens
= 65536 bytes
= 64 KiB

50 MiB / 64 KiB = 800 个 KV blocks
```

scheduler 再估算当前 sequence 每个 head 有多少 KV blocks：

```python
num_n_blocks ≈ num_m_blocks * tile_m_effective / r / tile_n
```

除以 `r` 是因为 packed M rows 包含 Q-head 维，必须还原成真实 token 数。当前实现用 Q blocks 推算 K blocks，隐含 `seqlen_q ≈ seqlen_k`；对 causal self-attention通常合理，对 cross-attention 只是性能近似。

接着从：

```text
16, 8, 4, 2, 1
```

中选择最大的 `nheads_in_l2`，满足：

```text
num_n_blocks * nheads_in_l2 <= max_kvblock_in_l2
```

并限制不超过实际 scheduler heads。

具体例子：

```text
seqlen_k = 8192  → 64 KV blocks/head
800 / 64 = 12.5 → 从候选中选择 8 heads/section

seqlen_k = 32768 → 256 KV blocks/head
800 / 256 = 3.125 → 选择 2 heads/section
```

序列越长，每个 KV head 的工作集越大，所以同一 section 中允许同时活跃的 heads 越少。

50 MiB 不是硬件保证的独占容量。L2 还可能被 Q/O、其他 kernel、系统流量和别的 SM 工作集占用，因此这只是经验启发式。

### 6.9 work-ID 如何被 swizzle

对某个 varlen batch，flat ID 先变成段内编号 `mh_block`。假设：

```text
num_m_blocks = 4
num_head     = 8
nheads_in_l2 = 2
```

则每个 L2 section 包含：

```text
2 heads * 4 blocks = 8 work tiles
```

映射公式的核心是：

```python
mh_in_l2   = nheads_in_l2 * num_m_blocks
section    = mh_block // mh_in_l2
l2_mod     = mh_block % mh_in_l2
block      = l2_mod // nheads_in_this_section
head_inner = l2_mod %  nheads_in_this_section
head       = section * nheads_in_l2 + head_inner

if lpt:
    block = num_m_blocks - 1 - block
```

得到的执行次序近似：

```text
section 0：
(head0, block3), (head1, block3),
(head0, block2), (head1, block2),
(head0, block1), (head1, block1),
(head0, block0), (head1, block0)

section 1：heads 2,3，仍从重 block 到轻 block
...
```

意义是：

- LPT 让重 causal blocks 先开始。
- section 让有限数量的 KV heads 在一段时间内集中活跃。
- section 内交错 heads，既保留并行度，又使其联合 K/V footprint 尽量留在 L2。
- Pack-GQA 开启时，scheduler head 就是 KV head，这个容量模型最自然。
- Pack-GQA 关闭时，scheduler head 是 Q head；多个相邻 Q heads 可能共享 KV head，但模型没有显式去重，因此通常更保守，也更依赖具体排列。

CLC 开启后，硬件决定哪个 Worker 实际取得某个尚未启动的 work-ID，所以时间顺序不再是严格串行的上述列表；但 swizzle 仍决定每个 ID 对应哪个逻辑 tile，从而限制某一 ID 区间涉及的 K/V 工作集。

### 6.10 Cache thrashing 是什么

你说的 “cache crushing” 在这里通常指 **cache thrashing（缓存抖动）**。它表示工作集频繁互相驱逐，导致数据刚进入 cache、还没被有效复用就被挤出，随后又从 HBM 重新加载。

一种典型坏调度：

```text
1. CTA 读取 batch A / KV head 0 的大段 K/V
2. scheduler 跳到 batch B / KV head 5
3. 又跳到 batch C / KV head 2
4. 大量新 K/V 占满 L2，A/0 被逐出
5. 再回到 batch A / KV head 0
6. 本可复用的 K/V 只能重新从 HBM 读取
```

这不是数值正确性问题，而是性能问题。常见来源包括：

- **capacity thrashing**：同时活跃的 K/V 总量超过 L2 容量。
- **reuse-distance 过长**：两次访问同一 K/V 之间插入了太多其他工作集。
- **并发干扰**：其他 SM、其他 kernel 或 Q/O 流量也在占用共享 L2。
- **调度打散**：CLC 或过宽的 head section 改善了负载均衡，却破坏了 K/V 的时间 locality。

可能观察到：

```text
L2 hit rate 下降
DRAM/HBM read bytes 上升
memory/long-scoreboard stall 增加
Tensor Core 等待 K/V 数据的时间增加
即使 SM tail 变短，kernel 总时间仍可能变差
```

所以 CLC、LPT、Pack-GQA 和 L2 swizzle 的关系是：

```text
LPT        → 优化任务轻重顺序
CLC        → 优化运行时 Worker 负载均衡
Pack-GQA   → 改变 CTA 内 GQA 复用与任务粒度
L2 swizzle → 控制跨 CTA 的 K/V 工作集和复用距离
```

它们可能协同，也可能互相抵消。最终必须比较 kernel time，并结合 L2 hit rate、HBM bytes 和 tail 分布解释结果，不能只看某一项理论优势。

此外 varlen scheduler 会估算 L2 能同时容纳多少 KV-head 工作集：

```python
kv_block_size = (headdim + headdim_v) * element_size * tile_n
max_kvblock_in_l2 = 50 MiB // kv_block_size
```

再从 `16/8/4/2/1` 中选择 `nheads_in_l2`，将 head 分成 section。目的不是改变正确性，而是避免同时展开太多不同 KV heads 导致 cache thrashing。

需要注意，当前估算用 Q blocks 近似 K blocks，隐含 `seqlen_q ≈ seqlen_k`。代码里也有 TODO 指出应读取真实 KV 长度。这个近似对 causal self-attention 通常成立，对 cross-attention 可能影响 locality，但不影响数学正确性。

### 6.11 对应代码位置

```text
flash_attn/cute/interface.py
  pack_gqa=None 的自动选择，以及 feature/kernel 限制

flash_attn/cute/pack_gqa.py
  pack_gqa_layout()：构造 packed layout view
  PackGQA.compute_ptr()/load_Q()/store_O()/store_LSE()：packed row 寻址

flash_attn/cute/flash_fwd_sm100.py
  use_tma_Q/use_tma_O 条件
  Q/O/LSE layout packing
  未 pack 时 q_head → kv_head 映射
  scheduler 参数 qhead_per_kvhead_packgqa

flash_attn/cute/tile_scheduler.py
  SingleTileVarlenScheduler.Params.create()：50 MiB 容量预算
  _get_num_m_blocks()：packed effective Q rows
  _varlen_coord_map()：nheads_in_l2、section swizzle 与 LPT block 反转
```

## 7. CLC 是怎么实现的

`ClcDynamicPersistentTileScheduler` 不是 FlashAttention 自己定义的类，而是 NVIDIA CUTLASS/CuTeDSL 提供：

```python
from cutlass.utils import (
    ClcDynamicPersistentTileScheduler,
    ClcDynamicPersistentTileSchedulerParams,
)
```

FlashAttention 自己实现 `ClcState` 作为适配层，组合：

```text
ClcDynamicPersistentTileScheduler
PipelineClcFetchAsync
producer/consumer PipelineState
```

### 7.1 CLC 的工作语义

kernel 仍启动覆盖完整任务空间的 grid。每个实际启动的 CTA 首先处理自己的 `blockIdx` 对应任务。完成当前任务前，scheduler warp 发起异步 CLC 查询：

```python
pipeline.producer_acquire(...)
mbarrier = pipeline.producer_get_barrier(...)
hw_scheduler.advance_to_next_work(mbarrier)
```

Blackwell 硬件尝试取消一个尚未开始运行的 CTA，并返回被取消 CTA 的 grid 坐标：

```text
当前 CTA 完成工作
       ↓
CLC try_cancel
       ↓
成功：取得尚未启动 CTA 的 work-ID，继续处理
失败：没有剩余可取消任务，退出
```

CLC 不会抢占已经在运行的 CTA，也不理解哪个 attention tile 更重。

### 7.2 CLC 是否只有在任务 CTA 数大于 SM 数时才有意义

结论是：**方向上接近，但准确条件不是“CTA 数大于物理 SM 数”，而是“grid 中的逻辑 Worker 数超过 GPU 同时可驻留的 Worker 数，因此仍有尚未启动、可被取消的 CTA/cluster”。**

需要区分三个数量：

```text
N_grid     ：本次 kernel launch 的 grid 中共有多少 CTA
N_valid    ：其中真正映射到 attention tile 的有效 work 数
R_resident ：GPU 同一时刻最多能驻留多少 Worker CTA
N_pending  ：已经进入 launch grid、但还没有开始执行的 CTA
```

在本文 hd128 1CTA 场景中，可以近似写成：

```text
R_resident ≈ SM 数量 × 每个 SM 可同时驻留的该 kernel CTA 数
N_pending  ≈ max(N_grid - R_resident, 0)
```

对 fixed-length dense grid，`N_grid` 和 `N_valid` 通常接近；对 varlen，grid 需要覆盖逐 sequence 向上取整的上界，可能包含 padding CTA。硬件层是否能执行 `try_cancel` 由 `N_grid` 中是否还有 pending CTA 决定，而性能层是否还有有用工作可接管则要看 pending CTA 映射后是否属于 `N_valid`。

但这里的 `R_resident` 由 threads、register、shared memory、tensor memory、barrier 和 cluster 约束共同决定，并不恒等于 SM 数。尤其要注意：

> “1CTA kernel”表示一次 MMA/attention work tile 由一个 CTA 承担，而不是保证每个 SM 只能驻留一个 CTA。

如果资源占用使一个 SM 只能驻留一个 CTA，那么才有：

```text
R_resident ≈ SM 数
```

此时“逻辑 CTA 数大于 SM 数”可以作为直观近似。若一个 SM 能同时驻留两个 CTA，则即使 `N_grid > SM 数`，也可能所有 CTA 已经同时启动，仍然没有可供 CLC 取消的 pending CTA。

CLC 的基本必要条件是：CTA 发出 `try_cancel` 时，launch queue 中还有尚未启动的 CTA/cluster：

```text
N_grid <= R_resident
  → grid 可能一开始就全部驻留
  → 没有 pending CTA 可以取消
  → 每个 CTA 只处理自己的初始 tile
  → CLC 查询很快失败，通常只有额外开销而没有调度收益

N_grid > R_resident
  → 至少存在第二波或后续 wave
  → 先完成的 CTA 可以取消一个 pending CTA
  → 当前 CTA 不退出，直接接管其 grid ID 并继续处理
```

例如假设 GPU 有 120 个 SM，并且这个 kernel 的有效 occupancy 是每 SM 一个 Worker CTA：

```text
N_grid = 80
  80 个 CTA 都可以进入首波
  没有未启动任务，CLC 基本无事可做

N_grid = 300
  首波约 120 个 CTA，约 180 个仍 pending
  某个 CTA 完成后可以通过 CLC 接管 pending ID
```

但 `N_grid > R_resident` 只是 CLC 能工作的必要条件，不是一定变快的充分条件：

- tile 几乎等长时，普通硬件 block scheduler 本来就能在 CTA 结束后补发下一波，CLC 的额外收益可能很小。
- varlen、causal、local 让 tile 耗时差异明显时，短 tile 的 CTA 可以较早接管 pending work，CLC 才更有机会降低 wave transition 和尾部空闲。
- CLC 查询、pipeline/barrier 和 scheduler warp 有成本；它也可能打散 K/V locality，引发更多 L2 miss。
- CLC 不能拆分或抢占一个已经运行的重 tile。若最后只剩几个已经启动的 straggler，且 launch queue 已空，CLC 无法把这些 tile 分给其他空闲 SM。
- 任务仅比一波略多一点时，虽然技术上可以取消 pending CTA，但可接管次数少，收益一般不稳定；多 wave 且负载不均衡的 workload 更有发挥空间。

因此更准确的判断是：

```text
有 pending work
    是 CLC 能发挥作用的前提；

有足够多的 wave + tile 耗时不均衡
    才是 CLC 更可能产生可测收益的场景；

CLC 调度收益 > 查询/同步成本 + locality 损失
    才会转化成最终 kernel 加速。
```

### 7.3 LPT 与 CLC 在这里如何分工

“重任务优先来自 work-ID 映射中的 LPT；CLC 只取得一个仍未启动的 work-ID”的完整含义是：

```text
LPT：定义 flat ID 解码成哪个 (batch, head, m_block)
     并通过 block 反转让较重 causal block 位于更靠前的逻辑顺序

CLC：请求取消一个尚未启动的 grid CTA/cluster
     成功后只得到该 CTA 的 grid ID
     再由相同的 varlen/L2/LPT 映射把 ID 解码成 attention tile
```

CLC 硬件不知道：

```text
某条 sequence 有多长
某个 causal tile 要扫描多少 K/V blocks
哪个 ID 对应重任务还是轻任务
```

它也不是一个按 tile cost 排序的 priority queue。LPT 在坐标映射层表达“重任务优先”，CLC 在执行层让已驻留 Worker 接管尚未开始的 grid work，两者是正交机制。

还要精确区分两种 valid：CLC response 的 `is_valid_tile=True` 只表示成功取消了一个未启动的 **grid CTA**。varlen grid 为容纳逐 sequence 向上取整可能含 padding，因此返回的 grid ID 仍要经过 `_varlen_coord_map()`；只有映射后的 `WorkTileInfo.is_valid_tile=True` 才表示真实有效的 attention tile。

### 7.4 shared-memory pipeline

FA4 为 CLC 增加：

- 16-byte response buffer。
- full/empty mbarrier。
- 当前实现一个 scheduler pipeline stage。

原本空闲的 warp 15 被用作 scheduler warp：

```python
while work_tile.is_valid_tile:
    tile_scheduler.prefetch_next_work()
    work_tile = tile_scheduler.advance_to_next_work()
```

load、MMA、softmax、correction、epilogue 等 warp 都消费相同的 scheduler state：

```python
consumer_wait()
work = get_current_work()
consumer_release()
```

这样 CTA 内所有角色会同步切换到同一个 `(batch, head, m_block, split)`。

核心代码：

```text
flash_attn/cute/tile_scheduler.py: ClcState
flash_attn/cute/flash_fwd_sm100.py: CLC pipeline 初始化
flash_attn/cute/flash_fwd_sm100.py: clc_scheduler_warp()
```

## 8. varlen + causal + GQA 下 CLC 与 LPT 的组合

完整执行链为：

```text
Host
  ├─ 判断这是 varlen → SingleTileVarlenScheduler
  ├─ causal=True     → lpt=True
  ├─ GQA + FA_CLC=1  → scheduling_mode=CLC
  └─ causal/varlen   → 1CTA

GPU initial work
  └─ blockIdx / initial CLC work
          ↓
    flat tile ID
          ↓
    读取 cu_seqlens_q
          ↓
    warp prefix-sum 定位 batch
          ↓
    解出 head 与 local m_block
          ↓
    L2 head section / GQA mapping
          ↓
    LPT：m_block = num_m_blocks - 1 - m_block
          ↓
    执行 attention tile
          ↓
    scheduler warp 发起下一次 CLC query
          ↓
    成功则把新的 flat ID 再走一遍相同映射；失败则退出
```

### 8.1 一个简化例子

假设只有一个 scheduler head：

```text
Seq A：4 blocks
Seq B：2 blocks
Seq C：3 blocks
```

flat 空间按各 sequence 的真实 block 数打包。LPT 映射后的逻辑顺序近似：

```text
A3, A2, A1, A0, B1, B0, C2, C1, C0
```

初始 CTA 各自从 grid 坐标获得任务。某个 CTA 较早完成后，CLC 取消一个尚未启动的 CTA 并返回其 flat ID。当前 CTA 随后用相同的 varlen/L2/LPT 映射解码这个 ID。

需要避免一种常见误解：CLC 不是从显式优先队列里“找最重 tile”。重任务优先来自 work-ID 映射中的 LPT；CLC 只取消一个仍未启动的 grid CTA 并取得其 ID，attention 层是否为有效 tile 还要经过 varlen 坐标映射判断。

## 9. 当前 host 的实际启用条件

CLC 默认关闭：

```python
_fa_clc_enabled = os.environ.get("FA_CLC", "0") == "1"
```

当前 `muse` branch 还会过滤两个已知回退场景：

```python
is_varlen_mha = is_varlen and qhead_per_kvhead == 1
is_dense_noncausal = not is_varlen and not causal and not local

use_clc_scheduler = (
    requested_use_clc_scheduler
    and not is_varlen_mha
    and not is_dense_noncausal
)
```

因此当前矩阵是：

| 场景 | `FA_CLC=1` 后实际 CLC | LPT |
| --- | --- | --- |
| hd128 varlen causal MHA | 否，host 回退 STATIC | 是 |
| hd128 varlen causal GQA/MQA | 是 | 是 |
| hd128 fixed causal/local | 是 | 是；`local` 指 sliding-window attention |
| hd128 dense non-causal | 否，host 回退 STATIC | 否 |

kernel 内还要求：

- 使用 TMA KV。
- 不进入 `overlap_sO_sQ` 不兼容路径。
- cluster N 为 1；本场景 cluster M 也是 1。

## 10. PR #2218 做了什么

PR #2218（2026-03-28 合并）给 FA4 SM100/SM110 forward 接入 CLC work stealing，约 `+890/-85`，涉及 8 个文件。

主要改动：

1. `tile_scheduler.py`
   - 新增 `SchedulingMode`。
   - 新增 `ClcState`。
   - 新增统一 `TileSchedulerProtocol`。
   - 让 `SingleTileLPTScheduler` 支持 STATIC/CLC。
   - 将 CLC grid 坐标映射回 FA 的 `(block, head, batch, split)`。

2. `flash_fwd_sm100.py`
   - 新增 `use_clc_scheduler`。
   - 分配 CLC response 和 mbarrier shared storage。
   - 创建 CUTLASS CLC scheduler 与 async pipeline。
   - 使用空闲 warp 15 发起 CLC 查询。
   - 让 load/MMA/softmax/epilogue 共享同一个 scheduler 实例。
   - 接入 1CTA/2CTA 同步框架。

3. `interface.py` / `utils.py`
   - 新增 `FA_CLC=1`。
   - 新增 `FA_DISABLE_2CTA=1`。
   - 把调度开关加入 JIT compile key。

4. 测试
   - 新增 `tests/cute/test_clc_fuzz.py`。
   - 覆盖不同 seqlen、GQA、head dim、causal/local、1CTA/2CTA 和重复执行。

PR #2218 本身主要完成 fixed-length/LPT CLC 集成。真正把 CLC 接入 `SingleTileVarlenScheduler` 的是后续 PR #2346（2026-03-30 合并）。

## 11. PR #2346 的 varlen 扩展

PR #2346 给 `SingleTileVarlenScheduler` 增加：

- `SchedulingMode.CLC` 参数。
- `clc_problem_shape()`。
- `ClcState` 成员。
- initial/current CLC work 到 flat ID 的转换。
- `prefetch_next_work()`、consumer wait/release、producer tail。
- varlen + causal/GQA/SplitKV 等正确性测试。

CLC 返回的 invalid response 不能信任其中的 tile 坐标，所以实现会用 `grid_dim` 作为 one-past-end flat ID，再让 `_varlen_coord_map()` 返回 invalid。这是避免读取垃圾 response 坐标的重要边界处理。

## 12. 与 FA1、FA2、FA3 的关系

LPT 不是 FA4 首创：

| 版本 | CTA 级 LPT | 调度方式 |
| --- | --- | --- |
| FA1 | 没有当前这种 CTA 级 LPT | 固定 grid/任务组织 |
| FA2 | 没有 | `m_block = blockIdx.x` 静态映射 |
| FA3 | 有 | 软件 dynamic persistent scheduler + global semaphore/counter |
| FA4 | 有 | CuTeDSL scheduler；Blackwell 上可再结合 CLC |

FA2 中一个 CTA 内部可能倒序遍历 K/V blocks，但那是单 tile 内的计算循环顺序，不等于 CTA 任务级 LPT。

FA3 的 Hopper scheduler 明确写有：

```cpp
// We use longest-processing-time-first scheduling:
// the longest remaining tile is assigned to the first SM that's free.
```

它通常只启动约等于可驻留 SM 数的 persistent CTAs。CTA 做完当前 tile 后，用全局原子计数器领取下一个逻辑 tile：

```cpp
next_tile = atomicAdd(tile_count_semaphore, 1);
```

这里变量叫 semaphore，但作用更像任务取号器或 work-queue head：原子性保证不同 CTA 不会领取到相同编号。

FA3 与 FA4 CLC 的区别：

```text
FA3 软件动态调度：
  启动少量 persistent CTA
  CTA 通过 atomicAdd 从软件任务队列领取下一个 ID

FA4 Blackwell CLC：
  启动覆盖完整任务空间的 grid
  已启动 CTA 用硬件 try_cancel 取消尚未启动的 CTA
  接管被取消 CTA 的 ID
```

目的相同，机制不同。

## 13. 如何确认一次运行真的用了 CLC

不能仅凭 kernel 源码中存在 CLC 分支判断运行时已经启用。应同时确认：

1. 启动进程前设置环境变量：

   ```bash
   FA_CLC=1
   ```

2. workload 没有被 host filter 回退，例如当前 varlen MHA 会回退。

3. 打开 host logging 后看到类似：

   ```text
   TileScheduler=SingleTileVarlenScheduler
   scheduling_mode=CLC
   USE_2CTA=False
   ```

4. 如需 kernel 级 trace，打开高等级 FA logging，确认出现真实运行输出：

   ```text
   [CLC] query sm=... cta=... (m_blk=...,h=...,b=...,s=...) valid=...
   ```

源码里存在 `"[CLC] query ..."` 字符串不算运行证据。

## 14. 常见误区

1. **“varlen + causal 一定用了 CLC”是错的。**  
   LPT 默认用于 causal/local；CLC 默认关闭，且当前 varlen MHA 即使请求也会回退。

2. **“CLC 会选择最重 tile”是错的。**  
   CLC 只返回尚未启动的 work-ID；LPT 映射才表达重 tile 优先。

3. **“LPT 是一个昂贵排序”是错的。**  
   当前 causal LPT 核心是 `num_m_blocks - 1 - block` 的 O(1) 隐式映射。

4. **“flat tile 就是一块 Q/K/V 数据”是错的。**  
   flat tile 只是任务 ID，device scheduler 解码后才确定数据范围。

5. **“使用 CLC 就不需要 L2 swizzle”是错的。**  
   CLC 解决动态负载均衡，L2 swizzle 解决 K/V locality；两者可能互相影响，必须一起 benchmark。

6. **“varlen causal 是 2CTA”是错的。**  
   当前通用 hd128 路径在 varlen/causal 下是 1CTA。

7. **“1CTA 就等于每个 SM 只能运行一个 CTA”是错的。**  
   1CTA 描述的是一个 work tile/MMA 的 CTA 协作粒度；实际每 SM 驻留数由 kernel resource occupancy 决定。

8. **“grid CTA 数只要大于 SM 数，CLC 就一定加速”是错的。**  
   首先要超过实际同时可驻留 Worker 数，产生 pending CTA；最终还要让负载均衡收益覆盖 CLC 开销和 L2 locality 损失。

## 15. 代码索引

FlashAttention：

```text
flash_attn/cute/interface.py
  - CLC 请求、host filter、1CTA/2CTA 选择、compile key

flash_attn/cute/flash_fwd_sm100.py
  - hd128 通用 SM100 forward
  - scheduler 选择、lpt 参数
  - CLC shared storage 与 pipeline
  - scheduler warp

flash_attn/cute/tile_scheduler.py
  - SchedulingMode
  - ClcState
  - TileSchedulerProtocol
  - SingleTileLPTScheduler
  - SingleTileVarlenScheduler

flash_attn/cute/pack_gqa.py
  - GQA packing/layout

tests/cute/test_clc_fuzz.py
  - CLC correctness/adversarial cases

AI/CLC_TRACE_DEBUG.md
  - CLC trace 与 deadlock 调试
```

CUTLASS/CuTeDSL：

```text
cutlass.utils.ClcDynamicPersistentTileScheduler
cutlass.utils.ClcDynamicPersistentTileSchedulerParams
cutlass.pipeline.PipelineClcFetchAsync
```

## 16. 验证边界

本文把四种状态明确区分：

```text
代码具备某分支
host 当前允许该分支
测试覆盖正确性
benchmark 实际启用了该分支
```

前面三项不能自动证明第四项。性能结论必须记录完整环境变量、scheduler log、GPU/CUDA/CuTeDSL 版本和目标 shape。
