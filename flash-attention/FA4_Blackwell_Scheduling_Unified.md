# FA4 Blackwell 前向调度：varlen、GQA、LPT、L2 swizzle 与 CLC

> 整理日期：2026-07-14
>
> 范围：FlashAttention-4（CuTeDSL）在 Blackwell SM100/SM110 上的 forward kernel。
>
> 本文整合 `FA4_hd128_1CTA_CLC_LPT.md` 与 `FA4_varlen_GQA_causal_CLC_LPT.md`。两个原文档保留，不作为本文写作过程中的删除或覆盖对象。

## 1. 范围与核心结论

本文讨论的不是单个优化开关，而是一条完整的 attention work 调度链：

```text
线性 work-ID
    ↓
varlen 映射：找到真实 batch/sequence
    ↓
GQA/Pack-GQA 映射：确定 Q head 与 KV head 的关系
    ↓
L2 swizzle：限制同时活跃的 K/V 工作集
    ↓
LPT：让较重的 causal/local Q tiles 尽早启动
    ↓
CTA 或 CTA cluster 执行 attention tile
    ↓
STATIC 结束，或由 CLC 获取尚未启动的下一份 grid work
```

其中各机制的职责必须分开理解：

| 机制 | 核心职责 | 不负责什么 |
| --- | --- | --- |
| varlen mapping | 将扁平 ID 映射到不同长度的 sequence | 不负责判断 tile 轻重 |
| GQA mapping | 表达多个 Q heads 共享一个 KV head | 不自动保证 K/V 留在 L2 |
| Pack-GQA | 将同组 Q heads 折入 packed M 维，形成 CTA 内 K/V 复用 | 不等价于 L2 swizzle |
| L2 swizzle | 控制邻近 work-ID 涉及的 KV-head 工作集 | 不执行动态 work stealing |
| LPT | 通过坐标反转让重 causal tile 位于较早的逻辑顺序 | 不是任务队列，也不做通用排序 |
| CLC | 让已启动 Worker 接管尚未启动的 grid CTA/cluster | 不知道哪个 tile 最重，也不保证 locality |

### 1.1 当前代码中已经成立的路径

对于通用 SM100 forward 的典型组合：

```text
head_dim=128
varlen=True
causal=True
GQA/MQA
pack_gqa=True（默认自动选择）
```

当前代码已经具备：

- `SingleTileVarlenScheduler` 的 varlen 坐标映射。
- causal/local 对应的 LPT block 反转。
- 基于约 50 MiB K/V 预算的 L2 head section。
- Pack-GQA 的 packed layout 和 CTA 内 K/V 共享。
- 通过 `FA_CLC=1` 请求 CLC；满足 host/kernel 条件时使用 CLC scheduling mode。

因此在这条路径中，CLC、LPT、Pack-GQA 和 L2 swizzle 可以同时存在，但它们仍是四个不同层次的机制。

### 1.2 当前代码中尚未完整成立的路径

当前非 FP8 的 `head_dim=256` 会进入专用 Blackwell forward。该 kernel 已包含部分 CLC 相关类型、buffer 和 scheduler plumbing，但构造函数当前明确固定：

```python
self.use_clc_scheduler = False
```

同时它还具有以下现状：

- 使用专用的协作 CTA cluster 路径。
- `pack_gqa=False`，GQA 采用 unpacked Q-head 表达。
- 接受 varlen 输入，但当前不是通用 hd128 的 compact varlen/LPT/L2 组合。
- CLC 骨架存在不等于运行时已经启用 CLC。

因此不能把 hd128 通用路径的完整调度结论直接套到 hd256 专用 kernel。本文会先解释共同机制，最后一章再集中比较两类 CTA 协作模式及其当前支持边界。

### 1.3 三条最重要的判断原则

第一，源码中存在某段实现，不代表一次运行实际使用了它：

```text
代码具备某分支
≠ host 允许该分支
≠ JIT specialization 选择该分支
≠ benchmark 实际运行了该分支
```

第二，CLC 有 pending work 才能执行取消与接管，但这只是必要条件：

```text
存在尚未启动的有效 grid work
    → CLC 才有任务可接管

负载均衡收益 > CLC 同步成本 + L2 locality 损失
    → 才可能产生最终 kernel 加速
```

第三，LPT 和 CLC 不是替代关系：

```text
LPT：决定一个 work-ID 解码成哪个重/轻 attention tile
CLC：取得一个尚未启动的 grid ID，再交给同一套映射解码
```

CLC 不是按 tile cost 排序的 priority queue；它既不读取 sequence 长度，也不计算 causal tile 的 K/V trip count。

## 2. Work tile、flat ID 与 varlen 映射

### 2.1 一个 attention work tile 是什么

前向 FlashAttention 将 Q 的 sequence 维切成多个 M blocks。一个逻辑 work tile 可以抽象成：

```text
(batch_idx, scheduler_head_idx, m_block, split_idx)
```

其中：

- `batch_idx` 表示哪条 sequence。
- `scheduler_head_idx` 的语义取决于 Pack-GQA：可能是 Q head，也可能是 KV head。
- `m_block` 表示 Q sequence 上的第几个 tile。
- `split_idx` 只在 SplitKV 等路径中有实际意义。

一个 work tile 内部不是只做一次矩阵乘。它会固定一个 Q tile，然后扫描该 tile 允许访问的 K/V blocks：

```text
加载 Q tile
    ↓
for n_block in 可见的 K/V 范围：
    加载 K/V tile
    Q @ Kᵀ
    online softmax 更新 row max / row sum
    P @ V 并累计输出
    ↓
写回 O 和 LSE
```

所以调度器分配的是一个 Q-side attention tile，而不是单独的一块 K、V 数据。

### 2.2 逻辑 tile、物理 CTA 与 scheduler Worker

下面三个概念不能混为一谈：

```text
逻辑 work tile
  attention 算法中的一份工作，例如 (batch, head, m_block)

物理 CTA
  CUDA grid 中实际启动的 thread block

scheduler Worker
  能够独立领取下一份逻辑工作的执行单位
```

在简单路径中，一个物理 CTA 对应一个 Worker，并处理一个逻辑 tile。使用 persistent/CLC 时，同一个已驻留 Worker 可以连续处理多个逻辑 tile。使用协作 cluster 时，一个 Worker 也可能由多个物理 CTA 共同组成。

因此文档中所说“tile 数”“CTA 数”和“Worker 数”必须结合上下文：

- `num_work_tiles` 是算法工作量。
- `gridDim` 是 launch 的物理 CTA 空间。
- `resident_workers` 是同一时刻真正驻留并可以执行工作的单位数。

### 2.3 flat tile 只是线性任务编号

多维 tile 坐标不适合直接作为统一调度队列，因此 scheduler 先使用一个扁平编号：

```text
flat_tile_id = 0, 1, 2, ...
```

再由 device scheduler 解码：

```text
flat_tile_id
    ↓
(batch_idx, head_idx, m_block, split_idx)
```

flat ID 本身：

- 不包含 Q/K/V 数据。
- 不表示显存地址。
- 不直接说明任务轻重。
- 只有经过 varlen、head section 和 LPT 映射后才确定实际 tile。

这也是 CLC 能与 FA scheduler 组合的基础：CLC 只需要返回一个 grid coordinate/ID，FA 再用自己的映射解释这个 ID。

### 2.4 fixed-length 的矩形任务空间

固定长度时，每个 batch/head 的 Q blocks 数相同。假设：

```text
batch = 2
scheduler heads = 3
num_m_blocks = 4
```

逻辑空间可以看成规则矩形：

```text
2 batches × 3 heads × 4 blocks = 24 work tiles
```

此时 flat ID 可以通过常量除法和取模直接解码，通常不需要读取每条 sequence 的长度。

### 2.5 varlen 为什么不能直接使用最大长度矩形

varlen batch 中每条 sequence 的有效 Q 长度不同：

```text
Seq A = 512 tokens
Seq B = 2048 tokens
Seq C = 128 tokens
```

若按 `max_seqlen_q=2048` 为每条 sequence 建相同矩形 grid，A/C 会产生大量越界 tiles。通用 varlen scheduler 会根据每条 sequence 的真实长度计算：

```python
num_m_blocks[b] = ceil(seqlen_q[b] / tile_m_effective)
```

若开启 Pack-GQA，则有效 M rows 还要乘以同一个 KV head 对应的 Q-head ratio：

```python
effective_q_rows = seqlen_q[b] * qhead_per_kvhead
num_m_blocks[b] = ceil(effective_q_rows / tile_m_effective)
```

### 2.6 prefix-sum 如何定位 sequence

假设三个 sequence 分别有：

```text
Seq A：4 blocks
Seq B：2 blocks
Seq C：3 blocks
```

忽略 head 后，其累计边界为：

```text
prefix = [0, 4, 6, 9]
```

一个 flat ID 落在哪个区间，就属于哪条 sequence：

```text
[0, 4) → Seq A
[4, 6) → Seq B
[6, 9) → Seq C
```

FA4 的 `SingleTileVarlenScheduler._varlen_coord_map()` 使用 warp-level prefix-sum、ballot、shuffle 等操作定位 batch，再计算 batch 内部的 `mh_block`，最后解出 head 与 local block。

完整职责可以写成：

```text
flat ID
    ↓
读取 cu_seqlens_q 或 seqused_q
    ↓
计算各 sequence 的真实 num_m_blocks
    ↓
warp prefix-sum 找到目标 batch
    ↓
减去前面 batches 的累计工作量
    ↓
得到 batch 内 mh_block
    ↓
L2/head/block/LPT 映射
```

### 2.7 compact mapping 的准确边界

“varlen compact”并不意味着 launch grid 中绝对没有任何 padding。host 计算 grid 时需要覆盖每条 sequence 向上取整后的 block 数上界，因此可能存在少量不能映射为真实 attention tile 的 grid IDs。

应区分：

```text
CLC response valid
  表示硬件成功取消了一个尚未启动的 grid CTA/cluster

WorkTileInfo valid
  表示该 grid ID 经过 varlen 映射后确实对应真实 attention work
```

当前实现对 CLC exhaustion 还有一个重要保护：CLC 返回 invalid response 时，其中的 tile coordinate 不能被信任。代码会把 tile ID 设为 `grid_dim` 这个 one-past-end 值，再让 `_varlen_coord_map()` 安全地产生 invalid work。

### 2.8 本章对应代码

```text
flash_attn/cute/tile_scheduler.py
  WorkTileInfo
  SingleTileVarlenScheduler._get_num_m_blocks()
  SingleTileVarlenScheduler._varlen_coord_map()
  SingleTileVarlenScheduler.get_current_work()

flash_attn/cute/seqlen_info.py
  SeqlenInfoQK：真实 Q/K 长度与 offset

flash_attn/cute/flash_fwd_sm100.py
  TileSchedulerArguments 构造
  scheduler 返回坐标后的 Q/K/V tile 选择
```

## 3. Causal、local 与 LPT

### 3.1 causal tile 为什么有轻重差异

对方形 causal self-attention，位置越靠后的 Q block 能看到越长的 K/V 前缀。假设每个 Q/KV tile 大小相同：

```text
Q block 0 → KV block 0
Q block 1 → KV blocks 0..1
Q block 2 → KV blocks 0..2
Q block 3 → KV blocks 0..3
```

因此单 tile 的 K/V loop 次数近似为：

```text
1, 2, 3, ..., B
```

一条 sequence 的 causal 总工作量近似为：

```text
1 + 2 + ... + B = B(B+1)/2
```

这意味着自然 block 顺序下，最重的任务位于后面。如果最后一波才启动这些重 tile，其他 SM 可能已经空闲，只剩少量重 tile 形成很长的 kernel tail。

### 3.2 LPT 的核心思想

LPT 是 Longest Processing Time First：先启动预计耗时最长的任务，再用轻任务填补执行尾部。

对 causal Q blocks，`m_block` 与工作量近似单调相关，因此不需要真的计算每个 tile 的精确 FLOPs，也不需要构造数组排序。FA4 使用坐标反转：

```python
if params.lpt:
    block = num_m_blocks - 1 - block
```

例如：

```text
自然映射：0, 1, 2, 3
LPT 映射：3, 2, 1, 0
```

这是一种 O(1) 的隐式 LPT：

- 不分配任务数组。
- 不执行 comparison sort。
- 不使用全局优先队列。
- 只改变 work-ID 到 `m_block` 的解码关系。

### 3.3 host 和 kernel 各自做什么

通用 SM100 forward 在 host/JIT 配置阶段设置：

```python
lpt = self.is_causal or self.is_local
```

这里 host 做的是选择 specialization：

```text
causal/local → 编译 lpt=True 的 scheduler
dense global → 通常编译 lpt=False 的 scheduler
```

`lpt` 是 compile-time constant。真正执行 `num_m_blocks - 1 - block` 的是 GPU device scheduler；非 LPT 版本中的分支会被编译器消掉。

所以“LPT 由 host 根据 causal/local 设置”的准确含义是：host 决定是否编译 LPT 坐标映射，而不是 host 预先生成并上传一张排好序的任务表。

### 3.4 这里的 local 是 sliding-window attention

`local` 指局部/滑动窗口注意力，不是 Python 局部变量，也不是“只在本 sequence 内运行”。公开接口通常通过左右窗口表达：

```python
window_size=(window_size_left, window_size_right)
```

对 Q token `i`，可见 K/V 范围近似为：

```text
[i - window_size_left, i + window_size_right]
```

并在 sequence 两端截断。例如：

```text
window_size=(128, 0)    → 当前位置及左侧最多 128 tokens
window_size=(128, 128)  → 双向 sliding window
未设置窗口              → dense/global attention
```

local attention 也会产生 tile 负载差异：

```text
sequence 开头：窗口被左边界截断，工作较少
sequence 中间：窗口完整，工作较多
sequence 末尾：窗口可能被右边界截断，工作较少
```

但需要注意：双向 local 的负载形状更接近“轻 → 重 → 重 → 轻”，不一定随 `m_block` 单调增加。因此同一个 block 反转只是低成本启发式，不是对 local tile cost 的精确全局 LPT。

### 3.5 varlen 中是 sequence 内近似 LPT

假设：

```text
Seq A：4 blocks
Seq B：2 blocks
Seq C：3 blocks
```

每条 sequence 内的 block 可以被映射成：

```text
A3, A2, A1, A0
B1, B0
C2, C1, C0
```

但当前通用 varlen scheduler 不会把所有 batches、heads 的 tile 先做一次全局耗时估计，再执行严格全局 sort。实际流程是：

1. flat ID 通过 prefix-sum 定位 sequence。
2. 在该 sequence 内得到 `mh_block`。
3. 执行 L2 head section 映射。
4. 对该 sequence 的 local block 做反转。

因此它是结构化、低成本的近似 LPT。若 batch 是明显长尾：

```text
[64K, 16K, 4K, 2K]
```

每条 sequence 内重到轻，并不等价于整个 batch 所有 tiles 已按真实耗时全局排序。batch 输入顺序、head section、cache hit 和运行时发射顺序仍会产生残余不均衡。

### 3.6 LPT 改变的是启动顺序，不改变数学结果

LPT 只置换独立 Q tiles 的调度顺序：

```text
相同输入 Q/K/V
相同 causal/local mask
相同 tile 内 K/V trip range
不同 tile 启动先后顺序
```

每个 tile 的 online softmax 和输出仍在自己的 Q rows 上计算，因此 LPT 不改变 attention 的数学语义。它优化的是 makespan、wave 分布和尾部利用率。

### 3.7 LPT 不等于 tile 内倒序扫描 K/V

需要区分两种“倒序”：

```text
CTA 任务级 LPT
  改变哪个 Q block 先被某个 CTA/Worker 执行

tile 内 K/V loop 顺序
  改变同一个 Q tile 内先扫描哪个 K/V block
```

后者可能影响 pipeline、mask 或数值累加次序，但不是本文所说的 CTA 任务级 LPT。

### 3.8 本章对应代码

```text
flash_attn/cute/flash_fwd_sm100.py
  TileSchedulerArguments(..., lpt=self.is_causal or self.is_local)

flash_attn/cute/tile_scheduler.py
  SingleTileLPTScheduler
  SingleTileVarlenScheduler._varlen_coord_map()
  block = num_m_blocks - 1 - block

flash_attn/cute/block_info.py
  causal/local 下每个 Q tile 的 K/V block range

flash_attn/cute/mask.py
  AttentionMask 的 causal/local 语义
```

## 4. GQA、Pack-GQA 与 TMA

### 4.1 GQA 的共享语义

设：

```text
Hq  = 32
Hkv = 8
r   = Hq / Hkv = 4
```

Q heads 与 KV heads 的关系为：

```text
Q heads  0..3   → KV head 0
Q heads  4..7   → KV head 1
...
Q heads 28..31  → KV head 7
```

每个 Q head 仍然产生独立输出：

```text
O[q_head] = softmax(Q[q_head] @ K[kv_head]ᵀ) @ V[kv_head]
```

共享的是 K/V head，不是 Q 或 O。优化机会来自让同组 Q heads 在 CTA 内或 L2 中复用同一份 K/V。

### 4.2 GQA、Pack-GQA 和 L2 swizzle 不是一回事

```text
GQA
  模型数学语义：多个 Q heads 共享一个 KV head

Pack-GQA
  kernel layout/计算粒度：把同组 Q heads 折进 MMA 的 M 维

L2 swizzle
  调度顺序：让共享或重叠 K/V 的跨 CTA 工作在时间上靠近
```

GQA 总是存在于模型输入关系中；Pack-GQA 和 L2 swizzle 是两个可以独立选择的实现优化。

### 4.3 `pack_gqa=None/True/False`

forward host 首先计算：

```python
qhead_per_kvhead = num_head // num_head_kv
```

默认选择为：

```python
if pack_gqa is None:
    pack_gqa = qhead_per_kvhead > 1
```

因此：

| 参数 | 实际含义 |
| --- | --- |
| `pack_gqa=None` | 自动选择；GQA/MQA forward 默认尝试开启，MHA 无需 pack |
| `pack_gqa=True` | 请求 packed GQA，前提是目标 kernel 和 feature 组合支持 |
| `pack_gqa=False` | 每个 Q head 作为独立 scheduler head/tile 处理 |

这里的结论针对 forward。当前 backward 尚不支持 Pack-GQA，会将其设为 `False`。

此外，“请求 `True`”不保证最终所有路径都保持为 `True`。例如某些 block-sparse head layout 或专用 kernel 会关闭/拒绝 Pack-GQA。

### 4.4 `pack_gqa=False`：scheduler head 是 Q head

未 pack 时，一个逻辑 tile 近似为：

```text
(batch, q_head, q_block, split)
```

kernel 内再计算：

```python
head_idx_kv = head_idx // qhead_per_kvhead
```

例如 `r=4`：

```text
Q-head CTA 0 ─┐
Q-head CTA 1 ─┼─→ 都使用 KV head 0
Q-head CTA 2 ─┤
Q-head CTA 3 ─┘
```

其影响是：

- 每个 CTA 的 M rows 只来自一个 Q head，Q/O 寻址规则。
- 同组 Q-head CTAs 会分别请求相同 K/V。
- CTA 内没有把多个 Q heads 合并成一次 K/V 使用。
- 跨 CTA 的 K/V 复用依赖调度邻近性和共享 L2 命中。
- scheduler 看到 `Hq` 个 heads，而真实不同 K/V 工作集只有 `Hkv` 个。

最后一点对 L2 容量模型很重要：如果直接把每个 scheduler head 当成独立 KV 工作集，就会高估 GQA 的 K/V footprint。

### 4.5 `pack_gqa=True`：把 Q-head-in-group 折入 M 维

Pack-GQA 不会预先复制或物理重排整块 Q/O tensor。`pack_gqa_layout()` 构造一个新的 CuTe layout view：

```text
Q/O 原始逻辑布局：
(seqlen_q, headdim, Hq, batch)

packed view：
((r, seqlen_q), headdim, Hkv, batch)
```

LSE 从：

```text
(seqlen_q, Hq, batch)
```

变为：

```text
((r, seqlen_q), Hkv, batch)
```

packed M 维中，`q_head_in_group` 是快变化坐标。以 `r=4` 为例：

```text
packed row 0 → token 0, q_head_in_group 0
packed row 1 → token 0, q_head_in_group 1
packed row 2 → token 0, q_head_in_group 2
packed row 3 → token 0, q_head_in_group 3
packed row 4 → token 1, q_head_in_group 0
...
```

此时 scheduler head 直接表示 KV head：

```text
(batch, kv_head, packed_m_block, split)
```

一个 CTA 的多行 QK/PV MMA 可以属于同组的多个 Q heads，而这些行共享同一个 KV head，因此一次载入的 K/V tile 可以服务多个 Q heads。

输出写回时再解码：

```text
token          = packed_row // r
q_head_in_group= packed_row % r
q_head         = kv_head * r + q_head_in_group
```

### 4.6 两条路径的直接对比

| 项目 | `pack_gqa=False` | `pack_gqa=True` |
| --- | --- | --- |
| scheduler head | Q head，数量 `Hq` | KV head，数量 `Hkv` |
| M 维语义 | token rows | `(q_head_in_group, token)` packed rows |
| KV head 映射 | `q_head // r` | scheduler head 本身 |
| CTA 内 K/V 复用 | 单个 Q head 使用 | 多个同组 Q heads 共享 |
| 跨 CTA K/V 复用 | 主要依赖 L2 | 仍存在，但 CTA 内先完成一层复用 |
| Q/O/LSE 寻址 | 普通 layout | packed view 与指针解码/scatter |
| head 轴并行度 | 较多 | head 轴缩小，packed M 轴变长 |
| L2 工作集模型 | 需识别多个 Q heads 共享 KV | scheduler head 天然对应 KV 工作集 |

### 4.7 Pack-GQA 不代表 CTA 数或 HBM 流量固定下降 `r` 倍

varlen scheduler 在 packed 模式下先计算：

```python
effective_q_rows = seqlen_q * r
num_m_blocks = ceil(effective_q_rows / tile_m_effective)
```

总 tile 数大致为：

```text
unpacked：Hq  × ceil(seqlen_q     / tile_m_effective)
packed：  Hkv × ceil(seqlen_q × r / tile_m_effective)
```

由于 `Hq = Hkv × r`，长序列下两者通常同阶。pack 后 head 轴缩小了，但 M 轴按 `r` 扩大；边界取整会进一步影响实际 tile 数。

更准确的收益描述是：

> Pack-GQA 将同组 Q heads 的部分 K/V 复用从“多个 CTA 之间依赖 L2 命中”变成“一个 CTA 内显式共享已载入 K/V”。最终 HBM 流量和 kernel 时间仍取决于 packed tile 数、序列长度、并行度、TMA 路径、边界浪费和指针计算成本。

### 4.8 对 TMA Q/O 路径的影响

通用 SM100 forward 中，packed Q 能使用规则 TMA tile 的基本条件之一是：

```python
m_block_size % qhead_per_kvhead == 0
```

若不整除：

```text
use_tma_Q=False
→ 使用普通 async copy
→ PackGQA.compute_ptr()/load_Q() 逐行计算 packed 指针
```

O store 还受以下条件影响：

```text
packed M 是否整除
是否 SplitKV
是否 varlen Q
```

varlen Q 会使 `use_tma_O=False`，epilogue/correction warps 使用普通 global store；packed 路径通过 `PackGQA.store_O()` 和 `store_LSE()` 将行写回真实 `(token, q_head)`。

因此 Pack-GQA 不只是 scheduler head 数变化，还会改变：

- Q load warp 的 copy 类型。
- O/LSE epilogue 寻址。
- shared-memory/barrier 路径。
- 每个 tile 的有效 row 判断。

### 4.9 unpacked GQA 仍可进行 GQA-aware swizzle

没有 Pack-GQA 不代表不能利用 GQA locality。调度器仍可显式计算：

```text
kv_head         = q_head // r
q_head_in_group = q_head % r
```

然后让共享 KV 的 Q heads 在 work-ID 空间中相邻：

```text
KV head 0：Q heads 0,1,2,3
KV head 1：Q heads 4,5,6,7
```

但这只能改善跨 CTA 的 L2 reuse distance，不能获得 packed 路径的一次 CTA 内 K/V 共享。

### 4.10 当前实现边界

当前代码中：

- 通用 hd128 GQA forward 默认可使用 Pack-GQA。
- 专用非 FP8 hd256 forward 明确不支持 Pack-GQA，host 会设为 `False`，kernel 也有 assert。
- backward 当前统一关闭 Pack-GQA。
- SplitKV、block sparsity、tile divisibility 等 feature 组合可能限制 packed load/store 路径。

### 4.11 本章对应代码

```text
flash_attn/cute/interface.py
  pack_gqa=None 的自动选择
  feature/kernel 限制与 hd256 强制关闭

flash_attn/cute/pack_gqa.py
  pack_gqa_layout()
  PackGQA.compute_ptr()
  PackGQA.load_Q()
  PackGQA.store_O()
  PackGQA.store_LSE()

flash_attn/cute/flash_fwd_sm100.py
  use_tma_Q/use_tma_O
  packed Q/O/LSE view
  unpacked q_head → kv_head 映射
  qhead_per_kvhead_packgqa scheduler 参数
```

## 5. L2 swizzle 与 cache thrashing

### 5.1 FlashAttention 为什么关心 L2

K/V 原始数据位于 HBM。CTA 读取 K/V 时通常经过所有 SM 共享的 L2，再进入 SMEM/TMEM/寄存器。

Attention 中存在两类重要的跨工作复用：

```text
GQA 复用
  多个 Q heads 使用同一个 KV head

causal Q-block 复用
  同一 KV head 下，相邻 Q blocks 扫描高度重叠的 K/V 前缀
```

例如：

```text
Q block 7 → KV blocks 0..7
Q block 6 → KV blocks 0..6
Q block 5 → KV blocks 0..5
```

若这些 tiles 在时间上接近，后一个 CTA 可能从 L2 命中此前读取的 K/V；若中间插入大量其他 batch/KV heads，原数据可能在复用前被驱逐。

### 5.2 L2 swizzle 改变什么

L2 swizzle 是对 work-ID 到 tile 坐标映射的置换：

```text
raw/flat work-ID
    ↓ locality-aware mapping
(batch, head, m_block)
```

它：

- 不改变 Q/K/V tensor layout。
- 不改变 attention mask。
- 不增删 work tile。
- 不改变数值结果。
- 只改变哪些 K/V 工作集在 work-ID 空间中彼此接近。

目标是在 locality 与并行度之间取平衡：

```text
locality：同时展开的不同 KV 工作集尽量少
parallelism：同时仍要有足够 heads/tiles 填满 GPU
```

### 5.3 reuse distance

同一份 K/V 两次访问之间插入的其他 cache lines 越多，它仍留在 L2 的概率越低。这个间隔称为 reuse distance。

```text
较好：KV-A → 少量相关工作 → KV-A
较差：KV-A → KV-B/C/D/... → KV-A
```

L2 swizzle 的本质是构造 locality 邻域：

```text
相近 work-IDs
  → 尽量属于同一 batch
  → 只涉及少量 KV heads
  → 包含相邻 causal Q blocks
  → unpacked 时让共享 KV 的 Q heads 靠近
```

这是概率性优化，不是 cache residency 保证。L2 replacement、其他 SM 流量和 CLC 实际返回顺序都会影响最终命中率。

### 5.4 当前 50 MiB 容量启发式

通用 varlen scheduler 使用约 50 MiB 作为 K/V L2 budget：

```python
size_l2 = 50 * 1024 * 1024

kv_block_size = (
    (headdim + headdim_v)
    * element_size
    * tile_n
)

max_kvblock_in_l2 = size_l2 // kv_block_size
```

这不是硬件向 kernel 独占保证的容量。真实 L2 还同时容纳 Q/O、page table、其他 SM 流量、其他 kernel 和系统访问，因此 50 MiB 只是经验预算。

### 5.5 hd128 容量例子

对：

```text
headdim=headdim_v=128
BF16/FP16
tile_n=128
```

一个 K+V block 为：

```text
(128 + 128) × 2 bytes × 128 tokens
= 65536 bytes
= 64 KiB
```

50 MiB 大约容纳：

```text
50 MiB / 64 KiB = 800 KV blocks
```

如果 `seqlen_k=8192`：

```text
8192 / 128 = 64 KV blocks/head
800 / 64 = 12.5 heads
```

当前候选只能从 `16/8/4/2/1` 选择，因此选 8 heads/section。

如果 `seqlen_k=32768`：

```text
32768 / 128 = 256 KV blocks/head
800 / 256 = 3.125 heads
```

因此选择 2 heads/section。

### 5.6 hd256 的同类容量直觉

对 hd256 BF16/FP16、`tile_n=128`：

```text
(256 + 256) × 2 × 128
= 131072 bytes
= 128 KiB/KV block
```

相同 50 MiB 预算大约只容纳：

```text
50 MiB / 128 KiB = 400 KV blocks
```

这说明 head dimension 增大后，同一 L2 budget 能同时维持的 KV heads 更少。该计算只是设计 hd256 GQA-aware swizzle 时的容量直觉；不能据此声称当前专用 hd256 kernel 已经采用通用 varlen 的 section 算法。

### 5.7 `nheads_in_l2` 如何选择

当前通用 varlen scheduler 先用 Q blocks 近似每 head 的 K/V block 数：

```python
num_n_blocks = (
    num_m_blocks
    * tile_m_effective
    // qhead_per_kvhead_packgqa
    // tile_n
)
```

除以 Pack-GQA ratio 是为了把 packed M rows 还原成真实 token 长度。然后从：

```text
16, 8, 4, 2, 1
```

中选择最大的 `nheads_in_l2`，满足：

```text
num_n_blocks × nheads_in_l2 <= max_kvblock_in_l2
```

并限制不超过实际 scheduler head 数。

当前公式有一个明确近似：它使用 Q 长度推算 K/V blocks，隐含 `seqlen_q ≈ seqlen_k`。对 causal self-attention 通常成立；对 cross-attention：

- `K > Q` 时可能低估 footprint，展开过多 heads。
- `K < Q` 时可能过于保守。

这只影响性能映射，不影响 kernel 使用真实 Q/K 长度计算 attention 的正确性。

### 5.8 section swizzle 的精确映射

对某个 varlen batch，flat ID 定位 batch 后得到段内编号 `mh_block`。核心映射为：

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

假设：

```text
num_m_blocks = 4
num_head = 8
nheads_in_l2 = 2
```

work-ID 空间近似排列为：

```text
section 0:
  (head0, block3), (head1, block3)
  (head0, block2), (head1, block2)
  (head0, block1), (head1, block1)
  (head0, block0), (head1, block0)

section 1:
  heads 2,3，同样从重 block 到轻 block
...
```

这样同时实现：

- LPT：重 causal blocks 位于 section 前部。
- L2 限流：一个 ID 区间只涉及有限数量的 heads。
- wave 并行：同一 block 上交错多个 heads。
- reuse distance 控制：同一 head 的相邻 blocks 之间只插入 section 内少量 heads。

### 5.9 Pack-GQA 对 L2 模型的影响

Pack-GQA 开启时：

```text
scheduler head = KV head
```

所以 `nheads_in_l2` 直接对应不同 K/V 工作集，容量模型最自然。

Pack-GQA 关闭时：

```text
scheduler head = Q head
多个相邻 Q heads 可能映射到同一个 KV head
```

若容量模型没有按 `kv_head = q_head // r` 去重，就会将共享 KV 的 Q heads 当成多个工作集，通常偏保守，并可能选择过小的 section。

因此专用 unpacked GQA 路径若要实现生产级 swizzle，应显式拆分：

```text
kv_head
q_head_in_group
```

并优先按 KV head 估算容量、按 Q-head group 保持邻近。

### 5.10 L2 swizzle 的启用边界

Pack-GQA 与 L2 swizzle 不是绑定开关。Pack-GQA 改变 layout、scheduler head 和 CTA 内 K/V 复用；当前通用 varlen 坐标中的 section swizzle 位于：

```python
if params.lpt or params.head_swizzle:
    # 计算 nheads_in_l2 并重排 head/block
```

在本文重点的 `varlen + causal/local` forward 中，因为 `lpt=True`，该分支会同时执行 L2 sectioning 和 block 反转。但仅仅设置 `pack_gqa=True`，不能单独证明一定进入这段 swizzle。

### 5.11 cache thrashing 是什么

有时口语中会说 “cache crushing”，这里通常真正指 **cache thrashing（缓存抖动）**：多个工作集频繁互相驱逐，导致数据刚进入 L2、还没有被有效复用就被挤出，稍后又从 HBM 重新加载。

典型坏调度：

```text
1. 读取 batch A / KV head 0 的大量 K/V
2. 跳到 batch B / KV head 5
3. 再跳到 batch C / KV head 2
4. 新工作集挤占 L2，A/0 被驱逐
5. 再回到 A/0
6. 原本可复用的 K/V 重新从 HBM 读取
```

常见类型：

- **capacity thrashing**：同时活跃的 K/V 总量超过 L2。
- **reuse-distance 过长**：两次访问同一 KV 之间插入过多其他数据。
- **并发干扰**：其他 SM/kernel/Q/O 流量共同占用 L2。
- **调度打散**：CLC 或过大的 head section 扩大同时活跃工作集。

可能观察到：

```text
L2 hit rate / sector hit rate 下降
HBM read bytes 上升
memory 或 long-scoreboard stall 增加
单 tile latency 上升
即使 SM tail 变短，kernel 总时间仍可能持平或回退
```

### 5.12 LPT、CLC、Pack-GQA 与 L2 swizzle 的关系

```text
LPT        → 优化轻重任务在 ID 空间中的顺序
CLC        → 让 resident Worker 接管 pending grid work
Pack-GQA   → 改变 CTA 内 GQA 复用和任务粒度
L2 swizzle → 控制跨 CTA 的 K/V 工作集和 reuse distance
```

四者可能协同，也可能互相抵消。最终应同时看 kernel time、tail、L2 hit 和 HBM bytes，不能只凭某一个理论优势判断。

### 5.13 本章对应代码

```text
flash_attn/cute/tile_scheduler.py
  SingleTileVarlenScheduler.Params.create()
    size_l2 / kv_block_size / max_kvblock_in_l2
  SingleTileVarlenScheduler._varlen_coord_map()
    num_n_blocks / nheads_in_l2 / section mapping / LPT

flash_attn/cute/pack_gqa.py
  packed scheduler head 与 M-row 语义
```

## 6. CLC 的原理、实现与生效条件

### 6.1 CLC 是什么

CLC 是 Blackwell Cluster Launch Control。它允许已经运行的 CTA/cluster 请求取消一份尚未启动的 grid CTA/cluster，并取得被取消工作的原始 grid coordinate。

可以抽象成：

```text
当前 Worker 完成 attention tile
        ↓
发起 CLC try_cancel
        ↓
成功：获得一个尚未启动 grid work 的 ID，继续处理
失败：没有可取消的 pending work，退出
```

CLC 不会：

- 抢占已经运行的 CTA/cluster。
- 拆分一个正在运行的重 attention tile。
- 理解 sequence 长度、causal mask 或 GQA。
- 按 tile cost 选择“最重任务”。
- 自动保证返回 ID 与当前 K/V 工作集相邻。

### 6.2 `ClcDynamicPersistentTileScheduler` 从哪里来

FA4 使用的硬件 scheduler 类来自 NVIDIA CUTLASS/CuTeDSL：

```python
from cutlass.utils import (
    ClcDynamicPersistentTileScheduler,
    ClcDynamicPersistentTileSchedulerParams,
)
```

不是 FlashAttention 自己重新实现的同名硬件机制。

FA4 自己提供 `ClcState` 适配层，将以下组件组合起来：

```text
ClcDynamicPersistentTileScheduler
PipelineClcFetchAsync
producer/consumer PipelineState
FA-specific WorkTileInfo mapping
```

硬件层返回 grid coordinate；FA-specific scheduler 再将它映射成 `(m_block, head, batch, split)`。

### 6.3 为什么 launch full grid 还能称为 dynamic persistent

FA4 CLC 路径仍会 launch 覆盖完整任务空间的 grid。每个实际启动的 Worker 先处理自己的初始 `blockIdx`。处理完成后，它不立即退出，而是尝试取消一份尚未启动的 grid work并接管其 ID。

```text
完整 grid 中：

已启动 CTA/cluster
  → 形成 resident Workers
  → 每个先处理自己的 initial work
  → 完成后尝试接管 pending grid work

尚未启动 CTA/cluster
  → 仍在 launch queue
  → 可被一个 resident Worker 取消和接管
```

所以 persistent 的含义是已驻留 Worker 可以循环处理多份逻辑工作；不是 host 只 launch 一个小 grid，也不是另建软件原子计数器队列。

### 6.4 scheduler warp 与异步 pipeline

通用 SM100 forward 将原本空闲的 warp 用作 CLC scheduler warp。它提前发起下一次查询，以尽量隐藏调度延迟：

```python
pipeline.producer_acquire(...)
mbarrier = pipeline.producer_get_barrier(...)
hw_scheduler.advance_to_next_work(mbarrier)
```

FA4 为 CLC 分配：

- response buffer。
- full/empty mbarrier storage。
- producer/consumer pipeline state。
- 当前实现中的 scheduler pipeline stage。

load、MMA、softmax、correction 和 epilogue warps 都通过同一个 scheduler state 消费下一份 work：

```python
consumer_wait()
work = get_current_work()
consumer_release()
```

这样 CTA 内不同 warp roles 会同步切换到相同的 attention tile，不会出现 load warp 和 MMA warp 使用不同 `(batch, head, block)` 的情况。

### 6.5 “CTA 数大于 SM 数”为什么不够准确

需要区分：

```text
N_grid     ：launch grid 中 CTA/cluster work 的数量
N_valid    ：映射后真正有效的 attention work 数
R_resident ：GPU 同时最多能驻留的 Worker 数
N_pending  ：尚未开始运行、仍可被取消的 grid work 数
```

对普通单 CTA Worker，可粗略写成：

```text
R_resident ≈ SM 数 × 每 SM 可同时驻留的该 kernel CTA 数
```

但每 SM 驻留数由 threads、register、shared memory、TMEM、barrier 和 cluster 约束决定。因此：

```text
物理 CTA 数 > SM 数
```

不一定表示已经有第二波 pending work。如果每个 SM 能驻留两个 CTA，那么 grid CTA 数虽然超过 SM 数，也可能仍全部进入首波。

更准确的必要条件是：

```text
CLC 请求发生时，launch queue 中仍存在尚未启动的 grid CTA/cluster
```

而要产生有用计算，还要求被取消 ID 映射成有效 attention tile，而不是 varlen padding。

### 6.6 两种规模下的行为

假设 GPU 有 120 个 SM，并且目标 kernel 的有效 occupancy 是每 SM 一个 Worker：

```text
N_valid = 80
  → 80 个 Worker 可以全部进入首波
  → 没有真实 pending attention work
  → CLC 查询无法提供额外工作
  → 通常只有查询/同步开销

N_valid = 300
  → 首波约 120 Workers
  → 后续仍有约 180 份 work pending
  → 先完成的 Worker 可以通过 CLC 接管 pending ID
```

如果实际 occupancy 是每 SM 两个 Workers，则 `R_resident≈240`，上述阈值也随之变化。

### 6.7 pending work 只是必要条件，不是加速保证

`N_valid > R_resident` 说明 CLC 有机会接管后续 wave，但最终是否更快还取决于：

- work 是否足够多，能产生多次接管。
- tile duration 是否明显不均衡。
- CLC query/pipeline/barrier 成本。
- resident Worker 复用是否减少退出、重启和初始化间隙。
- 返回 work-ID 是否扩大 K/V reuse distance。
- L2 miss/HBM 增量是否抵消 tail 改善。

典型趋势：

```text
单 wave 或不足一 wave
  → 几乎没有可接管任务，CLC 通常无益

仅比一 wave 多一点
  → 可以接管，但机会少，收益不稳定

多个 waves，tile 几乎等长
  → 普通 block scheduler 已能持续补发，CLC 增量可能较小

多个 waves，varlen/causal tile 差异明显
  → 短任务 Worker 能较早接管 pending work，CLC 更有发挥空间
```

### 6.8 CLC 无法修复已经运行的最后几个 stragglers

若 launch queue 已空，只剩少数已经启动的重 tile：

```text
SM 0：正在运行一个很重 tile
其他 SM：已经空闲
pending queue：空
```

CLC 无法把 SM 0 的 tile 拆给其他 SM，也无法抢占它。因此 LPT 仍然重要：尽量避免重 tile 被留到最后才启动。

### 6.9 CLC 与 LPT 的完整分工

```text
LPT
  定义 flat ID 如何映射到重/轻 m_block
  在 causal 情况下用 block 反转表达重任务优先

CLC
  请求取消尚未启动的 grid work
  成功后只得到该 work 的 grid ID
  再用同一套 varlen/L2/LPT 映射解码
```

因此“CLC 取得一个尚未启动的 work-ID”不等于“CLC 自己选择了最重 tile”。CLC 硬件看不到 attention cost；重任务优先来自软件定义的 ID 空间。

### 6.10 CLC 与 STATIC 的区别

#### STATIC single-tile

```text
CTA/cluster 启动
→ 处理自己的一个 tile
→ 退出
→ GPU launch scheduler 在空闲资源上启动其他 pending block
```

这种基线本身已经具有硬件层面的下一波补发。CLC 的增量主要来自 resident Worker 直接接管、减少生命周期切换/初始化间隙，以及与专用 pipeline 的组合。

#### STATIC persistent

```text
启动固定数量 resident Workers
worker i：i, i+R, i+2R, ...
```

后续 ID 由静态公式决定。某个 Worker 分到较轻任务而提前完成时，不能接管其他 Worker 静态序列中的任务。

#### CLC dynamic

```text
先处理 initial grid work
→ 完成后由硬件取消 pending grid work
→ resident Worker 接管返回 ID
```

CLC 的动态性来自运行时完成先后，而 LPT/L2 的语义仍由 ID 映射决定。

### 6.11 CLC 的 locality 边界

软件可以让相邻 IDs 具有良好 locality，但不能完全指定硬件下一次取消哪个 pending CTA/cluster。因此：

```text
软件可以保证：
  ID 空间按 batch/KV-head section 组织

软件不能严格保证：
  实际执行时序永远按相邻 ID 前进
```

CLC 可能减少 SM tail，也可能同时扩大活跃 KV-head/batch 数量。性能分析必须把负载均衡和 locality 两条证据一起看。

### 6.12 当前 host 过滤条件

CLC 默认由环境变量请求：

```bash
FA_CLC=1
```

当前 host 会过滤已知回退场景：

```python
is_varlen_mha = is_varlen and qhead_per_kvhead == 1
is_dense_noncausal = not is_varlen and not causal and not local

use_clc_scheduler = (
    requested_use_clc_scheduler
    and not is_varlen_mha
    and not is_dense_noncausal
)
```

通用 kernel 内还要求 CLC 与当前数据通路兼容，例如：

- 使用 TMA K/V。
- 不进入不兼容的 `overlap_sO_sQ` 路径。
- cluster shape 与 CTA group size 一致。

因此 `FA_CLC=1` 是请求，不是最终运行证据。

### 6.13 本章对应代码

```text
flash_attn/cute/tile_scheduler.py
  ClcState
  TileSchedulerProtocol
  SingleTileLPTScheduler 的 CLC mode
  SingleTileVarlenScheduler 的 CLC mode

flash_attn/cute/flash_fwd_sm100.py
  self.use_clc_scheduler 条件
  CLC response/mbarrier storage
  PipelineClcFetchAsync 初始化
  clc_scheduler_warp()

flash_attn/cute/interface.py
  FA_CLC 请求后的 host filter

flash_attn/cute/utils.py
  _get_use_clc_scheduler_default()
```

## 7. varlen + causal + GQA 的完整调度流程

### 7.1 host/JIT 阶段

以通用 hd128 forward 为例，host 先完成以下决策：

```text
输入包含 cu_seqlens/seqused
  → is_varlen=True
  → 选择 SingleTileVarlenScheduler

causal=True
  → lpt=True

Hq > Hkv 且 pack_gqa=None
  → pack_gqa=True
  → scheduler head 变成 KV head

FA_CLC=1 且未被 host/kernel filter
  → scheduling_mode=CLC

否则
  → scheduling_mode=STATIC
```

这些值进入 JIT compile key 和 compile-time specialization。不同组合可能编译为不同 kernel，而不是在同一个 binary 中保留所有动态分支。

### 7.2 初始 work-ID 的解码链

每个已启动 Worker 首先得到 initial grid ID：

```text
blockIdx / initial CLC work
        ↓
flat tile ID
        ↓
读取 cu_seqlens_q 或 seqused_q
        ↓
计算每个 sequence 的 num_m_blocks
        ↓
warp prefix-sum 定位 batch
        ↓
得到 batch 内 mh_block
        ↓
估算 num_n_blocks 和 nheads_in_l2
        ↓
解出 L2 section、scheduler head 和 block
        ↓
若 lpt=True：block = num_m_blocks - 1 - block
        ↓
形成 WorkTileInfo(block, head, batch, split)
```

之后 kernel 才根据这些坐标选择 Q/K/V tensor slice 和 mask range。

### 7.3 Pack-GQA 开启时

设：

```text
Hq=32, Hkv=8, r=4
```

Pack-GQA 开启后：

```text
scheduler head 0
  → KV head 0
  → packed rows 中包含 Q heads 0..3

scheduler head 1
  → KV head 1
  → packed rows 中包含 Q heads 4..7
```

完整 tile 语义变成：

```text
(batch, kv_head, packed_m_block)
```

其优势是：

- L2 section 的 head 数直接对应不同 K/V 工作集。
- 一个 CTA 内多 Q heads 显式共享 K/V tile。
- 相邻 packed M blocks 仍可通过 L2 复用同一 KV head 的 causal 前缀。

### 7.4 Pack-GQA 关闭时

unpacked 路径的 tile 为：

```text
(batch, q_head, m_block)
```

执行前再得到：

```text
kv_head = q_head // r
```

此时完整调度若要保持 GQA locality，应尽量让：

```text
相同 kv_head 的 q_head_in_group 相邻
相同 kv_head 的 causal m_blocks 相邻
```

否则同一 K/V 会由多个相隔较远的 CTA 重复请求，并更依赖 L2 是否仍然命中。

### 7.5 一个简化的 varlen 例子

假设只有一个 scheduler head：

```text
Seq A：4 blocks
Seq B：2 blocks
Seq C：3 blocks
```

batch 内 LPT 后的逻辑序列近似为：

```text
A3, A2, A1, A0,
B1, B0,
C2, C1, C0
```

这不是全局 cost sort；只是 compact varlen 顺序中的每条 sequence 内 block 反转。加入多个 heads 后，还会在每个 L2 section 内交错 heads。

### 7.6 STATIC 下如何前进

`SingleTileVarlenScheduler` 的 STATIC single-tile 路径中，一个 CTA 只处理自己的 initial work：

```text
解码 blockIdx
→ 执行一个 attention tile
→ scheduler 标记无下一份 work
→ CTA 退出
```

后续 grid CTA 由普通 GPU launch scheduler 在资源可用时启动。

### 7.7 CLC 下如何前进

CLC 路径中，同一个已驻留 Worker 在执行当前 tile 的同时/结束前预取下一份 work：

```text
执行当前 tile
    ↓
scheduler warp 发起 CLC query
    ↓
成功取消 pending grid CTA
    ↓
获得新的 flat/grid ID
    ↓
再次执行相同 varlen + L2 + LPT 解码
    ↓
所有 warp roles 同步切换到新 tile
    ↓
继续循环
```

若 CLC response invalid，则使用安全的 one-past-end ID，让映射返回 invalid work，然后所有参与 warp 正确退出 pipeline。

### 7.8 四个机制的最终分工

```text
varlen mapping
  解决不同 sequence 的 tile 数不同

Pack-GQA/GQA mapping
  解决 Q head 与 KV head 的共享关系和 CTA 内复用

L2 swizzle + LPT
  决定 flat ID 对应哪个 batch/head/block，并兼顾 locality 与轻重顺序

CLC
  决定一个 resident Worker 是否继续接管尚未启动的 grid ID
```

正确的理解不是“CLC 调度一切”，而是 CLC 提供 work-ID 来源，FA scheduler 负责赋予 ID attention 语义。

## 8. 当前仓库的实际支持状态

### 8.1 通用 SM100 forward

通用实现位于：

```text
flash_attn/cute/flash_fwd_sm100.py
FlashAttentionForwardSm100
```

它当前具备：

- fixed-length 和 varlen scheduler。
- causal/local LPT。
- varlen L2 head section。
- Pack-GQA 与 unpacked GQA。
- STATIC 与 CLC scheduling mode。
- CLC response/mbarrier pipeline 和 scheduler warp。

对典型 hd128 varlen causal GQA：

```text
pack_gqa=None → 通常自动变为 True
lpt=True
TileScheduler=SingleTileVarlenScheduler
FA_CLC=1 且满足过滤条件 → scheduling_mode=CLC
```

### 8.2 当前 CLC host matrix

以通用 hd128 forward 为主，当前行为可概括为：

| 场景 | `FA_CLC=1` 后的当前倾向 | LPT |
| --- | --- | --- |
| varlen causal MHA | host 过滤，回退 STATIC | 是 |
| varlen causal GQA/MQA | 可进入 CLC，仍需满足 kernel 条件 | 是 |
| fixed causal/local | 可进入 CLC | 是 |
| fixed dense non-causal | host 过滤，回退 STATIC | 否 |

此外，非 TMA K/V 或与 `overlap_sO_sQ` 冲突的组合会在 kernel 对象中继续关闭 CLC。

### 8.3 专用 hd256 forward

当前非 FP8、`head_dim=head_dim_v=256` 会选择：

```text
flash_attn/cute/sm100_hd256_2cta_fmha_forward.py
BlackwellFusedMultiHeadAttentionForward
```

当前事实是：

```python
self.is_persistent = False
self.use_clc_scheduler = False
```

同时：

- 固定 `tile_m=tile_n=128`。
- 不支持 Pack-GQA，host 将其关闭。
- 不支持 SplitKV。
- 接受 `cu_seqlens_q/k` varlen 输入。
- 当前调度不是通用 `SingleTileVarlenScheduler` 的 compact varlen + LPT + L2 section。
- 文件中存在 CLC scheduler、buffer 和 pipeline 分支，但因为开关固定为 `False`，不能据此声称当前运行使用了 CLC。

### 8.4 hd256 varlen GQA causal 的当前边界

该组合当前可以在专用 kernel 的 STATIC/unpacked 路径中表达，但下面这个完整组合尚未落地：

```text
compact varlen mapping
+ GQA-aware KV-head section
+ causal LPT
+ cluster-level CLC
```

若未来实现生产级映射，需要：

1. 按真实 sequence 长度紧凑映射 batch。
2. 用真实 `seqlen_k` 或可靠近似估算 K/V footprint。
3. unpacked 模式下按 KV head 对 Q heads 分组。
4. causal Q blocks 使用 LPT。
5. CLC 以完整协作 Worker 为单位接管 work。
6. 覆盖 tail、barrier、pipeline exhaustion 和重复运行正确性。

### 8.5 FP8 hd256 路由不要与上述结论混淆

当前 interface 对 FP8 hd256 还有单独的开发/实验路由，包括专用 1CTA、可选 2CTA 和回到通用 main kernel 的开关。这些路径与非 FP8 的专用 hd256 kernel 不是同一个实现对象。

因此分析日志时应同时记录：

```text
dtype
head_dim/head_dim_v
实际 kernel class/name
USE_2CTA log
scheduling_mode log
FA_CLC/FA_DISABLE_2CTA/hd256 专用环境变量
```

仅凭“hd256”或“源码里出现 CLC”无法判断实际路径。

## 9. PR 演进与 FA 各代调度关系

### 9.1 PR #2218 的主要作用

PR #2218 给 FA4 Blackwell forward 接入 CLC work stealing 基础设施，主要包括：

```text
tile_scheduler.py
  → SchedulingMode
  → ClcState
  → 统一 scheduler protocol
  → CLC grid coordinate 到 FA WorkTileInfo 的适配

flash_fwd_sm100.py
  → use_clc_scheduler
  → response/mbarrier shared storage
  → CUTLASS CLC scheduler 与 async pipeline
  → 使用空闲 warp 发起 CLC query
  → 各 warp roles 共享 scheduler state

interface.py / utils.py
  → FA_CLC 请求开关
  → 相关 JIT compile key

tests
  → CLC fuzz/correctness coverage
```

其核心价值不是发明 CLC 硬件类，而是把 CUTLASS/CuTeDSL 的 CLC scheduler 接入 FA4 的 warp-specialized attention pipeline。

### 9.2 varlen 扩展

后续 varlen 扩展让 `SingleTileVarlenScheduler` 支持 CLC mode，关键内容包括：

- 接收 `SchedulingMode.CLC`。
- 定义 CLC problem shape。
- 将 initial/current CLC grid ID 送入 varlen 坐标映射。
- 增加 `prefetch_next_work()`。
- 增加 consumer wait/release 和 producer tail。
- 对 invalid response 使用 one-past-end 安全 ID。
- 增加 varlen、causal、GQA 等正确性覆盖。

因此要区分：

```text
CLC 基础 plumbing
varlen scheduler 接入 CLC
host 对具体 workload 开放 CLC
某次 benchmark 实际选择 CLC
```

它们是四个不同层次。

### 9.3 LPT 是否只有 FA4 才有

不是。各代可粗略比较：

| 版本 | CTA 任务级 LPT | 主要调度方式 |
| --- | --- | --- |
| FA1 | 没有当前这种 CTA 级 LPT | 固定 grid/task 组织 |
| FA2 | 没有当前形式 | Q block 静态映射到 CTA |
| FA3 | 有 | Hopper 软件 dynamic persistent scheduler |
| FA4 | 有 | CuTeDSL 坐标 scheduler；Blackwell 可结合 CLC |

FA2 中某个 CTA 内部可能改变 K/V loop 顺序，但这不等于改变 Q-tile 启动顺序的任务级 LPT。

### 9.4 FA3 的软件 dynamic persistent scheduler

FA3 通常只启动接近 resident worker 数的 persistent CTAs。Worker 完成当前 tile 后，通过全局原子计数器领取下一编号：

```cpp
next_tile = atomicAdd(tile_count_semaphore, 1);
```

这里代码中的 `semaphore` 更接近软件 work-queue head/任务取号器：

- 原子加保证不同 CTAs 不会领取相同编号。
- 返回旧值作为下一个 work-ID。
- scheduler 再把 ID 映射成实际 tile。

它不是 CPU 线程语义中只负责 sleep/wakeup 的传统 semaphore。

### 9.5 FA3 软件调度与 FA4 CLC 的差异

```text
FA3：
  host launch 少量 persistent CTAs
  CTA 使用 global atomicAdd 获取下一个软件队列 ID

FA4 CLC：
  host launch 覆盖任务空间的 grid
  resident Worker 请求取消尚未启动的 grid CTA/cluster
  接管被取消工作的原始 grid ID
```

两者目标都包括保持 Worker 忙碌、减少尾部空闲，但工作来源机制不同：一个是软件原子队列，一个是 Blackwell launch-control 硬件。

### 9.6 LPT 在两种动态调度中的位置

无论 work-ID 来自：

```text
FA3 atomic counter
FA4 CLC response
STATIC blockIdx
```

LPT 都属于：

```text
work-ID → attention tile 坐标
```

这一层。动态取号机制决定“谁拿到下一 ID”，LPT 决定“该 ID 表示哪个重/轻 tile”。

## 10. 性能收益应如何判断

### 10.1 正确的比较基线

判断 CLC 收益时必须固定其他机制：

```text
STATIC：varlen mapping + Pack-GQA/GQA mapping + LPT + L2 swizzle
CLC：   varlen mapping + Pack-GQA/GQA mapping + LPT + L2 swizzle
```

否则把“是否 CLC”和“是否 LPT/Pack-GQA/swizzle”同时变化，无法归因。

若要拆解各项贡献，可额外比较：

```text
STATIC，无 LPT/swizzle
STATIC + LPT
STATIC + LPT + L2 swizzle
CLC    + LPT + L2 swizzle
```

前提是目标 kernel 提供独立开关或实验分支；不能为了做表格而声称当前公开接口已经暴露所有组合。

### 10.2 uniform 超长序列

例如：

```text
[32K, 32K, 32K, 32K]
```

causal 总工作量随 block 数近似二次增长，而最重单 tile 只线性增长。tile 很多且各 sequence 长度相近时：

- 并行工作充足。
- LPT 已经把重 blocks 提前。
- 同一 wave 的 tile cost 较接近。
- STATIC single-tile 的普通 block scheduler 也能持续补发。

因此 CLC 的增量负载均衡空间可能较小；若它扩大 KV-head/batch 的活跃集合，locality 代价反而可能可见。

### 10.3 long-tail varlen

例如：

```text
[64K, 16K, 4K, 2K]
```

causal 总工作量近似与序列长度平方相关，因此长度相差 4 倍，理论总工作量可能相差约 16 倍。

当前 LPT 是 sequence 内近似映射，不是全 batch 精确 cost sort。长尾下仍可能出现：

```text
部分 Workers 持续处理长 sequence 的重 tiles
其他 Workers 已处理完短 sequence
```

只要 launch queue 仍有有效 pending work，CLC 可以让较早完成的 Workers 接管后续任务，因此 long-tail 通常提供更大的潜在收益空间。

### 10.4 GQA ratio 的影响

GQA ratio 越大，同一 KV head 对应的 Q heads 越多：

- Pack-GQA 可增强 CTA 内 K/V 复用。
- unpacked 路径更依赖 Q-head group swizzle。
- 若 CLC 把同组工作打散，损失的潜在 reuse 也更大。

所以 GQA ratio 既可能放大优化收益，也可能放大错误调度造成的 HBM 重读。

### 10.5 wave 数比“总 CTA 数”更有解释力

定义：

```text
num_waves ≈ N_valid / R_resident
```

大致趋势：

| wave 特征 | CLC 机会 |
| --- | --- |
| `< 1` | 无真实 pending work，通常没有收益 |
| `≈ 1` | 仅 tail/padding，机会很少 |
| `1~2` | 有少量接管机会，结果对 tile 分布敏感 |
| 多 waves 且不均衡 | 更有机会摊薄 CLC 开销并改善利用率 |

精确 `R_resident` 应从目标 kernel 的实际 occupancy 和 cluster 约束获得，而不是直接用 GPU SM 数代替。

### 10.6 必须同时观察两组指标

负载均衡指标：

```text
各 Worker/cluster 实际处理 tile 数
各 SM/cluster 完成时间
最后一波 tail latency
active warps/SM utilization
CLC 成功/失败响应数量
```

locality/访存指标：

```text
L2 hit rate / sector hit rate
HBM K/V read bytes
memory/long-scoreboard stalls
单 tile latency
不同 batch/KV-head 在时间线上的交错程度
```

若 tail 变短但 HBM bytes 大幅上升，说明 CLC 的负载均衡收益与 L2 locality 冲突。只看总 TFLOPS 无法解释原因。

### 10.7 建议 workload 组

```text
uniform short：多个相同短 sequence
uniform long： 多个相同超长 sequence
moderate varlen：长度有变化但无极端长尾
long-tail：     一条/少量超长 + 多条短 sequence

每组再覆盖：
MHA / GQA ratio 2/4/8 / MQA
causal / local
足一 wave / 多 waves / 不完整最后一 wave
```

每次结果必须记录 dtype、head dims、tile shape、GPU、CuTeDSL/CUDA 版本、环境变量和实际 scheduler log。

## 11. 验证方法、常见误区与代码索引

### 11.1 如何确认实际用了 CLC

不能因为源码存在 CLC 分支就下结论。至少确认：

1. 进程启动前设置请求开关：

   ```bash
   FA_CLC=1
   ```

2. workload 没有被 host filter 回退。

3. kernel 内条件没有再次关闭 CLC，例如非 TMA K/V 或不兼容 overlap 路径。

4. host log 显示实际 scheduler，例如：

   ```text
   TileScheduler=SingleTileVarlenScheduler
   scheduling_mode=CLC
   USE_2CTA=False
   ```

5. 如需 kernel trace，确认看到真实运行产生的 CLC query/response，而不是只在源码里搜索到字符串。

6. kernel name/class 与预期一致，尤其避免将通用 hd128、专用 hd256 和 FP8 hd256 实验路径混为一谈。

### 11.2 正确性验证

建议至少覆盖：

- FP16/BF16。
- MHA、GQA ratio 2/4/8、MQA。
- uniform、long-tail、零长度和非 tile 整数倍长度。
- causal 与 local/sliding-window。
- sequence 数超过一个 warp prefix-sum group 的情况。
- 重复运行和 timeout，排查 pipeline/barrier deadlock。
- CLC exhaustion 和 varlen padding ID。
- 明确确认没有 silent fallback。

### 11.3 常见误区

1. **“varlen + causal 一定使用 CLC”是错的。**  
   LPT 可默认启用；CLC 默认关闭，且 host/kernel 还有过滤条件。

2. **“CLC 会选择最重 tile”是错的。**  
   CLC 只取消 pending grid work；LPT 映射表达 tile 轻重。

3. **“LPT 是 kernel 内昂贵排序”是错的。**  
   当前 causal 核心是 O(1) 的 block 坐标反转。

4. **“flat tile 是一块 Q/K/V 数据”是错的。**  
   flat tile 是线性 ID，经过坐标映射后才确定数据范围。

5. **“Pack-GQA 会把 Q tensor 预先复制重排”是错的。**  
   当前实现主要构造 CuTe layout view，并在必要时执行 packed 指针寻址。

6. **“Pack-GQA 后 CTA 数和 HBM 流量固定下降 `r` 倍”是错的。**  
   head 轴缩小但 packed M 轴扩大，最终取决于 tile 数、TMA 和 locality。

7. **“只要 GQA 就一定执行 L2 swizzle”是错的。**  
   GQA 是语义；Pack-GQA 与 L2 section 是不同开关/映射条件。

8. **“CLC 后不再需要 LPT/L2 swizzle”是错的。**  
   CLC 解决 work 接管；LPT 和 swizzle 分别解决轻重顺序和 locality。

9. **“grid CTA 数大于 SM 数就一定加速”是错的。**  
   应比较有效 work 与实际 resident Worker 数，并计入 CLC 和 locality 成本。

10. **“hd256 文件中有 CLC 代码，所以当前 hd256 已使用 CLC”是错的。**  
    当前专用 kernel 将 `self.use_clc_scheduler` 固定为 `False`。

### 11.4 FA4 代码索引

```text
flash_attn/cute/interface.py
  public API
  Pack-GQA 自动选择
  CLC host request/filter
  kernel 与 CTA 路径分流

flash_attn/cute/flash_fwd_sm100.py
  通用 SM100 forward
  LPT/varlen scheduler 选择
  Pack-GQA 与 TMA 条件
  CLC response/pipeline/scheduler warp

flash_attn/cute/tile_scheduler.py
  SchedulingMode
  WorkTileInfo / TileSchedulerProtocol
  ClcState
  SingleTileLPTScheduler
  SingleTileVarlenScheduler
  L2 section 与 LPT 映射
  专用 FMHA static/CLC scheduler 类型

flash_attn/cute/pack_gqa.py
  pack_gqa_layout()
  packed Q/O/LSE load/store

flash_attn/cute/block_info.py
  causal/local 的 K/V block range

flash_attn/cute/seqlen_info.py
  varlen Q/K 长度与 offset

flash_attn/cute/sm100_hd256_2cta_fmha_forward.py
  非 FP8 hd256 专用 forward
  当前 static 调度与未启用 CLC plumbing

tests/cute/test_clc_fuzz.py
  CLC correctness/fuzz 覆盖

AI/CLC_TRACE_DEBUG.md
  CLC trace 与调度调试

AI/DEBUG_2CTA.md
  协作 CTA pipeline/deadlock 调试
```

### 11.5 CUTLASS/CuTeDSL 代码索引

```text
cutlass.utils.ClcDynamicPersistentTileScheduler
cutlass.utils.ClcDynamicPersistentTileSchedulerParams
cutlass.pipeline.PipelineClcFetchAsync
```

CUTLASS 的 Blackwell CLC/cluster 示例可以证明硬件机制如何组合，但不能替代 FA 自己的 varlen、causal、GQA 和 L2 映射设计。

### 11.6 文档结论边界

本文将四种状态严格区分：

```text
源码存在某分支
host/JIT 当前允许该分支
测试覆盖了正确性
目标 benchmark 实际启用并获得收益
```

前三项不能自动证明第四项。性能结论必须来自目标 B200/B300、目标 shape 和完整运行配置。

## 12. 1CTA 与 2CTA 的区别

### 12.1 这里的 1CTA/2CTA 描述什么

这里的 1CTA/2CTA 描述的是：**一个逻辑 MMA/attention work tile 需要多少个 CTA 协作完成。**

```text
1CTA
  一个 CTA 独立承担一个逻辑 attention tile

2CTA
  两个 CTA 组成 cluster，共同承担一个逻辑 pair-tile
```

它不直接描述：

- GPU 总共 launch 多少 CTA。
- 每个 SM 同时驻留多少 CTA。
- kernel 有多少 waves。
- 一个 CTA 是否会通过 persistent/CLC 连续处理多个 tiles。

因此：

> “1CTA kernel”不等于“每个 SM 只能驻留一个 CTA”；occupancy 仍由 threads、register、SMEM、TMEM、barrier 等资源决定。

### 12.2 1CTA 的执行模型

1CTA 中，一个 Worker 通常就是一个 CTA：

```text
CTA
  ├─ load warp：加载 Q/K/V
  ├─ MMA warp：执行 QK/PV UMMA
  ├─ softmax warps：online softmax
  ├─ correction/epilogue warps：输出修正与写回
  └─ scheduler warp：可选 CLC query
```

逻辑 tile 可以直接写成：

```text
(batch, head, m_block, split)
```

CTA 内所有 warp roles 同步切换 tile，但不需要另一个 CTA 参与同一 MMA work。

### 12.3 2CTA 的执行模型

2CTA 中，两个 CTAs 组成 `(2,1,1)` cluster。它们通过 2SM UMMA、cluster barrier、SMEM/TMEM pipeline 协作：

```text
一个逻辑 pair-tile
    ├─ cluster rank 0 / CTA0：负责一部分 M rows 和 pipeline 角色
    └─ cluster rank 1 / CTA1：负责另一部分 M rows 和 pipeline 角色
```

在通用实现的抽象中：

```text
每 CTA 的 cta_tiler M       = m_block_size
2CTA MMA 覆盖的联合 M 范围 = 2 × m_block_size
cluster_shape_mn            = (2, 1)
cta_group_size              = 2
```

调度器必须将两个 CTAs 看成同一个 Worker：

```text
错误：CTA0 领取 tile A，CTA1 领取 tile B
正确：整个 cluster 领取 pair-tile A，rank 0/1 再得到各自物理部分
```

### 12.4 核心差异表

| 维度 | 1CTA | 2CTA |
| --- | --- | --- |
| 最小协作单位 | 单个 CTA | 两个 CTA 的 cluster |
| 典型 cluster shape | `(1,1,1)` | `(2,1,1)` |
| CTA group size | 1 | 2 |
| 一个逻辑 tile 的物理参与者 | 1 CTA | CTA0 + CTA1 |
| MMA 覆盖 M 范围 | 单 CTA M tile | 两 CTA 联合 M tile |
| scheduler Worker | CTA | cluster |
| STATIC launch 坐标 | CTA 自己的 ID | cluster origin + rank offset |
| CLC 取消/接管粒度 | CTA work | 整个 cluster work |
| 同步范围 | CTA 内 barrier/pipeline | CTA 内 + cluster 间 barrier/pipeline |
| tail 处理 | 单 CTA 判断有效 rows | 即使一侧无有效 rows，也可能必须参与 cluster 协议 |
| deadlock 风险 | 主要是 CTA 内 pipeline | 还包括两个 CTA trip count/barrier 不一致 |
| 调度粒度 | 较细 | 较粗，一个 work 占用两个 CTA/SM 资源 |

### 12.5 2CTA 为什么要求整个 cluster 同步换任务

两个 CTA 共同执行同一个 QK/PV pipeline。如果它们拿到不同 tile，会立刻破坏：

- Q/K/V 地址一致性。
- 2SM UMMA participant 关系。
- TMEM/SMEM producer-consumer 配对。
- cluster mbarrier 到达计数。
- causal/local K/V trip count。

因此下一个 work 的正确流程是：

```text
CTA0 + CTA1 完成当前 pair-tile
        ↓
cluster 共同消费一个 scheduler/CLC response
        ↓
获得下一个 cluster-origin work-ID
        ↓
两 CTA 映射到同一逻辑 pair-tile
        ↓
rank 0/1 计算各自 M-row offset
```

CLC 工作在 cluster 粒度时，返回的是被取消 cluster 的 origin/第一个 CTA 坐标，而不是给两个 CTA 分别返回互不相关的 ID。

### 12.6 2CTA barrier 与 `tx_count`

2CTA pipeline 中，某些 transaction barrier 由两个 CTA 共同参与。计算 TMA/CLC pipeline 的 expected transaction count 时，必须按实际 CTA group/cluster participant 数处理。

如果本应统计两个 CTA 的传输，却仍使用单 CTA 的 `tx_count`，可能出现：

```text
barrier 过早满足
  → consumer 读取尚未完整到达的数据

barrier 永远不满足
  → cluster hang/deadlock
```

因此 2CTA 调试不能只检查 tensor 数值；还要核对 cluster layout、producer/consumer group、arrival count 和两个 CTA 的循环次数。

### 12.7 causal/varlen tail 为什么对 2CTA 更敏感

在 sequence 尾部，联合 pair-tile 可能只有 CTA0 的部分 Q rows 有效，而 CTA1 对应 rows 全部越界：

```text
pair-tile 联合范围：256 rows
真实剩余 Q rows：80

CTA0：部分有效
CTA1：可能完全无有效输出
```

CTA1 不能简单提前 return，因为 CTA0 仍可能等待它参与：

- cluster barrier。
- TMEM alloc/dealloc 协议。
- 2SM UMMA participant synchronization。
- pipeline producer/consumer arrival。

正确做法通常是让无有效输出的一侧跳过越界数据操作，但继续参与必要同步，直到整个 cluster 安全结束或切换 work。

### 12.8 causal trip range 必须按联合 pair-tile 设计

对 causal attention，两个 CTA 覆盖相邻 M rows。若每个 CTA仅按自己的局部 rows 独立计算 K/V loop 次数，两个 CTA 可能进入不同数量的 pipeline iterations：

```text
CTA0 认为需要 N 次 K/V loop
CTA1 认为需要 N+1 次
```

这会破坏协作 pipeline。2CTA kernel 必须定义整个 pair-tile 的一致 semantic trip range，并使用 predication/mask 处理各 CTA 内部的局部有效行。

### 12.9 当前通用 hd128 为什么在 varlen + causal 下是 1CTA

通用 SM100 forward 的 2CTA 选择条件包含：

```text
Blackwell SM100/SM110
未设置 FA_DISABLE_2CTA
非 causal
非 local
非 SplitKV
非 varlen Q / seqused Q
非 block sparse
兼容的 page size
padded head_dim 为 128 或 192
head_dim_v padded 为 128
sequence/tile 和 Pack-GQA divisibility 条件满足
```

因此：

```text
head_dim=128 + varlen + causal
```

会同时违反“非 causal”和“非 varlen Q”条件，通用路径使用 1CTA。CLC 若启用，也是以单 CTA Worker 粒度接管 work-ID。

### 12.10 通用 2CTA 的典型适用范围

通用 kernel 中，2CTA 更偏向：

```text
fixed-length
dense non-causal
兼容的 head_dim/head_dim_v
足够长的 Q sequence
无 SplitKV/varlen/block-sparse 冲突
```

这不是“所有 hd128 都使用 2CTA”。同一个 head dimension 会因为 causal、varlen、local、SplitKV、page size 和环境变量进入不同协作模式。

### 12.11 专用 hd256 为什么是另一条规则

非 FP8 `head_dim=head_dim_v=256` 不依赖通用 2CTA heuristic，而是直接路由到专用 kernel。该 kernel 固定使用协作 cluster/2SM UMMA 结构，即使是 causal 或 varlen 也仍属于专用 2CTA 路径。

但当前它同时固定：

```python
self.is_persistent = False
self.use_clc_scheduler = False
```

并关闭 Pack-GQA。因此当前准确表述是：

```text
hd256 专用 2CTA + STATIC + unpacked GQA
```

而不是：

```text
hd256 2CTA + compact varlen + LPT + L2 swizzle + CLC
```

后一个组合是可设计的目标方案，不是当前运行事实。

### 12.12 hd256 未来接入 CLC 时的必要条件

若将专用 hd256 2CTA 真正接入 CLC，至少需要保证：

1. CLC problem shape 和 launch grid 按 `(2,1,1)` cluster 对齐。
2. 返回 ID 表示 cluster origin。
3. 两 CTA 共享同一逻辑 tile，只按 rank 拆分物理 rows。
4. varlen 映射以 pair-tile 为单位，不生成半个 cluster 的独立任务。
5. causal trip range 对整个 pair 一致。
6. unpacked GQA swizzle 按 KV head 分组，而不是将所有 Q heads 当独立 K/V 工作集。
7. tail 中无有效 rows 的 CTA 仍参与必要同步。
8. CLC exhaustion 后所有 warp/CTA 的 pipeline state 一致退出。
9. response buffer、mbarrier `tx_count` 和 consumer group 覆盖两个 CTA 的参与者。
10. 在 B200/B300 上验证重复运行、timeout、L2 locality 和实际收益。

### 12.13 1CTA 与 2CTA 的性能取舍

2CTA 并不天然比 1CTA 快，1CTA 也不天然更灵活。一般取舍为：

```text
1CTA：
  调度粒度更细
  cluster 同步更少
  tail 和 varlen 映射更直接
  单 tile 可用的协作计算资源较少

2CTA：
  可用 2SM UMMA 和更宽的联合 tile
  适合某些大 head dimension/计算形状
  调度粒度更粗
  cluster barrier、tail、trip range 和 CLC 映射更复杂
```

最终选择由 head dimensions、mask、varlen、tile shape、数据类型、并行度、occupancy 和实际 benchmark 共同决定，不能只根据“CTA 越多计算越快”推导。

### 12.14 最终记忆方式

```text
1CTA / 2CTA
  回答：一份逻辑 work 由几个 CTA 协作？

STATIC / persistent / CLC
  回答：Worker 完成当前 work 后，下一份 work 从哪里来？

LPT / L2 swizzle
  回答：一个 work-ID 应映射成哪个轻重/locality tile？

GQA / Pack-GQA
  回答：Q heads 如何共享 KV，以及共享发生在 CTA 内还是主要依赖跨 CTA cache？
```

四组概念分别位于协作粒度、work 分发、坐标映射和数据复用层，只有分层理解，才能准确判断一次 FA4 kernel 实际采用了什么调度方案。
