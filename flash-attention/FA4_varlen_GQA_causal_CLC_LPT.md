# FA4 Blackwell：varlen GQA causal 的 2CTA、CLC、LPT 与 L2 swizzle

> 整理日期：2026-07-14
>
> 范围：FlashAttention-4（CuTeDSL）、Blackwell SM100/SM110、前向 attention。
>
> 目标场景：`head_dim=256 + 2CTA + varlen GQA + causal + CLC`。

## 1. 先看结论

当前 FA4 仓库还没有完整支持目标组合：

```text
hd256 + 2CTA + varlen GQA + causal + CLC
```

但这不是 Blackwell 硬件限制：

- CLC 可以调度 `(2,1,1)` CTA cluster。
- FA4 hd256 专用 kernel 已经有 CLC pipeline、response buffer 和 scheduler warp 骨架。
- 真正缺少的是适合 `varlen GQA causal` 的 cluster-level 调度映射，以及正确性和性能验证。

这个场景需要同时处理三类问题：

| 机制 | 解决的问题 | 不解决的问题 |
| --- | --- | --- |
| LPT | causal tile 计算量不均，重任务不能留到最后 | 不负责动态 work stealing |
| CLC | 不同 cluster 实际完成速度不同，需要动态补任务 | 不知道哪个 tile 更重，也不保证 L2 locality |
| L2 swizzle | 控制同时活跃的 K/V 工作集，增加 K/V 复用 | 不保证负载均衡 |

因此理想方案不是三选一，而是：

```text
varlen compact mapping
    + GQA-aware L2 swizzle
    + causal LPT
    + cluster-level CLC
```

性能上，超长序列不自动意味着 CLC 收益很大：

- 长度接近的超长序列：LPT 通常更重要，CLC 的增量收益可能较小。
- 长尾 varlen，例如 `[64K, 16K, 4K, 2K]`：CLC 更可能有明显收益。
- CLC 如果破坏 K/V locality，也可能抵消负载均衡收益。

## 2. 先建立 FA 的工作模型

### 2.1 一个 FA work tile 是什么

前向 FlashAttention 通常把 Q 按块切分。一个逻辑 work tile 可以抽象成：

```text
(batch, head, Q-block)
```

处理一个 Q-block 时，kernel 在内部循环扫描它需要的 K/V blocks：

```text
Q tile 常驻
  → 依次读取 K/V block
  → QK
  → online softmax
  → PV
  → 输出 O tile
```

tile scheduler 负责决定这些 `(batch, head, Q-block)` 以什么顺序交给 CTA 或 CTA cluster。

### 2.2 varlen 带来的变化

varlen batch 中每条序列长度不同：

```text
batch 0: 64K
batch 1: 8K
batch 2: 2K
```

如果按最大长度建立矩形 grid，短序列会产生大量无效 tiles。更好的方式是根据每条序列的真实长度计算 Q-block 数，并用 prefix-sum 将它们紧凑排列：

```text
每个 batch 的真实 block 数
    ↓ prefix-sum
紧凑的全局 work-ID 空间
    ↓
(batch, batch 内 Q-block)
```

varlen scheduler 的第一个职责就是把线性 work-ID 正确映射到不同长度的 batch。

### 2.3 GQA 带来的变化

GQA 中多个 Q heads 共享一个 KV head。例如：

```text
Q heads  = 32
KV heads = 8
qhead_per_kvhead = 4

Q heads 0..3   → KV head 0
Q heads 4..7   → KV head 1
...
```

这产生了很强的 K/V 复用机会：共享同一个 KV head 的 Q-head tiles 会读取相同的 K/V。

调度时如果把这些 Q-head tiles 放得很远，中间插入很多其他 KV heads，K/V 可能被 L2 驱逐；如果放得相近，后执行的 tile 更可能直接命中 L2。

### 2.4 causal 带来的变化

causal attention 中，越靠后的 Q-block 需要扫描越长的 K/V 前缀：

```text
Q-block 0 → KV blocks [0]
Q-block 1 → KV blocks [0, 1]
Q-block 2 → KV blocks [0, 1, 2]
...
```

因此不同 Q-block 的计算量不同。若一条序列有 `B` 个 Q-block：

```text
单 tile 工作量约为：1, 2, 3, ..., B
总工作量约为：B(B+1)/2
```

这个三角形工作量分布正是 LPT 存在的原因。

### 2.5 hd256 2CTA 带来的变化

hd256 专用 kernel 使用两个 CTA 协作完成一个逻辑 pair-tile：

```text
一个 (2,1,1) cluster
    ├─ CTA0：处理逻辑 tile 的一部分
    └─ CTA1：处理逻辑 tile 的另一部分
```

两个 CTA 通过 2SM UMMA、barrier、TMEM/SMEM pipeline 协作。调度器必须把它们看成一个整体：

```text
最小调度单位 = 一个 2CTA cluster
             = 一个逻辑 pair-tile
```

不能让 CTA0 和 CTA1 各自领取不同的 attention tile。

## 3. 三种调度机制分别做什么

### 3.1 LPT：先执行重的 causal tile

LPT 是 Longest Processing Time first。

对 causal attention，Q-block 越靠后，扫描的 K/V 越多。因此最简单的 LPT 就是把 block 顺序反转：

```text
自然顺序：block 0, 1, 2, 3
LPT 顺序：block 3, 2, 1, 0
```

LPT 的主要价值是避免下面的尾部情况：

```text
大多数 SM 已经空闲
只剩少数最重 Q-block 仍在运行
→ kernel tail 很长
```

把重 tile 提前后，重任务可以更早地分散到不同 SM；末尾更可能剩下轻任务。

#### LPT 不是任务队列

LPT 只定义：

```text
raw work-ID → 哪个 causal Q-block
```

它不负责生成下一个 raw work-ID，也不是简单地让 CTA 每次执行 `current_block - 1`。

在 FA4 中，host 只选择 causal/LPT 策略，真正的 work-ID 反转映射会随 CuTeDSL kernel 编译到 device 端。

### 3.2 CLC：完成当前任务后动态取得新任务

CLC 是 Blackwell 的 Cluster Launch Control。它允许正在运行的 CTA/cluster 尝试取消一个尚未启动的 CTA/cluster，并接管被取消工作的 grid coordinate。

可以把它理解为硬件支持的 work stealing：

```text
cluster A 提前完成
    ↓
尝试取消一个尚未启动的 cluster B
    ↓
成功：取得 B 的 work-ID，并继续执行
失败：没有可窃取工作，退出
```

CLC 解决的是运行时负载不均：有些 cluster 因为 tile 较轻、访存更快或调度时机不同而提前完成，可以继续承担剩余工作。

CLC 本身不知道：

- 哪个 causal tile 最重。
- 哪些 Q heads 共享 K/V。
- 哪些数据当前仍在 L2。
- 哪个 batch 是长序列。

这些语义必须由软件 scheduler 的 work-ID 映射提供。

### 3.3 L2 swizzle：让相近任务复用 K/V

L2 swizzle 是对 work tile 执行顺序的置换：

```text
raw work-ID
    ↓ swizzle
(batch, KV head, Q-head group, Q-block)
```

它不改变 tensor layout，不改变 attention 结果，也不增删 work tile。它只改变哪些 tile 在时间上更接近。

其目标是：

- 缩短同一份 K/V 两次访问之间的距离。
- 限制同时活跃的 KV heads 数量。
- 避免多个不同 K/V 工作集互相驱逐。

第 5 节会详细解释它在 FA 中的使用方式。

## 4. 一个 CTA/cluster 做完后，下一个 tile 是谁

这个问题取决于调度模式，不由 LPT 单独决定。

### 4.1 STATIC single-tile

当前 generic varlen STATIC 路径通常是一个 CTA 只做一个 tile：

```text
CTA 启动
  → 根据自己的 block ID 得到一个 tile
  → 完成 tile
  → 退出
```

对这个 CTA 而言没有“下一个 tile”。SM 空闲后，GPU 正常 launch scheduler 会启动另一个尚未启动的 CTA。

LPT 的作用是在 CTA 启动时，就把它的 raw ID 映射成适当的重/轻 Q-block。

### 4.2 STATIC persistent

静态 persistent scheduler 会让固定数量的 resident workers 循环处理任务。常见形式是：

```text
next_raw_id = current_raw_id + num_resident_workers
```

每个 worker 的后续任务由静态公式预先决定，不会根据其他 worker 实际完成速度调整。

### 4.3 CLC dynamic

CLC 下，下一个 raw work-ID 来自硬件取消响应：

```text
当前 tile 运行
    ↓
scheduler warp 可提前发出 CLC 请求以隐藏调度延迟
    ↓
硬件选择一个尚未启动的 cluster
    ↓
返回该 cluster 的 origin / raw grid coordinate
    ↓
软件执行 varlen + GQA + L2 + LPT 映射
    ↓
得到下一个实际 tile
```

硬件选择哪个尚未启动的 cluster，不由 kernel 软件指定。因此不能假设下一个 raw ID 一定与当前 ID 相邻。

例如最简单的 LPT 映射为：

```text
raw ID 0 → Q-block 3
raw ID 1 → Q-block 2
raw ID 2 → Q-block 1
raw ID 3 → Q-block 0
```

如果 CLC 返回 raw ID 2，下一个实际任务就是 Q-block 1。若返回的 ID 属于另一个 batch 或 KV head，下一步也会跨到那里。

### 4.4 2CTA 时是整个 cluster 换任务

hd256 2CTA 的正确语义是：

```text
CTA0 + CTA1 完成同一个 pair-tile
        ↓
整个 cluster 消费一个 CLC response
        ↓
取得下一个逻辑 pair-tile
        ↓
cluster rank 0/1 再得到各自的物理部分
```

CLC 对 cluster 的取消粒度有明确依据：CUTLASS Blackwell 文档说明 CLC works on the granularity of clusters。CLC 返回的是被取消 cluster 的第一个 CTA ID，也就是 cluster origin。

## 5. L2 swizzle 的原理，以及如何用在 FA

### 5.1 L2 cache 在这里解决什么问题

K/V 原始数据位于显存。CTA 通过内存系统读取 K/V 时会经过 L2，再进入每个 SM 私有的 SMEM/TMEM 或寄存器。

L2 是所有 SM 共享的，因此两个不同 SM 上的 CTA 只要在相近时间访问相同 K/V，后执行者也可能命中 L2。

如果同时活跃的数据超过 L2 容量，会发生 cache thrashing：

```text
加载 KV-A
加载 KV-B、KV-C、KV-D ...
KV-A 被驱逐
再次需要 KV-A
只能重新访问 HBM
```

cache thrashing 不影响数值正确性，但会：

- 降低 L2 hit rate。
- 增加 HBM read bytes。
- 增大 tile latency。
- 抵消 CLC 带来的负载均衡收益。

### 5.2 FA 中有哪几类 K/V 复用

#### GQA Q heads 之间的复用

共享同一个 KV head 的多个 Q heads 读取相同 K/V：

```text
Q heads 0..3 → KV head 0
```

如果这些 Q-head tiles 相邻执行，K/V 复用非常直接。

#### 相邻 causal Q-block 之间的复用

同一 KV head 下，相邻 causal Q-block 的 K/V 前缀高度重叠：

```text
block 7: KV [0..7]
block 6: KV [0..6]
block 5: KV [0..5]
```

LPT 的重到轻顺序仍保持这种嵌套关系，因此可以与 L2 locality 一起设计。

超长序列的完整 K/V 通常无法全部留在 L2；目标是复用仍然驻留的部分，并减少其他 KV heads 的干扰。

### 5.3 Swizzle 的核心：控制 reuse distance

同一份 K/V 两次使用之间插入的数据越少，它仍留在 L2 的概率越高。这个距离可以称为 reuse distance。

```text
较好：KV-A → 少量其他工作 → KV-A
较差：KV-A → 大量不同 K/V → KV-A
```

L2 swizzle 本质上是在 work-ID 空间中建立 locality 邻域：

```text
相邻或接近的 raw IDs
    → 相同 batch
    → 少量 KV heads
    → 相邻 causal Q-block
    → 共享同一 KV head 的 Q-head group
```

这是一种 locality-friendly 映射，不是严格保证。GPU/CLC 的实际执行时序仍可能打乱邻近 ID。

### 5.4 为什么要对 KV heads 分 section

设一个 KV head 的活跃 K/V 工作集约为 `W`，同时展开 `H_active` 个 KV heads：

```text
active_KV_bytes ≈ H_active × W
```

为了减少互相驱逐，应选择满足下面关系的 `H_active`：

```text
H_active × W <= L2_budget
```

然后把 KV heads 切成 section：

```text
section 0: KV heads [0 .. H_active-1]
section 1: 后续 KV heads
...
```

一个 section 处理到一定程度后，再展开下一个 section。这样不会一次把所有 KV heads 的工作集压进 L2。

FA4 generic scheduler 当前使用一个简单启发式：

- 预留约 50 MiB 作为 K/V 的 L2 budget。
- 根据 `head_dim`、dtype、`tile_n` 和序列长度估算每个 KV head 的 blocks/bytes。
- 从 `16/8/4/2/1` 中选择可同时活跃的 head 数量。

这是低开销近似，不是硬件 cache 的精确模型。

#### 一个容量例子

对 hd128 BF16、`tile_n=128`：

```text
一个 K/V block
= (K head_dim + V head_dim) × 2 bytes × tile_n
= (128 + 128) × 2 × 128
= 64 KiB
```

按 50 MiB 预算，大约能容纳 800 个这样的 K/V blocks。

若序列为 16K，共 128 个 KV blocks/head：

```text
4 heads × 128 = 512  blocks，可以
8 heads × 128 = 1024 blocks，超过预算
→ 同时活跃约 4 个 heads
```

若序列为 64K，共 512 个 KV blocks/head：

```text
1 head × 512 = 512 blocks
2 heads × 512 = 1024 blocks，超过预算
→ 同时活跃约 1 个 head
```

`H_active=1` 不表示完整 64K K/V 能常驻 L2，只表示尽量不再引入其他 KV head 与它竞争。

### 5.5 causal LPT 和 L2 section 如何组合

假设一个 batch 有：

```text
4 个 Q-block
8 个 scheduler heads
每个 L2 section 放 4 个 heads
```

一种典型排列是：

```text
section 0:
  block 3, heads 0..3   # 最重
  block 2, heads 0..3
  block 1, heads 0..3
  block 0, heads 0..3

section 1:
  block 3, heads 4..7
  block 2, heads 4..7
  ...
```

它同时实现：

- LPT：Q-block 从重到轻。
- wave 均衡：相邻 work-ID 的计算量相近。
- L2 限流：只展开一个 head section。
- 有界 reuse distance：同一 head 的相邻 Q-block 之间只隔着 section 内少量 heads。

### 5.6 varlen 下如何使用 swizzle

varlen 不能直接使用固定矩形下标。正确顺序是：

```text
raw work-ID
  → 根据真实 seqlen_q 做 prefix-sum，找到 batch
  → 得到该 batch 内部 work-ID
  → 按该 batch 的 seqlen_k 估算 K/V footprint
  → 选择 KV-head section
  → 应用 causal LPT
  → 得到 Q-head/KV-head/Q-block
```

每个 batch 应使用自己的长度估算 section 大小。长序列通常使用更少的同时活跃 KV heads，短序列可以展开更多 heads。

batch 之间是否重排是另一层决策。当前 generic 方案主要保证 batch 内 locality，并不会全局按序列长度重新排序所有 batch。

### 5.7 packed GQA：generic hd128 的做法

generic hd128 默认可使用 `pack_gqa=True`。多个共享 KV 的 Q heads 被折叠进 tile 的 M 维：

```text
一个 scheduler head = 一个 KV head
一个 tile 内包含这个 KV head 对应的多个 Q heads
```

它有两层复用：

1. CTA 内：一份 K/V 服务 tile 内多个 Q heads。
2. CTA/SM 间：相邻 Q-block 继续通过 L2 复用同一 KV head。

CTA 内复用比单纯依赖 L2 更强，因为它避免了重复发起多个独立 Q-head tiles。

### 5.8 unpacked GQA：hd256 2CTA 应如何做

当前 hd256 专用 kernel 不支持 `pack_gqa=True`。要运行 GQA，需要显式使用 unpacked 路径；scheduler head 仍是 Q head：

```text
Q head 0 ─┐
Q head 1 ─┼→ KV head 0
Q head 2 ─┤
Q head 3 ─┘
```

因此不能直接把每个 Q head 当成独立 K/V 工作集，否则会：

- 高估实际 K/V footprint。
- 选择过小的 head section。
- 错过共享 KV 的 Q-head locality。

hd256 的 swizzle 应显式拆出：

```text
kv_head = q_head // qhead_per_kvhead
q_head_in_group = q_head % qhead_per_kvhead
```

推荐逻辑顺序：

```text
batch
  → KV-head L2 section
  → causal Q-block（重到轻）
  → KV head
  → q_head_in_group
  → 2CTA cluster rank
```

例如两个 KV heads、每个共享 4 个 Q heads：

```text
block 3, KV head 0, Q heads 0..3
block 3, KV head 1, Q heads 4..7
block 2, KV head 0, Q heads 0..3
block 2, KV head 1, Q heads 4..7
...
```

若超长序列只适合同时激活一个 KV head，则可以使用更强 locality 的顺序：

```text
KV head 0:
  block 3, Q heads 0..3
  block 2, Q heads 0..3
  ...
KV head 1:
  block 3, Q heads 4..7
  ...
```

### 5.9 L2 swizzle 与 CLC 的边界

STATIC 下，raw IDs 通常按较稳定的顺序启动，swizzle 更容易得到预期 locality。

CLC 下，软件不能指定硬件下一次返回哪个尚未启动 cluster。因此软件只能保证：

```text
raw-ID 空间中的邻近任务具有 locality
```

不能保证：

```text
运行时严格按邻近 raw ID 执行
```

这就是为什么 CLC 的负载均衡收益必须和 L2 hit rate、HBM 流量一起评估。

## 6. 四个机制如何组合成一次完整调度

目标 scheduler 可以理解为下面的映射链：

```text
CLC / STATIC 提供 raw work-ID
        ↓
varlen compact mapping：确定真实 batch
        ↓
GQA mapping：确定 KV head 和共享它的 Q-head group
        ↓
L2 section：限制同时活跃的 KV heads
        ↓
LPT：把 causal Q-block 从重到轻排列
        ↓
得到一个逻辑 2CTA pair-tile
        ↓
cluster rank：CTA0/CTA1 得到各自物理部分
```

这里的顺序是逻辑职责，不要求代码必须分成完全相同的函数层次。

最重要的正确性约束有两个：

1. 两个 CTA 必须得到同一个逻辑 pair-tile，只在 cluster 内拆物理部分。
2. causal K/V trip range 必须按整个 pair-tile 的联合 Q 范围计算，不能让两个 CTA 进入不同数量的 pipeline iterations。

如果 CTA1 位于序列 tail、实际没有有效 Q 行，它仍可能必须参与必要的 cluster barrier 和 pipeline 协议。

## 7. 超长 varlen GQA causal 中收益如何判断

### 7.1 先明确比较基线

判断 CLC 收益不能只看“序列是否很长”，而要比较：

```text
CLC 净收益
≈ 减少的空闲/tail 时间
 + 省掉的部分 cluster 重启与初始化开销
 - CLC scheduler/pipeline 开销
 - locality 变差造成的额外 HBM 时间
```

如果基线是 **STATIC persistent**，固定 resident workers 的后续任务由静态公式决定。分到轻任务的 worker 提前完成后不能接管重 worker 的任务，CLC 的动态 work stealing 可能明显减少空闲。

如果基线是 **STATIC single-tile**，一个 CTA/cluster 做完一个 tile 就退出，GPU 的正常 launch scheduler 会自动在空闲 SM 上启动下一个 pending cluster。这种基线本身已经能动态平衡不同 tile，CLC 的额外价值主要是让 resident cluster 直接接管任务，减少退出、重启和部分初始化间隙。

因此，同一个 workload 相对 static persistent 可能有明显 CLC 收益，相对 single-tile STATIC 的增量却可能小得多。后文所说“长尾更适合 CLC”是趋势，不代表必然有很大百分比收益。

### 7.2 为什么长度接近的超长序列中，LPT 通常已经很有效

例如：

```text
[32K, 32K, 32K, 32K]
```

设每条序列有 `B` 个 Q-block。单个 KV head 的 causal 总工作量近似为：

```text
1 + 2 + ... + B ≈ B²/2
```

最重的单 tile 工作量约为 `B`。序列越长，总工作量按 `B²` 增长，而最大单 tile 只按 `B` 增长。因此最大 tile 相对总工作的比例大约按 `1/B` 下降。

LPT 又把重 tile 提前：

```text
早期 waves：block B-1、B-2 等重任务
后期 waves：block 1、0 等轻任务
```

当各序列长度和 KV-head 数接近时，同一 wave 中不同 cluster 拿到的 tile 通常也比较接近：

```text
wave 0: 多个相近的最重 tiles
wave 1: 多个相近的次重 tiles
...
最后:   多个轻 tiles
```

于是总 Q-block 数很多、并行度充足，LPT 已经消除了大部分“最后只剩少数重 tile”的风险。不同 cluster 的完成时间比较接近，CLC 没有太多空闲时间可以回收。

如果基线还是 single-tile STATIC，GPU 本身也会持续向空闲 SM 发射 pending clusters。因此 CLC 主要只能优化最后几波的小残差。

预期：

```text
STATIC + LPT + L2 swizzle 已经较强
CLC 的增量可能较小、持平，甚至因 locality 略有回退
```

### 7.3 为什么长尾 varlen 给 CLC 更多机会

例如：

```text
[64K, 16K, 4K, 2K]
```

causal 总工作量近似与序列长度平方相关：

```text
64K 相对 16K：约 (64/16)² = 16 倍
64K 相对 4K： 约 (64/4)²  = 256 倍
64K 相对 2K： 约 (64/2)²  = 1024 倍
```

所以长度差 4 倍时，causal 工作量可能相差约 16 倍，而不是 4 倍。

LPT 通常是在每个 batch/head 的 Q-block 范围内做重到轻排列，并不等价于把所有 batch 的 tiles 做一次全局精确耗时排序。batch 输入顺序、KV-head section 和实际访存时间仍会造成残余不均衡。

在 static persistent 分配中，可能出现：

```text
部分 workers 持续处理 64K 序列的重 tiles
另一些 workers 完成 4K/2K 序列后已经空闲
```

CLC 可以让提前完成的 workers 接管尚未启动的 64K/16K 工作。这正是长尾更符合 CLC 设计目标的原因。

但如果基线是 single-tile STATIC，GPU 原生 scheduler 本来就会在空闲 SM 上启动其他 pending tiles。此时 CLC 的额外收益主要来自减少 cluster 生命周期和初始化间隙，而不是从“不能接活”变成“可以接活”。

所以更准确的结论是：长尾给 CLC 提供了更大的潜在负载均衡空间，但实际增量仍由基线调度方式决定。

### 7.4 为什么 CLC 可能破坏 K/V locality

负载均衡只关心“cluster 不要空闲”，L2 locality 关心“接下来的 tile 最好继续使用相同 K/V”。两者的最优选择不一定一致。

假设 hd256 BF16、序列 64K。一个 KV head 的完整 K/V 大小约为：

```text
K: 64K × 256 × 2 bytes = 32 MiB
V: 64K × 256 × 2 bytes = 32 MiB
K + V = 64 MiB
```

这已经大于当前 scheduler 使用的 50 MiB L2 预算。理想 swizzle 会尽量只展开一个 KV head，并连续处理共享它的 Q heads/Q-blocks：

```text
batch A, KV head 0, block 255, Q-head group
batch A, KV head 0, block 254, Q-head group
batch A, KV head 0, block 253, Q-head group
...
```

如果 CLC 取得的 work-ID 被映射成：

```text
batch A, KV head 0
batch B, KV head 3
batch C, KV head 1
batch A, KV head 0
```

多个巨大的 K/V 流会同时经过 L2，batch A / KV head 0 的 cache line 可能在复用前被驱逐。结果可能是：

- cluster 空闲时间减少。
- HBM K/V 读取量增加。
- 单 tile 执行时间变长。
- 总时间最终持平或回退。

CLC 并不必然破坏 locality。如果 work-ID 映射让被窃取的 IDs 仍集中在同一 batch/KV-head section，就有机会同时获得负载均衡和 K/V 复用。

所以关键不是简单的“是否使用 CLC”，而是：

```text
CLC 取得的 raw work-ID
    → 是否仍映射到 locality-friendly 的 tile
```

### 7.5 为什么 uniform 更容易被 locality 代价抵消

uniform 超长时，LPT 后的空闲/tail 本来就不大，因此 CLC 能节省的时间较少；但每个 KV head 的工作集很大，轻微扩大同时活跃的 KV-head/batch 数量，也可能增加大量 HBM 流量：

```text
可获得的负载均衡收益：小
潜在的 locality 代价：不一定小
```

所以可能小幅提升、持平或回退。

长尾场景中，原始负载不均衡更大，CLC 可节省的空闲时间也更大。即使付出一定 locality 代价，仍更可能得到正收益：

```text
可获得的负载均衡收益：较大
潜在的 locality 代价：仍然存在
```

“长尾更可能受益”表达的是这两个量的相对大小，不是说长尾一定更快。

### 7.6 判断时应看哪些条件

| workload 特征 | CLC 倾向 |
| --- | --- |
| 序列长度接近、tiles 很多、LPT 后 wave 均衡 | 增量可能较小 |
| batch 长度方差很大、static persistent 分配不均 | 更可能受益 |
| single-tile STATIC 已充分动态发射 | CLC 增量缩小 |
| KV 工作集远超 L2、CLC 同时展开很多 KV heads | 容易因 locality 回退 |
| GQA-aware swizzle 让 stolen IDs 集中在同一 KV group | 更可能同时获得均衡和复用 |
| 总 tile 数只有少数 waves、最后一波很不整齐 | 更可能改善 tail |

应分别测量：

```text
负载均衡：各 cluster 完成时间、idle/tail、每个 cluster 的 tile 数
locality：L2 hit rate、HBM K/V bytes、单 tile latency
```

只看 kernel 总时间，无法判断 CLC 为什么提升或回退。

### 7.7 GQA ratio 的影响

GQA ratio 越大，共享一个 KV head 的 Q heads 越多，理论 K/V 复用机会越强。

但当前 hd256 是 unpacked GQA：

- 如果 scheduler 做了 Q-head group swizzle，较大 ratio 可能提高 L2 复用。
- 如果没有 GQA-aware mapping，更多独立 Q-head tiles 也可能扩大调度混乱和重复读取。

### 7.8 2CTA 的影响

2CTA 让一个调度单位更粗：

- CLC 每次窃取一个 cluster，而不是一个 CTA。
- tile tail 和 cluster 空洞可能更明显。
- 但 2CTA 也减少了逻辑 worker 数，可能让最后几波负载均衡更重要。

最终收益必须通过目标 GPU 和目标 shape 实测，不能仅由序列长度判断。

## 8. 当前 FA4 仓库的支持状态

| 场景 | 当前状态 | 说明 |
| --- | --- | --- |
| generic hd128、varlen GQA causal | 已有参考实现 | compact varlen、LPT、L2 head section，可请求 CLC |
| hd256 2CTA、varlen causal | 有 STATIC 路径 | 矩形 grid，kernel 内跳过无效 tile |
| hd256 2CTA、GQA | 仅 unpacked | 必须 `pack_gqa=False`，Q head 单独成 tile |
| hd256 2CTA、CLC | 骨架存在但关闭 | 构造函数把 CLC 固定为关闭 |
| hd256 2CTA、varlen GQA causal CLC | 未完整支持 | 缺少完整 scheduler、测试和性能验证 |

需要特别区分两条路径：

### generic hd128 参考路径

- 支持 packed GQA。
- scheduler head 可以直接表示 KV head。
- 已有 varlen compact mapping、LPT 和 L2-aware section。

### hd256 专用 2CTA 路径

- 当前不支持 packed GQA。
- scheduler 看到的是 Q heads，需要显式恢复 KV-head 分组语义。
- CLC plumbing 已存在，但实际开关被关闭。
- 当前调度没有完整的 varlen GQA causal LPT/L2 组合。

因此 generic hd128 的算法可以作为设计参考，但不能不加修改地直接复制到 hd256。

## 9. 建议的 hd256 落地路线

### 9.1 第一阶段：验证 2CTA CLC 正确性

目标是证明已有 CLC plumbing 可以在 hd256 pair-UMMA kernel 中安全运行：

- 让 CLC 开关真正控制专用 kernel。
- 继续使用真实 `(2,1,1)` cluster shape。
- 确认 CLC 返回 cluster origin，两个 CTA 得到同一个逻辑 tile。
- 确认 causal trip range 使用 pair 的联合 Q 范围。
- 覆盖 tail CTA 无有效行但仍需同步的情况。

这一阶段可以暂时保留矩形 varlen grid，但它只证明正确性，不代表最终性能方案。

### 9.2 第二阶段：实现生产级映射

生产 scheduler 应具备：

1. 按真实 varlen 长度紧凑映射 batch。
2. 使用真实 `seqlen_k` 估算 K/V footprint。
3. 按 KV head 而不是 Q head 计算 GQA 工作集。
4. 让共享 KV 的 Q heads 相邻。
5. causal Q-block 使用 LPT。
6. swizzle 的逻辑单位是 2CTA pair-tile。
7. CLC exhaustion 后所有参与 warp/CTA 正确退出。

### 9.3 第三阶段：决定何时启用 CLC

CLC 不一定应该对所有 shape 默认开启。可以根据 workload 特征建立 heuristic：

- batch 长度方差或长尾程度。
- 总逻辑 tiles 相对可并发 clusters 的 wave 数。
- KV footprint 相对 L2 budget。
- GQA ratio 和 KV-head 数量。
- CLC 调度开销相对单 tile 计算时间。

可能的策略是：

```text
uniform 超长 → 优先 STATIC + LPT + swizzle
明显 long-tail → 考虑 CLC + LPT + swizzle
```

最终阈值应来自 B200/B300 benchmark，而不是只靠静态推导。

## 10. 如何验证

### 10.1 正确性

至少覆盖：

- FP16/BF16，`d=256`。
- GQA ratio 2/4/8 和 MQA。
- uniform、long-tail、零长度、奇数长度。
- Q tail 只覆盖 cluster 中部分 CTA 行。
- batch 大于 31，覆盖多轮 varlen prefix-sum 分组。
- 重复运行和 timeout，排查 2CTA deadlock。
- 明确确认实际运行的是 2CTA + CLC，而不是 silent fallback。

### 10.2 性能对照

当前 `benchmarks/configs/clc.yaml` 的 varlen sweep 使用 `causal: [false]`，不能直接回答目标场景。因此需要补充专门的 varlen GQA causal benchmark。

建议至少拆分以下配置：

```text
STATIC，无 LPT/swizzle          # 理解基础调度成本
STATIC + LPT
STATIC + LPT + L2 swizzle
CLC    + LPT + L2 swizzle
```

shape 至少分为：

```text
uniform:  [32K, 32K, 32K, ...]
longtail: [64K, 16K, 4K, 2K, ...]
```

指标不能只有 kernel time，还应观察：

- L2 hit rate / sector hit rate。
- HBM K/V read bytes。
- 各 cluster 完成时间分布。
- 最后一波 tail latency。
- 每个 cluster 实际执行的 tile 数。
- CLC 返回 work-ID 的 batch/KV-head 分布。

如果 CLC 降低了 tail latency，却增加了大量 HBM 流量，说明负载均衡和 locality 发生了冲突。

## 11. 常见疑问

### CLC 能否替代 LPT

不能。LPT 决定重任务优先顺序；CLC 决定空闲 cluster 如何动态取得未启动任务。

### LPT 是 host 算法还是 kernel 算法

host 选择策略并传入参数；真正的 work-ID 到 Q-block 的反转映射通过 CuTeDSL 编译进 GPU kernel。

### 为什么现有 L2 模型假设 `seqlen_q == seqlen_k`

当前 generic varlen scheduler 主要读取 Q 长度来估算 K/V blocks，没有把每个 batch 的 K 长度完整加入 scheduler 参数。因此它用 Q 长度近似 K 长度。

这影响 L2 section 的性能判断，不影响 attention 正确性。kernel 仍分别使用真实 Q/K 长度。

- `K > Q`：可能低估 K/V footprint，展开过多 heads。
- `K < Q`：可能过于保守。
- causal self-attention 通常 `Q == K`，该近似较合理。

生产级 cross-attention scheduler 应直接使用 `seqlen_k`。

### 为什么 CLC 必须以整个 2CTA cluster 为单位

CLC 取消粒度是 cluster。对 `(2,1,1)` cluster，返回的是 cluster origin。更重要的是，2CTA UMMA 和 pipeline 本身要求两个 CTA 协同处理同一个逻辑 tile。

### CLC 能否严格保持 L2 locality

不能。软件只能设计 locality-friendly 的 work-ID 空间；硬件实际返回哪个未启动 cluster 由 CLC 决定。因此需要通过 section 大小、ID 映射和启用 heuristic 降低 locality 风险。

## 12. 参考实现与最小代码索引

正文不依赖具体实现细节，以下路径用于继续核对。

### FA4

```text
flash_attn/cute/interface.py
  host 端 kernel/CLC 选择

flash_attn/cute/flash_fwd_sm100.py
  generic SM100 前向与 scheduler 接入

flash_attn/cute/sm100_hd256_2cta_fmha_forward.py
  hd256 专用 2CTA kernel、现有 CLC plumbing

flash_attn/cute/tile_scheduler.py
  varlen、LPT、L2 swizzle、FMHA CLC schedulers

flash_attn/cute/pack_gqa.py
  packed GQA 的逻辑布局

tests/cute/test_flash_attn.py
tests/cute/test_clc_fuzz.py
benchmarks/clc_bench.py
benchmarks/configs/clc.yaml
AI/CLC_TRACE_DEBUG.md
```

### CUTLASS 的 2CTA + CLC 例子

最直接的 CuTeDSL 参考：

```text
examples/python/CuTeDSL/cute/blackwell/tutorial/
tutorial_gemm/fp16_gemm_3_1.py
```

它同时使用：

```text
2CTA instructions
cluster_shape = (2,1,1)
CLC dynamic scheduler
```

其他参考：

```text
examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/
dense_gemm_persistent_dynamic.py

examples/82_blackwell_distributed_gemm/
82_blackwell_distributed_gemm.cu

test/unit/pipeline/
pipeline_cluster_launch_control_async_warp_specialized_blackwell.cu

media/docs/cpp/blackwell_cluster_launch_control.md
```

这些例子证明 2CTA 与 CLC 的机制可以组合，但不能替代 FA 自己的 causal、varlen 和 GQA scheduler 设计。

## 13. 验证边界

本文结论来自 2026-07-14 的静态代码审查。

当前本机为 macOS arm64，没有可运行 SM100 kernel 的 CUDA/CuTeDSL GPU 环境，因此：

- 已核对当前代码路径、开关和 scheduler 逻辑。
- 未在本机验证 hd256 2CTA CLC 的 GPU 正确性或无死锁性。
- 尚无目标场景的可靠性能百分比。
- 最终结论必须以 B200/B300 上的目标 workload benchmark 为准。
