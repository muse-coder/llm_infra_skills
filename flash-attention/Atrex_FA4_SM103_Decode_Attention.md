# FlashInfer CuTe DSL、TRTLLM-Gen Cubin 与 Atrex FA4 Decode Attention 优化

> 本文把 FlashInfer 的 CuTe DSL GQA decode、TRTLLM-Gen cubin backend 与 Atrex 的 SM103/HD256 serving 特化放在同一套分析框架中。重点是为什么需要 dispatch、可选 kernel 方案如何工作，以及这些方案怎样落到 CTA、warp、TMA、TMEM、SMEM 和同步。本文不展开 Python API 调用链，也不记录提交版本或检查日期。

---

## 1. 本文只讨论 GPU decode kernel

讨论范围：

- GQA decode 与短 Q MTP；
- paged KV cache；
- 沿 KV sequence 的 Flash-Decoding；
- single-split、atomic-cluster、external-reduction 三种方案；
- TRTLLM-Gen 的 cubin 交付边界、kernel selection、Multi-CTA KV 与归约模式；
- Atrex 的 CUDA Graph、ragged Q、N32/N64 和 page128/page256 特化；
- Blackwell 上的 TMA、TCGEN05 MMA、TMEM、DSM、mbarrier 与 PDL；
- dispatch 的依据、代价模型与验证方法。

不讨论：

- 上层请求调度器如何组 batch；
- Python/C++ binding 的逐层调用关系；
- cubin 内不可从当前仓库源码验证的指令级实现；
- prefill kernel 的完整流水线。Atrex 2CTA prefill 只作为 decode fast path 的 fallback 出现。

需要先区分两个实现的定位：

| 实现 | 定位 | 关键范围 |
|---|---|---|
| FlashInfer CuTe DSL GQA decode | 通用 Blackwell decode 基线 | GQA packing、KV split、三种归约、TMA、TMEM、TCGEN05、paged KV |
| FlashInfer TRTLLM-Gen | Blackwell 默认生产 backend | 预编译 cubin、runtime kernel selection、Multi-CTA KV、CGA/global reduction、量化与 speculative decode |
| Atrex FA4 decode | 面向具体 serving shape 的特化 | SM103、HD256、BF16/FP8、Q0～Q5、ragged Q、CUDA Graph、手工 dispatch |

三者使用同一 attention/Flash-Decoding 数学，但工程边界不同：CuTe DSL 的 kernel 主体可读；TRTLLM-Gen 的 loader、metadata、selection 和 launch policy 可读，真正执行的 kernel 以 cubin 交付；Atrex 的 kernel 与 serving dispatch 都在源码中可读。

---

## 2. 整体优化思路：先决定并行拓扑，再优化单 CTA

Decode 的优化顺序应当是：

```text
顶层目标：降低一轮 decode/MTP 的端到端延迟
  ↓
是否需要沿 KV 增加并行度
  ↓
选择 single / atomic-cluster / external-reduction
  ↓
选择 GQA×Q packing、KV tile 和 page 搬运方式
  ↓
设计 CTA 内 TMA → QK → softmax → PV → correction 流水线
  ↓
分配 TMEM / SMEM / registers，并只在真实依赖处同步
  ↓
用 correctness、NCU 和 SASS 验证
```

### 2.1 第一层：先解决 CTA 数不足

Prefill 有很多 Q rows，可以沿 Q 维产生大量 CTA。Decode 通常只有 Q1；MTP 也常只有 Q2～Q5。若 batch、KV heads 和 packed query tiles 共同产生的 CTA 数远小于 SM 数量，单 CTA 再快也无法利用整张 GPU。

因此先问：

```text
不拆 KV 时，能否产生大约一个 SM wave 的有效 CTA？
```

不能时，就把一条长 KV sequence 拆给多个 CTA。这是 Flash-Decoding。

### 2.2 第二层：选择 partial result 的归约位置

KV split 之后，每个 CTA 只有局部 softmax 状态，必须合并。主要有两条路径：

- atomic-cluster：同一 kernel 内通过 CTA cluster、DSM 和 reduce store 合并；
- external reduction：第一阶段写 global partial，第二个 kernel 合并。

前者减少 global traffic 和 launch，后者降低 CTA 耦合并提供灵活后处理。不存在对所有 batch、Q 长度和 split 数都最优的一种归约。

### 2.3 第三层：再提高每个 CTA 的效率

拓扑确定后，再优化：

- 把同一 KV head 下的 GQA query heads 与 MTP tokens 打包；
- 选择 S128 或 S256 KV tile；
- 用 TMA 搬 Q/K/V/O；
- 用 TCGEN05 Tensor Core 计算 QK 与 PV；
- 用 TMEM 保存 logits 和 O accumulator；
- 用 warp specialization overlap 搬运、MMA、softmax 与 correction；
- 控制 P/O pipeline stages、寄存器和 shared memory。

---

## 3. 共同数学：Flash-Decoding 到底拆了什么

### 3.1 普通 GQA decode

设：

```text
Q: [B, Sq, Hq, D]
K/V: [B, Skv, Hkv, D]
G = Hq / Hkv
```

GQA 中，一个 KV head 被 `G` 个 query heads 共享。对某个 query-token position `q` 和 query head `h`：

```text
S_j = scale * dot(Q, K_j)
P_j = exp(S_j) / sum_t exp(S_t)
O   = sum_j P_j * V_j
```

Decode 的问题不是公式特殊，而是 `Sq` 很小、`Skv` 很长。沿 Q 方向只有极少 work，沿 KV 方向却有大量串行循环。

#### query token、query vector 与 attention row

“row”在不同上下文容易产生歧义，本文统一使用以下定义。

对：

```text
Q[b, q, h, :]    # 长度为 D 的向量
```

- `q`：request 内的 query-token position；
- `h`：query-head index；
- `Q[b,q,h,:]`：一个 query vector；
- `(b,q,h)`：一条独立 attention row，因为它会产生一行沿 KV 位置展开的 logits/probabilities。

也就是说，一个 query token 在 tensor 中通常表现为：

```text
Q[b,q,:,:] : [Hq,D]
```

它包含 `Hq` 个 query vectors，因此对应 `Hq` 条 attention rows，而不是只有一条 head-level attention row。

#### prediction row/slot 是什么

在 MTP 或 speculative decode 中，一次 attention 调用可能同时处理一个 request 的多个待验证 token positions：

```text
Q1 → 每个 request 有 1 个 query-token position
Q2 → 每个 request 最多有 2 个 query-token positions
Q4 → 每个 request 最多有 4 个 query-token positions
```

这些位置有时被非正式地叫作 `prediction rows`，更准确的名称是：

```text
MTP query-token positions
或 prediction/query slots
```

它们不是四个 batch requests，也不是四个 query heads。它们是同一个 request 在本轮准备计算的四个 token positions。候选 token 最终可能被接受或拒绝，但进入 attention 时每个有效 position 都需要自己的 Q vector 和 attention result。

以 `Q4,Hq=32` 为例：

```text
query-token positions = 4
每个 position 的 query heads = 32
attention rows = 4 * 32 = 128
```

若是 GQA `Hq=32,Hkv=2`，每个 KV head 只负责其中 16 个 query heads：

```text
每个 KV head 对应的 attention rows
= 4 query-token positions * 16 grouped query heads
= 64 rows
```

这 64 rows 正是后文 Q4/N64 packing 中的 N64。

还要区分逻辑 Q 长度和物理 `prediction_tile`：

```text
实际 Q3:
  logical query positions = q0,q1,q2
  prediction_tile = 4
  physical slots = q0,q1,q2,padding
```

因此：

- `query-token position`：真实需要 attention 的 token 位置；
- `prediction slot`：kernel tile 中预留的物理位置；
- `padding slot`：物理存在但没有真实 query，会被 mask；
- `attention row`：一个 `(query-token position, query head)` 组合。

后文提到“Q2～Q5 带来更多 Q 方向并行度”，指的是有效 query-token positions 增多，并进一步乘上 GQA head tiles 后产生更多独立 attention work。

### 3.2 Flash-Decoding：沿长 KV sequence 并行

假设 `Skv=8192`，sequence tile 为 128：

```text
num_kv_tiles = ceil(8192 / 128) = 64
```

若 `kv_splits=4`，不是把 Q 复制出四个不同 attention，而是四个 CTA 对同一条 attention 的不同 KV tiles 并行计算：

```text
split 0: tile 0, 4, 8,  ...
split 1: tile 1, 5, 9,  ...
split 2: tile 2, 6, 10, ...
split 3: tile 3, 7, 11, ...
```

因此：

```text
Flash-Decoding = 保持 Q/GQA work 不变，增加 KV sequence 方向的 CTA 数
kv_splits      = 每条 KV sequence 同时由多少个 split CTA 扫描
```

### 3.3 每个 split 输出什么

第 `i` 个 split 扫描自己的 KV 子集后产生：

```text
mi = 该 split 的最大 logit
li = sum exp(score - mi)
Ui = sum exp(score - mi) * V
```

`Ui` 是未归一化的 weighted-V numerator，不是最终 output。

跨 split 正确合并为：

```text
M = max_i(mi)
L = sum_i exp(mi - M) * li
U = sum_i exp(mi - M) * Ui
O = U / L
```

不能直接做 `O = O0 + O1 + ...`。原因是每个 split 使用不同的局部最大值 `mi`，其指数尺度不同。

例如：

```text
split 0: m0=10
split 1: m1=2
M=10
```

split 1 的 numerator 必须先乘 `exp(2-10)`，否则会把本应很小的贡献错误地放大。

### 3.4 KV 很短时为什么会出现空 split

split 分配的单位是 KV tile，不是单 token：

```text
num_kv_tiles = ceil(kv_length / sequence_tile)
```

若 `KV=512`、`S128`、`kv_splits=8`：

```text
num_kv_tiles = 4
split 0..3: 各得到一个 tile
split 4..7: 没有 tile
```

后四个 CTA 会跳过 QK、softmax 和 PV，但并不完全免费：CTA 已 launch，仍可能读取 metadata、初始化 barrier，并在 cluster 或外部归约协议中表示“无有效 partial”。

所以 split 的目标不是越大越好，而是：

```text
用足够少的 split 填满 GPU，同时避免空 CTA 和过高归约成本。
```

---

## 4. Work geometry 与 dispatch 的共同框架

### 4.1 GQA packing 把什么放进一个 CTA

#### 先区分 N、S 和 D

这里的 `N16/N32/N64` 来自 CTA 内两次矩阵乘的 tile 形状，不是模型 tensor 的第 N 维，也不是 page size。

对一个 KV head，decode 主循环可以写成：

```text
QK GEMM:
  [S, D] K tile × [D, N] packed-Q tile → [S, N] logits

PV GEMM:
  [D, S] V tile × [S, N] probability  → [D, N] output
```

三个字母分别表示：

```text
S = 当前 CTA 一轮处理的 KV token 数，例如 S128/S256
D = head dimension，例如 HD256
N = 共享同一 KV head 的 query-head × query-token rows
```

所以：

```text
N32 = 一个 CTA 同时为 32 个 packed query rows 计算 attention
```

它不表示 32 个 KV tokens，不表示 page32，也不一定表示 32 个 query heads。

#### 一个 packed row 是什么

GQA 中，同一 KV head 被多个 query heads 共享：

```text
grouped_heads = Hq / Hkv
```

MTP 又让每个 request 同时有多个 query tokens。对固定的 request 和 KV head，每个独立 attention row 由二元坐标确定：

```text
(local_query_head, prediction_token)
```

kernel 把这个二维坐标线性化到 GEMM 的 N 维：

```text
N rows = grouped_head_tile × prediction_tile
```

例如一个 CTA 覆盖 8 个 grouped query heads 和 4 个 prediction slots：

```text
N = 8 * 4 = 32

N rows:
  (head0,q0) (head0,q1) (head0,q2) (head0,q3)
  (head1,q0) (head1,q1) (head1,q2) (head1,q3)
  ...
  (head7,q0) (head7,q1) (head7,q2) (head7,q3)
```

实际内存中的先后顺序由 CuTe layout/stride 决定；关键是每个 N column 对应一个独立的 `(query head, query token)` attention row。这些 rows 使用相同 KV head，所以可以复用同一块 K/V tile。

#### 为什么叫“规则 tile”

Tensor Core MMA、SMEM layout、TMEM allocation 和 TMA box 都针对固定编译期 tile。kernel 不会为任意 `N=19` 临时生成一个不规则 fragment，而是选固定宽度，例如：

```text
N16 / N32 / N64
```

实际 Q 长度也先映射到 power-of-two prediction tile：

```text
Q1 → prediction_tile=1
Q2 → prediction_tile=2
Q3 → prediction_tile=4
Q4 → prediction_tile=4
Q5 → prediction_tile=8
```

若实际是 Q3，第四个 prediction slot 只是物理 tile 中的 padding：

```text
physical slots: q0 q1 q2 q3
valid slots:    ✓  ✓  ✓  ×
```

无效 slot 会被 mask，不产生有效 output。这就是“逻辑 rows”与“物理 tile rows”的区别。

FlashInfer 通用实现中：

```text
blk_tile_n = grouped_head_tile * prediction_tile
```

底层 MMA 还有最小 N 宽度约束：FP8 N-major operand 至少使用 N16，BF16 路径的底层最小宽度可以更小。若逻辑 `blk_tile_n` 小于 MMA 最小宽度，物理 MMA tile 会扩到最小合法宽度，并只保留有效 columns。

#### 用同一个 GQA shape 展开 N16/N32/N64

若：

```text
Hq=32, Hkv=2
grouped_heads=16
```

则每个 KV head 对应 16 个 query heads。不同 Q 长度和 tile 选择可以得到：

| 场景 | prediction tile | 每 CTA 的 grouped heads | 物理 N | 每 KV head 的 head tiles | 有效情况 |
|---|---:|---:|---:|---:|---|
| Q1 | 1 | 16 | N16 | 1 | 16×1 全有效 |
| Q2 | 2 | 16 | N32 | 1 | 16×2 全有效 |
| Q3，N32 | 4 | 8 | N32 | 2 | 每 CTA 8×3=24 有效，8 padding |
| Q4，N32 | 4 | 8 | N32 | 2 | 每 CTA 8×4 全有效 |
| Q4，N64 | 4 | 16 | N64 | 1 | 16×4 全有效 |
| Q5，N32 | 8 | 4 | N32 | 4 | 每 CTA 4×5=20 有效，12 padding |

这张表还说明 N tile 会改变 CTA 数。Q4/N32 需要两个 CTA 才覆盖一个 KV head 下的 16 个 grouped heads；Q4/N64 只需要一个 CTA。

#### N 越宽并不总是越好

宽 N 的收益：

- 一个 CTA 复用同一 K/V tile 服务更多 query rows；
- 减少重复 page-table/TMA/KV mainloop 工作；
- 增大单次 MMA 的有效工作量；
- 减少同一 KV head 需要的 head tiles。

宽 N 的代价：

- CTA 数减少，小 batch 时更难填满 SM wave；
- logits、softmax state 和 O accumulator 变宽；
- TMEM、SMEM 和 registers 压力增加；
- softmax/correction 单 CTA 工作变多；
- 可能需要减少 P/O stages 才能放下宽 fragment。

窄 N 则反过来：单 CTA 复用较少，但产生更多独立 CTA，更适合 request parallelism 不足的小 batch。

因此 GQA packing 的意义不是简单“把 N 做到最大”，而是选择：

```text
K/V 复用与单 CTA 效率
        versus
CTA 数量与整卡并行度
```

原先的简写例子：

```text
Q1: 16 × 1 = 16 rows
Q2: 16 × 2 = 32 rows
Q4: 16 × 4 = 64 rows
```

只是“把全部 16 个 grouped heads 放进同一个 CTA”时的逻辑上限。实际 dispatch 可能把 grouped heads 再切成多个 head tiles，例如 Q4/N32 的 `8 heads × 4 tokens`，以换取更多 CTA。

### 4.2 CTA、SM residency 与 SM wave

CTA 就是一个 CUDA thread block。GPU 把 CTA 调度到 SM 上执行；一个 CTA 在执行期间会占用该 SM 的 threads、registers、shared memory，Blackwell kernel 还可能占用 TMEM。

一个 SM 能同时驻留多少个当前 kernel 的 CTA，近似为：

```text
resident_ctas_per_sm = min(
    architectural_max_blocks_per_sm,
    floor(max_threads_per_sm / threads_per_cta),
    floor(registers_per_sm / registers_per_cta),
    floor(smem_per_sm / smem_per_cta),
    TMEM / barrier / cluster 等资源限制
)
```

这里的“驻留”不是已经 launch 就算，而是 CTA 的资源已经分配、可以在 SM 上推进指令。若 shared memory 只能容纳一份 CTA，即使 thread 数允许两个 CTA，最终仍是：

```text
resident_ctas_per_sm = 1
```

SM wave 不是 CUDA 编程模型中的正式对象，而是性能分析术语：

```text
一个 CTA wave = 整张 GPU 同一时刻最多能驻留的一批 CTA
```

普通、非 cluster kernel 的理论 wave capacity 为：

```text
ctas_per_wave = SM_count * resident_ctas_per_sm
num_waves = ceil(launched_ctas / ctas_per_wave)
```

例如只为说明计算，假设 GPU 有 148 个 SM，当前 decode kernel 因 SMEM/TMEM/register 限制每 SM 只能驻留一个 CTA：

```text
resident_ctas_per_sm = 1
ctas_per_wave = 148

128 CTAs → 0.86 个 wave，约 20 个 SM 没有该 kernel 的 CTA
148 CTAs → 1 个完整 wave
256 CTAs → ceil(256/148)=2 个 waves
```

最后一个 wave 只有 `256-148=108` 个 CTA，所以第二个 wave 是不满的。更多 CTA 并非免费：它可能延长到第二轮调度，还可能来自更多 KV splits，从而增加归约成本。

FlashInfer/Atrex 的 decode CTA 通常很重，设计目标经常接近每 SM 一个 CTA。因此 dispatch 中常用：

```text
ctas_per_wave ≈ SM_count
```

这是一条有意简化的 occupancy heuristic，不是完整的 CUDA occupancy 计算。若某个 specialization 实际能每 SM 驻留两个 CTA，真正的一 wave 容量应接近 `2*SM_count`；必须用编译资源报告和 profiler 验证。

#### cluster kernel 的 wave

Atomic KV-split 中，同一 attention 的 `kv_splits` 个 CTA 组成一个 cluster。若：

```text
cluster_size = 8
base logical works = 16
```

则：

```text
cluster 数 = 16
总 CTA 数 = 16 * 8 = 128
```

在每 SM 驻留一个 CTA 的简化假设下，一个 split8 cluster 大约同时占用 8 个 CTA residency slots：

```text
clusters_per_wave ≈ floor(SM_count / 8)
```

对 148 SM，理论上界约为 18 个 split8 clusters，所以 16 个 clusters、共 128 CTA 接近一 wave。实际 cluster placement 还受 GPC 边界、portable cluster size、co-scheduling 与每 CTA 资源限制影响，不能仅靠 `floor(SM_count/cluster_size)` 得到精确值。

External reduction 没有 cluster 约束；它的 split CTAs 可以作为独立 CTA 被调度，因此 placement 更自由。

### 4.3 基础 CTA 数与有效 CTA 数

Atrex 使用的通用估算可写为：

```text
prediction_tile = next_power_of_2(max_q)

query_head_tiles =
    ceil(grouped_heads / grouped_head_tile)
  * ceil(max_q / prediction_tile)

base_ctas = graph_batch * Hkv * query_head_tiles
```

`base_ctas` 是没有 KV split 时 launch 的独立 CTA 数。加入 split 后：

```text
launched_ctas = base_ctas * kv_splits
```

但 launch 出来的 CTA 不一定都有 QK/PV 工作。设实际 active slots 为 `active_batch`：

```text
useful_ctas ≈
    active_batch
  * Hkv
  * query_head_tiles
  * min(kv_splits, num_kv_tiles)
```

这个公式忽略了 tile 分配不均，但能揭示两个主要损失：

- `active_batch < graph_batch`：inactive graph slots 对应的 CTA early-exit；
- `num_kv_tiles < kv_splits`：部分 split CTA 没有 KV tile。

理想目标不是让 `launched_ctas` 看起来等于 SM 数，而是：

```text
useful_ctas 接近 SM_count * resident_ctas_per_sm
```

例如 graph B8 只激活五个 slots，即使按 B8 launch 出 128 CTA，粗略有效值也可能只有：

```text
128 * 5/8 = 80 useful CTAs
```

CUDA Graph 为了保持 topology 固定仍必须 launch B8 grid，因此 dispatch 只能用静态信息近似有效 wave。

### 4.4 split 数的第一层估算

Atrex 对 Q1 更积极填满一个 wave，对 Q2～Q5 更保守：

```text
Q1:     occupancy_splits = ceil(SM_count / base_ctas)
Q2-Q5: occupancy_splits = floor(SM_count / base_ctas)

target_splits = min(16, occupancy_splits)
kv_splits = floor_power_of_two(target_splits)
```

Q2～Q5 已经从多个有效 query-token positions 获得额外 Q 方向并行度；若只差几个 CTA 就填满 wave，不值得为此多一倍 split 和归约成本。

例如仍假设 148 SM、每 SM 一个 CTA，某 shape 有：

```text
base_ctas = 16
```

则：

```text
split8  → 128 CTAs，接近一个 wave
split16 → 256 CTAs，需要两个 waves
```

split16 虽然 CTA 更多，却多了一轮不满的 wave，并加倍 partial/reduction 参与者，所以 dispatch 可能更偏向 split8。Q1 使用 `ceil` 是为了积极补足不足的一 wave；Q2～Q5 使用 `floor` 是为了避免只为填补少量空闲 SM 就把 split 数提升一档。

power-of-two 的原因不是数学必须，而是 atomic cluster 的 butterfly reduction 和 cluster size 只支持 `{1,2,4,8,16}` 这组实现形态。

### 4.5 为什么还需要 shape-specific dispatch

上式只考虑 CTA 数，没有考虑：

- split2 与 split16 的 DSM 同步差异；
- N32 与 N64 的 Tensor Core 和寄存器效率；
- page size 导致的 TMA 次数；
- external workspace 流量；
- batch 增大后原生 CTA 已经足够；
- CUDA Graph bucket 中 inactive slots 的浪费。

所以 Atrex 再对主要 serving shapes，例如 `(Hq,Hkv)=(32,2)` 和 `(16,1)`，按 batch/TP/MTP 实测加 split cap、归约模式和 N32/N64 规则。这里的 dispatch 是性能模型加测量结果，不是 attention 正确性分支。

---

## 5. 三种主 kernel 方案

### 5.1 方案 A：single-split decode

```text
kv_splits = 1
reduction = none
```

一个 CTA 扫描完整 KV tile 序列：

```text
Q load once
  → 循环 K/V tiles
  → online softmax
  → PV accumulate
  → normalize/store O
```

它没有跨 split partial、DSM、global workspace 或第二个 kernel。适合 `base_ctas` 已足够，或 KV 很短、split 收益覆盖不了额外成本的场景。

### 5.2 前置概念：CTA cluster、DSM 与 mbarrier

#### 普通 shared memory 为什么不够

普通 CUDA kernel 中，每个 CTA 有自己的 shared memory：

```text
CTA 0 → SMEM 0
CTA 1 → SMEM 1
CTA 2 → SMEM 2
CTA 3 → SMEM 3
```

CTA 0 默认不能直接读取 `SMEM 1`。如果四个独立 CTA 要交换 partial softmax state，传统做法只能写 global memory，再由其他 CTA或第二个 kernel 读取。

#### CTA cluster 是什么

CTA cluster 把若干 CTA 声明为一个需要协作调度的 group：

```text
cluster 0 = {CTA 0, CTA 1, CTA 2, CTA 3}
```

硬件/runtime 保证这些 CTA 以 cluster topology 被共同调度，并提供 cluster-scope 的同步与远端 shared-memory 访问。cluster 是协作和调度单位，不表示四个 CTA 合并成一个 CTA：每个 CTA 仍有自己的 threads、registers 和本地 shared memory。

#### DSM 是什么

DSM 全称 Distributed Shared Memory。它把同一 cluster 中每个 CTA 的 shared-memory segment 暴露给 peer CTA：

```text
cluster DSM address space
  ├─ rank 0 segment: CTA 0 的 SMEM
  ├─ rank 1 segment: CTA 1 的 SMEM
  ├─ rank 2 segment: CTA 2 的 SMEM
  └─ rank 3 segment: CTA 3 的 SMEM
```

“distributed”的含义是这些数据物理上仍分布在各 CTA 所在 SM 的 shared memory 中；它不是先把四块 SMEM 复制到一块更大的集中式 cache。CTA 0 访问 rank 1 segment 时，发生的是 cluster 内 remote SMEM access。

DSM 与其他存储的区别：

| 存储 | 谁能访问 | 生命周期 | 典型延迟/用途 |
|---|---|---|---|
| registers | 当前 thread | thread/CTA 执行期 | 私有 fragment |
| local SMEM | 当前 CTA | CTA 执行期 | CTA 内 producer/consumer |
| DSM | 同一 cluster 的 CTA | cluster 执行期 | split CTA 间交换少量状态 |
| global memory | 全 GPU/后续 kernel | allocation 生命周期 | 大 workspace、跨 kernel 数据 |

DSM 适合 `m/l` 这种尺寸小但需要低延迟交换的 partial state。它不适合自动替代所有 global workspace：DSM 容量受每 CTA shared memory 限制，并且要求 cluster CTAs共同驻留和同步。

#### 有 DSM 为什么仍需要 mbarrier

“能够访问 peer 地址”不等于“peer 已经写完且数据已经可见”。正确协议必须建立生产者到消费者的 happens-before：

```text
1. 初始化 cluster/DSM 使用的 mbarrier
2. producer CTA 写自己的 local/remote DSM slot
3. producer 对 mbarrier arrive，发布本轮完成
4. consumer CTA wait，直到对应 peer 到达
5. wait 完成后读取 DSM 并合并状态
6. buffer 再次使用前，完成下一轮 phase 协议
```

`mbarrier` 是 memory barrier object：它既跟踪到达/phase，也用于把异步或远端写入与后续读取建立正确的内存可见性。只做普通 `__syncthreads()` 不够，因为它只同步一个 CTA 内的 threads，不能同步 cluster 中另一个 CTA。

#### split4 中 DSM 实际传什么

每个 CTA 完成局部 KV mainloop 后有：

```text
CTA i: mi, li, Ui
```

不必先通过 DSM 搬运完整 HD256 `Ui`。cluster reduction 主要交换每个 packed row 的小型 softmax state：

```text
state_i = (mi, li)
```

split4 butterfly：

```text
round 0:
  CTA0 与 CTA1 交换 (m,l) → 得到 state01
  CTA2 与 CTA3 交换 (m,l) → 得到 state23

round 1:
  pair01 与 pair23 交换 → 所有 CTA 得到共同 (M,L)
```

每次合并不是直接相加，而是：

```text
m = max(ma, mb)
l = exp(ma-m)*la + exp(mb-m)*lb
```

共同 `(M,L)` 得到后，每个 CTA 在本地修正自己的宽 `Ui`，再通过 TMA reduce/atomic 累加最终 O。这样 DSM 负责小状态，reduce store 负责大 output，避免在 DSM 中广播完整 partial O。

#### DSM 的收益与代价

收益：

- 不把小型 reduction state 往返 global memory；
- 可以在同一个 kernel 内完成跨 split softmax merge；
- 配合 reduce store 避免完整 FP32 partial-O workspace 和第二 kernel。

代价：

- cluster CTAs 必须按 topology 共同调度；
- remote SMEM access 比本地 SMEM 更贵；
- 每轮需要 cluster-scope barrier/phase；
- 最慢 CTA 决定其他 CTA 的等待时间；
- cluster 越大，placement 和 occupancy 越受约束。

#### DSM 同步开销到底大不大

没有一个脱离 shape 的固定答案。Atomic-cluster 路径的延迟可以粗略拆成：

```text
T_atomic ≈
    T_KV_mainloop_per_split
  + log2(kv_splits) * (T_remote_DSM + T_barrier + T_peer_wait)
  + T_correct_partial_O
  + T_reduce_store_contention
```

增加 split 会缩短第一项，因为每个 CTA 扫描的 KV tiles 变少；但会增加 reduction rounds、peer 数和最终 O 的写入者。这两部分方向相反。

DSM 同步通常容易被摊薄的场景：

- KV 很长，每个 split 仍有足够多 QK/PV mainloop；
- split2 或 split4，reduction rounds 少；
- KV tiles 能比较均匀地分给各 CTA；
- 省掉的 FP32 partial-O global traffic 和第二 kernel launch 很可观。

DSM 同步可能成为主要成本的场景：

- KV 很短，QK/PV 本身很少，固定 barrier 成本占比高；
- split8/split16，分别需要 3/4 轮 butterfly；
- `kv_splits > num_kv_tiles`，存在没有 mainloop 工作但仍要参与协议的空 split；
- 各 split 工作不均，快 CTA 在 barrier 等最慢 CTA；
- cluster placement 降低 occupancy；
- 多个 CTA 最后竞争同一个 O tile 的 TMA reduce/atomic store。

例如 `KV=512,S128,split8` 只有四个 KV tiles：

```text
CTA0..3: 各处理一个 tile
CTA4..7: 没有 QK/PV tile
```

但一个合法的 split8 cluster 不能让 CTA4～7 在初始化前直接消失，否则其他 CTA 可能等待永远不会 arrive 的 peer。这时三轮 cluster protocol 相对四个有效 tiles 就很重。

所以“DSM 比 global reduction 快”不是恒等式。Atomic 的优势是少 global workspace 和少一次 launch；external 的优势是没有细粒度 peer wait、CTA placement 更自由。Atrex 在部分 Q1 buckets 选择 external，以及 Q2～Q5 最终强制 external，正说明 DSM/atomic 路径的收益必须按 shape 衡量；Q2～Q5 的选择还同时考虑 ragged pack 和 LSE 融合，并非只由 DSM 慢决定。

#### DSM 是否只能用于 2-CTA pair

不是。DSM 的可访问范围是整个 CTA cluster：cluster 内 CTA 可以按 rank 访问任意 peer 的 shared-memory segment。当前 KV-split atomic kernel 使用的 cluster size 可以是：

```text
2 / 4 / 8 / 16 CTAs
```

“2-CTA pair”只是 butterfly reduction 在某一轮中的通信关系，不是 cluster 的永久划分，也不是 DSM 的能力上限。

split8 的三轮关系可以写成：

```text
round 0, distance 1:
  CTA0↔1  CTA2↔3  CTA4↔5  CTA6↔7

round 1, distance 2:
  group(0,1)↔group(2,3)
  group(4,5)↔group(6,7)

round 2, distance 4:
  group(0..3)↔group(4..7)
```

从 rank 角度，round `r` 的 partner 可理解成：

```text
partner_rank = my_rank XOR (1 << r)
```

每轮看起来是两个 peer/group 配对，三轮合起来却让全部八个 CTA 获得整个 cluster 的 `(M,L)`。完整调度单位始终是一个 split8 cluster，不是四个独立的 split2 clusters。

#### 不要把 DSM pair 与 2CTA MMA 混为一谈

| 概念 | 两个 CTA 在协作什么 | 是否等于 KV split2 |
|---|---|---|
| DSM butterfly pair | 某轮交换/合并 `(m,l)` 状态 | 否；它可以属于 split4/8/16 的一轮 |
| 2CTA MMA | 两个 CTA 协作完成一个 Tensor Core MMA tile | 否；它划分的是一次矩阵乘工作 |
| KV split2 | 两个 CTA 分别扫描同一 attention 的不同 KV tiles | 只有采用 atomic 时才可能再用 DSM 合并 |

FlashInfer CuTe GQA decode 的 QK/PV MMA 使用 1CTA MMA；即使 `kv_splits=8`，也是八个各自执行 1CTA MMA 的 split CTAs，再通过 DSM 合并 softmax state，而不是一个 8CTA MMA。

TRTLLM-Gen metadata 中的 `uses2CtaMma` 也是独立 trait。当前 selection 明确不把 2CTA MMA 与 `CgaSmemReduction` 同时使用。Atrex 的“2CTA prefill fallback”同样不是 decode 的 `kv_splits=2`。

TRTLLM-Gen 中的 `CgaSmemReduction` 使用的 CGA/cluster shared-memory 思路与这里相同；具体 buffer layout 和指令流水线则由各自 kernel 实现决定。

### 5.3 方案 B：atomic-cluster Flash-Decoding

```text
kv_splits ∈ {2,4,8,16}
cluster_x = kv_splits
```

同一 attention 的 split CTAs 组成一个 CTA cluster：

```text
每个 CTA: local mi/li/Ui
        ↓
DSM butterfly: merge M/L
        ↓
每个 CTA: Ui *= exp(mi-M)/L
        ↓
TMA reduce/atomic: 累加 final O
```

它省掉：

- `[splits,B,Q,H,D]` FP32 partial-O workspace；
- partial O 的 global write/read；
- 第二个 reduction kernel launch。

但增加两类成本。

第一类是 DSM/mbarrier。概念与内存协议已在上一节说明；split4 的 reduction topology 类似：

```text
round 1: CTA0 ↔ CTA1, CTA2 ↔ CTA3
round 2: pair(0,1) ↔ pair(2,3)
```

轮数为：

```text
split2 → 1
split4 → 2
split8 → 3
split16 → 4
```

每轮包含 remote DSM write/read、mbarrier 通知和 peer wait。若某个 CTA 扫描的有效 tiles 更多或到达较晚，同 cluster 的 CTA 会等待它。

第二类是 output write contention。得到共同 `M/L` 后，所有 split CTA 都向相同 O tile 累加 corrected `Ui`。split 越多，同一输出 cache line 的 reduce/atomic 写入者越多。

因此“atomic/DSM 同步增加”应精确理解为：

```text
DSM cost    = 合并 softmax max/sum 的远端 shared-memory 通信与 barrier 等待
atomic cost = 多个 corrected partial-O 写入同一最终 O 的竞争
```

不是说 attention 主循环里的每条指令都变成 atomic。

### 5.4 方案 C：external-reduction Flash-Decoding

第一阶段 CuTe kernel 的 split CTA 完全独立：

```text
split i → global mi, li, Ui
```

第二阶段 kernel：

```text
读取所有 partial
  → M=max(mi)
  → L=sum(exp(mi-M)*li)
  → U=sum(exp(mi-M)*Ui)
  → O=U/L
  → optional LSE=M+log(L)
```

FlashInfer 的通用实现把这称为 kernel reduction；Atrex 使用专门的 Triton reduction，并把更多 serving 后处理融合进去。

代价：

- FP32 workspace；
- partial O 的 global write/read；
- 第二次 kernel launch。

收益：

- split CTA 不要求 cluster co-residency；
- 不进行细粒度 DSM peer wait；
- 容易处理任意 split 数的后处理；
- 可融合 ragged pack、LSE 和格式转换。

### 5.5 为什么不能固定只用 atomic 或 external

| 维度 | Atomic cluster | External reduction |
|---|---|---|
| kernel launch | 1 个主 kernel | 主 kernel + reduction |
| partial O | 不落 global workspace | FP32 global workspace |
| CTA 关系 | cluster 内强耦合 | 第一阶段相互独立 |
| 同步 | DSM + mbarrier | kernel boundary |
| output merge | TMA reduce/atomic | 第二阶段普通写回 |
| ragged/LSE 后处理 | 不方便 | 容易融合 |
| 大 split 风险 | cluster 与写竞争 | workspace 流量增加 |

小 cluster、Q1、无需 LSE 时，atomic 常有低延迟优势。split 较大、ragged MTP 或需要 LSE 时，external 的解耦与融合可能更划算。

---

## 6. FlashInfer CuTe DSL GQA decode

### 6.1 它提供的是通用机制

FlashInfer 的 `GroupedQueryAttentionDecode` 暴露三种 reduction mode：

```text
none   : kv_splits 必须为 1
atomic : cluster reduction + reduce store，无 workspace
kernel : 写 partial workspace，再启动 reduction kernel
```

核心静态 tile 参数为：

```text
head dimension
grouped_head_tile
prediction_tile
sequence_tile
reduction_mode
```

这使相同主循环可覆盖 Q1、MTP、GQA 和不同 KV split 方案。

### 6.2 CTA 与 warp specialization

通用实现每 CTA 使用 512 threads，即四个 warpgroups：

```text
warpgroup 0: MMA/TMA roles
  warp 0: QK MMA
  warp 1: PV MMA
  warp 2: K/V TMA
  warp 3: Q/O TMA

warpgroup 1-2: softmax
warpgroup 3: correction / epilogue
```

这里不是所有线程按同一循环执行。不同 warp 长期负责不同角色，通过 TMEM、SMEM 和 named barrier 交换阶段状态。

这样可以 overlap：

```text
下一 K/V tile 的 TMA
当前 tile 的 QK MMA
上一 logits tile 的 softmax
更早 P tile 的 PV MMA
旧 O accumulator 的 online-softmax correction
```

### 6.3 Tensor Core 与 TMEM

QK 与 PV 都通过 Blackwell TCGEN05 MMA：

```text
QK: [S,D] × [D,packed_rows] → logits S
PV: [D,S] × [S,packed_rows] → output accumulator O
```

FP32 MMA accumulator 位于 TMEM。TMEM 主要承载：

- 多 stage logits `S`；
- softmax column sum/state；
- 多 stage output accumulator `O`。

stage 数不是随意设置。实现先为 P 和 O 保留 TMEM columns，再用剩余 columns 决定 S stages，并把总 allocation 向上取 power-of-two。增加 packed rows 或 O stages 会减少可留给 S 的 TMEM。

### 6.4 SMEM 与 TMA

SMEM 保存：

- Q tile；
- K/V ring stages；
- softmax max/sum exchange；
- P tiles；
- O store staging；
- pipeline mbarriers；
- atomic 模式的 cluster reduction buffers。

Q/K/V 使用 global-to-shared TMA。普通输出使用 shared-to-global TMA store；atomic 模式使用带 reduce 语义的 TMA store。

KV stages 由 SMEM 剩余容量计算，而不是固定拍脑袋：先扣除 Q/P/O、barrier 与 softmax 状态，再把剩余空间分给 K/V stages。

### 6.5 online softmax 与 correction

第一个 KV tile 得到 `(m0,l0,U0)`。下一个 tile 得到局部 `(m1,l1,U1)` 后，更新为：

```text
m_new = max(m_old, m1)
alpha = exp(m_old - m_new)
beta  = exp(m1 - m_new)

l_new = alpha*l_old + beta*l1
U_new = alpha*U_old + beta*U1
```

`correction` 就是对旧 O accumulator 乘 `alpha`。它必须与新的 PV accumulate 正确排序，否则新旧 numerator 会处于不同指数尺度。

### 6.6 Paged-KV layout 与 page size

FlashInfer 公共 CuTe paged decode 支持：

```text
page_size ∈ {8,16,32,64}
```

逻辑 KV 仍是：

```text
[request, token, kv_head, head_dim]
```

物理 cache 通过 page table 映射：

```text
logical_token
  → logical_page = token // page_size
  → offset       = token % page_size
  → physical_page = page_table[request, logical_page]
```

page size 影响的是一次 sequence tile 需要跨多少物理 pages：

```text
pages_per_S128 = 128 / page_size

page8  → 16 pages
page16 → 8 pages
page32 → 4 pages
page64 → 2 pages
```

小 page 提高 cache 分配与 prefix sharing 粒度，但每个 KV tile 需要更多 page-table lookup 和更多 TMA/gather 片段。大 page 减少寻址与搬运指令，但可能增加尾部碎片。

### 6.7 BLASST 的边界

BLASST 的源码行为是：先得到 QK tile max，再根据阈值决定是否跳过贡献很低的 softmax/PV/V 路径。它是近似优化，不应自行扩展其缩写。

FlashInfer 暴露可选 threshold，默认关闭。它不属于 Flash-Decoding 必需机制：KV split、softmax merge 和 BLASST 是三个不同层次的问题。

---

## 7. Atrex FA4 decode：在通用机制上做 serving 特化

### 7.1 支持边界

Atrex decode fast path 有意缩窄范围：

```text
architecture: SM103
head_dim:     256
Q/K/V dtype:  BF16 或 FP8 E4M3FN，且匹配
query length: Q1～Q5；ragged 时允许某些 request 为 Q0
attention:    causal
page size:    {16,32,64,128,256}
```

严格边界换来的收益是可以把 tile、stage、workspace 和 dispatch 针对真实模型固定下来，避免通用 kernel 的大量组合与动态分支。

若 decode fast path 不支持某个输入，但仍在 Atrex FA4 2CTA prefill 的支持范围内，才 fallback 到 prefill。非 SM103 并不会因为“有 fallback”就变成可用的 FA4 路径。

### 7.2 Atrex 继承了什么

主要继承机制：

- GQA packing；
- KV split / Flash-Decoding；
- TMA、TMEM、TCGEN05 Tensor Core；
- warp specialization；
- online softmax 与 correction；
- DSM cluster reduction；
- TMA reduce/atomic output merge；
- PDL；
- 可选 BLASST 框架。

因此把这些全部称为“Atrex 新算法”是不准确的。

### 7.3 Atrex 新增或明显强化了什么

- SM103 + HD256 的窄特化；
- true packed ragged Q0～Q5；
- CUDA-Graph-stable long-KV split topology；
- 按 batch/TP/MTP shape 测量的 dispatch table；
- Q4 的 N64 packing；
- page128 与物理 page256 处理；
- Triton 融合 reduction、dense-to-packed 和自然对数 LSE；
- Q4/N64 的 P/O stage 调整；
- decode 与 2CTA prefill 的统一生命周期。

### 7.4 N32/N64 packed rows

Atrex 基础 `GROUPED_HEAD_TILE=16`，dispatch 的主要 packed-row budget 是 32 或 64：

```text
Q4 + grouped_heads=16 + batch>=16 → N64
其他主要短 Q shape                 → N32
```

这里的 N32 是“允许一个 CTA 最多打包约 32 个 `(head,token)` rows”的 specialization budget，不表示所有 Q 长度最终都执行物理 N32。例如 Q1 只有：

```text
grouped_head_tile = min(16, packed_rows / prediction_tile)
                  = min(16, 32/1)
                  = 16

blk_tile_n = grouped_head_tile * prediction_tile
           = 16 * 1
           = N16
```

Q4/N32 则得到：

```text
grouped_head_tile = min(16, 32/4) = 8
blk_tile_n = 8*4 = N32
```

因此一个 KV head 下的 16 个 grouped heads 要分成两个 CTA。Q4/N64 得到：

```text
grouped_head_tile = min(16, 64/4) = 16
blk_tile_n = 16*4 = N64
```

只需一个 CTA 就覆盖全部 16 heads。N64 提高 Q/KV 复用和 MMA N 维工作量，但也加倍 logits/O fragments 的宽度，增加 TMEM、register 和 softmax 工作；同时 CTA 数减半。因此只有 batch 已经提供足够 request parallelism 时，Atrex 才选择 N64。

### 7.5 S128/S256 sequence tile

Atrex 默认 S128，以下主要路径使用 S256：

```text
Q1, Hq=32, Hkv=2
page128 + FP8 + N32 + Q1..Q5
```

S256 的收益：

- 单 CTA 每次处理更多 KV tokens；
- 减少 sequence loop、page metadata 和 pipeline 循环开销；
- 对 page128 恰好一次覆盖两个 physical pages。

代价：

- 同一 KV 长度的 tile 数减半，更容易出现空 split；
- 单 stage 占用更多片上资源；
- 短 KV 或宽 packed rows 时可能降低并行度。

### 7.6 P/O pipeline specialization

Atrex 的主要配置为：

```text
Q4    → P stages = 2
其他  → P stages = 4

N64   → O stages = 1
N32   → O stages = 2

softmax warpgroups = 1
```

Q4/N64 已经把 packed row 维做宽，继续保留通用实现的深 P/O stages 会推高 TMEM/SMEM 与寄存器压力。减少 stages 是在“更宽 tile”和“更深 overlap”之间重新平衡，不表示 stage 越少越先进。

### 7.7 page128/page256

Atrex 支持：

```text
page_size ∈ {16,32,64,128,256}
```

kernel 内原生 page tile 最大为 128。物理 page256 不复制、不重排，而是解释成两个 page128 子页：

```text
physical page P, offset 0..127   → subpage 0
physical page P, offset 128..255 → subpage 1
```

page table 仍指向原始 physical page；kernel 通过 `page_table_factor` 计算子页偏移。这就是“zero-copy two page128 subpages”。

大 page 的优势是一个 S128/S256 tile 只涉及一个或两个物理 page；小 page 则要聚合多次 page-sized TMA。page size 因而会直接影响 TMA 数、page-table loads、边界 mask 与地址计算。

---

## 8. FlashInfer TRTLLM-Gen cubin backend

### 8.1 哪些部分开源，哪些部分不是源码

TRTLLM-Gen 不能简单回答成“完全闭源”或“kernel 全部开源”。当前仓库中能直接读到：

- Python backend 选择、参数检查和 workspace 管理；
- C++ cubin loader；
- kernel metadata 的匹配条件；
- SM compatibility、tile、scheduler、Multi-CTA KV 与 reduction 的选择逻辑；
- grid/cluster 计算和 launch；
- separate reduction 等辅助代码。

真正被 GPU 执行的 FMHA kernel 主体以预编译 `.cubin` 下载并加载，当前仓库没有与每个 cubin 一一对应的完整 CUDA/CuTe kernel 源码。因此可以确认“选择了什么策略和参数”，但不能仅凭 loader 代码断言 cubin 内每一条 TMA/MMA 指令或精确流水线。

```text
开源 Python/C++ policy
  → 用 shape/dtype/SM 生成 kernel key
  → metadata 查找匹配 cubin symbol
  → 从本地 cache 或 artifact repository 取得 cubin
  → 校验并通过 CUDA Driver API 加载
  → 按计算出的 grid/cluster launch
```

这也解释了为什么它可以同时拥有“可审查的 dispatch”与“不可直接修改的 kernel 本体”。修改 selection/launch 可以重新编译 host wrapper；修改 cubin 内 mainloop 则需要 NVIDIA 重新发布相应 artifact。

### 8.2 为什么 B200/B300 默认选它

FlashInfer 的自动 backend 逻辑对 compute capability 10.x，也就是 SM100/SM103，选择 `trtllm-gen`；其他架构再走对应 backend。loader 接受 SM100 family cubin，并在存在架构专用版本时优先匹配具体 SM100 或 SM103 kernel。

因此它对 B200/B300 的适配不是一句泛化宣传，而落实在两层：

```text
API dispatch: CC major == 10 → trtllm-gen
kernel loader: SM100 family compatibility + 优先具体 SM kernel
```

“专门适配”仍不等于 B200 与 B300 永远使用不同 kernel；某些 shape 可以共享 SM100-family cubin，另一些 shape 若发布了 SM103-specific cubin则优先选后者。

### 8.3 kernel key 明确包含哪些优化维度

metadata/hash key 包含的主要维度有：

```text
QKV layout 与 mask type
context / generation kernel type
tile scheduler
Multi-CTA KV reduction mode
headDimQK / headDimV / headDimPerCtaV
tileSizeQ / tileSizeKV
tokens per page / dynamic page-size mode
是否复用 K/V shared memory
是否使用 2CTA MMA
sparse MLA type
是否 skip softmax
dtype 与 SM architecture
```

这意味着 TRTLLM-Gen 不是一个固定 cubin，而是 cubin 集合加 runtime selector。所谓“自动选择 kernel”，就是先根据输入与代价模型确定这些 traits，再从已发布 metadata 中找到完全匹配的 symbol。

### 8.4 Q/head packing 与 tile-size cost model

普通 GQA generation 会先计算：

```text
numTokensHeadsQ = numHeadsQPerKv * maxSeqLenQ
```

再在 Q tile `{8,16,32,64,128}` 中选择能容纳 grouped heads 与 speculative tokens 的形态。较小 tile 往往产生更多 CTA、mainloop 单次成本更低；较大 tile 提高复用并减少 reduction work。

对 `maxSeqLenQ>1`，selector 还会估算候选 tile：

```text
estimated_time =
    mainloop_cost(tileQ) * seqLenPerCtaKv
  + reduction_cost(tileQ) * reduction_factor * numCtasPerSeqKv

estimated_total = estimated_time * num_waves
```

它同时考虑单 CTA 成本、KV split reduction 和总 wave 数，而不是只按 `Hq/Hkv` 做一个 if/else。这是文档中可以从开源 selection 代码明确确认的优化方案。

### 8.5 Multi-CTA KV 如何选择

TRTLLM-Gen 的 Multi-CTA KV 与本文前面的 Flash-Decoding是同一顶层思想：当 Q/head/batch 产生的 CTA 不足时，沿 KV 再拆 CTA。

基础估算：

```text
base_ctas = numCtasPerSeqQ * numCtasForAllHeadsQ
          * numCtasPerHeadDim * batch

occupancy_splits = floor(SM_count / base_ctas)
```

它还显式限制每个 split 至少要有约两个 KV steps 的工作：

```text
maxNumCtasPerSeqKv =
    ceil(maxAttentionWindow / (2 * stepKv))

numCtasPerSeqKv = min(maxNumCtasPerSeqKv, occupancy_splits)
```

其中因子 2 的源码理由是：避免 reduction overhead 超过缩短 mainloop 带来的收益。

这个规则比“尽量填满所有 SM”更完整。以 `attention_window=512, stepKv=128` 为例：

```text
ceil(512 / (2*128)) = 2
```

即使 occupancy 允许 split8，selector 也最多使用 2 个 KV CTAs，避免产生四个以上几乎没有主循环工作的 split。

若最终只有一个 KV CTA，Multi-CTA mode 被关闭，并改选适合单 CTA 的 persistent scheduler。

### 8.6 三类 Multi-CTA reduction

TRTLLM-Gen metadata 中可见三类有效路径：

```text
Disabled
CgaSmemReduction
GmemReduction / GmemReductionWithSeparateKernel
```

`CgaSmemReduction`：

- `numCtasPerSeqKv` 在 2～16；
- split CTAs 组成 cluster；
- partial state 通过 cluster shared memory/DSM 合并；
- 避免完整 global partial + standalone reducer；
- 当前只对特定 generation kernel type 和 headDimV 范围启用；
- 2CTA MMA 与 CGA reduction 当前不能同时使用。

`GmemReduction`：

- partial stats/O 通过 global scratch 与 counter 协调；
- 不受最大 cluster 16 的相同约束；
- 某些 kernel 可在主 kernel 生命周期内完成最终合并。

`GmemReductionWithSeparateKernel`：

- 主 kernel 写 global partial；
- 单独 reduction kernel 合并；
- 用于某些 wide-head、KeepsMmaAb 或 MLA shape。

这里的 CGA 是 Cooperative Group Array，在 CUDA launch 层对应 CTA cluster。其 shared-memory reduction 与前文 atomic-cluster 的 DSM 思想相近，但具体 cubin 内 reduction layout 不能由 host selector 完整还原。

### 8.7 static/persistent scheduler 的真实含义

selector 的 kernel key包含 tile scheduler。Multi-CTA KV 关闭、每条 sequence 只有一个 KV CTA 时，代码会切到 persistent scheduler，以便固定数量的常驻 CTAs循环领取 work，降低大量小 work 的 launch/scheduling overhead。

Multi-CTA KV 开启时，grid 需要显式表达一条 sequence 的多个 KV CTAs及可能的 cluster topology，使用何种 scheduler 由匹配的 cubin traits 决定。

因此“支持 static/persistent scheduler”是 metadata 与选择逻辑明确体现的能力；但 persistent kernel 内 work queue 的每条指令仍属于 cubin 实现细节。

### 8.8 page size 到底支持多大

TRTLLM-Gen 没有像 CuTe DSL wrapper 那样把公共 page size 限死为 `{8,16,32,64}`。开源 selector 给出的规则是：

```text
page size 必须是 2 的幂
page size < 128  → page size 直接参与 kernel key
page size >= 128 → 使用 dynamic-page kernel key 128，真实 page size runtime 传入
```

测试覆盖了 `128/256/512/1024` 的 dynamic page-size decode。这证明这组大 page 可用；不要把它外推成“任意无限大的 2 次幂都已验证”。最终上限还受 TMA descriptor、tensor shape、已发布 cubin 和接口检查约束。

page size 的影响仍与前文一致：小 page 增加 page-table 与跨页搬运次数；大 page 减少寻址，但增加 cache 分配碎片。dynamic page kernel 的意义是让 `>=128` 的多个物理 page sizes 复用同一个 cubin specialization，避免为每个大 page 单独发布 kernel。

### 8.9 KV layout、量化与 speculative decode

TRTLLM-Gen paged decode 以 HND 为优选物理 layout：

```text
K/V: [num_pages, num_kv_heads, page_size, head_dim]
```

普通 NHD 输入可以通过 view transpose 表达；NVFP4 的 NHD 数据与 block scales 需要 transpose 后 contiguous copy，文档明确提示这会产生额外 allocation 和 copy，因此量化 KV 应优先直接生成 HND。

公开接口还能确认：

- uniform `q_len_per_req` 与 `cum_seq_lens_q/max_q_len` ragged speculative decode；
- FP8、NVFP4 KV 与可选 NVFP4 output；
- per-block KV scales；
- attention sinks 与 LSE output；
- PDL；
- block-sparse per-KV-head page table；
- 可选 skip-softmax 近似路径。

这些是 cubin backend 的接口/metadata 能力。是否在某个具体 shape 下选中某个专用 cubin，仍由 dtype、SM、head dims、tile、mask 与 layout 的完整 key 决定。

### 8.10 counter buffer 与 workspace

Multi-CTA KV 需要 counter buffer 追踪同一 output 的 split CTAs 是否完成。buffer 首次分配必须为零；kernel 每次 launch 末尾自行复位，因此后续调用无需 host 反复清零。

workspace 则承载 softmax stats、global partial 或其他 scratch，具体用途依 reduction mode 而变。二者不要混为一谈：

```text
counter buffer = completion/ownership 协议状态
workspace      = partial stats、partial O 或 kernel scratch payload
```

若使用 CUDA Graph，二者都应提前分配并保持地址稳定。

### 8.11 哪些说法是证据，哪些只能算推断

可以明确确认：

- CC 10.x 自动选择 TRTLLM-Gen；
- SM100-family 与 SM103-specific kernel matching；
- tile/scheduler/Multi-CTA/page/dtype 等 metadata 维度；
- Multi-CTA KV 数量公式与“至少约两个 KV steps/split”的 cap；
- CGA shared-memory、global-memory、separate-kernel reduction 三类选择；
- Q tile cost model、dynamic page size、ragged Q、FP8/NVFP4、PDL、LSE。

不能只凭当前开源代码确定：

- 每个 cubin 的精确 TMA stage 数；
- TMEM/SMEM 的逐地址布局；
- warp role 的完整分配；
- 每条 barrier 与 MMA 指令的时序；
- 某个接口能力必然对应一个独立 kernel，而非共享 kernel 的 runtime 分支。

所以正确表述是“TRTLLM-Gen 的 host-side optimization policy 明确选择这些方案；kernel 本体以 cubin 交付”，而不是把可见的 dispatch 直接当成完整 kernel 源码。

---

## 9. Atrex 的 CUDA Graph、ragged Q 与归约选择

### 9.1 ragged Q 是什么

不同 request 的 query 长度可不同。packed 表示为：

```text
Q: [total_q, Hq, D]
cu_seqlens_q: [batch+1]
```

例如 query lengths 为 `[0,1,2,3]`：

```text
cu_seqlens_q = [0,0,1,3,6]
```

request 0 没有 query；request 1 使用 `Q[0:1]`；request 2 使用 `Q[1:3]`；request 3 使用 `Q[3:6]`。

MTP 系统中完全可能出现 batch 内 Q 长度不同：请求可能处于不同 speculative depth，某些 token 已被 mask/reject，或 CUDA Graph bucket 中只有部分 slots active。

### 9.2 CUDA Graph slot 与 inactive slot

Graph B8 固定预留八个 batch positions，也就是八个 slots。若当前只有五个 Q1 请求：

```text
cu_seqlens_q = [0,1,2,3,4,5,5,5,5]
```

slots 0～4 的 query length 为 1；slots 5～7 的起止 offset 相同，query length 为 0，所以 inactive。

inactive 的准确含义是：

```text
graph 中该固定 batch 位置仍存在，但本轮没有绑定有效 query；
对应 CTA 仍可能 launch，读取长度后 early-exit。
```

它不是空 KV page，也不是 KV cache 中未使用的 token slot。

### 9.3 为什么 dispatch 使用 graph batch

Graph replay 期间 kernel 数量、grid 和依赖 topology 不能因 active count、KV length 改变。Atrex 因此用 graph bucket 的静态 batch 计算 `base_ctas`，而不是本轮 active requests：

```text
Graph B8, active=5
dispatch batch=8
```

这可能高估有效 CTA 数，从而选择比 active-count scheduler 更少的 splits；代价是少量 replay 效率，收益是同一 graph 可稳定复用。

同理，Atrex 不根据 replay 时真实 KV 长度改变 split topology。长 KV 时配置合理，短 KV replay 可能产生空 split。

### 9.4 为什么 Q2～Q5 强制 external

Atrex 最终公开行为是：

```text
Q2～Q5       → external reduction
请求 LSE     → external reduction
Q1 无 LSE    → 按 batch/head bucket 选择 none/atomic/external
```

内部 heuristic 中即使存在 Q2/Q3/Q4 atomic 候选，最终 runtime 规则仍覆盖为 external。分析真实行为应看最终有效规则，而不是中间候选。

external 的 Triton kernel 可同时：

```text
merge m/l/U
  + dense graph slot → packed ragged token
  + 跳过 inactive/padding rows
  + 输出自然对数 LSE
```

第一阶段 workspace 可保持 graph-friendly dense shape：

```text
[splits, graph_batch, max_q, heads, ...]
```

第二阶段读取 `cu_seqlens_q`，把有效 `(batch,query_slot)` 映射到 packed output index。这样 ragged bookkeeping 不侵入高性能 CuTe mainloop。

### 9.5 为什么 LSE 也强制 external

归约本来就得到：

```text
LSE = M + log(L)
```

CuTe 主循环内部采用 base-2 指数状态；公开接口需要自然对数 LSE。Triton reduction 可在合并时直接完成尺度/对数底转换。这个选择是实现组织和融合收益，不表示 atomic 路径在数学上无法计算 LSE。

---

## 10. CTA 内主流水线：TMA → QK → softmax → PV → correction

### 10.1 数据流

```text
Q: GMEM ─TMA→ SMEM ───────┐
                           ├─ TCGEN05 QK → logits in TMEM
K: paged GMEM ─TMA→ SMEM ─┘
                                      ↓
                            softmax in registers
                                      ↓
                              P staged in SMEM
                                      ↓
V: paged GMEM ─TMA→ SMEM ─ TCGEN05 PV → O in TMEM
                                      ↓
                         online-softmax correction
                                      ↓
                         SMEM epilogue ─TMA→ GMEM
```

Q 对一个 CTA work 通常只加载一次；K/V 随 sequence tiles 循环。优化重点是让 K/V 搬运、QK、softmax、PV 和 correction 在不同 stages 上重叠。

### 10.2 为什么 QK 与 PV 分给不同 warp

QK producer 生成 TMEM logits；softmax consumer 把 logits 转成 P；PV consumer 再读取 P。若同一 warp 串行承担 QK 和 PV，两个 MMA 链之间难以 overlap。专门 warp 让 QK 可以继续生产后续 tile，同时 PV 消费较早 tile。

### 10.3 correction 为什么是独立角色

online softmax 每看到更大的新 max，都必须重缩放旧 numerator：

```text
U_old *= exp(m_old - m_new)
```

O accumulator 很宽，HD256 × N32/N64 的 FP32 fragment 需要大量 TMEM/register bandwidth。把 correction 独立出来，可与 MMA/TMA 分工，并通过 barrier 精确约束“先 rescale 还是先 accumulate”。

### 10.4 stage 不是单一概念

至少要区分：

- K/V stages：隐藏 paged global memory latency；
- S stages：TMEM 中可并行存在的 logits tiles；
- P stages：softmax 到 PV 的 producer/consumer ring；
- O stages：PV accumulator 与 correction 的 phase 数。

调大某个 stage 会消耗 SMEM/TMEM/barriers，可能挤压其他 stage 或降低 occupancy。必须看完整资源账本，不能只说“更多 stages 更好”。

---

## 11. 片上资源：SMEM、TMEM 与 registers

### 11.1 SMEM 账本

主要项目：

```text
Q tile
K/V pipeline stages
P stages
O store staging
softmax max/sum buffers
atomic cluster DSM buffers
TMA/pipeline mbarriers
TMEM pointer与控制状态
```

atomic 模式还需为最多 `log2(16)=4` 轮 cluster reduction 预留 max/sum 和 barrier 空间。

### 11.2 TMEM 账本

TMEM 保存 Tensor Core accumulator：

```text
S: sequence_tile × packed_rows × s_stages
O: head_dim × packed_rows × o_stages
L/state: packed_rows × phases
```

N64 会显著增加 O/S columns，因此 Atrex 把 N64 的 O stages 降到 1。Q4 的 P stages 降到 2也是同一资源平衡的一部分。

### 11.3 register redistribution

warp specialization 造成角色间寄存器需求不均：

- TMA/control warps 只需少量 registers；
- softmax warps 保存 max/sum 和 fragments；
- correction warpgroup 读取宽 O fragment，需求更高。

实现使用 `setmaxregister_decrease/increase` 把额度从轻角色让给重角色。验证时必须同时看编译 spill 和动态 occupancy；仅看到源码中的 register hint 不代表最终没有 local-memory spill。

---

## 12. 同步设计：只在真实数据依赖处等待

### 12.1 CTA 内同步

| 生产者 → 消费者 | 数据 | 同步目的 |
|---|---|---|
| TMA → QK/PV | Q/K/V in SMEM | 确认异步搬运完成 |
| QK → softmax | logits in TMEM | 确认 MMA accumulator 可读 |
| softmax → PV | P in SMEM | 防止 PV 读取未完成 P |
| PV → correction | O in TMEM | 防止 rescale 与 accumulate 乱序 |
| epilogue → 下一 phase | O staging | 防止 buffer 被过早复用 |

TMA 和 TCGEN05 MMA 都是异步操作。发出指令不等于数据已可被下一角色消费。

### 12.2 cluster 同步

atomic 模式还包含：

```text
cluster barrier initialization
cluster arrive/wait
remote DSM max/sum exchange
每轮 butterfly mbarrier
```

即使一个 split 没有有效 KV tile，它也必须安全完成协议要求的初始化/到达，否则 peer CTA 可能永久等待。这正是空 split 在 atomic 模式下比普通 early-exit 更敏感的原因。

### 12.3 external 的同步边界

external 模式不做 split CTA 间细粒度同步。第一 kernel 完成是全局 partial 可见的边界，第二 kernel 才合并。慢 split 会推迟第二 kernel 开始，但不会让其他 split CTA 在 DSM barrier 中驻留等待。

---

## 13. FP8 与数值路径

QK logits、softmax max/sum 和 O accumulation 应保持 FP32 状态。FP8 主要影响 Q/K/V/P 的 Tensor Core 输入与 scale。

关键原则：

- QK scale 与 `log2(e)` 的转换必须一致；
- online softmax correction 必须在同一指数底下计算；
- P cast 到 FP8 前需要与 PV 使用的 scale/offset 成对设计；
- external reduction 的 `m/l/U` workspace 保持 FP32；
- LSE 输出若要求自然对数，必须完成 base-2 到 natural-log 转换；
- masked/empty split 用有限的最小值或严格 identity，避免 `-inf - -inf` 产生 NaN。

Atrex 当前公开 fast path 给 BLASST threshold 传 `None`，因此 BLASST 不参与实际 dispatch。不能把性能差异归因于一个未启用的近似跳 tile 方案。

---

## 14. 三条实现路线的直接对比

| 维度 | CuTe DSL | TRTLLM-Gen cubin | Atrex FA4 decode |
|---|---|---|---|
| 目标 | 通用可读 GQA kernel | SM100/SM103 生产 backend | 特定 SM103 serving fast path |
| kernel 本体 | CuTe DSL 源码可读 | 预编译 cubin | CuTe DSL/Triton 源码可读 |
| dispatch | wrapper heuristic | metadata + cost model + cubin selector | graph bucket + shape-specific table |
| head dim | 参数化，满足 kernel 约束 | key 支持多种 QK/V dim 与 split-V | 固定 HD256 |
| Q 长度 | prediction tile 参数化 | uniform/ragged speculative decode | Q1～Q5，ragged 可 Q0 |
| reduction | none / atomic / kernel | CGA SMEM / GMEM / separate kernel | none / atomic / external Triton |
| KV split | atomic 为 1/2/4/8/16 | Multi-CTA KV，按 occupancy 与 KV work cap | 1/2/4/8/16，graph-stable + shape caps |
| Q/head tile | grouped heads × prediction | Q8/Q16/Q32/Q64/Q128 cost model | 主要 N32；Q4 特定场景 N64 |
| sequence tile | 参数化，通常 128 倍数 | KV64/KV128 metadata | 主要 S128/S256 dispatch |
| public page 规则 | 8/16/32/64 | power-of-two；>=128 dynamic key，测试到 1024 | 16/32/64/128/256 |
| page256 | 不在公共 paged decode 范围 | dynamic page kernel | 两个零拷贝 page128 子页 |
| ragged Q | wrapper 支持相应形态 | `cum_seq_lens_q/max_q_len` | true packed Q0～Q5 |
| CUDA Graph | 可作为基础 kernel 使用 | 固定 buffers 可 capture | 显式固定 long-KV topology/workspace |
| LSE | kernel 路径可输出 | 直接可选输出 | 请求 LSE 强制 external 并转换 |
| 低精度 | 与具体 CuTe kernel specialization 相关 | FP8/NVFP4 KV、可选 NVFP4 O | BF16/FP8 E4M3FN |
| skip softmax | BLASST 可选、默认关 | 可选近似 cubin trait | fast path threshold=None |

最重要的判断是：

```text
CuTe DSL 提供可读、可组合的 kernel building blocks；
TRTLLM-Gen 提供覆盖更广的生产 cubin 集合与 runtime cost model；
Atrex 把可读 kernel 收窄、重排并接入固定 serving topology。
```

三者的差异不主要是 attention 公式，而是开放边界、支持矩阵、调度模型与 serving 集成。Atrex 的差异集中在 ragged/CUDA Graph 生命周期、page128/256、N64 Q4、external fused reduction 和实测 dispatch 的组合；TRTLLM-Gen 的差异集中在大规模 cubin specialization、Multi-CTA 模式与 runtime cost model。

---

## 15. 如何评价 dispatch 是否合理

### 15.1 先看有效 CTA，而不是 launch CTA

需要区分：

```text
launched_ctas = graph_batch * head/query tiles * kv_splits
useful_ctas   = active slots * head/query tiles * min(kv_splits,num_kv_tiles)
```

inactive graph slots 与空 KV splits 都会让 launched 数高于 useful 数。

### 15.2 再看归约占比

对短 KV：主循环小，第二 kernel launch 或 DSM 初始化可能占比很高。

对长 KV：主循环大，split 带来的并行收益更容易覆盖归约成本，但 partial-O workspace 流量也随 splits、Q rows、heads、D 增加。

workspace 的主要量级近似为：

```text
O_partial bytes ≈ splits * B * Sq * Hq * D * 4
m/l bytes       ≈ splits * B * Sq * Hq * 2 * 4
```

HD256 下 O partial 是主项。

### 15.3 看 cluster 是否成为限制

atomic 路径应观察：

- cluster launch/active clusters；
- barrier stall；
- split 间 workload imbalance；
- DSM traffic；
- TMA reduce/atomic 写竞争；
- 输出预清零成本。

external 路径应观察：

- partial global write/read bytes；
- reduction kernel duration；
- launch gap；
- workspace 是否已预热并 graph-stable；
- dense-to-packed/LSE 融合是否抵消额外 launch。

---

## 16. 修改 kernel 时的验证矩阵

### 16.1 正确性维度

至少覆盖：

```text
Q length: 0/1/2/3/4/5
batch: active 满 bucket / 部分 inactive / 单 request
KV: 0、短于 tile、刚好 tile、跨 page、长 KV
GQA: grouped_heads=1/8/16/32
split: 1/2/4/8/16，包含 splits > num_kv_tiles
reduction: none/atomic/external
dtype: BF16/FP8
page: FlashInfer 8/16/32/64；Atrex 16/32/64/128/256
LSE: off/on
```

检查 O 与 reference；检查 LSE；检查 inactive rows 不写越界；检查 empty split 不导致 NaN 或 deadlock；检查 page tail 地址即使被 mask 也物理安全。

### 16.2 CUDA Graph

- warmup 时完成 JIT 与 workspace allocation；
- capture 内不得动态编译或分配；
- 相同 bucket 用不同 active slots、KV lengths replay；
- grid、kernel 数和依赖保持不变；
- atomic 输出每轮正确清零；
- external workspace 不跨 layer/request 错误 alias。

### 16.3 性能矩阵

不能只测一个长 KV。至少交叉：

```text
batch bucket × Q1..Q5 × KV length × head shape × page size × dtype
```

对每个点同时记录：

- selected splits/reduction/N tile/S tile/stages；
- main kernel time；
- reduction/clear time；
- 端到端 attention time；
- useful CTA 比例。

---

## 17. NCU 与 SASS 的诊断顺序

### 17.1 先确认跑的是哪个 kernel

先记录实际选择：

```text
none / atomic / external
kv_splits
N32 / N64
S128 / S256
P/O stages
page size
ragged / dense
```

否则很容易用一个方案的指标解释另一个方案。

### 17.2 再确认 work geometry

计算：

```text
base_ctas
launched_ctas
active-slot ctAs
num_kv_tiles
empty splits
waves per SM
```

若性能差来自只有一半 CTA 有效，继续微调单条 MMA 指令没有意义。

### 17.3 再判断瓶颈

- memory bound：看 TMA bytes、page metadata、L2/DRAM throughput；
- Tensor Core bound：看 TCGEN05 issue/active；
- synchronization bound：看 mbarrier、cluster wait、warp stall；
- reduction bound：看 partial traffic、第二 kernel 和 atomic contention；
- resource bound：看 registers、spill、SMEM/TMEM allocation、active clusters。

### 17.4 最后用指令证明路径

SASS/编译产物至少确认：

- Q/K/V 是否确实使用预期 TMA；
- QK/PV 是否为预期 TCGEN05 MMA；
- atomic 路径是否使用 reduce store；
- page128/page256 是否没有意外 gather/copy；
- register spill 是否产生 local load/store；
- FP8 路径是否出现预期转换而无多余往返。

---

## 18. 代码位置

FlashInfer：

```text
flashinfer/cute_dsl/attention/gqa_decode.py
flashinfer/cute_dsl/attention/gqa_decode_paged.py
flashinfer/cute_dsl/attention/wrappers/batch_decode.py
flashinfer/decode.py
include/flashinfer/trtllm/fmha/fmhaKernels.cuh
include/flashinfer/trtllm/fmha/fmhaRunnerParams.h
include/flashinfer/trtllm/fmha/kernelParams.h
csrc/trtllm_fmha_kernel_launcher.cu
flashinfer/jit/cubin_loader.py
flashinfer/artifacts.py
```

Atrex：

```text
src/fa4/decode_runtime.py
src/fa4/decode_cutedsl.py
src/fa4/decode_reduce_triton.py
python/atrex/api/fa4.py
tests/test_fa4_vllm_gpu.py
```

阅读顺序建议：先看 runtime 中的支持边界和最终 dispatch，再看 CuTe mainloop 的 tile/stage/warp roles，最后看 Triton reduction 如何合并 partial 并 pack ragged output。

---

## 19. 核心优化结论

1. Decode 的第一瓶颈通常是 Q 太短导致 CTA 数不足；Flash-Decoding 通过沿 KV sequence 增加 split CTA 解决。
2. `kv_splits` 是同一条 attention 的 KV 并行份数，不是独立 query 数。每个 split 输出局部 `m/l/U`，必须按稳定 softmax 公式合并。
3. single-split、atomic-cluster 和 external-reduction 是三种拓扑，不是同一 kernel 的无成本开关。
4. Atomic 用 DSM/mbarrier 与 TMA reduce 减少 global workspace和第二 launch；代价是 cluster 耦合、barrier wait 和输出写竞争。
5. External 用 FP32 global partial 和第二 kernel 换取 split CTA 解耦，并能融合 ragged packing、LSE 与格式转换。
6. GQA packing 把共享同一 KV head 的 query heads 与 MTP tokens放到一个 CTA；Atrex 的 N64 Q4 是这一思路的 shape-specific 扩展。
7. CUDA Graph 的 dispatch 必须基于固定 bucket，而不是 replay 时 active slots 或 KV length；因此会出现 inactive CTA 和短 KV 空 split。
8. Page size 会改变一个 sequence tile 涉及的物理 pages、TMA 数和 page-table loads。FlashInfer 公共路径支持 8/16/32/64，Atrex 扩展到 128/256。
9. Atrex 的主要新颖处不是新的 attention 数学，而是 SM103/HD256 特化、ragged Q0～Q5、graph-stable split、N64、page128/256 与 fused external reduction 的组合。
10. TRTLLM-Gen 在 FlashInfer 中是 CC 10.x 的默认生产路径；可读的 host policy 明确包含 Q tile cost model、Multi-CTA KV、CGA/GMEM/separate reduction 和 dynamic page-size 选择，但 cubin 内完整流水线不能从当前源码直接证明。
11. 优化应按“有效 work geometry → 归约成本 → CTA 内流水线 → 片上资源 → 指令”下钻；只看单 kernel 的 Tensor Core 利用率不足以判断端到端 decode 是否更快。
