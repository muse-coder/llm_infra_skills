# B300 FA4 HD256 FP8 Causal Varlen Prefill Kernel 优化

> 代码基线：Atrex `0381108a`
> 目标：NVIDIA HGX B300 / SM103、FP8 E4M3 Q/K/V、BF16 O、`head_dim=256`、causal varlen prefill、GQA16

## 1. 本文只讨论 GPU kernel

本文只记录算子内部优化：

- B300/SM103 使用了哪些硬件能力；
- Q/K/V/O 如何在 GMEM、SMEM、TMEM、register 之间搬运；
- `tcgen05.mma`、TMA、TMEM load/store、barrier 对应哪些 PTX/SASS；
- QK、softmax、P cast、PV、online-softmax correction、epilogue 如何编排；
- 哪些工作异步 overlap，哪些依赖必须显式同步；
- 每个 tile 搬多少数据、做多少计算；
- 2CTA、CLC、PackGQA、K ping-pong、Paged-KV TMA 为什么有效。

本文不覆盖 host/API 工程。

当前 kernel 的核心配置为：

```text
physical cluster       (2, 1, 1)
threads / CTA          320 = 10 warps
logical QK tile        M256 × N128 × K256（CTA pair）
per-CTA Q rows         128
Q stage                1
S/P K-direction slots  2
K/V TMA stages         5
input                   FP8 E4M3
accumulator             FP32 in TMEM
output                  BF16
scheduler               CLC + causal LPT
GQA                     PackGQA, ratio 16
```

## 2. B300 硬件预算与算子瓶颈

### 2.1 官方硬件数据

NVIDIA Blackwell Ultra Datasheet 对 HGX B300 的公开规格为：

| 单 GPU 指标 | HGX B300 Blackwell Ultra |
| --- | ---: |
| FP8/FP6 Tensor Core | 9 PFLOPS sparse，即 4.5 PFLOPS dense |
| HBM3E | 270 GB |
| HBM bandwidth | 7.7 TB/s |
| HGX B300 GPU 数 | 8 |
| HGX B300 总 FP8/FP6 | 72 PFLOPS sparse |

因此 dense FP8 的理论 ridge point 约为：

```text
4.5 PFLOP/s ÷ 7.7 TB/s ≈ 584 FLOP/byte
```

这个数字只适合判断“最终是否可能受 HBM 限制”。本 kernel 的 K/V 会跨 Q tiles 在 L2 中复用，所以实际 HBM arithmetic intensity 高于单 cluster 的 SMEM staging intensity；测量中 DRAM SOL 往往很低，主要瓶颈更常是 tensor-pipe 利用率、TMA/地址 latency 和 barrier。

来源：

- [NVIDIA Blackwell Ultra Datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet)
- [NVIDIA HGX B300 specifications](https://www.nvidia.com/en-us/data-center/hgx.md)

### 2.2 SM 片上资源为什么决定当前 tile

NVIDIA Blackwell Tuning Guide 对 compute capability 10.0 给出的主要资源为：

```text
register file             64K × 32-bit registers / SM
max registers / thread    255
shared memory / SM        228 KB
max shared memory / CTA   227 KB
max concurrent warps      64 / SM
```

当前 FP8 kernel 每 CTA 使用约 `182.27 KB` dynamic shared memory，因此一个 SM 只能驻留一个 CTA。当前测试设备由 CUDA/NCU 报告 148 SM；2CTA cluster 需要两个 SM，所以同一时刻最多约：

```text
148 CTA / 2 CTA per cluster = 74 resident clusters
```

这也是 PackGQA、work count 和 wave cliff 特别重要的根本原因。

来源：[NVIDIA Blackwell Tuning Guide — Occupancy](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy)

### 2.3 单个 logical K block 的计算量与片上搬运

CTA pair 每次处理：

```text
M = 256 packed Q rows
N = 128 KV tokens
D = 256
```

一个 K block 的 QK 与 PV 总 FLOPs：

```text
QK = 2 × M × N × D
PV = 2 × M × N × D
总计 = 4 × 256 × 128 × 256
     = 33,554,432 FLOPs
```

每个 K block 从 GMEM/L2 搬入 cluster SMEM 的 FP8 payload：

```text
K = 128 × 256 × 1B = 32 KB / cluster
V = 128 × 256 × 1B = 32 KB / cluster
合计                    64 KB / cluster
```

只按 K/V staging 计算：

```text
33.55 MFLOPs ÷ 64 KB = 512 FLOP/byte
```

它接近但低于 584 FLOP/byte 的理论 ridge point，因此 K/V TMA、L2 命中和 pipeline depth 都很重要。与此同时，多 Q tiles 会在 L2 中复用 K/V，实际 HBM 流量可远低于所有 cluster staging 流量；所以最终要用 NCU 区分 HBM、L2/TMA latency 和同步瓶颈。

## 3. 2CTA、warp specialization 与片上存储

### 3.1 2CTA 不是两个独立的 HD128 attention

physical cluster 为 `(2,1,1)`：

```text
CTA0 owns Q rows [0,   128)
CTA1 owns Q rows [128, 256)

CTA pair jointly executes one M256 × N128 × K256 tcgen05 MMA
```

两个 CTA 合作完成完整 `D=256` reduction。它们不是各算一半 head dimension 后在软件中归约；`tcgen05.mma.cta_group::2` 直接访问 CTA pair 的 TMEM/SMEM operand。

NVIDIA PTX ISA 规定：

- CTA pair 是 cluster rank 只差最低 bit 的两个 CTA；
- `tcgen05` 的 pair-level 操作会访问两个 CTA 的 Tensor Memory；
- `tcgen05.mma.cta_group::2` 在当前 CTA 和 peer CTA 的 TMEM 上执行；
- `tcgen05.mma` 是异步、single-thread-semantics 指令，一条指令即可发起完整 MMA。

来源：

- [PTX ISA — CTA Pair](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair)
- [PTX ISA — tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)

### 3.2 当前 10-warp 分工

`q_stage=1 + TMA Q/K/V + varlen vector epilogue` 允许把通用 16-warp 布局压缩到 10 warps：

| Warp | 数量 | 任务 | 寄存器配置 |
| --- | ---: | --- | ---: |
| softmax warps | 4 | TMEM→register 读取 S；mask、row max、`exp2`、row sum；FP32→FP8 P；写回 TMEM | 160 |
| correction/epilogue warps | 4 | online-softmax O rescale；最终 normalize；TMEM→SMEM→GMEM | 128 |
| MMA warp | 1 | 发射 QK/PV `tcgen05.mma.cta_group::2` | 96 |
| load/CLC warp | 1 | Q/K/V TMA；leader CTA 上预取下一 CLC work | 96 |
| 合计 | 10 | 320 threads / CTA | — |

关键技巧：

- `tcgen05.mma` 是 single-thread-semantics，不需要用多个 warps 发射；一个 MMA warp 已足够。
- softmax 与 correction 是标量/SIMT 重工作，分别分配四个 warps，并提高寄存器预算。
- correction warps 同时承担 epilogue，删除独立 epilogue warp 和空闲 softmax group。
- load warp 在 TMA 发射完当前 tile 后，替 leader CTA 预取 CLC response，避免为 CLC 再增加第 11 个 warp。

### 3.3 用 warp-role register redistribution 消除 spill

最终配置不是给 320 个 threads 统一设一个较低上限，而是按 warp role 重新分配 CTA register pool：

```python
# softmax warpgroup, warps 0..3
cute.arch.setmaxregister_increase(160)

# correction/epilogue warpgroup, warps 4..7
cute.arch.setmaxregister_decrease(128)

# MMA warp 8 and load/CLC warp 9
cute.arch.setmaxregister_decrease(96)
```

对应 PTX 是：

```text
setmaxnreg.inc.sync.aligned.u32 160;
setmaxnreg.dec.sync.aligned.u32 128;
setmaxnreg.dec.sync.aligned.u32 96;
```

在当前 SM103 NCU SASS 中对应：

```text
USETMAXREG.TRY_ALLOC.CTAPOOL ..., 0xa0   # 160
USETMAXREG.DEALLOC.CTAPOOL       0x80   # 128
USETMAXREG.DEALLOC.CTAPOOL       0x60   # 96
```

因此验证时应区分三层名字：CuTeDSL `setmaxregister_*` → PTX `setmaxnreg.inc/dec` → SM103 SASS `USETMAXREG.*.CTAPOOL`。

这不是“每个 warp 实际一直使用这么多寄存器”，而是修改该 warp 每线程可拥有的最大寄存器数。NVIDIA PTX 规定 register pool 以 CTA 为单位维护；`.dec` 把寄存器归还池，`.inc` 从池中申请，若池中数量不足，`.inc` 会阻塞。立即数必须在 24～256 之间且为 8 的倍数；同一 warpgroup 的所有 warps 必须执行相同的 `setmaxnreg`，否则行为未定义。

来源：[PTX ISA — `setmaxnreg`](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)

#### 小 trick：把寄存器给真正保存大 fragment 的 warps

softmax 的 register pressure 最大。每 CTA 的 S tile 为 `128×128 FP32`，由 128 个 softmax threads 处理；只算 score fragment 就约为：

```text
128 × 128 / 128 = 128 FP32 values/thread
                    ≈ 128 32-bit registers/thread
```

还需要 row max/sum、mask predicate、地址、scale 和 FP8 packing 临时量，所以最终给 softmax 160，而不是只按 score payload 给 128。

correction 若一次把整个 `128×256 FP32 O` 搬进 128 个 threads，会需要约 256 values/thread，必然超过当前 128 配额。最终代码不这么做，而是：

```text
correction_rescale: corr_tile_size = 16 columns
epilogue normalize: corr_tile_size = 16 columns

TMEM load 16 columns → register multiply/convert → TMEM/SMEM store
然后复用同一批 registers 处理下一段 16 columns
```

也就是说，解决 spill 的核心不是盲目增大上限，而是把 O live range 切成 16-column fragments。correction warps 随后复用同一寄存器配额做 epilogue；两个阶段按时间串行，不同时保留 rescale fragment 和完整 output fragment。

MMA warp 只负责组装 descriptor、发异步 `tcgen05.mma` 和推进 barrier，矩阵 accumulator 放在 TMEM；load/CLC warp 主要保存 TMA descriptor、pipeline state 和少量 page IDs。因此两者都限制在 96，把释放的 pool 留给 softmax/correction。

#### 小 trick：编译前先做两层 register budget

第一层检查硬件/CTA pool，不预测编译器：

```text
B300 register file                  = 65,536 × 32-bit registers / SM
threads / CTA                       = 10 warps × 32 = 320
NCU launch allocation              = 160 registers/thread
粗略 CTA launch pool               = 320 × 160 = 51,200 registers

role maxima requested
  = 32 × (4×160 + 4×128 + 1×96 + 1×96)
  = 43,008 registers / CTA

CTA-pool headroom before rounding   = 51,200 - 43,008 = 8,192 registers
occupancy headroom before rounding  = 65,536 - 51,200 = 14,336 registers
```

这里必须使用“warp 数量 × 该 role 的 registers/thread”，不能只看 kernel 报告的最大 `160 registers/thread`。当前 occupancy 已被约 182.27 KB SMEM 固定为一 CTA/SM，因此把闲置 register capacity 给 softmax，通常不会再降低 residency。

第二层按 **同时 live 的 thread-local 数据** 估算 role 下界：

```text
R_live_est
  = Σ(同时存活的 fragment elements/thread × 每 element 的 32-bit register 数)
  + scalar state
  + address/descriptor state
  + predicate/index temporaries
  + compiler scheduling margin
```

只累计同一时间窗口内的 fragment。已经写回 TMEM/SMEM 的前一段不要重复计算；相反，若为了 overlap 同时保留 `current` 和 `next` 两份地址、page ID 或 fragment，就必须两份都计入。特别检查：

- FP32/Int32/pointer low/high 通常各占一个或两个 32-bit registers；
- runtime-indexed local array 可能直接落到 local memory，不能假设会完全 registerize；
- 循环 unroll 会复制临时量并拉长 live range；
- helper inline 后，调用点两侧原本不重叠的变量可能同时 live；
- register-local page-ID cache 应只留在 load warp，不应跨到 softmax/correction role。

这一步只能判定“明显放不下”或给 sweep 起点，不能精确预测 spill。寄存器分配、live-range splitting、指令调度和 allocation granularity 都由编译器决定。

#### 小 trick：用编译结果和动态 counter 双重确认

最终接受条件是：

```text
compile/launch:
  registers per thread allocated = 160
  spill stores / spill loads      = 0 / 0

NCU dynamic:
  sass__inst_executed_register_spilling           = 0
  sass__inst_executed_register_spilling_mem_local = 0
  sass__inst_executed_register_spilling_mem_shared= 0
  derived__local_spilling_requests                 = 0
```

同时检查 SASS 中没有由寄存器溢出产生的 `LDL/STL`。普通 local-memory 指令也可能来自显式/编译器放置的 thread-local object，所以以 NCU 的 register-spilling 分类为最终依据。NVIDIA Nsight Compute 对 `derived__local_spilling_requests` 的定义就是“register spilling 对 L1 发出的已执行指令和请求数”。

若出现 spill，按下面顺序处理：

1. 先定位是 softmax、correction/epilogue 还是 load role，而不是全 kernel 一起加寄存器。
2. 以 8 registers/thread 为步长调整对应 `setmaxnreg`，每次重新检查 CTA pool 预算。
3. 优先缩短 live range：像当前 O 路径一样按 16 columns load→compute→store，避免保留完整 tile。
4. 把大 accumulator/中间矩阵留在 TMEM，把跨 warp 数据留在 SMEM；register 只保存当前 fragment。
5. 对地址和 page IDs，只缓存真正复用且长度编译期固定的 tuple；廉价、低复用值可以重算。
6. 每次同时复查 wall time 和 `setmaxnreg.inc` 等待；“spill 为零”不代表寄存器配置已最优。

来源：[Nsight Compute Profiling Guide — register spilling metrics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)

### 3.4 Shared memory 账本

每 CTA 的主要 SMEM：

| Buffer | 逻辑尺寸 | 每 CTA bytes | 说明 |
| --- | --- | ---: | --- |
| sQ | `128 × 256 × FP8` | 32 KB | `q_stage=1` |
| sO | `128 × 256 × BF16` | 64 KB | 当前不能与 sQ alias |
| sK/sV physical storage | `5 × 16 KB` | 80 KB | K/V 共享同一块 physical SMEM，5-stage pipeline |
| barrier/CLC/scratch | — | 约 6.27 KB | mbarrier、CLC response、scale 等 |
| 合计 | — | 约 182.27 KB | NCU 实测 dynamic SMEM |

K/V 每 stage 是 `16 KB / CTA`：logical cluster K 或 V tile 为 32 KB，由 CTA pair 各持有一半 operand slice。

sQ 和 sO 不做 alias：在 2CTA CLC persistent 生命周期内，只有在所有相关 warp 和下一 work 都不再访问 Q 后才能安全复用；当前同步协议没有提供这个证明，所以为 sO 保留独立 64 KB。

### 3.5 Tensor Memory 账本

PTX ISA 说明，Blackwell 5th-generation Tensor Core 的 TMEM 每 CTA 为：

```text
128 lanes × 512 columns × 32 bits
```

当前 HD256 kernel 使用完整 512 columns：

```text
S/P slot 0      128 columns
S/P slot 1      128 columns
O accumulator   256 columns
合计            512 columns
```

这解释了为什么：

- `q_stage` 只能为 1：再保留第二个 O accumulator 会超过 TMEM；
- 仍然可以做 K-direction ping-pong：S/P 使用两个 128-column slot；
- O 始终以 FP32 留在 TMEM，直到 correction/epilogue。

来源：[PTX ISA — Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)

## 4. PackGQA、physical cluster 与 wave

### 4.1 先区分三个概念

```text
work tile
    scheduler 的一个逻辑任务：一个 packed M256 tile × 一个 KV head

physical cluster
    执行一个 work tile 的两个 CTA，即 cluster=(2,1,1)

wave
    GPU 同一时刻能驻留的一批 physical clusters
```

由于每 SM 只能驻留一个 CTA，B300 一 wave 最多运行 74 个 2CTA clusters。wave 不是一次新的 kernel launch，而是 launch 中的并发批次：超过 74 的 work 必须等前一批 cluster 释放资源后才能继续。

### 4.2 不 PackGQA 时为什么产生大量 tail padding

以 `Q=1084, HQ=16, HKV=1, GQA ratio=16` 为例。

不 pack 时，每个 Q head 单独切 M256 tiles：

```text
每个 head 的 tiles = ceil(1084 / 256) = 5
总 work clusters    = 5 × 16 = 80

实际 Q rows         = 1084 × 16 = 17,344
调度 Q rows         = 80 × 256   = 20,480
padding             = 3,136 rows = 18.1%
```

80 clusters 超过单 wave 容量 74：

```text
wave 0: 74 clusters
wave 1:  6 clusters  ← 严重低利用率 tail
```

即使 CLC 能让先结束的 resident cluster steal 后续 work，最后只剩 6 个 work 时，最多也只有 6 个 clusters 在做有效计算。

### 4.3 PackGQA 如何重排 M 维

PackGQA 把同一 KV head 对应的 16 个 Q heads 折入 packed M：

```text
packed_M = q_tokens × q_heads_per_kv_head
         = 1084 × 16
         = 17,344 rows

work clusters = ceil(17,344 / 256) × HKV
              = 68 × 1
              = 68

调度 Q rows   = 68 × 256 = 17,408
padding       = 64 rows = 0.37%
```

68 clusters 可以全部放进一个 74-cluster wave，消除了 `74+6` 尾波。

PackGQA 不改变下面这些硬件配置：

- physical cluster 仍是 2CTA；
- 每 CTA 仍拥有 128 packed rows；
- `UTCQMMA.2CTA` 仍计算 M256；
- K/V tile 仍是 N128×D256。

它改变的是 Q row 的逻辑含义：M 维现在同时编码 token 与 Q-head，但同一个 packed tile 中的 rows 都属于同一 KV head，因此仍可共享同一 K/V tile。

### 4.4 通用 work-count 公式

设 GQA ratio 为 `R=HQ/HKV`，logical cluster M tile 为 `T=256`：

```text
unpacked clusters = HKV × R × ceil(Q / T)
packed clusters   = HKV × ceil(Q × R / T)
```

两者在没有 tail rounding 时理论 work 相同；PackGQA 的主要收益来自把 `R` 次独立取整改成一次联合取整。

典型边界：

| Q | HQ/HKV | unpacked | packed | 74-cluster waves |
| ---: | --- | ---: | ---: | --- |
| 1 | 16/1 | 16 | 1 | 都是 1 wave，但 packed 几乎消除空算 |
| 128 | 16/1 | 16 | 8 | 都是 1 wave，work 减半 |
| 256 | 16/1 | 16 | 16 | 无 tail，cluster 数相同 |
| 1084 | 16/1 | 80 | 68 | `2 waves → 1 wave` |
| 1084 | 32/2 | 160 | 136 | `3 waves → 2 waves` |

### 4.5 PackGQA 还带来什么

- 减少每个 Q head 独立 tile 的 causal/tail mask 工作；
- 减少因 padding 产生的无效 QK/PV；
- 同一 packed tile 内复用 K/V shared-memory operand；
- 改善 cluster wave 几何和尾部利用率。

但 PackGQA 不保证所有 shape 都减少 cluster 数。`Q` 已对齐 256 时，work count 可能不变；收益要结合 padding、head 数、L2 locality 和 wave boundary 实测。

最终 specialization 让 PackGQA 与 K ping-pong 同时生效：前者减少 work/padding，后者保持 work 内 QK-softmax-PV overlap；只启用其中一个都不能得到当前完整路径。

## 5. Q/K/V 数据搬运：TMA 指令与字节账

### 5.1 当前路径使用的搬运指令

| 数据 | 方向 | CuTe DSL primitive | 典型 SASS | 完成机制 |
| --- | --- | --- | --- | --- |
| Q | GMEM→SMEM | `CopyBulkTensorTileG2SOp(CtaGroup.TWO)` + MMA-aware TMA A atom | `UTMALDG.3D.2CTA` | transaction mbarrier |
| contiguous/page128/page256 K/V | GMEM→SMEM | CTA-group TWO + MMA-aware TMA B atom | `UTMALDG.4D(.2CTA)` | transaction mbarrier |
| page16/32/64 K/V | GMEM→SMEM | CTA-local page-sized TMA atom | `UTMALDG.4D` | 每 CTA 本地 mbarrier + cluster notification |
| irregular fallback | GMEM→SMEM | non-bulk async tiled copy | `LDGSTS.E...128` | `cp.async` commit/wait 或 mbarrier |
| current O | SMEM→register→GMEM | 128-bit `CopyUniversalOp` | `LDS.128` + `STG.E.128` | correction-warp named barrier |
| regular dense O alternative | SMEM→GMEM | `CopyBulkTensorTileS2GOp` | `UTMASTG` | bulk async-group commit/wait |

PTX ISA 对 `cp.async.bulk.tensor` 的定义是非阻塞 tensor copy，支持 1D–5D、`.cta_group::1/2`、cluster multicast，并以 `mbarrier::complete_tx::bytes` 或 bulk async-group 报告完成。

来源：[PTX ISA — cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)

### 5.2 Q：一次 work 只加载一次，64 KB / cluster

logical M tile 为 256 rows：

```text
Q payload / cluster = 256 × 256 × 1B = 64 KB
Q payload / CTA     = 128 × 256 × 1B = 32 KB
```

Q 使用 `CtaGroup.TWO` 的 MMA-aware TMA A atom，生成 `UTMALDG.3D.2CTA`。Q 只在 work-tile 开始时加载一次，然后被所有 K blocks 重用。

关键点：

- TMA destination 与 2CTA QK operand 的 swizzled SMEM layout 直接一致，避免额外 transpose/repack；
- `q_stage=1`，只有一个 32 KB/CTA sQ buffer；
- Q TMA 与首个 K TMA 可以并行发出；MMA warp 必须等 Q/K 两个 transaction barrier 后才能发 QK；
- descriptor 在 kernel 入口由 `UTMACCTL.PF` 预取，避免首个 TMA 暴露 descriptor fetch latency。

### 5.3 K/V：每个 K block 各 32 KB / cluster

每个 logical K block：

```text
K = 128 tokens × 256 × FP8 = 32 KB / cluster
V = 128 tokens × 256 × FP8 = 32 KB / cluster

per CTA K slice = 16 KB
per CTA V slice = 16 KB
```

K 与 V 的 2CTA 分工不同：

```text
K: 沿 token/page 维拆
   CTA0 loads K tokens [0, 64)
   CTA1 loads K tokens [64, 128)

V: 沿 output-D 维拆
   CTA0 loads V dims [0, 128)
   CTA1 loads V dims [128, 256)
```

这个分工由 2CTA QK/PV 的 B-operand layout 决定，不能把 K 的 source-tile 公式直接用于 V。

### 5.4 page128/page256：一页包含一个或多个 logical tiles

FP8 `tile_n=128`：

```text
page128: 1 physical page = 1 logical K/V tile
page256: 1 physical page = 2 logical K/V tiles
```

page256 必须同时计算：

```text
tiles_per_page = page_size / tile_n = 2
logical_page   = n_block / 2
tile_in_page   = n_block % 2
```

page table 选择 physical page，`tile_in_page` 再选择该页内部的前/后 128 tokens。只用 `n_block` 索引 page table 或固定页内 offset=0，会让第二个 tile 重复读取第一页的前半部分。

这条路径使用 MMA-aware TMA B atom；一次 logical K 或 V tile 的有效 payload 仍为 32 KB/cluster。

### 5.5 page16/32/64：一个 logical tile 聚合多条 page-sized TMA

small page 无法用一条 tensor tile 跨任意 physical pages，因此为每种 page size 构造 compile-time page-sized descriptor，并对 physical pages 发多条 TMA。

#### K 的每 CTA TMA 数与单条 bytes

| page size | 单条 K TMA tile | bytes / TMA | TMA / CTA | 总 bytes / CTA |
| ---: | --- | ---: | ---: | ---: |
| 16 | `16 × D256 × FP8` | 4 KB | 4 | 16 KB |
| 32 | `32 × D256 × FP8` | 8 KB | 2 | 16 KB |
| 64 | `64 × D256 × FP8` | 16 KB | 1 | 16 KB |

K 沿 token/page 拆，所以两个 CTA 各负责 logical tile 中一半 pages。

#### V 的每 CTA TMA 数与单条 bytes

| page size | 单条 V TMA tile | bytes / TMA | TMA / CTA | 总 bytes / CTA |
| ---: | --- | ---: | ---: | ---: |
| 16 | `D128 × 16 × FP8` | 2 KB | 8 | 16 KB |
| 32 | `D128 × 32 × FP8` | 4 KB | 4 | 16 KB |
| 64 | `D128 × 64 × FP8` | 8 KB | 2 | 16 KB |

V 沿 D 拆，所以每个 CTA 都遍历 logical tile 的所有 pages，但只搬自己的一半 D。

最终 page16 的一个 K/V logical block 在 cluster 内动态发出：

```text
K: 4 TMA/CTA × 2 CTA = 8 TMA, total 32 KB
V: 8 TMA/CTA × 2 CTA = 16 TMA, total 32 KB
```

虽然 TMA 条数增加，但它替代了大量 per-thread page pointer、divmod、shuffle 和 `LDGSTS`。最终 SASS 记录中 page16 K/V 有 32 个静态 `UTMALDG.4D` instruction sites、`LDGSTS=0`。

### 5.6 为什么 small page 的多 TMA 仍优于 LDGSTS gather

旧 `LDGSTS` 路径的症状：

```text
global sectors/request ≈ 13.9
L1 hit                 ≈ 1.13%
DRAM SOL               ≈ 2.2%
主要 stall             long scoreboard
```

它不是 HBM 带宽满，而是每个线程独立做 page-table/index/address 工作，发出很多不连续 128-bit copy。page-sized TMA 把地址生成提升到 tensor descriptor 层，减少 load-warp 指令并让 5-stage pipeline 更容易隐藏 latency。

选择原则：

- 规则 tile、单次至少数 KB：优先 TMA；
- arbitrary gather、每行不同 pointer、强细粒度 predicate：保留 `LDGSTS`；
- 不能为了使用 TMA 而让 masked tail 指向未初始化页；物理 load 必须数值安全；
- `.3D`/`.4D` 是 descriptor rank，不是性能等级；重点看 TMA 条数、有效 bytes、barrier 和地址指令数。

### 5.7 TMA load 的异步完成协议

GMEM→SMEM TMA 使用 transaction mbarrier：

```text
producer:
    wait empty(stage, phase)
    mbarrier.arrive.expect_tx(total_bytes)
    issue UTMALDG(..., mbarrier)

hardware:
    async copy
    complete_tx(copied_bytes)

consumer:
    mbarrier.try_wait(stage, phase)
    read SMEM / issue MMA
    release stage
```

PTX ISA 明确规定，TMA 完成时以实际 copy bytes 对 mbarrier 执行 `complete_tx`。因此 `expect_tx` 必须填写该 generation 的总 transaction bytes，而不是 TMA 指令条数。

来源：

- [PTX ISA — mbarrier.expect_tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx)
- [PTX ISA — TMA complete-tx bytes](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)

small-page 2CTA 每个 CTA 使用本地 TMA barrier。non-leader 完成本地 TMA 后，再用一次 16-byte cluster remote store 通知 leader；leader barrier 的 expected bytes 需要额外包含这 16 bytes。

TMA `cute.copy()` 本身必须保持 warp-uniform，CUTLASS DSL 会自动选择单线程发射，不应再手工包 `elect_one()`；barrier 初始化和 `expect_tx` 才需要 elect-one。

## 6. Tensor Core：QK 与 PV 都使用 `UTCQMMA.2CTA`

### 6.1 QK 数据路径

```text
Q: FP8, SMEM, logical shape M256 × K256
K: FP8, SMEM, logical shape K256 × N128
S: FP32, TMEM, logical shape M256 × N128

instruction class:
tcgen05.mma.cta_group::2.kind::f8f6f4
SASS: UTCQMMA.2CTA
```

`tcgen05.mma` 的 exact E4M3 types、M/N/K shape 和 accumulate 配置编码在 instruction descriptor (`idesc`) 中。SASS 只显示统一的 `UTCQMMA.2CTA` 类别。

### 6.2 Softmax/P 数据路径

QK 完成后，S 留在 TMEM：

```text
S TMEM FP32
    ↓ tcgen05.ld / register fragment
mask + row max + scale + exp2
    ↓ FP32 → FP8 E4M3
P register
    ↓ tcgen05.st
P TMEM FP8
```

softmax 使用 B300 的 native `MUFU.EX2`，最终 reciprocal 使用 `MUFU.RCP`/`rcp_approx`。SM103 tuning 中关闭 software exp2 emulation，避免用额外 ALU 指令替代硬件 MUFU。

### 6.3 PV 数据路径

```text
P: FP8, TMEM, logical shape M256 × N128
V: FP8, SMEM, logical shape N128 × D256
O: FP32, TMEM, logical shape M256 × D256

instruction class:
tcgen05.mma.cta_group::2.kind::f8f6f4
SASS: UTCQMMA.2CTA
```

这正好利用 PTX ISA 支持的 operand placement：A 可以来自 TMEM 或 SMEM，B 来自当前/peer CTA 的 SMEM，D accumulator 位于 TMEM。

来源：[PTX ISA — tcgen05 MMA operand placement](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma)

### 6.4 为什么使用一个 MMA warp

与 `mma.sync`/`wgmma` 的 collective semantics 不同，PTX ISA 规定 `tcgen05.mma` 是 single-thread semantics；一个线程发起即可启动完整 MMA。kernel 仍分配一个 MMA warp，是为了：

- 统一发射 QK/PV；
- 管理 TMEM allocation/free；
- 管理 TMA consumer state；
- 发出 UMMA completion barrier；
- 避免多个 warp 竞争同一异步 MMA pipeline。

`tcgen05.mma` 是异步指令，不能在发出后立即让 softmax 读取 S 或 correction 读取 O；必须用 UMMA→async mbarrier/`UTCBAR.2CTA.MULTICAST` 完成通知。

## 7. 主流水线：TMA、QK、softmax、PV、correction 的 overlap

### 7.1 两个独立的 pipeline 维度

```text
q_stage = 1
    只保留一个 Q tile 和一个 O accumulator

s_pp = 2
    沿 K-block 方向保留两个 S/P TMEM slots

kv_stage = 5
    沿 K/V load 方向保留五个 SMEM/TMA stages
```

`q_stage=1` 不等于没有 double buffering。HD256 的 O accumulator 已占 256 TMEM columns，无法再放第二个 O；但 S/P 仍可以用剩余 256 columns 做两个 128-column ping-pong slots。

#### 三个 stage 参数控制不同资源

```text
                         M / output direction
                         ────────────────────►
q_stage = 1          [ Q tile 0 ] [ O accumulator 0 ]
                         同一个 work 只有一组 Q/O

                         K-block / time direction
                         ─────────────────────────►
s_pp = 2             [ S/P slot 0 ] [ S/P slot 1 ] [ reuse slot 0 ] ...
                         QK/softmax/PV 在两个 slot 间交替

                         memory-prefetch direction
                         ─────────────────────────►
kv_stage = 5         [stage0][stage1][stage2][stage3][stage4]
                         K/V TMA 与 MMA consumer 的环形队列
```

三者不能混为一谈：

- 增加 `q_stage` 需要另一份 sQ、sO 和 O TMEM accumulator，HD256 放不下；
- 增加 `s_pp` 只需要额外 S/P TMEM slot，不复制 O；
- 增加 `kv_stage` 只增加 K/V SMEM 和 TMA barrier，不增加 TMEM O。

#### S/P slot 的状态机

每个 KPP slot 在一轮中的状态：

```text
       QK producer                softmax producer              PV consumer
            │                            │                           │
            ▼                            ▼                           ▼
   ┌──────────────┐  S_full    ┌────────────────┐  P_partial  ┌─────────────┐
   │ QK writes S  │ ─────────► │ read S, write P│ ──────────► │ PV starts   │
   └──────────────┘             └────────────────┘              │ with P/V    │
                                            │       P_last      └──────┬──────┘
                                            └─────────────────────────►│
                                                                      ▼
                                                               slot reusable
```

同一个 TMEM slot 先保存 FP32 S，再在不同 column offset 保存/覆盖为 FP8 P。只有 PV 已消费对应 P generation 后，这个 slot 才能被下一次 QK 复用。

#### 为什么两个 slot 就能 overlap

单 slot 会形成严格串行：

```text
QK(i) → softmax(i) → PV(i) → QK(i+1)
```

双 slot 后：

```text
slot 0: softmax/PV(i)
slot 1: QK(i+1)
```

QK(i+1) 不依赖 P(i)，它只依赖相同 Q 和新 K(i+1)；因此可以在 softmax warps 处理 S(i) 时由 MMA warp发出。真正的限制是同一个 MMA issue stream 中 QK 与 PV 的发射顺序，以及 O accumulator 在 PV 之间的 correction dependency。

#### 简化时序图

```text
time ─────────────────────────────────────────────────────────────────────►

KV TMA    K0  V0  K1  V1  K2  V2  K3  V3 ...       (5-stage ring prefetch)

MMA       QK0       QK1  PV0       QK2  PV1       QK3  PV2 ...
           │          │    ▲         │    ▲
S slot     S0         S1    │        S0    │        S1
           │          │     │        │     │
Softmax    └─SM0/P0───┘     │  SM1/P1┘     │  SM0/P2 ...
                    P0 ready┘       P1 ready┘

Correction init-ready      rescale O0      rescale O1      ...
                           ▲               ▲
PV dependency              └─ next PV waits O_rescaled
```

图中的 `QK1` 与 `SM0/P0` 可以 overlap；`QK2` 与 `SM1/P1` 可以 overlap。PV0 必须等 P0 partial-ready，PV1 还必须等 O0 correction 完成。

### 7.2 Load warp 的 K/V 发射顺序

K/V 共享 5-stage physical SMEM，load warp 按下面顺序推进 pipeline state：

```text
K0 → Q → V0 → K1 → V1 → K2 → V2 → ...
```

关键点：

- Q 只加载一次；
- K 与 V 分别占用连续 pipeline generations；
- MMA warp 在使用 V(i) 时保持该 stage 不释放；
- 同时等待/消费下一 stage 的 K(i+1)，发出 QK(i+1) 后立即释放 K stage；
- 5 stages 给 load warp 足够距离预取后续 K/V，不降低当前一 CTA/SM residency。

stage 4→5 的实测收益为 `0.8%–2.4%`；3 stages 无法隐藏 latency，16K/64K 可退化约 26%/24%；6 stages 没有进一步收益并增加同步/资源压力。

### 7.3 K-direction ping-pong 的 steady state

初始阶段：

```text
1. wait Q ready
2. wait K0 ready
3. issue QK0 → S slot 0
4. signal S0 ready
```

稳定迭代 `i`：

```text
MMA warp                         Softmax warps                    Correction warps
──────────────────────────────   ──────────────────────────────   ──────────────────────
hold V(i) stage
wait K(i+1)
issue QK(i+1) → S slot next  ──► wait S(i)
release K(i+1) stage             TMEM→register load S(i)
                                  causal mask / row max
                                  MUFU.EX2 + row sum
                                  FP32→FP8 P(i)
                                  register→TMEM store P(i)
wait P(i) partial-ready       ◄── signal P(i) at 75%
wait O accumulator rescaled   ◄────────────────────────────────── rescale O(i-1) if needed
issue PV(i) → O TMEM          ──────────────────────────────────► wait O(i) ready
signal O(i) full
release V(i) stage
                                  finish/write last 25% P(i)
```

与此同时 load warp 正在用 TMA 预取更远的 K/V stages；CLC response 也可由 leader CTA 的 load warp 在空隙中预取。

#### 代码层：MMA 是 S producer、P consumer

下面是当前 `mma()` fast path 的简化形式：

```python
# 跨 CLC work 保留，不能每个 work 清零
kpp_iter_global = 0

# 第一个 K block
pipeline_q.consumer_wait_w_index_phase(0, q_phase)
pipeline_kv.consumer_wait(kv_state)                  # wait K0
s_slot = kpp_iter_global % 2
gemm_Si[s_slot](K0)                                  # async QK0
pipeline_s_full_kpp.producer_commit_w_index(s_slot) # QK completion → softmax
pipeline_kv.consumer_release(kv_state)
kv_state.advance()

for i in range(num_k_blocks - 1):
    pipeline_kv.consumer_wait(kv_state)              # hold V(i)
    v_state = kv_state.clone()
    kv_state.advance()

    pipeline_kv.consumer_wait(kv_state)              # wait K(i+1)
    k_state = kv_state.clone()
    next_slot = (kpp_iter_global + i + 1) % 2
    gemm_Si[next_slot](K_next)                        # async QK(i+1)
    pipeline_s_full_kpp.producer_commit_w_index(next_slot)
    pipeline_kv.consumer_release(k_state)
    kv_state.advance()

    # 当前 P(i) 和上一轮 corrected O 都 ready 后才能 PV
    pipeline_p_full_kpp.consumer_wait(p_state)
    pipeline_o_rescaled_kpp.consumer_wait(o_rescaled_state)
    gemm_Pi[cur_slot](P_i, V_i, lastsplit_mbarrier)  # async PV(i)
    pipeline_o_acc.producer_commit_w_index(0)        # PV completion → correction

    pipeline_p_full_kpp.consumer_release(p_state)
    pipeline_o_rescaled_kpp.consumer_release(o_rescaled_state)
    pipeline_kv.consumer_release(v_state)
    p_state.advance()
    o_rescaled_state.advance()
```

这里有两个容易忽略的点：

1. V(i) stage 在 QK(i+1) 期间保持 occupied，因为随后 PV(i) 仍要读 V(i)；
2. K(i+1) 只用于 QK，QK 发出并建立正确 completion dependency 后即可 release，从而让 TMA producer 更早复用该 SMEM stage。

#### 代码层：softmax 是 S consumer、P producer

当前 `softmax_step()` 的简化形式：

```python
# slot/phase 来自 global K iteration
pipeline_s_full_kpp.consumer_wait_w_index_phase(s_slot, s_phase)
pipeline_p_full_kpp.producer_acquire_w_index_phase(s_slot, p_phase)

# TMEM S → registers
cute.copy(tmem_load_atom, S_tmem, S_regs)
apply_causal_mask(S_regs)
row_max, acc_scale = online_softmax_update_max(S_regs)

# 先把 acc_scale 交给 correction
sScale[row] = acc_scale
sm_stats_barrier.arrive_w_index(...)

P_regs = exp2_and_cast_e4m3(S_regs, row_max, max_offset=4)

for fragment in P_fragments:
    cute.copy(tmem_store_atom, P_regs[fragment], P_tmem[fragment])
    if fragment_reaches_75_percent:
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_warp()
        pipeline_p_full_kpp.producer_commit_w_index(s_slot)

# 后 25% 完成
cute.arch.fence_view_async_tmem_store()
cute.arch.sync_warp()
pipeline_p_full_lastsplit_kpp.producer_commit_w_index(s_slot)
```

`producer_acquire` 防止 softmax 覆盖仍被前一轮 PV 使用的 P slot；`S consumer_wait` 防止在 QK 完成前读取 S；TMEM async fence 保证 arrival 之前对应 P stores 对 MMA 可见。

#### 代码层：correction 是 O consumer、O-rescaled producer

```python
# 第一轮没有旧 O 需要 rescale，先发一个 ready token
pipeline_o_rescaled_kpp.producer_acquire_w_index_phase(0, phase)
pipeline_o_rescaled_kpp.producer_commit_w_index(0)

for i in range(num_k_blocks - 1):
    sm_stats_barrier.arrive_and_wait_w_index(...)    # wait acc_scale
    scale = sScale[row]

    pipeline_o_acc.consumer_wait_w_index_phase(0, o_phase)  # wait PV O_full
    if warp_vote_any(scale < 1.0):
        correction_rescale_O_in_tmem(scale)

    pipeline_o_rescaled_kpp.producer_acquire_w_index_phase(0, phase)
    pipeline_o_rescaled_kpp.producer_commit_w_index(0)       # next PV may accumulate
    pipeline_sm_stats.consumer_release_w_index(0)
```

第一轮的 artificial ready token 很重要：PV0 之前不存在旧 O，不需要 correction，但 MMA consumer 仍使用统一的 `O_rescaled` protocol。

#### index 与 phase 如何推进

两-slot ring 的状态可写成：

```text
global_iter  slot = iter % 2  phase = (iter / 2) & 1
     0              0                  0
     1              1                  0
     2              0                  1
     3              1                  1
     4              0                  0
```

slot 决定访问哪个 mbarrier/TMEM region，phase 区分同一 slot 的不同 generation。只有 slot 没有 phase，会把 iteration 2 误认为 iteration 0 已完成；只有 phase 没有 slot，则无法同时保留两个在飞的 S/P。

### 7.4 overlap 的实质

理想 steady state 同时存在五类工作：

```text
TMA:         load K/V(i+2, i+3, ...)
Tensor Core: QK(i+1)
SIMT/MUFU:   softmax + P cast(i)
Tensor Core: PV(i)
SIMT/TMEM:   O correction(i-1)
```

这不是让 QK 与 PV 在同一个 Tensor Core 上完全并发；二者仍由同一 MMA warp 排序发射。收益来自异步 `tcgen05.mma` 与不同执行单元/warp 的重叠：MMA warp 发出操作后，softmax/correction/load warps 可以独立工作，barrier 只在真正的数据依赖处阻塞。

### 7.5 split-P：75% 时提前启动 PV

`split_P_arrive = 96`，即 N128 的 75%。softmax warps 写 P TMEM 时：

```text
write P columns [0, 96)
fence async TMEM store
signal P partial-ready
    ↓
MMA warp 可以开始 PV
    ↓
write P columns [96, 128)
signal P last-split-ready
```

PV 的 partial MMA 路径带 last-split mbarrier，确保需要后 32 columns 时数据已经可见。

实验结果：

- 75%→50% 只有 `-0.3%～+0.1%`，基本中性；
- 25% 触发 SM launch failure；
- 当前保留 75%。

split point 不是纯性能常量，它同时改变 TMEM producer fence、partial/full arrival 次数和 PV consumer 的合法同步节奏。

### 7.6 online-softmax correction 如何与 PV 配合

softmax 每处理一个新 K block，会更新 running row max。若 max 变化，已有 O accumulator 必须乘：

```text
acc_scale = exp(old_max - new_max)
```

流水线为：

```text
softmax warps:
    计算 acc_scale
    写 sScale
    named-barrier 通知 correction

MMA warp:
    PV 完成后 signal O_full

correction warps:
    wait softmax scale
    wait O_full
    TMEM→register load O fragment
    packed FP32 multiply by acc_scale
    register→TMEM store
    fence_view_async_tmem_store
    signal O_rescaled

MMA warp:
    下一次 PV 前 wait O_rescaled
```

`should_rescale` 是 warp-wide ballot：任意一行需要 rescale，整个 warp 都执行。因此尝试只在少数迭代跳过 correction wait 几乎没有收益。

### 7.7 KPP phase 是 persistent-cluster 状态

KPP 的 S/P slot、phase 和 producer/consumer state 都在 work loop 外创建，不能在取得新 CLC work 时清零。完整的 physical cluster、logical work、CLC KPP 和 odd-block dummy handshake 见 10.7～10.9。

## 8. 同步设计：只在真实数据依赖处等待

### 8.1 同步关系总表

| Pipeline / barrier | Producer | Consumer | 保护的数据 | 等待点 |
| --- | --- | --- | --- | --- |
| Q TMA pipeline | load warp / TMA | MMA warp | sQ | 首次 QK 前 |
| K/V 5-stage TMA pipeline | load warp / TMA | MMA warp | sK/sV stage | 对应 QK/PV 前 |
| `S_full_kpp[2]` | MMA warp | softmax warps | S TMEM slot | softmax 读 S 前 |
| `P_full_kpp[2]` | softmax warps | MMA warp | P 前 75% | PV 启动前 |
| `P_lastsplit_kpp[2]` | softmax warps | PV MMA | P 后 25% | PV 使用后半 P 前 |
| `O_full` | MMA warp | correction warps | O TMEM accumulator | correction 读 O 前 |
| `O_rescaled` | correction warps | MMA warp | rescaled O TMEM | 下一次 PV 前 |
| softmax stats named barrier | softmax warps | correction warps | `acc_scale/row_sum/row_max` | correction/最终 normalize 前 |
| epilogue named barrier | 4 correction warps | 同一组 correction warps | sO tile | LDS/STG 前 |
| cluster init barrier | CTA0/CTA1 | 全体 pipeline | mbarrier/TMEM/cluster state | 第一个 work 前 |
| CLC response mbarrier | hardware CLC request | cluster scheduler | 16-byte response | advance work 前 |

### 8.2 哪些操作是异步的

- `UTMALDG.*`：异步 GMEM→SMEM，硬件完成后对 transaction mbarrier 做 `complete_tx(bytes)`；
- `UTCQMMA.2CTA`：异步 Tensor Core MMA；
- `tcgen05.st`：register→TMEM 属于 async proxy，需要 `fence_view_async_tmem_store`；
- CLC cancellation/fetch：异步返回 16-byte response，通过 shared mbarrier 同步；
- cluster remote store：异步写 peer/leader CTA 的 shared/barrier state。

### 8.3 哪些位置必须同步

1. MMA 读取 sQ/sK/sV 前，必须等对应 TMA generation 完成。
2. softmax 读取 S TMEM 前，必须等 QK UMMA 完成。
3. PV 读取 P TMEM 前，必须等 partial P fence/arrival；读后 25% 前还要等 last-split。
4. correction 修改 O 前，必须等本轮 PV 写 O 完成。
5. 下一轮 PV 累加 O 前，必须等 correction 写回 TMEM 完成。
6. epilogue 读 O 前，必须等最后一次 PV 和 correction 完成。
7. sO→GMEM 前，四个 correction warps 必须确认整块 sO 已写完。
8. TMEM free 前，MMA、softmax、correction 全部 warp 必须到达 allocation barrier。

### 8.4 常见错误

- 在 TMA `cute.copy()` 外再套 `elect_one()`：DSL 已隐式单线程发射，可能破坏同步。
- `expect_tx` 填 TMA 数量而非 bytes：barrier 永远不能正确完成。
- 只同步本 CTA，不包含 peer CTA 的 softmax/correction producer 数：2CTA barrier 提前完成。
- TMEM store 后没有 async-proxy fence：MMA 可能读到旧 P/O。
- work 边界重置 phase：CLC persistent 下跨 work 死锁。
- 提前释放 V stage：PV 尚未完成就被后续 K/V TMA 覆盖。

PTX 依据：

- [tcgen05 asynchronous operations](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-async-operations)
- [tcgen05 pipelined instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions)
- [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)

## 9. Epilogue：FP32 TMEM → BF16 GMEM

### 9.1 当前 epilogue 的完整路径

每个 CTA 输出 128 rows：

```text
O payload / CTA     = 128 × 256 × 2B = 64 KB
O payload / cluster = 256 × 256 × 2B = 128 KB
```

最终路径：

```text
O FP32 accumulator in TMEM
    ↓ wait final O_full
tcgen05.ld → register fragment
    ↓ multiply final_scale = rcp(row_sum) × v_descale
packed FP32 multiply
    ↓ FP32 → BF16 conversion
register → sO
    ↓ fence_view_async_shared
4 correction warps named-barrier
    ↓
sO → register via 128-bit LDS
    ↓
register → GMEM via 128-bit STG
```

典型 SASS 为 `LDS.128` + `STG.E.128`。四个 correction warps 同时负责：

- 最终 normalize；
- FP32→BF16 conversion；
- TMEM→SMEM 搬运；
- PackGQA row/head 地址还原；
- varlen row predicate；
- GMEM vector store。

这样可以复用 correction warps 的寄存器和线程，避免独立 epilogue warp。

### 9.2 为什么必须经过 sO

TMEM 适合 Tensor Core accumulator，但普通 global store 不能直接从 TMEM 发出。中间使用 sO 有三个作用：

1. 把 tcgen05 TMEM fragment 重排成连续 128-bit GMEM vector layout；
2. 在 shared-memory store 时完成 FP32→BF16 conversion；
3. 让四个 correction warps 合作生成完整 output tile，再统一进行 coalesced global store。

`fence_view_async_shared` 只保证 async shared writes 的视图顺序；它不替代线程间 barrier。因此后面仍需要 128-thread named barrier，确保所有 warp 的 sO 分片都完成。

### 9.3 当前为什么不用 TMA O-store

规则 dense O 可以使用：

```text
CopyBulkTensorTileS2GOp
    → UTMASTG
    → cp.async.bulk.commit_group
    → cp.async.bulk.wait_group.read
```

但当前输出同时有 PackGQA 与 packed varlen：

- packed M row 需要还原为 `(token, q_head)`；
- sequence base 由 runtime `cu_seqlens_q` 决定；
- 最后一个 tile 有逐 row predicate；
- 一个规则 dense descriptor 不能自然表达所有地址映射。

所以当前选择 vector epilogue，而不是强开 TMA。

### 9.4 TMA store 与 TMA load 的同步不同

TMA load 使用 transaction mbarrier；TMA store 使用 bulk async-group：

```text
issue UTMASTG
cp.async.bulk.commit_group
cp.async.bulk.wait_group.read N
```

`wait_group.read` 用于确认 TMA store 已读完 sO，之后该 shared-memory stage 才能安全复用。它不是 load-side `expect_tx/complete_tx` 协议。

## 10. Causal 调度：LPT + Cluster Launch Control

### 10.1 先定义 work、K block、cluster 和 wave

当前 scheduler 返回的 logical work 坐标是：

```python
m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
```

一个 work 表示：

```text
一个 packed-M256 Q/O tile
× 一个 KV head
× 一条 sequence
× 该 tile 对应的 causal K-block range
```

它不是一个 K block，也不是一个 CTA：

- **K block** 是 work 内层循环的一次 KV128 迭代；一个 work 通常包含多个 K blocks。
- **physical cluster** 是执行 work 的硬件执行组，固定为两个 CTA：CTA0 负责 packed rows `[0,128)`，CTA1 负责 `[128,256)`。
- **work** 是逻辑任务坐标；同一个 physical cluster 在 persistent loop 中可以先后处理多个 work。
- **wave** 是某一时刻可同时 resident 的 physical clusters 集合。B300 有 148 SM，而当前资源占用使每个 CTA 独占一个 SM，所以一 wave 最多为 74 个 2CTA clusters。

关系可以画成：

```text
logical work W(m, hkv, batch, split)
│
├─ Q/O ownership: packed M256 × D256
│    ├─ CTA0: rows   0..127
│    └─ CTA1: rows 128..255
│
└─ K loop: K0, K1, ... K(n-1), each KV128

one physical cluster = CTA0 + CTA1 = one work at a time
one physical cluster lifetime = zero or more consecutive works
one wave on B300 = at most 74 such physical clusters resident together
```

PackGQA 改变 logical work 的数量和每个 work 中 row→Q-head 的映射，但不改变 `cluster=(2,1,1)`，也不把一个 work 拆成两个独立 CTA work。

### 10.2 causal work 的计算量不相等

右下角对齐的 causal attention 中，越靠后的 Q tile 能看到越多 K blocks：

```text
early Q tile  → 少量 K blocks
late Q tile   → 大量 K blocks
```

varlen batch 又叠加不同 sequence lengths。若按自然顺序执行，轻 tile 先完成，重 tile 留在 grid 尾部，会产生少数长任务拖尾。

### 10.3 LPT 与 CLC 分工

```text
LPT (Longest Processing Time first)
    根据 causal K-block count，把重 tile 映射到较早 work ID

CLC (Cluster Launch Control)
    resident cluster 完成当前 work 后，异步取消一个尚未启动的 cluster，
    取得其 block/cluster ID，并在当前 resident cluster 上继续执行该 work
```

LPT 改善初始排序，CLC 做动态 work stealing。二者缺一不可：LPT 不会处理运行时不均衡，CLC 也不知道哪个 work 更重。

NVIDIA CUDA Programming Guide 说明，CLC 是 Blackwell compute capability 10.0 引入的功能；取消请求是异步操作，通过 shared-memory barrier 返回，成功后 resident block/cluster 使用被取消任务的 index 执行 work。

来源：[CUDA Programming Guide — Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)

### 10.4 当前如何隐藏 CLC 开销

CLC response 为 16 bytes，使用一个 stage 的 async pipeline：

```text
leader CTA load warp:
    issue next-work cancellation/fetch
    mbarrier.expect_tx(16)

current work:
    TMA / QK / softmax / PV / correction continues

work boundary:
    wait response only when advance_to_next_work()
```

load warp 在发完当前 tile 的 TMA 后预取 CLC response，把 request latency 与当前 work 的 compute overlap。非 leader CTA 不独立请求；整个 physical cluster 必须消费同一个 work ID。

### 10.5 CLC 的正确调度单位是 2CTA cluster

必须同时满足：

```text
launch cluster shape         = (2,1,1)
CLC problem cluster shape    = (2,1,1)
scheduler coordinate divisor = 2 CTA
```

若 CLC descriptor 错写成 cluster M=1，CTA0/CTA1 可能取得不同 work，随后却共同发 `UTCQMMA.2CTA`，结果是 partial final cluster 错乱或 hang。

causal K bound 也必须使用 logical cluster M=256，而不是 per-CTA M=128；否则 CTA1 的后 128 rows 会缺少本应可见的晚 K blocks。

### 10.6 CLC 与 PackGQA/wave 的关系

- PackGQA 先减少 work clusters 和 padding，可能直接跨过 74-cluster wave cliff；
- LPT 把 remaining heavy works 提前；
- CLC 让完成较早的 resident clusters steal 尚未启动的 work；
- KPP global phase 保证同一个 resident cluster 连续执行多个 work 时 pipeline 仍正确。

三者分别处理 work 数、work 顺序和动态负载均衡，不能互相替代。

### 10.7 CLC KPP 到底是什么

这里的两个缩写处理不同的循环层级：

```text
CLC: Cluster Launch Control
     在 work 边界复用 resident physical cluster，动态取得下一个 work ID

KPP: K-direction ping-pong
     在单个 work 内沿 K-block 方向轮换两个 S/P TMEM slots，
     overlap QK(i+1) 与 softmax/PV(i)

CLC KPP:
     physical cluster 连续执行多个 CLC works 时，仍保留 KPP fast path，
     并让 KPP barrier generation 跨 work 正确延续
```

CLC 不会在同一个 cluster 内并发执行两个 works。它只是在 work A 结束后，让同一组 CTA 和 warps 继续执行 work B。KPP 才是 work 内部的 K-direction pipeline。

例如 grid 有 80 个 logical works，而第一 wave 最多 resident 74 个 clusters：

```text
grid work IDs:  W0  W1  W2  ...  W73  W74  W75 ... W79

physical C0:    W0 ───────────────► cancel/fetch W74 ─────► ...
physical C1:    W1 ───────────► cancel/fetch W75 ─────────► ...
physical C2:    W2 ────────────────────────────────────────► exit
                 ^                         ^
                 |                         |
          physical cluster 不变       logical work ID 改变
```

图中 W74/W75 只是示意；硬件返回的是成功取消、尚未启动的 cluster index。关键是 SMEM、TMEM allocation、mbarriers 和各 warp 的 pipeline state 都属于 physical cluster 生命周期，不会因为 work ID 改变而重新构造。

代码结构等价于：

```python
# physical-cluster lifetime: 只执行一次
allocate_smem_and_tmem()
init_mbarriers()
init_q_kv_s_p_o_pipeline_states()
kpp_iter_global = 0

work = tile_scheduler.initial_work_tile_info()
while work.is_valid_tile:
    run_q_kv_tma(work)
    run_kpp_qk_softmax_pv_correction(work)
    # leader CTA 的 load warp 已经异步预取下一个 CLC response
    work = tile_scheduler.advance_to_next_work()

drain_pipelines_and_free_tmem()
```

因此“persistent CLC 下维护跨 work-tile 的 pipeline epoch”不是保存 attention 数值；Q、O、row max、row sum 都会为新 work 重新开始。需要保存的是同一批物理 mbarriers 下一次应使用的 slot/phase，以及 Q、KV、S/P/O 各 producer/consumer ring 已推进到哪一代。

### 10.8 为什么 odd K-block work 会破坏 KPP epoch

两个 KPP slots 的映射是：

```text
global K iter    slot = iter % 2    phase = (iter // 2) & 1
      0                 0                       0
      1                 1                       0
      2                 0                       1
      3                 1                       1
      4                 0                       0
```

`slot` 选择两组 S/P TMEM region 和 mbarrier；`phase` 区分同一 mbarrier 的相邻 generation。假设 work A 有三个 K blocks：

```text
physical cluster lifetime ─────────────────────────────────────────►

work A
  K0                K1                K2
  slot0/phase0       slot1/phase0      slot0/phase1
                                          |
                                          | slot1/phase1 尚未闭合
                                          v
                                     dummy slot1/phase1
                                          |
work B                                    v
  K0                K1                ...
  slot0/phase0       slot1/phase0
```

若 work B 把局部计数强制重置为 `slot0/phase0`，MMA 和 softmax 可能把 work A 的旧 arrival 当成 work B 的新 completion，或者分别等待不同 generation。表现通常是 hang、CUDA 912，或更危险的旧 S/P 数据被误消费。

仅仅让 `global_iter += 3` 也不够理想：下一 work 会从 `slot1/phase1` 开始，所有独立 warp role 都必须无误地继承这个半周期。当前最终方案在 odd work 后补齐另一个 slot，把 work 边界恢复到完整的双-slot周期。

### 10.9 代码层如何闭合 dummy epoch

MMA warp 与 softmax warpgroup 各自维护同构的 `kpp_iter_global`，变量在各自的 persistent work loop 外。work 的真实 K blocks 完成后：

```python
# MMA side: S producer, P consumer
kpp_iter_global += block_iter_count

if block_iter_count % 2 != 0:
    dummy_slot = kpp_iter_global % 2

    # 不发 QK；只生成该 dummy slot 的 S-ready token
    pipeline_s_full_kpp.producer_commit_w_index(dummy_slot)

    # 等 softmax side 回送 dummy P-ready，再完成 consumer release
    pipeline_p_full_kpp.consumer_wait(p_state)
    pipeline_p_full_kpp.consumer_release(p_state)
    p_state.advance()

    kpp_iter_global += 1
```

softmax warpgroup 执行互补的一半：

```python
# softmax side: S consumer, P producer
if block_iter_count % 2 != 0:
    dummy_slot  = k_iter % 2
    dummy_phase = (k_iter // 2) & 1
    p_phase     = dummy_phase ^ 1

    pipeline_s_full_kpp.consumer_wait_w_index_phase(
        dummy_slot, dummy_phase
    )
    pipeline_p_full_kpp.producer_acquire_w_index_phase(
        dummy_slot, p_phase
    )

    cute.arch.sync_warp()
    with cute.arch.elect_one():
        pipeline_p_full_kpp.producer_commit_w_index(dummy_slot)
        pipeline_p_full_lastsplit_kpp.producer_commit_w_index(dummy_slot)

    k_iter += 1

kpp_iter_global = k_iter
```

这个 dummy iteration：

- 不发 QK UMMA；
- 不读 S、不算 softmax、不写真实 P；
- 不发 PV UMMA，也不修改 O；
- 只让 S producer→consumer 与 P producer→consumer 的 barrier generation 成对闭合。

Q TMA phase、5-stage KV producer/consumer state、`O_full/O_rescaled` phase 也都在各 warp role 的 work loop 外持续推进；它们不能因为 `work_tile.tile_idx` 更新就归零。不同 pipeline 可以有不同 ring size 和 phase，但必须遵守同一原则：**复用 physical storage，就必须继承该 storage 的下一合法 generation。**

优化和验证顺序应固定为：

1. CLC descriptor、launch cluster shape 和 scheduler divisor 都使用 `(2,1,1)`。
2. CLC response 只由 leader CTA 请求，并在所有 warp roles 的 work 边界一致消费。
3. S、P、O 使用独立 barrier contract；不要用一个 phase 猜另一个 pipeline 的 generation。
4. slot/phase 来自跨 work 的 global counter。
5. odd K-block work 执行 dummy S/P closure。
6. load warp 提前发 16-byte CLC async request，让返回延迟与当前 work compute overlap。
7. 同时覆盖 PackGQA、odd/even K-block、连续多次 steal 和 partial-final-cluster 测试。

## 11. Paged-KV 寻址：减少 payload 之外的 load 指令

### 11.1 page、tile、physical page 是三个单位

```text
n_block        logical KV128 tile 编号
logical page   page_table 列编号
physical page  page_table 中保存的实际页号
```

任何优化都必须先把三者分开。page256 是“一页多 tile”，page16/32/64 是“一 tile 多页”；两类映射不能共用同一个简化公式。

### 11.2 K/V page ID register cache

page64 的一个 KV128 tile 有两个 physical page IDs。优化前：

```text
K: CTA0 查 page0，CTA1 查 page1               = 2 loads / cluster
V: CTA0 查 page0/page1，CTA1 查 page0/page1   = 4 loads / cluster
合计                                           = 6 loads / cluster / KV tile
```

K 与 V 使用相同 physical page IDs。当前先在 load-warp registers 中构造 compile-time tuple，再复用于 K/V TMA：

```text
page_ids = load_page_table_once(n_block)
    ├─ issue K TMA(page_ids)
    └─ issue V TMA(page_ids)
```

page64 从 6 次降到 4 次 page-table loads；实测 kernel global-load instructions 下降约 32.8%。

关键点：

- cache page IDs，而不是完整 K/V address；K/V source-tile 公式不同；
- tuple 长度由 page size 在编译期决定：page16/32/64 分别为 8/4/2；
- register-local reuse 不需要新增 cluster SMEM/barrier；
- 指令数下降必须同时带来 wall-time 改善，否则不值得增加 live registers。

### 11.3 masked tail 的物理 load 仍必须安全

TMA 或 CTA-local async copy 可能对最终 partial tile 的 masked columns 继续发物理 load。若 invalid page entry 指向 physical page0，而 page0 未初始化或含 FP8 NaN：

```text
masked tail load NaN
    → FP8 MMA accumulator
    → score mask 未必能消除 NaN 污染
```

正确做法是让 invalid tail 指向该 sequence 最后一个已初始化、有限的 physical page，然后再由 causal/seqlen mask 丢弃越界 columns。

截至 `0381108a`，safe-tail 修复仍只在 `0d56d09f` / `8bbeb01d` side branches；主线部分路径仍返回 page0。这是当前 kernel 的 P0 correctness risk。

必须用下面的 adversarial case 验证：

- physical page0 主动填 FP8 NaN；
- 有效 sequence 不引用 page0；
- shuffled page table；
- `seqlen_k = page_size±1, tile_n±1`；
- page64 CTA1-only partial half；
- page16/32/64/128/256。

## 12. FP8 数值路径也是 pipeline 设计

### 12.1 为什么 P 需要放大后再 cast FP8

softmax probability P 直接 cast E4M3 容易让小概率下溢。kernel 在写 P TMEM 前放大：

```text
P_fp8 = cast_e4m3(P_fp32 × 2^max_offset)
```

PV/O 最终再补偿该 scale。当前：

```text
max_offset         = 4
rescale_threshold  = 4
```

online softmax 的 running max 最多允许落后 4 个 log2 units，而 P 又放大 4 bits。近似保持：

```text
max_offset + rescale_threshold <= 8
```

避免放大后的 P 超过 E4M3 最大有限范围，同时减少 underflow。

### 12.2 P 与 O 必须使用同一个 offset

P cast 前使用 `2^4=16`，最终 normalization 必须用相同 `max_offset` 修正 row sum/LSE/O scale。只改 P 路径或只改 epilogue 会产生整体 scale error，而不一定产生 NaN。

因此正确性不能只检查 finite，还要看 cosine、relative L1/L2 和输出幅值。

### 12.3 数值 knob 会改变同步频率

`rescale_threshold` 决定 running max 变化多大时触发 correction。它同时影响：

- correction warps 执行多少次 TMEM load/mul/store；
- MMA warp 等待 `O_rescaled` 的频率；
- `O_full`/softmax-stats barrier 压力；
- P saturation 与 underflow。

曾尝试 `(max_offset, threshold)=(2,6)` 并 conditional wait，希望减少 correction；实测改善只有 `0.01%–0.13%`。原因是 `should_rescale` 为 warp-wide ballot，32 行中任意一行命中就让整个 warp 执行，实际几乎每轮为真。

结论：调数值阈值前必须统计 warp-wide predicate 命中率，不能仅根据标量概率判断同步会减少。

### 12.4 B300 native EX2

SM103 路径把 `ex2_emu_freq=0`，softmax exponent 使用硬件 `MUFU.EX2`，不再像部分早期 Blackwell 配置那样混入 software exp2 emulation。最终 reciprocal 使用 `MUFU.RCP`/approx reciprocal。

SASS 验证应看到：

```text
MUFU.EX2
MUFU.RCP
UTCQMMA.2CTA
```

而不是用大量额外 ALU 序列模拟 exponent。

PTX 指令语义参考：[PTX ISA — ex2](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2)

## 13. 有效优化与性能证据

不同报告的 shape、page size 和 baseline 不同，下面只记录能由 counter/SASS 解释的 kernel feature。

| Kernel feature | 代表收益 | 机制证据 |
| --- | --- | --- |
| CLC + LPT + warp-role registers | 8K `514.85→463.62 us` | 降低 causal tail；最终 role allocation 的 local/shared spill 为 0 |
| page128/256 TMA | 32K prefix `3200→2980 us` | LSU traffic `168.9M→13.0M`；tensor pipe `67.7%→76.3%` |
| K/V stage 4→5 | `0.8%–2.4%` | tensor active/eligible warps 上升，barrier stall 下降，occupancy 不变 |
| page16/32/64 TMA | FP8 60/60 快于当时 nightly TRTLLM-gen，`1.090x–1.338x` | `LDGSTS→0`，出现 page-sized `UTMALDG.4D` |
| PackGQA + KPP | 真实 shape 避免 wave cliff | work 80→68 clusters；仍进入 KPP fast path |
| K/V page-ID cache | global-load instructions 约 `-32.8%` | K/V 复用 register page-ID tuple |

### 13.1 最终 small-page SASS 检查点

代表性 FP8 page16、HQ32/HKV2、K4096 报告：

```text
block / cluster       320 threads / cluster X=2
dynamic SMEM          182.27 KB / CTA
registers             160 / thread（NCU allocation）
UTCQMMA.2CTA           40 static instruction sites
UTMALDG.3D.2CTA        2  static sites（Q）
UTMALDG.4D             32 static sites（paged K/V）
LDGSTS                  0
local/shared spill      0 / 0
```

静态 instruction-site 数不等于每个 work 的动态执行次数；它用于确认 specialization 生成了预期指令类别。

### 13.2 当前残余瓶颈

page64 的 short-Q / very-long-K / sub-wave shapes 中：

- K/V DRAM payload 与 TRT baseline 基本相同；
- Atrex barrier stall 更高；
- SourceCounters 把主要 stall 定位到 intermediate O-full wait；
- long-scoreboard 不是最主要差距。

这说明下一步不应继续盲目减少 K/V payload，而应优先研究 correction/O-rescaled 同步、short-Q 下的 cluster occupancy，以及是否存在不破坏统一 2CTA topology 的调度办法。

## 14. 修改 kernel 时的验证矩阵

### 14.1 2CTA / causal / PackGQA

- Q length：`128±1`、`256±1`、1084、1408；
- HQ16/HKV1 与 HQ32/HKV2；
- PackGQA 前后 work clusters、physical CTAs、waves；
- CTA0/CTA1 rows 分别比较；
- partial final cluster；
- Q<K 的 prefix-cache causal 对齐。

### 14.2 KPP / persistent phase

- odd/even K-block count；
- 同一 resident cluster 连续取得多个 CLC works；
- causal-mask loop、no-mask loop、最后一次 iteration；
- P partial/full arrival；
- O correction 发生/不发生；
- work A 为 odd blocks，work B 紧随其后。

### 14.3 TMA / Paged-KV

- page16/32/64/128/256；
- identity 与 shuffled page table；
- page256 的第二个 tile；
- page16 一 tile 八页；
- physical page0=NaN；
- partial page/tile；
- NCU/SASS 确认 `UTMALDG`、`LDGSTS`、TMA bytes 与 barrier stall。

### 14.4 Epilogue

- PackGQA row/head 映射；
- ragged sequence start 非 8/16/128 对齐；
- final partial M tile；
- `LDS.128/STG.E.128` coalescing；
- TMEM/SMEM async fence 与 named barrier；
- 若实验 TMA O，必须验证 `UTMASTG`、ragged descriptor base 和 sO reuse wait。

### 14.5 FP8 数值

```text
finite
cosine > 0.99
relative L1 < 0.08
relative L2 <= 0.05
```

额外覆盖：P underflow、E4M3 saturation、rescale frequent/rare、contiguous vs paged、不同模型幅值。

## 15. NCU/SASS 诊断顺序

### 15.1 先确认 kernel identity

```text
kernel             FlashAttentionForwardHd256_2CTA_Sm103
SM                 103
block              320
cluster            (2,1,1)
dynamic SMEM       ≈182 KB
input/output       FP8 / BF16
```

### 15.2 再看 work geometry

```text
work clusters
physical CTAs = clusters × 2
wave occupancy = clusters / 74
PackGQA padding
partial final wave
```

如果 work 从 75 降到 74，wave cliff 通常比少几条指令更重要。

### 15.3 再判断瓶颈

| 症状 | 优先检查 |
| --- | --- |
| long scoreboard 高、DRAM SOL 低 | TMA page size、page-table loads、KV stages |
| barrier 高 | O_full/O_rescaled、KPP phase、2CTA arrival count |
| tensor active 低 | QK/PV bubble、softmax latency、work/wave 不足 |
| local load/store 非零 | warp-role register allocation |
| payload 相同但 global loads 多 | page-ID/address bookkeeping |
| DRAM SOL 高 | K/V payload、PackGQA/L2 reuse |

### 15.4 用指令证明代码路径

```text
UTMACCTL.PF             descriptor prefetch
UTMALDG.3D.2CTA        Q TMA
UTMALDG.4D             small-page K/V TMA
UTCQMMA.2CTA            QK/PV Tensor Core
UTCBAR.2CTA.MULTICAST   2CTA UMMA completion
SYNCS.*                 mbarrier arrive/trywait
USETMAXREG.*.CTAPOOL    `setmaxnreg` 对应的 warp-role register redistribution
MUFU.EX2 / MUFU.RCP     softmax math
LDS.128 / STG.E.128     current vector epilogue
LDL / STL               must not be generated by register spilling
```

## 16. 官方资料与代码位置

### 16.1 NVIDIA 官方资料

- [Blackwell Ultra Datasheet](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet)：B300 FP8 compute、HBM capacity/bandwidth。
- [HGX B300 specifications](https://www.nvidia.com/en-us/data-center/hgx.md)：HGX B300 system compute 与 dense/sparse 说明。
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)：register、SMEM、cluster/occupancy。
- [PTX ISA — Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)：TMEM 512 columns × 128 lanes。
- [PTX ISA — tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)：2CTA Tensor Core、single-thread semantics、operand placement。
- [PTX ISA — cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)：TMA、CTA group、multicast、completion mechanism。
- [PTX ISA — mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)：transaction bytes 与 phase wait。
- [PTX ISA — setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)：CTA register pool、warpgroup 一致性、`.inc/.dec` 约束。
- [CUDA Programming Guide — CLC](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)：Blackwell work stealing。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：register allocation 与 local/shared spilling 动态 metrics。

### 16.2 Atrex kernel 与性能证据

```text
src/cutedsl/flash_fwd_hd256_2cta_sm103.py
op_test/test_flash_attn_hd256_cute.py

output/fa4_hd256_2cta_prefix_cache_30pct/final_results.md
output/fa4_prefill_ncu_20260717/summary.md
output/fa4_hd256_small_pages_20260720/summary.md
output/test_csv_fa4_20260721/report.md
kernel_opt_fa4_hd256_fp8_small_pages/profiles/
```

## 17. 核心优化结论

1. HD256 的核心不是增加 `q_stage`，而是利用 TMEM 的 `S/P×2 + O×1` 做 K-direction ping-pong。
2. `tcgen05.mma.cta_group::2` 让 CTA pair 直接完成完整 D256 QK/PV，不需要软件跨 CTA reduction。
3. 5-stage K/V TMA、2-slot S/P KPP 和 single O accumulator 是三个独立 pipeline 维度。
4. PackGQA 不改变 physical cluster；它减少 per-head tail rounding，并可让 work count 跨过 74-cluster wave cliff。
5. TMA 的价值主要是减少 per-thread 地址/指令并建立深异步 pipeline，不是“任何 load 换 TMA 都更快”。
6. correction 是 online softmax 的真实数据依赖；`O_full → rescale → O_rescaled` 不能省略，只能尽量与 QK/softmax/TMA overlap。
7. persistent CLC 让 barrier generation 跨 work 存活，KPP slot/phase 必须作为全局状态机维护。
8. 当前 epilogue 为 TMEM→register→SMEM→register→GMEM；PackGQA/ragged mapping 是使用 vector store 而非 TMA store 的主要原因。
9. Paged-KV 优化不仅是 payload，还包括 page-table loads、page/tile 换算和 masked tail 的物理安全。
10. warp-role register 上限为 softmax/correction/other=`160/128/96`；O correction/epilogue 按 16 columns 流式处理，最终 spill metrics 必须为 0。
11. 判断优化是否生效，应同时看 work geometry、SASS instruction class、TMA bytes、tensor active、barrier、spill 和 wall time。
