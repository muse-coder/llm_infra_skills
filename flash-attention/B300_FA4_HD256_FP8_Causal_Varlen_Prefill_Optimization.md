# B300 FA4 HD256 FP8 Causal Varlen Prefill 优化复盘

> 最后整理：2026-07-25
> 代码基线：Atrex `0381108a`
> 目标场景：NVIDIA B300 / SM103、FP8 E4M3 Q/K/V、BF16 输出、`head_dim=256`、causal varlen prefill、GQA16、contiguous/paged KV

## 1. 文档目的与阅读方式

本文不是按 commit 顺序记录实验，而是按“大的优化方向 → 可复用的小 trick”整理。每个 trick 都回答五个问题：

1. 它解决什么瓶颈；
2. 具体如何实现；
3. 最关键的工程技巧或 invariant 是什么；
4. 有什么收益和证据；
5. 它目前属于产品主路径、历史经验，还是已回退实验。

正文分成五个大的优化方向：

```text
一、缩小问题与做真 varlen
    ├─ 专用 kernel / 依赖收敛
    ├─ native varlen，删除 dense 绕路
    └─ ragged O TMA-store（原型线经验）

二、调度与工作几何
    ├─ CLC + LPT
    ├─ 2CTA logical cluster geometry
    ├─ PackGQA 与 wave cliff
    └─ 固定 2CTA + CLC 产品 topology

三、计算流水线与片上资源
    ├─ K-direction ping-pong
    ├─ CLC 跨 work-tile phase
    ├─ KV pipeline stage
    ├─ sQ/sO 生命周期复用
    ├─ warp-role 寄存器重分配
    └─ split-P arrival

四、Paged-KV 数据搬运与寻址
    ├─ TMA / multicast / LDGSTS / vector copy 的选择
    ├─ 一页多 tile 的 TMA
    ├─ 一 tile 多页的 small-page TMA
    ├─ K/V page ID 缓存
    └─ masked tail 的安全物理页

五、FP8 数值与服务热路径
    ├─ P scaling / correction 联合设计
    ├─ prepared launcher
    ├─ CUDA Graph-safe runtime metadata
    └─ 编译、部署和性能测量收敛
```

提交时间线只放在附录中，用于查找代码，不作为正文结构。

## 2. 当前产品路径：先明确“现在运行的是什么”

截至 Atrex `0381108a`，目标路径为：

| 维度 | 当前配置 |
| --- | --- |
| GPU | B300 / SM103 |
| 输入 | 主要验证为 FP8 E4M3 Q/K/V；接口也已支持部分 mixed-dtype vLLM 路径 |
| 输出 | BF16 |
| Attention | causal varlen forward |
| Head | `head_dim=head_dim_v=256` |
| GQA | ratio=16；典型为 HQ16/HKV1、HQ32/HKV2 |
| CTA | physical cluster `(2, 1, 1)` |
| Scheduler | CLC + varlen mapping + causal LPT |
| Pipeline | `q_stage=1`、K ping-pong、KV TMA stage=5 |
| Paged KV | page16/32/64/128/256 |
| GQA organization | 默认 PackGQA |
| Split KV | `num_splits=1` |
| Host path | `prepare()` 固化静态属性，`run()` 只传动态 tensor |

当前路径有几个容易误解的事实：

- 2CTA 不是两个独立的 HD128 attention。两个 CTA 共同执行完整 HD256 的 `UTCQMMA.2CTA`；CTA pair 合作完成 QK/PV，输出行仍由各 CTA 本地负责。
- CLC 的 worker 是整个 physical cluster，不是单个 CTA。两个 CTA 必须消费同一个 dynamic work。
- `q_stage=1` 不等于没有 ping-pong。Q/O 只保留一个 stage，但 S/P 可以沿 K-block 方向使用两个 slot。
- 当前 varlen 产品路径并没有使用 ragged TMA O-store。ragged O TMA 是原型分支上验证有效的技术；当前 2CTA + PackGQA 路径仍由 correction/epilogue warp 完成 O store。
- page-table 的单位是 page，kernel 调度的单位是 logical KV tile；二者不能混用。

### 2.1 三层状态必须分开

这条 kernel 最容易出错的地方，不是某条 MMA 指令，而是混淆不同层次的状态：

```text
Host 静态状态
    dtype / head / page_size / PackGQA / kernel topology
    → 必须进入 prepare 或 compile key

Scheduler work 状态
    batch / head / logical Q cluster tile / causal K range
    → CLC 以 cluster 为单位动态发放

Pipeline epoch 状态
    KPP slot / barrier phase / producer-consumer generation
    → persistent CLC 下可能跨 work-tile 存活
```

优化时如果把本层状态错误地当成局部循环变量，通常会表现为：partial cluster 算错、odd K-block 后 hang、CUDA 912、随机精度错误或 JIT cache 复用错误 CUBIN。

### 2.2 Paged-KV 的三个单位必须分开

```text
n_block        kernel 的 logical KV tile 编号
logical page   page_table 的列编号
physical page  page_table 中记录的实际页号
```

FP8 路径 `tile_n=128`：

| page size | tile/page 关系 |
| ---: | --- |
| 16 | 一个 tile 聚合 8 页 |
| 32 | 一个 tile 聚合 4 页 |
| 64 | 一个 tile 聚合 2 页 |
| 128 | 一页正好一个 tile |
| 256 | 一页包含两个 tile |

后续所有 Paged-KV 优化都建立在这个分层上。

## 3. 优化总览

| 大优化方向 | 小 trick | 主要收益 | 当前状态 |
| --- | --- | --- | --- |
| 做真 varlen | native varlen，删除 route-to-dense | 统一 ABI 和维护路径 | 当前原则 |
| 做真 varlen | ragged O TMA-store | multi-sequence 代表 case `-12%` | 原型线有效，当前产品未使用 |
| 做真 varlen | 最小化 vendored kernel | 可安装、可复现、减少无关 specialization | 当前产品基础 |
| 调度与几何 | CLC + LPT | 降低 causal/varlen long tail | 当前主路径 |
| 调度与几何 | 修正 2CTA logical tile | 修复 CTA1 causal K 范围 | 当前正确性基础 |
| 调度与几何 | physical cluster-aware CLC | 修复 partial cluster / prefix-cache | 当前正确性基础 |
| 调度与几何 | PackGQA | 降低 padding、KV 重读和 wave cliff | 当前主路径 |
| 流水线 | K ping-pong | overlap `QK(i+1)` 与 `softmax/PV(i)` | 当前主路径 |
| 流水线 | 跨 work phase + dummy handshake | CLC 下恢复 KPP 且保持正确 | 当前主路径 |
| 流水线 | KV stage 4→5 | 代表 case `0.8%–2.4%` | 当前主路径 |
| 流水线 | sQ/sO alias | 1CTA 代表 case `1%–5.2%` | 历史有效，当前 2CTA 禁用 |
| Paged-KV | 按规则性选择 TMA/LDGSTS/vector copy | 降低地址指令并控制同步成本 | 当前设计方法 |
| Paged-KV | page128/256 TMA | LSU 指令约降 13 倍 | 当前主路径 |
| Paged-KV | page16/32/64 TMA | FP8 60/60 快于当时 TRT baseline | 当前主路径 |
| Paged-KV | page ID register cache | global loads 约降 32.8% | 当前主路径 |
| Paged-KV | safe tail page | 避免 masked load 读取 FP8 NaN | side branch，尚未进入 HEAD |
| FP8 数值 | `max_offset=4, threshold=4` | 平衡 underflow、saturation 和 correction | 当前配置 |
| Host | prepared launcher + graph-safe metadata | dispatch/validation 移出热路径 | 当前主路径 |

## 4. 大优化点一：缩小问题，并把 varlen 做成真正的主路径

这一组优化的核心不是某个微小 knob，而是先让产品语义、kernel specialization 和部署边界一致。否则局部 benchmark 再快，也会被双路径维护、错误 import 或 host sync 抵消。

### 4.1 小 trick：用专用 kernel 缩小支持面

#### 解决的问题

早期实现从 generic FA4 interface 继承了大量不相关组合：SM80/90/100/120、forward/backward、不同 head dim、MLA、SplitKV、1CTA/2CTA 等。对目标 workload 而言，这些分支会带来：

- 更大的依赖和源码面；
- 更多 JIT specialization 与 compile cache 维度；
- 更难证明测试命中了目标 kernel；
- 修改一个公共 helper 时引入无关回归。

#### 怎么做

提交 `80774d20` 建立最小化、自包含的 HD256 forward 路径：

- Atrex 只 vendor 上游当时没有的 HD256 专用 kernel；
- 通用 softmax、mask、PagedKV、scheduler helper 从已安装的 FA4 wheel 导入；
- interface 删除 backward、autograd、SM80/90/120、MLA 等无关 dispatch；
- 后续再收敛为 SM103、HD256、causal varlen、GQA16、2CTA、CLC-only。

#### 关键技巧

- specialization 的价值不仅是少几个 `if`，更重要的是可以围绕真实 workload 固化 invariant，并在构造阶段直接 assert。
- 不支持的输入应尽早失败，不能静默 fallback 到另一个未调优 kernel。
- kernel identity、threads/block、cluster shape 必须成为性能报告的一部分；只看 API 名无法证明走了专用路径。

#### 当前状态

这是当前产品路径的基础。当前构造函数明确约束 SM103、2CTA、varlen Q 和 CLC。

### 4.2 小 trick：先用 route-to-dense 找 ceiling，再删除它

#### 解决的问题

原型早期 single-sequence varlen 明显慢于 dense，主要原因是：

- varlen O 使用手工 register-to-global epilogue；
- varlen scheduler 有额外映射和寄存器压力。

提交 `dbb8fe45` 曾利用 `cu_seqlens_q.numel()==2` 判断 batch=1，并临时转到 dense kernel。它不读取 device sequence length，因此适合快速确认 dense 性能上限。

#### 怎么做

这个 trick 的正确用法是两阶段：

```text
阶段 1：route-to-dense
    → 估计 epilogue/scheduler 可达到的性能 ceiling

阶段 2：修复 native varlen
    → 获得相同能力后删除 dense 路由
```

提交 `6911ed40` 最终删除 dense entry 和 route-to-dense，让 ABI 与执行语义统一为 native varlen。

#### 关键技巧

- fast path 可以用于定位上限，但不能代替修主路径。
- 不要为了 dispatch 把 `cu_seqlens` 拷回 host；同步成本通常比 kernel 差异更大，并破坏 CUDA Graph。
- batch=1 仍然是 varlen ABI，未来可能带 `seqused_q/k`、paged KV 或其他 runtime metadata。

#### 当前状态

route-to-dense 已删除。当前产品坚持 varlen-only，并把动态长度留在 device side。

### 4.3 小 trick：ragged O TMA-store

#### 解决的问题

packed varlen 输出为：

```text
O: [total_q, num_heads, head_dim]

seq0 → rows [0, q0)
seq1 → rows [q0, q0+q1)
seq2 → rows [q0+q1, q0+q1+q2)
```

每条 sequence 的起点由 `cu_seqlens_q` 决定，可能不按 8/16/128 对齐。普通 dense TMA descriptor 无法直接表示任意 runtime ragged base，早期只能使用手工 `STG` epilogue，产生更多指令、寄存器压力和 spill。

#### 怎么做

原型提交 `487d3f3e` 从 SM90 路径移植 ragged tensor 机制：

```text
create_ragged_tensor_for_tma(ragged_dim=0, ptr_shift=True)
    ↓
按 sequence 在 runtime 平移 descriptor base pointer
    ↓
sequence 内部仍然是规则 tile，可以使用 TMA store
```

它不是让 TMA 支持不规则 stride，而是把不规则性转换成“每条 sequence 一个动态 base”。

#### 关键技巧

- 测试必须包含非对齐 segment start，例如 `[100, 127, 300, 4097]`；只测整齐 4K 分段无法发现地址错误。
- O、LSE 和 aux tensor 的 ragged offset 要分别确认，不能因为 O 正确就默认全部正确。
- TMA epilogue 会改变 shared-memory 生命周期、warp role 和寄存器分配，不能只切一个 boolean。
- SASS 中应看到 `UTMASTG` 替代 `STG.E`，并确认 local load/store 没有重新出现。

#### 收益与状态

原型线上，4×4096 代表 case 从 `303.97 us` 降到 `267.36 us`，约改善 12%；`STG.E` 从 32 个降为 0，出现 4 个 `UTMASTG`，并消除 local spill。

这是有价值的通用 varlen 技术，但不是当前 2CTA 产品配置。当前代码在 `is_varlen_q=True` 时 `use_tma_O=False`，PackGQA/varlen 输出仍由 correction/epilogue warp 写回。未来若重新引入，必须针对 2CTA + PackGQA 单独证明 layout 和生命周期。

### 4.4 小 trick：把源码、依赖和 wheel 行为收敛成同一个实现

#### 解决的问题

原型曾依赖本地 `/home/mudi/flash-attention`、`sys.path` 注入、`inspect.getsource`、字符串替换和 `exec`。它适合快速实验，但会导致：

- 本地未提交代码静默影响 Atrex；
- wheel 不包含实际执行源码；
- traceback、source mapping 和 cache key 难以追踪；
- 换机器或 clean environment 后无法复现。

#### 怎么做

- `06a506be`：固定 FA4 来源，把修改保存为 tracked patch，build 时幂等应用。
- `80774d20`：只 vendor 目标专用 kernel，其余依赖来自 wheel。
- `be1c2f0c`：对齐 CUTLASS DSL 4.5.1，并删除不再维护的 1CTA 产品分支。
- wheel 复现从 `/tmp` 启动、清空 `PYTHONPATH`，确认 import 来自安装包。

#### 关键技巧

- CUTLASS DSL、Quack、FA4 helper 与 vendored kernel 是一个版本组合，不能只升级其中一个。
- 版本变化后要清 JIT cache，避免旧 CUBIN 掩盖源码或依赖不兼容。
- `flash-attn-4` 使用 namespace package；环境里其他 `flash_attn` 根包可能遮蔽 `flash_attn.cute`。
- clean-wheel 验证要打印真实 `__file__`，不能只检查 import 成功。

## 5. 大优化点二：调度与工作几何

这一组优化决定“有多少工作、按什么顺序运行、一个 dynamic work 由谁消费”。对 B300 上的 2CTA kernel，work geometry 往往比局部少几条指令更重要。

### 5.1 小 trick：LPT 提前重 tile，CLC 动态接管尾部 work

#### 解决的问题

causal attention 的 Q tiles 工作量呈三角形：越靠后的 Q tile 能访问越多 K/V blocks。varlen 又叠加 batch 间长度不均。自然顺序会让轻 tile 先运行、重 tile 留在 grid 尾部，造成大部分 SM 提前空闲。

#### 怎么做

```text
LPT（Longest Processing Time first）
    把 causal 重 tile 映射到更早的 work ID

CLC（Cluster Launch Control）
    resident cluster 完成当前任务后，接管尚未启动的 grid work
```

二者解决不同问题：LPT 改善初始顺序，CLC 动态消除尾波；必须组合使用，不能把它们当成同一个机制。

#### 关键技巧

- CLC response valid 与 varlen work valid 是两层概念；取消到 padding grid ID 后仍要安全映射为 invalid work。
- CLC 可能改善负载均衡，也可能破坏 K/V L2 locality，不能仅凭“causal”就假设一定快。
- multi-CTA 下被调度的 worker 是 cluster，而不是 CTA。
- scheduler 的临时变量可能只让某类 warp spill，需按 warp role 看寄存器和 local load/store。

#### 收益与状态

原型 `45d1161b` 中，8K HQ32/HKV2 由 `514.85 us` 降到 `463.62 us`，接近 dense ceiling `463.01 us`。当前产品固定使用 CLC + LPT。

### 5.2 小 trick：区分 CTA tile 与 logical cluster tile

#### 解决的问题

2CTA kernel 中每个 CTA 拥有 128 个 Q rows，但一个 scheduler work 由 CTA pair 共同处理 256 rows：

```text
CTA0: Q rows [0, 128)
CTA1: Q rows [128, 256)
logical cluster tile: [0, 256)
```

早期 causal `BlockInfo` 使用单 CTA 的 `cta_tiler[0]` 计算 K 上界，导致 CTA1 后 128 行需要的较晚 K blocks 被错误裁掉。

#### 怎么做

提交 `1a10ace9` 将 causal/local bound 使用的逻辑 M tile 修正为：

```text
logical_m_tile = cta_tiler[0] * cta_group_size
```

同时明确以下尺寸各自服务不同语义：

- `cta_tiler`：单 CTA 的 shared-memory / row ownership；
- `mma_tiler_qk`：2CTA UMMA 覆盖范围；
- scheduler tile：一个 dynamic work；
- cluster tile：causal bounds 和 cluster coordinate 的逻辑范围。

#### 关键技巧

- 不要寻找一个“统一 M size”到处复用；causal bounds、grid count、epilogue ownership 本来就可能使用不同 M 尺寸。
- 必测 `128±1`、`256±1`、1408 等 partial final cluster；整 256 对齐 case 很容易漏 bug。
- CTA0 正确不代表 CTA1 正确，应按 row range 分别比较。

### 5.3 小 trick：CLC descriptor 必须描述 physical cluster

#### 解决的问题

早期 launch 使用 physical cluster `(2,1,1)`，但 CLC problem descriptor 仍声明 `cluster_shape_m=1`。结果 CTA0/CTA1 可能取得不同 dynamic work，随后却继续通过 2CTA UMMA 和 cluster barrier 合作，导致 partial final Q cluster 错乱或 hang。

#### 怎么做

`45e9bda1` 增加 2CTA 专用 varlen scheduler descriptor：

```text
CLC problem cluster shape = (params.cluster_shape_m, 1, 1)
```

并要求：

```text
launch cluster shape
    == CLC problem shape
    == scheduler coordinate divisor
    == cta_group_size
```

#### 关键技巧

- 两个 CTA 必须共享一次 CLC response，不能各自 request/advance。
- 加构造期 assert 比依赖调用方约定更可靠；当前代码已有 `cluster_shape_m == cta_group_size` 检查。
- partial final cluster 是这个问题的最小有效测试，完整 cluster 数不会暴露它。

### 5.4 小 trick：PackGQA 优先优化 work 数和 wave 几何

#### 解决的问题

GQA16 若每个 Q head 独立调度，会重复扫描同一 KV head，并分别向 Q tile 边界取整。B300 有 148 SM，2CTA cluster 每波最多约 74 个：

```text
148 SM / 2 SM per cluster = 74 clusters per wave
```

例如 unpacked 为 80 clusters 时，虽然只超过 74 六个 cluster，却必须启动低利用率的第二波。

#### 怎么做

PackGQA 把同一 KV head 的多个 Q heads 折入 packed M：

- 减少 scheduler work 数；
- 减少逐 Q-head 的 tile padding；
- 在 CTA 内复用 K/V；
- 让 cluster 数从两波边缘回到一波内。

#### 关键技巧

- PackGQA 的首要分析指标是 cluster/wave 数，不只是理论 K/V 复用。
- packed M 会改变 token/head 映射、causal indexing 和 epilogue layout，必须覆盖 HQ16/HKV1、HQ32/HKV2。
- 优化组合要检查 predicate。早期 KPP 条件包含 `not self.pack_gqa`，导致 PackGQA 虽减少 work，却切回慢 pipeline；最终删除该排除，让 PackGQA 与 KPP 同时启用。
- 记录 physical CTA、cluster 数和相对 74-cluster 容量的位置，单看 tensor shape 不足以解释性能。

### 5.5 小 trick：产品最终固定 2CTA + CLC，不保留 shape policy

#### 解决的问题

1CTA/2CTA 最优选择依赖 batch、total Q、Q/K 比、prefix hit、page size、绝对 head 数以及是否跨 wave。早期从 64K 阈值逐渐发展为复杂 shape policy，但规则很快膨胀并过拟合当前 shape 集。

#### 怎么做

`be1c2f0c` 后收敛为 SM103 2CTA-only，`68c6014e` 再固定 CLC，删除环境变量和 static fallback。

#### 关键技巧

- 产品选择的是稳定支持边界和整体矩阵，不等于每个 shape 的理论最优。
- 若未来重新 dispatch，只能使用 host 已知静态 metadata，不能同步读取 device `cu_seqlens`。
- 1CTA/2CTA、static/CLC 都会扩大 compile variants 和正确性矩阵；没有明确整体收益时不要长期保留双路径。
- 环境变量适合实验 branch，不应决定生产 kernel 行为。

#### 已知代价

真实 page64 shapes 中，short-Q/very-long-K 的 sub-wave case 仍受 2CTA barrier 成本影响；19 个生产 shapes 有 6 个略慢于当时 TRTLLM-gen，最差为 `+4.98%`，但全部进入“不慢于 5%”门槛。该残余是固定 topology 的明确 trade-off，而非未知噪声。

## 6. 大优化点三：计算流水线与片上资源

这一组优化的目标是让 QK、softmax、P cast、PV 和 K/V load 尽量重叠，同时不因 shared memory、寄存器或 barrier 增加新的瓶颈。

### 6.1 小 trick：把 Q-stage 与 K-direction ping-pong 解耦

#### 解决的问题

HD256 下 TMEM 资源只适合保留一个 Q tile / O accumulator，即 `q_stage=1`。如果把 `q_stage=1` 错解成“所有中间结果也只有一个 slot”，QK、softmax 和 PV 会完全串行。

#### 怎么做

沿 K-block 方向为 S/P 保留两个 slot：

```text
slot 0: softmax/PV(i)
slot 1: QK(i+1)
    ↓ 下一轮交换
slot 1: softmax/PV(i+1)
slot 0: QK(i+2)
```

对应两个独立概念：

```text
q_stage = 同时保留多少个 Q/O tile
s_pp    = 沿 K 方向保留多少个 S/P slot
```

当前 HD256 配置为 `q_stage=1, s_pp=2`。

#### 关键技巧

- slot 选择必须基于全局 K iteration parity，不能让 causal-mask loop、no-mask loop、末轮各自从 0 开始。
- softmax、P producer、PV consumer、correction warp 必须对同一个 slot/phase 达成一致。
- split-P 使一个 P slot 有 partial/full 两次通知，expected arrival 必须匹配。
- KPP 是否有效要看 tensor active、long scoreboard 和 barrier stall；代码中存在双缓冲不代表硬件真的形成 overlap。

### 6.2 小 trick：persistent CLC 下维护跨 work-tile 的 pipeline epoch

#### 解决的问题

static kernel 中，一个 cluster 通常只处理一个 work，pipeline state 随 block 结束销毁。CLC persistent cluster 会连续处理多个 work：

```text
work A: odd number of K blocks
    ↓ 同一个 resident cluster
work B: 从哪个 slot / phase 开始？
```

若 work B 每次假设从 slot0/phase0 开始，work A 的奇数 K-block 会让 producer/consumer generation 错一位，表现为 hang、CUDA 912 或随机精度错误。

#### 怎么做

`60e36fc3` 恢复 CLC KPP 时采用：

1. `kpp_iter_global` 跨 work-tile 保留；
2. slot 用 global iteration 对 `s_pp` 取模；
3. barrier phase 显式限制为一位 generation；
4. work 的 K-block 数为奇数时补一个 dummy slot handshake；
5. dummy 只闭合 barrier epoch，不发新的 QK/PV MMA；
6. 下一 work 从完整双 slot 周期开始。

#### 关键技巧

- pipeline phase 是代数状态，不是某个 `for` 循环的局部计数器。
- dummy handshake 的 producer/consumer commit、wait、release 必须成对；只修 MMA 或 softmax 一侧必然死锁。
- 先在 CLC 下关闭 KPP、隔离正确性，再恢复 fast path，是处理复合 pipeline bug 的可靠顺序。
- 必测 odd/even K-block，并确保同一 resident cluster 连续取得多个 CLC work；单 work grid 无法验证。

### 6.3 小 trick：KV pipeline 从 4 stage 加到 5 stage

#### 解决的问题

NCU 显示目标路径并非 DRAM 带宽饱和，而是 K/V 到 shared-memory 的 load latency、long scoreboard 和 barrier 暴露较多。

#### 怎么做

在不改变 residency 的前提下，把 2CTA HD256 KV TMA pipeline 从 4 stage 增加到 5 stage：

```text
FP8 dynamic shared memory:
165.888 KB/CTA → 182.272 KB/CTA

occupancy:
仍为每 SM 一个 CTA，没有下降
```

#### 收益

代表 shapes 改善约 `0.8%–2.4%`。B4 irregular case 中：

- tensor pipe active：`67.93% → 69.59%`；
- eligible warps/scheduler：`0.461 → 0.472`；
- barrier stall/issued：`1.672 → 1.561`。

#### 关键技巧

- stage 增加只有在不降低 residency、且瓶颈确为 load latency 时才有意义。
- 3 stage 在 2K/16K/64K 退化约 8%/26%/24%；6 stage 对 B=1 基本无收益、B=4 略慢。stage 不是越多越好。
- 同时测 B=1 长序列和 B=4 irregular；两者对 producer distance 的敏感度不同。
- 确认 dynamic shared-memory bytes 真正变化，防止 JIT cache 复用旧 specialization。

### 6.4 小 trick：用 sQ/sO 生命周期复用换更多 KV stage

#### 解决的问题

在历史 1CTA kernel 中，Q 在主循环早期消费完，O 只在 epilogue 写入 shared memory，二者生命周期不重叠。分别分配 sQ/sO 会浪费约 32KB shared memory。

#### 怎么做

`083776d8` 让 sQ/sO alias，同一块地址先存 Q、后存 O，从而让 1CTA KV pipeline 由 4 stage 增至 5 stage。

```text
时间：load/use Q ────┐       write/store O
                    └─空闲───┘
空间：[          sQ / sO alias          ]
```

代表 prefix shapes 改善约 `5.2% / 3.2% / 1.0%`。

#### 关键技巧与当前状态

- 必须证明所有 warp 都不再读 Q，而不只是 MMA warp。
- persistent loop、提前 epilogue、aux output 都可能延长生命周期。
- alias 会改变 layout address、对齐和 barrier storage 排布，需整体复核。
- 当前 2CTA CLC kernel 明确禁止 `overlap_sO_sQ`；persistent/cluster 生命周期不满足原来的不重叠证明。因此这是历史有效 trick，不应直接复制到当前路径。

### 6.5 小 trick：按 warp role 分配寄存器，而不是只看平均值

#### 解决的问题

varlen scheduler、correction、softmax 和 load warp 的局部变量完全不同。只看 kernel 的平均 registers/thread，可能看不到某类低配 warp 已发生 stack spill。

#### 怎么做

原型 CLC+LPT 调优中，把寄存器从 correction warp 向其他 warp 重分配：

```text
correction warp: 80 → 72
other warp:      48 → 56
```

目标不是减少总寄存器，而是让 scheduler 临时状态不落入 local memory。

#### 关键技巧

- 用 SASS/SourceCounters 检查 `LDL/STL`，不要只看 occupancy 表。
- scheduler 修改后要重新检查各 role 的 spill；增加一个坐标或 divmod 可能只伤害 load/scheduler warp。
- 寄存器重分配和 threads/block、warp role 数绑定，不能从 1CTA 机械迁移到 2CTA。

### 6.6 小 trick：split-P 让 PV 提前启动，但只保留有证据的切分点

#### 原理

softmax warp 可以先写一部分 P，发 partial-ready，让 PV 提前开始，再写剩余列并发 full-ready：

```text
write P[0:split] → partial arrival → PV begins
write P[split:N] → full arrival
```

#### 实验结论

- 当前 2CTA 默认 75%；
- 75%→50% 只有约 `-0.3%～+0.1%`，基本中性；
- 25% 触发 SM launch failure；
- 因此保留 75%。

#### 关键技巧

- split point 同时影响 overlap、barrier arrival 数和消费者最小连续工作量，不只是一个性能常量。
- 1CTA 的最优值不能直接迁移到 2CTA；pair-UMMA 和 cluster barrier 改变了消费节奏。
- 必测短 K、长 K、奇数 block count 和 causal 边界。

## 7. 大优化点四：Paged-KV 数据搬运与寻址

Paged-KV 的优化重点不是“减少 K/V payload”——多数 case 的 payload 已接近必要量——而是减少 gather 指令、地址依赖、重复 page-table load 和不安全的 masked load。

### 7.1 小 trick：先按数据形状选择搬运原语，不要只看指令名字

同样是 GMEM/SMEM 搬运，CuTe DSL 可以生成 TMA、non-bulk `cp.async`、普通 vector load/store 等不同指令。它们的功能有重叠，但寻址方式、发射线程、同步协议和固定成本不同。

#### 选择表

| 数据传输需求 | 优先原语 | 常见 SASS | 优点 | 主要代价/限制 |
| --- | --- | --- | --- | --- |
| 规则多维 GMEM→SMEM tile | Tensor TMA load | `UTMALDG.{N}D` | 单次描述大 tile，地址/指令开销低，适合深 pipeline | 需要 descriptor、规则 tile 和 mbarrier；小传输固定成本高 |
| 2CTA UMMA 的规则 operand | CTA-group TWO Tensor TMA | `UTMALDG.{N}D.2CTA` | descriptor/layout 与 2CTA MMA operand 对齐 | barrier bytes、cluster layout 必须匹配；并不自动等于 multicast |
| 同一 GMEM tile 被 cluster 多 CTA 复用 | multicast Tensor TMA | TMA multicast variant | 一次读取可投递多个 CTA 的 SMEM，减少重复 payload | 需要 multicast op、mask、remote SMEM 和 cluster 同步；仅适合消费者数据完全相同 |
| 规则 SMEM→GMEM tile | Tensor TMA store | `UTMASTG` | 消除大量逐线程 store，降低 epilogue 指令和寄存器压力 | 输出必须能表达成规则 descriptor tile；完成协议是 bulk commit/wait |
| 不规则、逐行 predicated GMEM→SMEM | non-bulk `cp.async` | `LDGSTS` | 每线程地址灵活，容易处理 gather、ragged 和细粒度 predicate | 指令与地址计算多，容易 long-scoreboard；warp 要共同搬完整 tile |
| 小块、强 predicate、SMEM→GMEM | vector register copy | `LDS` + `STG` 等 | 最灵活，可处理 PackGQA/ragged row 映射 | 占寄存器、指令多，可能 spill |
| 规则连续 byte range、无需 tensor descriptor | bulk non-tensor copy | bulk `cp.async` variant | 比逐线程 copy 更粗粒度，不需要多维 tensor map | 不能自动做多维坐标/边界映射；当前 HD256 主路径没有使用 |

#### 不要用 SASS 维度数直接判断快慢

`UTMALDG.3D`、`UTMALDG.4D` 中的 3D/4D 主要反映 tensor descriptor 和坐标的 rank，不是性能等级：

- Q 在当前 kernel 中通常生成 `UTMALDG.3D.2CTA`；
- small-page K/V 的 tensor 含 page/head/physical-page 等维度，生成 `UTMALDG.4D`；
- contiguous 或大页 K/V 还可能生成 `UTMALDG.4D.2CTA`。

同样 payload 下，较少动态坐标和较低 descriptor rank 可能减少少量寻址工作，但真正决定性能的是：

```text
每个 logical tile 发多少条 TMA
每条 TMA 搬多少有效 bytes
是否重复读取相同 payload
是否需要跨 CTA multicast/notification
mbarrier 和 pipeline 是否隐藏了 latency
是否为了使用 TMA 而搬了大量会被 mask 的无效数据
```

因此不能仅看到 `.3D` 就认定优于 `.4D`，也不能仅看到 `.2CTA` 就认定比单 CTA TMA 快。

#### 当前 kernel 的调用方式与生成指令

| 数据 | CuTe DSL 构造方式 | 发射方式 | 典型 SASS |
| --- | --- | --- | --- |
| Q | `CopyBulkTensorTileG2SOp(CtaGroup.TWO)` + `make_tiled_tma_atom_A(...)` | `tma_get_copy_fn()` 后传 `tma_bar_ptr` | `UTMALDG.3D.2CTA` |
| contiguous/page128/page256 K/V | CTA-group TWO op + `make_tiled_tma_atom_B(...)` | `tma_partition()` + `cute.copy(...)` | `UTMALDG.4D(.2CTA)` |
| page16/32/64 K/V | 默认 CTA-group ONE op + 通用 `make_tiled_tma_atom(...)`，descriptor tile 等于一个 physical page | 每 CTA 对自己负责的 page 循环 `cute.copy(...)` | `UTMALDG.4D` |
| dense/规则 O | `CopyBulkTensorTileS2GOp()` + `make_tiled_tma_atom(...)` | `cute.copy()` + bulk commit/wait | `UTMASTG` |
| 当前 varlen/PackGQA O | `CopyUniversalOp` 组成 128-bit tiled copy | sO→register→GMEM，逐 row predicate | `LDS/STG` 类指令 |

这里的 `make_tiled_tma_atom_A/B` 是 MMA-aware helper：它根据 MMA operand、CTA layout 和 SMEM layout 构造 TMA atom，适合规则 Q/K/V operand。通用 `make_tiled_tma_atom` 则允许显式指定 `(page_size, head_dim)` 这样的 physical-page tile，更适合 small-page specialization。

这两类 helper 都在 JIT 编译阶段构造 descriptor/layout，本身没有 Python runtime 调用开销。性能差异来自最终生成的 descriptor、TMA 条数、搬运范围和同步，而不是函数名字。

#### TMA load 的正确发射协议

规则 GMEM→SMEM TMA 的基本结构是：

```python
op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    op, gmem_tensor, smem_layout, tile_shape
)

# kernel 开始时预取 descriptor
if warp_idx == 0:
    cpasync.prefetch_descriptor(tma_atom)

# 每个 pipeline stage 先设置预计 transaction bytes
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive_and_expect_tx(barrier, tx_bytes)

# TMA copy 本身不要再包 elect_one；DSL 会自动选一个线程发射
cute.copy(
    tma_atom,
    gmem_tile,
    smem_tile,
    tma_bar_ptr=barrier,
)
```

关键点：

- `cute.copy(tma_atom, ...)` 必须保持 warp-uniform，不要手工包 `with elect_one()`；当前 CUTLASS DSL 会隐式选择发射线程，重复 elect 可能导致错误或 deadlock。
- barrier 初始化和 `expect_tx` 仍需要 `elect_one()`。
- `tx_bytes` 不是“发了几条 TMA”，而是该 barrier generation 预期完成的总字节数。
- 多 page/多 TMA 共用一个 stage barrier 时，必须把所有 transaction bytes 相加。
- current small-page 2CTA 路径中，每个 CTA 先等待自己的 TMA barrier；non-leader 再用一次 16-byte cluster remote store 通知 leader，因此 leader 的 expected bytes 还要包含这 16 bytes。
- descriptor 在 kernel 入口预取一次，可以避免第一次 TMA issue 暴露 descriptor fetch latency；不要在每个 K block 重复 prefetch。

#### TMA store 使用另一套完成协议

SMEM→GMEM TMA store 不使用 load-side 的 transaction mbarrier。典型调用为：

```python
store_O(src_idx=stage, dst_idx=stage)
cute.arch.cp_async_bulk_commit_group()

# 复用 sO buffer 前，等待对应 store group 已完成读取 shared memory
cute.arch.cp_async_bulk_wait_group(pending_groups, read=True)
```

`read=True` 的意义是确保 TMA store 已经读完 sO，之后 producer 才能安全复用该 shared-memory stage。若只等待“全局写最终可见”而忽略 sO 生命周期，persistent epilogue 可能覆盖尚未被 TMA 读取的数据。

#### `.2CTA` 与 multicast 的区别

当前 `CopyBulkTensorTileG2SOp(CtaGroup.TWO)` 选择的是 Blackwell 2SM/2CTA TMA 形式，用来匹配 2CTA MMA operand 和 cluster layout。它不等价于把同一 tile multicast 给两个 CTA。

真正的 multicast 需要：

```python
op = cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group)
tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    op, gmem_tensor, smem_layout, tile_shape,
    num_multicast=cluster_size,
)
cute.copy(
    tma_atom.with_(mcast_mask=cluster_mask),
    gmem_tile,
    smem_tile,
    tma_bar_ptr=barrier,
)
```

只有当多个 CTA 需要完全相同的 GMEM tile、各自 SMEM layout 兼容，并且节省的重复 payload 大于 multicast/remote synchronization 成本时，multicast 才值得使用。当前 small-page K/V 路径是 CTA-local 分工：K 按 token/page 拆，V 按 D 拆，两个 CTA 需要的目标 slice 不相同，因此选择多个 CTA-local TMA，而不是强行 multicast。

#### TMA 与 `LDGSTS` 的实用选择流程

```text
数据能否表示为 compile-time tile + 少量 runtime tensor coordinates？
    否 → 使用 LDGSTS / vector copy
    是
    ↓
每次能否搬一个有足够有效 bytes 的规则 tile？
    否 → 小传输下 TMA 固定成本可能不划算，A/B 测 LDGSTS
    是
    ↓
多个 CTA 是否消费相同 payload？
    是 → 评估 multicast TMA
    否 → CTA-local TMA 或 CTA-group TWO TMA
    ↓
边界是否会读取未初始化/NaN 数据？
    是 → descriptor 边界、safe page 或显式 predicated fallback
    ↓
用 SASS + NCU 确认：TMA 条数、有效 bytes、barrier stall、long scoreboard
```

对当前 HD256 workload，经验是：

- Q 和规则 contiguous/大页 K/V：优先 MMA-aware TMA；
- page16/32/64：优先按 physical page 构造多个小 TMA，而不是逐线程 gather；
- 真正任意 page gather、逐行不规则地址或复杂 predicate：保留 `LDGSTS`；
- dense/aligned O：优先 TMA store；PackGQA/ragged O 在 descriptor 无法自然表达时使用 vector epilogue，或重新设计 ragged descriptor，而不是强开 TMA。

#### 判断 TMA 是否真的更快，要同时看五项指标

1. **TMA issue 数**：把一个 tile 拆成过多小 TMA，可能把 descriptor/barrier 固定成本放大。
2. **有效字节比例**：TMA 搬得大但多数被 mask，未必优于 predicated `LDGSTS`。
3. **地址指令数**：TMA 的主要收益之一是减少 per-thread pointer/divmod/gather 指令。
4. **同步成本**：long-scoreboard 下降但 barrier stall 大增时，wall time 可能不变。
5. **pipeline overlap**：只有 consumer 能在后续 stage 工作时，异步搬运才能隐藏 latency。

当前 small-page 优化就是一个完整例子：`LDGSTS` 路径 DRAM SOL 很低但 long-scoreboard 高；换成 page-specialized `UTMALDG.4D` 后，静态 `LDGSTS` 从 32 个降到 0，并在最终矩阵中获得稳定收益。这说明收益来自减少地址/指令与改善 overlap，不是因为 TMA 的理论带宽一定更高。

### 7.2 小 trick：page128/256 使用一页多 tile 的 TMA

#### 解决的问题

早期实现隐含 `page_size == tile_n`：page-table 直接用 `n_block` 索引，页内 tile 固定为 0。对 FP8 `tile_n=128`：

- page128 恰好正确；
- page256 的第二个 KV tile 会重复读取该页前 128 tokens；
- 通用 gather 路径又有大量 LSU 指令和地址计算。

#### 怎么做

`fe0a5c5c` 对 `page_size >= tile_n` 使用：

```text
tiles_per_page = page_size / tile_n
logical_page   = n_block / tiles_per_page
tile_in_page   = n_block % tiles_per_page
physical_page  = page_table[logical_page]
```

TMA descriptor 选择 physical page，source tile 再使用 `tile_in_page` 选择页内偏移。

#### 收益

32K prefix 代表 case：

```text
LSU traffic: 168.9M → 13.0M（约 13 倍下降）
tensor pipe: 67.7%  → 76.3%
latency:     3200us → 2980us
```

这表明旧路径主要受 gather 指令、地址计算和 load latency 限制，而不是 HBM 带宽饱和。

#### 关键技巧

- page table 永远以 page 为单位，不能直接用 tile ID。
- page256 是验证 `tile_in_page` 的必要 case；只测 page128 无法覆盖页内第二 tile。
- page size 必须进入 compile key，因为它改变 descriptor 和 source layout。
- 使用 shuffled physical page table；identity mapping 会让忽略 page table 的错误实现也可能通过。

### 7.3 小 trick：page16/32/64 使用一 tile 多页的 page-specialized TMA

#### 解决的问题

small page 早期走 `PagedKVManager + LDGSTS/cp.async` gather。典型 NCU 症状为：

```text
global sectors/request ≈ 13.9（理想约 4）
L1 hit                 ≈ 1.13%
L2 hit                 ≈ 78.4%
DRAM read SOL          ≈ 2.2%
主要 stall             long scoreboard
```

DRAM 利用率很低，说明问题是大量不连续小 load、page-table/index 指令和 load latency，而非带宽。

#### 怎么做

`526b741e` 为 page size 做 compile-time specialization，把一个 logical tile 拆成多个 TMA transaction：

```text
FP8 tile N=128:
    page16 → 8 个 page transaction
    page32 → 4 个 page transaction
    page64 → 2 个 page transaction

BF16 tile N=64:
    page16 → 4 个 page transaction
    page32 → 2 个 page transaction
    page64 → 1 个 page transaction
```

2CTA 内保持明确分工：

- K load 按 KV token/page 范围分给 CTA0/CTA1；
- V load 按输出 D 范围分给 CTA0/CTA1；
- QK/PV 仍由 `UTCQMMA.2CTA` 完成完整 HD256 运算；
- 每个 CTA 使用本地 TMA barrier；
- non-leader 完成全部本地 TMA 后，以 cluster async store 通知 leader。

#### 收益

最终 FP8 page16/32/64 验证矩阵：

```text
60/60 finite
60/60 correct
60/60 faster than 当时 nightly TRTLLM-gen
speedup 1.090x–1.338x
```

SASS 验证包含 `UTCQMMA.2CTA` 和 `UTMALDG.4D`，`LDGSTS=0`，证明收益不是通过回退 1CTA 或重打包 page128 获得。

#### 关键技巧

- small page 不等于只能 gather；只要 page size 是 compile-time specialization，就可以聚合多次 TMA。
- K 和 V 的 CTA 分工不同，不能复用同一个 source-tile 公式。
- 每 CTA barrier 的 transaction bytes 必须等于本 CTA 实际发出的 TMA 总量。
- remote notification 必须在本 CTA 全部 TMA 完成之后。
- 新 page size 要先检查 `pages_per_tile` 与 CTA 分工的 divisibility。
- `seqlen_k < 2048` 时 fixed cluster/barrier 开销占比较大，历史只保证正确，不承诺一定更快。

### 7.4 小 trick：page ID 只查一次，在 K/V issue 间用寄存器复用

#### 解决的问题

page64 真实 shape 的 NCU 对比中：

```text
Atrex 与 TRT 读取的 DRAM payload 基本相同
Atrex global-load instructions: 228,720
TRT global-load instructions:     2,640
```

差距主要是 page-table/address bookkeeping。一个 KV128 tile 含两个 page64：

```text
K：CTA0/CTA1 各查一个 page ID   → 2 次/cluster
V：两个 CTA 再查 page0/page1    → 4 次/cluster
合计                              6 次/cluster/tile
```

K/V 使用相同 physical page IDs，没有必要重复查询。

#### 怎么做

`4f1d748a` 在 load warp 中一次加载当前 logical tile 的 page ID tuple，并在 K/V TMA issue 间复用：

```text
page_ids = load_page_table_once()
    ├─ issue K TMA(page_ids)
    └─ issue V TMA(page_ids)
```

#### 收益

global loads 约下降 32.8%，与理论从 6 次降到 4 次接近；最差真实 shape 被推进“不慢于 TRT 5%”的验收范围。

#### 关键技巧

- tuple 长度由 page size 决定，必须 compile-time specialization。
- register-local 复用优于为了共享索引新增 cluster shared-memory 同步。
- K/V 的 page ID 相同，但 source layout 和 CTA 分工不同；缓存索引，不要勉强缓存完整 address 计算结果。
- 指令下降不等于 latency 一定下降；必须同时看 wall time，否则回退。

### 7.5 小 trick：masked tail 必须指向“有限的有效页”

#### 解决的问题

最后一个 logical tile/page 常常不完整，但 TMA 或 CTA-local copy 仍可能对被 mask 的列发物理 load。当前 HEAD 的无效 entry 仍可能使用：

```text
physical_page = 0
```

page 0 未必初始化，也可能包含 FP8 NaN：

```text
masked tail → load page0 NaN → MMA accumulator → 后续 score mask 无法可靠消除
```

page64/tile128 下，CTA1 的 local predicate 还可能漏掉自身 64-token offset，使 partial half 继续发 load。

#### 怎么做

side-branch `0d56d09f` / `8bbeb01d` 的做法是：

```text
invalid tail entry
    → 指向该 sequence 最后一个有效 physical page
    → 再由 seqlen/causal score mask 丢弃越界列
```

最后有效页至少属于当前请求且已初始化，避免引入 NaN 源。

#### 关键技巧

- “数学上会被 mask”不代表物理 load 可以读取任意垃圾；masked load 也必须数值安全。
- page0=NaN 是必要的主动回归，不要依赖 allocator 恰好清零。
- 同时覆盖 small-page TMA 和 PagedKVManager 路径，以及 CTA1-only partial half。
- `seqlen_k=0` 若不支持，应提前 assert；不能直接计算 last valid page。

#### 当前状态：P0 风险

截至 `0381108a`，safe-tail fix 仍在 side branch，尚未进入当前主线；当前 `load_paged_tma_page_indices()` 和部分 PagedKVManager fallback 仍会为 invalid entry 返回 0。除非上层严格保证 physical page 0 始终初始化为有限值，否则这是需要优先合入或重新实现的正确性风险。

## 8. 大优化点五：FP8 数值路径与服务热路径

前四组优化主要处理 GPU kernel 内部。最后一组保证 FP8 cast 不损失有效动态范围，并把不会随请求变化的 Python/JIT 工作移出服务热路径。

### 8.1 小 trick：P scaling 与 online-softmax correction 联合设计

#### 解决的问题

softmax 概率 P 写入 FP8 E4M3 前容易下溢，因此 kernel 在 cast 前乘 `2^max_offset`，PV 后再反缩放。但 online softmax 的 running row max 可能落后真实 max；放大过多会让 P 超过 E4M3 最大有限值 448 并 saturation。

可以近似写成约束：

```text
max_offset + rescale_threshold <= 8
```

其中：

- `max_offset`：P cast 前主动放大的 log2 指数；
- `rescale_threshold`：running max 允许落后的 log2 阈值。

#### 怎么做

当前 FP8 配置为：

```text
max_offset = 4
rescale_threshold = 4
```

P producer 和最终 O normalization 都使用同一个 offset。correction warp 在 running max 跳变时重缩放已有 O accumulator。

#### 关键技巧

- 数值 knob 同时是同步 knob：threshold 会改变 correction 频率和 O-full barrier 行为。
- P cast 与 O normalization 必须使用同一个 offset；只改一处会产生整体 scale 错误。
- saturation 不一定产生 NaN，必须看 cosine、relative L1/L2，而不只是 finite。
- 减小 offset 防 saturation，却可能增加小概率下溢；两端都要测。
- TRTLLM 某些 FP8-output baseline 在模型尺度输入上会接近全零，只能作为 speed lower bound，不能当 correctness golden。

### 8.2 小 trick：优化条件分支前，先测 warp-wide predicate 命中率

#### 实验

曾尝试把 `(max_offset, threshold)` 从 `(4,4)` 改成 `(2,6)`，并仅在 `should_rescale=True` 时等待 O-full barrier，希望减少 correction 同步。

#### 为什么没有收益

`should_rescale` 是 warp-wide predicate：32 行中只要一行需要 rescale，整个 warp 就执行 correction。实测它几乎每轮都为真，最终改善只有约 `0.01%–0.13%`，属于噪声范围，方案回退。

#### 可复用技巧

看到源码里有 `if`，不代表大多数迭代能跳过。任何 conditional-wait、conditional-correction 优化，都应先统计真实 predicate 分布和 warp aggregation 行为。

### 8.3 小 trick：prepare 固化静态 dispatch，run 只做 launch

#### 解决的问题

普通 Python wrapper 每次调用可能重复：

- shape/dtype/layout validation；
- causal/window 参数归一；
- kernel variant 选择；
- compile-key 组装与 cache lookup；
- fake tensor / JIT compile 检查；
- 输出和 aux metadata 准备。

短 kernel 中，这些 CPU 工作会污染端到端延迟，也不利于 CUDA Graph capture。

#### 怎么做

`1eab0af9` 增加 `FlashAttentionHd256Prefill`：

```text
prepare()/plan():
    validation
    static dispatch
    compile/cache lookup
    固化 callable 与静态参数结构

run():
    接收本次 Q/K/V、长度 tensor、page table、out
    直接启动 prepared kernel
    不读取 device sequence lengths
```

#### 关键技巧

- prepare 只能固化 dtype、layout、page size、head 数、PackGQA 等静态属性。
- `cu_seqlens`、`seqused_k`、page-table 内容是动态数据，必须继续由 device tensor 提供。
- dtype/layout/page size/head 数变化时必须重新 prepare。
- compile key 必须覆盖所有影响 descriptor/layout 的属性；当前包含 Q dtype、KV dtype、head dim、HKV、page size、PackGQA、2CTA 等。
- benchmark 要明确是否计入 prepare/compile。kernel 对比通常排除，服务冷启动则单独报告。

### 8.4 小 trick：CUDA Graph 下替换 runtime metadata tensor，而不重做 dispatch

#### 解决的问题

vLLM 会复用 graph-stable metadata buffer，但每个 scheduler step 的内容不同。若 `run()` 固定使用 prepare 时的长度 tensor，或者根据新长度重新选择 CUBIN，都无法正确支持 graph replay。

#### 怎么做

`0381108a` 允许 `run()` 覆盖：

- `cu_seqlens_q/k`；
- `seqused_q/k`；
- `page_table`；
- output buffer。

静态 kernel 选择不变，runtime 只替换 tensor 指针/内容；page128 vLLM path 因而可以 graph-safe replay。

#### 关键技巧

- graph safety 的关键不是“所有参数不变”，而是 kernel topology 不依赖动态 sequence length。
- prepared callable 与 graph 生命周期中的 buffer 地址约束必须一致。
- 不要在 `run()` 中基于长度选择 1CTA/2CTA 或重新组 compile key。

### 8.5 小 trick：把调优开关留在实验层，不让环境变量决定产品行为

#### 解决的问题

早期存在 `FA_CLC`、`FA_HD256_2CTA`、`FA_HD256_USE_MAIN` 等环境开关。它们适合 A/B，但会让相同 wheel 在不同 shell 中运行不同 kernel，并扩大 compile/test 矩阵。

#### 怎么做

`68c6014e` 固定目标路径为 CLC，配合 2CTA-only 收敛，删除 static fallback 与产品环境开关。

#### 关键技巧

- A/B knob 保留在独立 benchmark、workspace kernel 或实验 branch。
- 产品源码只留下已经通过完整 correctness/performance matrix 的配置。
- profiler 首先确认 kernel class、block size 和 cluster size，避免测试脚本仍设置无效的旧环境变量。

## 9. 优化之间的依赖关系：单点有效不等于组合有效

这条 kernel 的多数问题发生在优化组合处。下面的依赖比单个 knob 更值得长期保留。

| 组合 | 必须满足的条件 | 失败表现 |
| --- | --- | --- |
| 2CTA + causal | causal bound 使用 logical cluster M=256 | CTA1 后半 rows 缺 K blocks |
| 2CTA + CLC | physical cluster、CLC descriptor、coordinate divisor 一致 | partial cluster 错乱或 hang |
| CLC + KPP | slot/phase 跨 work 保留；odd work 补 dummy epoch | CUDA 912、hang、随机误差 |
| PackGQA + KPP | fast-path predicate 不排除 PackGQA | work 变少但 kernel 反而慢 |
| small-page TMA + 2CTA | K/V 各自按正确维度拆分；本地 barrier bytes 正确 | 读错页或 barrier deadlock |
| page-index cache + tail | invalid tuple entry 仍指向有限页 | page0 NaN 污染 |
| P scaling + correction | P 与 O 使用相同 offset；threshold 满足 FP8 范围 | 静默 saturation/scale error |
| prepare + CUDA Graph | 只固化静态属性；动态长度留在 tensor | replay 错误或触发重新 dispatch |
| sQ/sO alias + persistent | 必须重新证明全体 warp 生命周期不重叠 | 覆盖仍在使用的 Q 数据 |

一个实用检查法是：每增加一个优化，先列出它改变了哪些状态边界——work geometry、layout、barrier epoch、shared-memory lifetime、compile key 或 runtime metadata——然后逐项检查与已有 fast path 的交集。

## 10. 没有形成产品收益的实验

失败实验不是独立的大优化点，而是每个大方向的边界条件。保留它们可以避免重复验证相同的错误假设。

| 所属方向 | 实验 | 预期 | 实测与结论 |
| --- | --- | --- | --- |
| varlen | single-seq route-to-dense | 绕过 varlen 固定开销 | 短期有效，只用于定位 ceiling；最终删除 |
| topology | 仅按 64K 阈值选择 2CTA | 用简单规则覆盖长序列 | batch/prefix/heads 改变结论，误判较多 |
| topology | 复杂 shape policy | 每个 shape 选 1CTA/2CTA 最优 | 规则快速膨胀、容易过拟合；产品固定 2CTA |
| pipeline | KV stage 3 | 降低 shared memory/barrier | 2K/16K/64K 退化约 8%/26%/24% |
| pipeline | KV stage 6 | 更深预取 | B=1 基本无收益，B=4 略慢于 stage 5 |
| pipeline | split-P 75%→50% | PV 更早开始 | `-0.3%～+0.1%`，基本中性 |
| pipeline | split-P 25% | 更激进 overlap | SM launch failure，说明同步组合可能非法 |
| pipeline | CLC 直接复用 static KPP | 保留 overlap | odd K-block 后 phase 错，必须维护 global epoch |
| work geometry | 只打开 PackGQA | 减少 cluster 数 | 因 predicate 关闭 KPP，某些 shape 反而退化 |
| FP8 | threshold 4→6/8 | 减少 correction | warp-wide predicate 抵消收益，boundary case 无改善 |
| FP8 | conditional O-full wait | 无 rescale 时跳过等待 | 仅 `0.01%–0.13%`，回退 |
| Paged-KV | invalid tail 指向 page0 | 简化边界逻辑 | 可能读取未初始化 FP8 NaN，不安全 |
| shared memory | 把 1CTA sQ/sO alias 直接移到 2CTA | 释放空间 | CLC/persistent 生命周期无法证明，当前明确禁用 |

这些实验反映出几个通用规律：

- 先测 predicate/occupancy/work geometry，再改源码；
- “更多 stage”“更早 arrival”“更少 work”都不是单调收益；
- correctness isolation 优先于保留全部 fast path；
- 产品配置追求可验证的整体矩阵，不追求每个 shape 的 oracle dispatch。

## 11. 性能证据应如何理解

不同报告的 baseline、page size、PackGQA、timer 和 GPU clock 不完全一致，不能把所有数字拼成一条单调优化曲线。下面只按证据类型归纳。

### 11.1 单项优化证据

| 优化 | 代表结果 | 说明 |
| --- | --- | --- |
| ragged O TMA | 4×4096 `303.97→267.36 us`，约 `-12%` | 原型 varlen epilogue；当前产品未使用 |
| varlen CLC+LPT + reg topology | 8K `514.85→463.62 us` | native varlen 接近 dense ceiling |
| page128/256 TMA | 32K prefix `3200→2980 us`；LSU 约降 13 倍 | 消除 gather/address bottleneck |
| 1CTA sQ/sO alias + stage5 | 三个 prefix case 改善约 `5.2%/3.2%/1.0%` | 历史 1CTA 路径 |
| 2CTA KV stage5 | 代表 shapes 改善 `0.8%–2.4%` | 当前产品路径 |
| 2CTA vs 1CTA，30% prefix | 22 cases 提升约 `1.037x–1.083x` | 说明目标 workload 中 2CTA 有整体收益 |
| page-ID cache | global load 约降 32.8% | 把最差真实 shape 推入 5% gate |

### 11.2 最终矩阵证据

- small-page FP8：page16/32/64 共 60 cases，60/60 finite、correct，并快于当时 nightly TRTLLM-gen，speedup `1.090x–1.338x`。
- page64 真实 19 shapes：19/19 进入 Atrex 不慢于 TRT 5% 的门槛，13/19 严格更快，几何平均 speedup `1.087x`。
- page128 历史 103 shapes：相对当时 custom FA4，103/103 更快，几何平均 speedup `1.389x`。
- `0381108a` 对 flash-attn-4 dev1503 的 page128 扩展测试：68/68 更快，报告的几何平均 speedup `1.375x`；NCU 显示主要来自更高 SM throughput，而不是 DRAM 饱和。

### 11.3 为什么不同新 wheel 对比会得到不同幅度

2026-07-24 的记录同时包含两类口径：

- 默认/production-facing prepared 或 graph 路径、PackGQA 生效时，对 dev1503 的扩展 page128 corpus 报告约 `1.375x` geomean；
- 双方都改成 direct public forward、`pack_gqa=False`、锁 2032MHz 后，对 dev1494 的 Qwen3.5 12 shapes 为 `1.069x`，Qwen3.7 34 shapes 的 do_bench/Graph 分别约 `1.038x/1.063x`。

这不是矛盾，而是说明 host path、PackGQA、wheel 版本、shape corpus 和 timer 会显著改变结果。引用性能时必须同时写清这些条件，不能只保留一个 speedup 数字。

### 11.4 当前仍有改进空间的区域

page64 short-Q / very-long-K / sub-wave shapes 中，2CTA correction barrier 仍是残余瓶颈。最差 case 的 SourceCounters 把多数 barrier samples 定位到 intermediate O-full wait；Atrex 与 TRT 读取的 DRAM payload 相近，因此继续减少 payload 不是第一优先级。更可信的后续方向是：

- 在不破坏统一 topology 的前提下降低 correction/cluster 同步成本；
- 改善 page/address locality；
- 用可解释的 work-geometry 指标判断是否值得重新引入有限 dispatch；
- 优先合入 safe-tail correctness fix，再做新的性能叠加。

## 12. 验证方法：按优化影响的层次设计 case

### 12.1 基础数值门槛

FP8 attention 不宜只用默认逐元素 `assert_close`。历史常用门槛为：

```text
finite output
cosine > 0.99（稳定产品结果通常 > 0.999）
relative L1 < 0.08
relative L2 <= 0.05
max abs 作为辅助指标
```

同一实现的 contiguous/paged 路径若运算顺序等价，可要求 bit-exact；与独立 baseline 比较则使用统计阈值。

### 12.2 native varlen 必测

- batch=1、batch=4 equal、batch=4 irregular、long-tail；
- segment start 非 8/16/128 对齐，例如 `[100,127,300,4097]`；
- tiny tail：`1/127/128/129`；
- `cu_seqlens` 和 `seqused` 两类 metadata；
- max sequence length 大于真实长度；
- empty Q/K 若不支持，应验证明确 assert。

### 12.3 2CTA / CLC / causal 必测

- Q rows 同时覆盖 CTA0 和 CTA1；
- Q length 为 `128±1`、`256±1` 和 partial final cluster；
- K-block 数分别为奇数和偶数；
- 同一个 persistent cluster 连续获取多个 CLC work；
- equal Q/K；
- Q<K 的 prefix-cache 右下角 causal 对齐；
- very-short-Q + very-long-K；
- PackGQA 的 HQ16/HKV1 与 HQ32/HKV2。

### 12.4 Paged-KV 必测

- page16/32/64/128/256；
- identity 和 shuffled page table；
- physical page0 主动填 FP8 NaN，且有效 sequence 不引用 page0；
- `seqlen_k` 位于 `page_size±1`、`tile_n±1`；
- page256 的第二个 tile；
- page16/32/64 的一个 tile 聚合多页；
- CTA1-only partial half；
- contiguous 与 paged 输出对比；
- 多次运行检查非确定性。

### 12.5 Pipeline 必测

- causal-mask loop、no-mask loop 和最后一次 iteration；
- split-P partial/full arrival；
- correction rescale 发生与不发生；
- odd K-block work 后紧跟另一个 CLC work；
- PackGQA + KPP 的组合路径；
- 清理 JIT cache 后重新编译；
- SASS 确认没有意外 `LDL/STL` spill。

## 13. 性能测量与 NCU 诊断顺序

### 13.1 分开三种时间

```text
compile time  CuTe DSL/JIT 生成 CUBIN
prepare time  validation、dispatch、cache lookup、callable 准备
run time      kernel launch 与 GPU execution
```

kernel 优化比较排除 compile/prepare；服务冷启动则三项分别报告。

### 13.2 do_bench、CUDA Graph 和 NCU 各自回答不同问题

```text
do_bench
    快速 sweep、发现候选和异常 shape

CUDA Graph replay
    服务态 launch 路径与 graph-safe 行为

NCU gpu__time_duration.sum
    kernel duration、identity、stall、pipe、cache、指令和 launch geometry
```

最终结论至少记录：kernel name、grid、threads/block、cluster dimension、dynamic shared memory、registers/thread、capture 次数、中位数、GPU 和频率状态。

### 13.3 第一步：确认测到目标 kernel

```text
kernel class = FlashAttentionForwardHd256_2CTA_Sm103
cluster X    = 2
block        = 320 threads
SM arch      = 103
input        = FP8 E4M3
output       = BF16
```

identity 不匹配时，后续 counter 没有解释价值。

### 13.4 第二步：先看 work geometry

```text
physical CTAs
clusters = CTAs / 2
waves ≈ clusters / 74
PackGQA 前后 cluster 数
Q-tail padding
```

如果恰好从 75 个 cluster 降到 74，优先解释 wave cliff；不要先归因于某条低级指令。

### 13.5 第三步：用 stall/pipe 判断瓶颈类型

| NCU 症状 | 常见原因 | 优先检查 |
| --- | --- | --- |
| long scoreboard 高、DRAM SOL 低 | 小 load、地址依赖、预取不足 | Paged TMA、stage、page-ID cache |
| barrier 高 | 2CTA/KPP/correction 同步 | phase、arrival、O-full wait |
| local ld/st 非零 | 某类 warp 寄存器不足 | warp-role register config |
| tensor active 低、grid 不满一波 | 并行度不足 | PackGQA、tile padding、topology |
| payload 相同但 global loads 多 | page-table/address bookkeeping | index cache、divmod、load warp |
| DRAM SOL 高 | 真正带宽受限 | 减少 payload、增加 K/V 复用 |

### 13.6 第四步：用 SASS/SourceCounters 证明改动真的生效

历史上的有效信号：

- `STG.E → UTMASTG`：ragged O TMA 生效；
- `LDGSTS → UTMALDG.4D`：small-page TMA 生效；
- `UTCQMMA.2CTA`：2CTA UMMA 未被意外移除；
- `LDL/STL`：寄存器 spill；
- barrier samples 聚集到 O-full trywait：correction 同步瓶颈；
- page-table global loads 远多于 payload：重复索引而非 HBM payload 问题。

### 13.7 baseline 对齐清单

至少对齐：

```text
Q/K/V dtype 与 output dtype
causal 语义和 Q/K 真实长度
page size、layout、page compaction 是否在计时内
GQA ratio 与绝对 head 数
PackGQA on/off
prefix-cache hit ratio
num_splits
prepare/plan/compile 是否在 timed region
GPU、频率、timer 与统计方式
```

短 kernel 必须先 warm up；before/after 使用同一物理 GPU。历史 GPU0 未及时升频的异常表明，显示相同设备型号并不足以保证可比性。

## 14. 推荐的优化工作流

### 14.1 先建立最小解释模型

每轮只回答一个问题：

```text
瓶颈在哪一层？
    work geometry
    compute pipeline
    data movement/addressing
    numerical correction
    host launch
```

不要同时修改 scheduler、page layout 和 barrier；否则即使变快，也无法知道哪项生效，正确性出错时也无法隔离。

### 14.2 候选优化的最小闭环

```text
1. 用 NCU/SASS 提出可证伪的瓶颈假设
2. 只改一个主要机制
3. 跑最小 adversarial correctness cases
4. 清 JIT cache，确认 kernel identity
5. 快速 sweep 看收益覆盖面
6. 对边界/最差 shape 做 paired NCU
7. 检查组合 fast path 是否仍启用
8. 通过完整矩阵后再删除旧 fallback
```

### 14.3 提交说明至少回答四组问题

#### 正确性

- 改变了哪些支持组合或 invariant？
- 是否覆盖 ragged、partial cluster、odd K-block、shuffled page 和 page0=NaN？
- contiguous/paged 是否一致，最差误差是多少？

#### 性能

- baseline、timer、GPU/clock 和 timed region 是什么？
- 提升覆盖多少 shape，最差退化多少？
- grid/cluster/wave 如何变化？
- counter 是否支持瓶颈解释？

#### 可部署性

- clean wheel / clean cwd / empty `PYTHONPATH` 是否通过？
- 实际 import path 与依赖版本是什么？
- 是否新增 compile-key 维度？
- prepared/CUDA Graph 路径是否仍有效？

#### 可维护性

- 能否删除旧 fallback，而不是永久保留双路径？
- cluster/tile/phase invariant 是否有 assert？
- 注释是否解释“为什么”，尤其是 dummy handshake 和页/tile 换算？
- benchmark shape catalog 与环境是否可追踪？

## 15. 当前风险与后续优先级

### P0：safe tail-page fix

当前 HEAD 对 invalid Paged-KV tail 仍可能使用 physical page0。优先工作：

1. 把 `0d56d09f` / `8bbeb01d` 的思路适配到当前 HEAD；
2. small-page TMA 与 PagedKVManager 两条路径都覆盖；
3. 增加 page0=NaN、CTA1 partial-half、shuffled page-table 回归；
4. 明确 `seqlen_k=0` 的支持边界；
5. 再跑 page16/32/64/128/256 全矩阵。

### P1：版本化性能资产

大量关键证据仍位于本地 `output/` 和独立优化目录。建议版本化：

- shape catalog；
- benchmark 命令和 timer；
- commit/wheel/dependency/GPU 版本；
- 汇总 CSV/Markdown；
- 大型 `.ncu-rep` 可不进 Git，但保存生成命令和摘要 counter。

### P1：固化依赖组合

提供明确的 CUTLASS DSL / Quack / FA4 wheel 安装入口或 lockfile。依赖不兼容时应输出可诊断错误，而不是静默让 `has_fa4_hd256=False`。

### P2：继续观察真实服务 shape

重点监控：

- prefix-cache 命中率和 Q/K 比；
- short-Q/very-long-K/sub-wave 分布；
- page allocator 是否保证所有已引用页有限；
- CUDA Graph replay 的 metadata buffer 更新；
- cold compile、prepare 和 warm run 的独立延迟。

## 16. 演进索引：只用于定位代码

### 16.1 原型线 `mudi_dev_fp8_hd256`

| 提交 | 作用 |
| --- | --- |
| `5a292fab` | 初始 Blackwell FP8 HD256 attention |
| `dbb8fe45` | single-seq varlen 临时 route-to-dense |
| `487d3f3e` | ragged O TMA-store |
| `06a506be` | vendor/patch，删除本地源码注入 |
| `6911ed40` | 删除 dense 绕路，native varlen-only |
| `45d1161b` | varlen CLC+LPT 与寄存器拓扑 |
| `13393859` | 自包含 correctness/performance benchmark |

### 16.2 产品线 `fa4_mudi`

| 提交 | 作用 |
| --- | --- |
| `80774d20` | 最小化 vendored HD256 1CTA kernel |
| `fe0a5c5c` | page128/256 multi-tile-per-page TMA |
| `083776d8` | 历史 1CTA sQ/sO alias、KV stage5 |
| `8d8a4d38` | FP8 split-P 调优 |
| `1a10ace9` | 2CTA causal logical tile 修复 |
| `0a60e703` | 历史 1CTA/2CTA shape dispatch |
| `8afe6eb6` | 专用 SM103 2CTA CLC+LPT kernel |
| `45e9bda1` | 2CTA CLC/prefix-cache/cluster geometry 修复 |
| `60e36fc3` | 恢复 CLC KPP，修跨 work phase |
| `be1c2f0c` | CUTLASS DSL 4.5.1、2CTA-only 收敛 |
| `526b741e` | page16/32/64 page-specialized TMA |
| `1eab0af9` | prepared launcher |
| `68c6014e` | CLC-only，删除环境分叉 |
| `4f1d748a` | PackGQA + K/V page-ID cache |
| `0381108a` | graph-safe native page128 vLLM path |
| `0d56d09f` / `8bbeb01d` | safe tail-page side-branch fix，未进入 HEAD |

## 17. 代码与证据位置

### 当前代码

```text
python/atrex/api/flash_attn_hd256_cute.py
src/cutedsl/interface.py
src/cutedsl/flash_fwd_hd256_2cta_sm103.py
op_test/test_flash_attn_hd256_cute.py
```

### 关键报告

```text
output/fa4_hd256_2cta_prefix_cache_30pct/final_results.md
output/fa4_hd256_2cta_sm103_wheel_ncu/final_reproduction.md
output/fa4_prefill_ncu_20260717/summary.md
output/fa4_hd256_small_pages_20260720/summary.md
output/test_csv_fa4_20260721/report.md
output/page128_fa4_history_compare_20260722/report.md
output/fa4_dev1494_benchmark_20260724/DIRECT_PUBLIC_LOCKED_DEV1494_VS_ATREX_FP8_PREFILL.md
output/fa4_dev1503_vs_atrex_page128_20260724/FA4_DEV1503_VS_ATREX_FP8_PREFILL_COMPARISON.md
kernel_opt_fa4_hd256_fp8_small_pages/profiles/
```

## 18. 最终可复用经验

1. **先优化问题定义。** 专用支持面、真实 varlen ABI 和单一产品 topology 会放大后续所有 kernel 优化的价值。
2. **把调度单位说清楚。** 2CTA 的 work、causal bound 和 CLC worker 都以 logical cluster 为单位。
3. **把 persistent state 当状态机。** KPP slot、barrier phase 和 generation 可能跨 work 存活，不能在局部循环任意归零。
4. **把 page、tile、页内 offset 分层。** page256 与 page16 分别覆盖“一页多 tile”和“一 tile 多页”两个方向。
5. **masked load 也要物理安全。** 后续 mask 不能为读取未初始化 FP8 NaN 兜底。
6. **优先看 work geometry。** 74→75 个 2CTA clusters 的 wave cliff 足以压过许多指令级优化。
7. **优化组合需要重新验证 fast path。** PackGQA、KPP、CLC、TMA、shared-memory alias 各自有效，不代表组合后仍进入相同代码路径。
8. **资源 knob 不单调。** stage、split point、offset、threshold 都存在同步和 occupancy 的反作用。
9. **Host 路径也是产品性能。** compile、prepare、run 和 CUDA Graph 必须分层，动态长度不应触发 host dispatch。
10. **用 profiler 证明机制，而不只证明变快。** wall time 给出结果，grid/SASS/counter 才能说明为什么以及是否可迁移。
