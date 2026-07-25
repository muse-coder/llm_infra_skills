# B300 FA4 HD256 FP8 Causal Varlen Prefill 优化复盘

> 整理日期：2026-07-23
>
> 来源仓库：`/home/mudi/atrex`
>
> 主要范围：NVIDIA B300 / SM103、FlashAttention-4 CuTe DSL、`head_dim=256`、FP8 E4M3 Q/K/V、BF16 输出、causal varlen prefill、GQA16，以及 contiguous/paged KV。

## 1. 文档目的

本文不是一份通用 FlashAttention 教程，而是对 Atrex 中 B300 HD256 FP8 causal varlen prefill 优化历程的工程复盘，重点记录：

- 每个有效优化解决了什么瓶颈；
- 为什么它能生效；
- 它与其他优化能否组合；
- 曾经出现过哪些正确性、调度、数值和部署问题；
- 哪些看似合理的实验没有收益；
- 后续修改这条 kernel 时应该使用什么验证矩阵。

本文覆盖两条历史线：

1. `mudi_dev_fp8_hd256` 原型线：从 generic FA4 subclass、dense 绕路发展到真正 native varlen。
2. `fa4_mudi` 产品线：从自包含 1CTA kernel 发展到当前 2CTA CLC+LPT、Paged-KV TMA、PackGQA 和 prepared launcher。

这两条线不是一条连续的 Git ancestry。原型线提供了 ragged TMA、varlen scheduler 和寄存器调优经验；产品线重新整理了依赖、kernel 边界和部署方式。

## 2. 当前产品路径

截至 Atrex HEAD `0381108a`，优化路径收敛为：

```text
GPU                    B300 / SM103
输入                   FP8 E4M3 Q/K/V
输出                   BF16
attention              causal varlen forward
head_dim               256
GQA ratio              16
CTA topology           physical cluster (2, 1, 1)
scheduler              CLC + varlen mapping + causal LPT
Q/KV pipeline          q_stage=1 + K ping-pong
KV storage             contiguous 或 paged
paged page size        16 / 32 / 64 / 128 / 256
GQA organization       默认 PackGQA
split KV               num_splits=1
```

当前实现刻意限定范围，不应当把它当成通用 FA4 forward：

- 只支持 SM103，而不是任意 SM100 family GPU；
- 只支持 HD256；
- 只支持 causal varlen；
- 只支持 GQA16；
- 当前 kernel 固定 2CTA 和 CLC；
- backward、任意 mask、任意 GQA ratio、SplitKV 不属于该优化路径。

范围收窄本身是一项重要工程优化：它减少了 JIT specialization 分支、依赖面和不可验证组合，使关键 pipeline 可以围绕 B300 的真实 workload 激进调优。

## 3. 演进时间线

### 3.1 原型线：从能运行到真正 varlen

| 提交 | 主要变化 | 结论 |
| --- | --- | --- |
| `5a292fab` | 增加 Blackwell FP8 HD256 attention | 建立 generic FA4 subclass 和 vLLM 兼容入口 |
| `08c78765` | 扩展 shape/GQA 覆盖 | 暴露 dense、varlen、paged fallback 的边界 |
| `dbb8fe45` | single-seq varlen 路由到 dense | 临时消除 `is_varlen_q` 路径开销，但不是真正解决方案 |
| `487d3f3e` | ragged O TMA-store | 真正解决 multi-sequence varlen 输出地址与性能问题 |
| `06a506be` | vendor FA4、移除硬编码路径和运行时源码替换 | 解决可部署性和可复现性 |
| `6911ed40` | 删除 dense 和 route-to-dense | 统一为 native varlen-only 路径 |
| `45d1161b` | varlen 使用 CLC+LPT，并重分寄存器 | 让 native varlen 达到 dense 1CTA 性能上限附近 |
| `13393859` | 自包含 correctness/perf benchmark | 固化 18 个生产 shape 和 GQA-2 覆盖 |

### 3.2 产品线：从 1CTA 到 2CTA CLC+LPT

| 提交 | 主要变化 | 结论 |
| --- | --- | --- |
| `80774d20` | vendor 最小化 1CTA FP8 varlen kernel | 从外部工作树转为可安装 wheel 路径 |
| `fe0a5c5c` | paged KV multi-tile-per-page TMA | page128/256 不再走低效 gather |
| `083776d8` | 1CTA sQ/sO 复用，KV stage 4→5 | 用 shared-memory 生命周期换更深预取 |
| `8d8a4d38` | B300 FP8 P split 调优 | 说明 P producer/consumer overlap 需要按架构和路径重测 |
| `1a10ace9` | 修正 2CTA causal tile bounds，并在长序列启用 2CTA | 发现逻辑 cluster tile 与单 CTA tile 不能混用 |
| `0a60e703` | 按 shape 调整 1CTA/2CTA dispatch | 证明 CTA 选择高度依赖 batch、Q/K 比例和页布局 |
| `8afe6eb6` | 引入专用 SM103 2CTA CLC+LPT kernel | 建立当前产品 kernel 的主体 |
| `45e9bda1` | 修复 2CTA CLC、prefix-cache 和页内 tile 索引 | 解决 cluster geometry、page256 和 partial cluster 正确性 |
| `60e36fc3` | 恢复 CLC K ping-pong | 补齐跨 CLC work-tile 的 barrier phase 状态 |
| `be1c2f0c` | 支持 CUTLASS DSL 4.5.1，删除 1CTA 产品路径 | 收敛到 SM103 2CTA-only |
| `526b741e` | small-page Paged-KV TMA | page16/32/64 从 gather 转为 page-specialized TMA |
| `1eab0af9` | prepared launcher | 把 validation、dispatch 和 compile 移出 `run()` 热路径 |
| `68c6014e` | 固定使用 CLC | 删除环境变量和 static/CLC 双路径 |
| `4f1d748a` | PackGQA + K/V page ID 缓存 | 改善 cluster 波次、Q-tail padding 和 page-table 指令数 |
| `0381108a` | graph-safe native page128 vLLM kernel | 完成生产调用路径和 CUDA Graph 适配 |

## 4. 总体性能结论

不同报告使用的 shape、page size 和 baseline 不完全相同，不能把所有数字拼成一条单调曲线。可以确认的代表性结果包括：

- ragged O TMA-store：4×4096 multi-sequence 从 `303.97 us` 降到 `267.36 us`，约 `-12.0%`；
- varlen CLC+LPT + 寄存器重分配：8K case 从 `514.85 us` 降到 `463.62 us`；
- page128/256 TMA：32K prefix case 从约 `3200 us` 降到 `2980 us`，LSU 指令量约下降 13 倍；
- 1CTA sQ/sO alias：典型 prefix shapes 改善约 `1.0%–5.2%`；
- 30% prefix-cache：2CTA 比 1CTA 快约 `1.037x–1.083x`；
- page16/32/64 FP8 最终矩阵：60/60 正确且快于 nightly TRTLLM-gen，最低 speedup `1.090x`；
- page64 真实 19 shapes：全部进入“不慢于 TRT 5%”门槛，13/19 严格更快，几何平均 speedup `1.087x`；
- page128 历史 103 shapes：103/103 快于对比的 custom FA4，几何平均 speedup `1.389x`。

这些结果同时说明：不存在一个对所有 shape 都最优的简单 knob。最终收益来自 varlen 输出、调度、pipeline、页寻址、work geometry 和 host 热路径共同优化。

## 5. 优化点一：真正的 varlen O TMA-store

### 5.1 原始问题

packed varlen 输出的物理 layout 是：

```text
O: [total_q, num_heads, head_dim]

sequence 0: rows [0, q0)
sequence 1: rows [q0, q0+q1)
sequence 2: rows [q0+q1, q0+q1+q2)
```

不同 sequence 的起始 row 由 `cu_seqlens_q` 决定，可能不是 8、16 或 tile 大小的整数倍。普通 dense TMA descriptor 假设规则 batch stride，不能直接表达任意 ragged segment 的 runtime base pointer。

早期 SM100 varlen 路径因此关闭 TMA O-store，改用 register-to-global 的手工 epilogue。后果是：

- 生成大量 `STG.E.128`；
- epilogue 占用更多寄存器；
- 容易出现 local spill；
- single-sequence varlen 明显慢于 dense；
- 若简单强开 `use_tma_O`，非对齐 segment 会写到错误的 global rows。

### 5.2 有效方案

提交 `487d3f3e` 从 SM90 路径移植 ragged TMA 机制：

```text
create_ragged_tensor_for_tma(ragged_dim=0, ptr_shift=True)
    ↓
每条 sequence 在 runtime 偏移 descriptor base pointer
    ↓
tile 内仍使用规则 TMA store
```

关键不是给 TMA 增加一个不规则 stride，而是对每条 sequence 动态平移 base pointer，使局部 tile 仍表现为规则 tensor。

### 5.3 收益

代表性 multi-sequence case：

```text
4 × 4096, HQ=16
manual epilogue: 303.97 us
ragged TMA O:    267.36 us
改善:            12.0%
```

SASS 变化：

```text
STG.E epilogue: 32 → 0
UTMASTG:          0 → 4
local spill:      消除
```

### 5.4 注意事项

- 不能通过 `cu_seqlens_q.numel() == 2` 推断“永远可以 dense 化”；它只能说明 batch=1，不能替代正确的 varlen ABI。
- 必须覆盖非对齐 segment start，例如 `100/127/300`，只测 `[0, 4096, 8192]` 很容易漏掉地址 bug。
- TMA store 和手工 epilogue会改变 warp role、寄存器分配和 shared-memory 生命周期，不能只改一个 boolean。
- `out`、LSE 和任何 aux tensor 都要分别确认 ragged offset 语义，不能只验证 O。
- 正确性比较需要逐 sequence 检查，避免错误 segment 恰好写入另一条 sequence 后整体 norm 仍看起来合理。

### 5.5 推荐验证

```text
single sequence: [8192]
aligned multi:   [4096, 4096, 4096, 4096]
ragged multi:    [100, 127, 300, 4097]
tiny tail:       [1, 127, 128, 129]
GQA heads:       HQ16/HKV1, HQ32/HKV2
```

除数值阈值外，应检查 SASS 中是否真正出现 TMA store，以及是否重新出现 local load/store。

## 6. 优化点二：删除 route-to-dense，统一 native varlen

### 6.1 临时方案为什么有效

提交 `dbb8fe45` 对 batch=1 的 varlen 调用做了 shape-only 判断，并转发到 dense kernel：

```text
cu_seqlens_q.numel() == 2
    → 单 sequence
    → dense TMA O + static scheduler
```

它避免了当时 varlen 路径的两个开销：

- 手工 O epilogue；
- `SingleTileVarlenScheduler` 的映射和寄存器压力。

该方案不需要把 `cu_seqlens` 拷回 host，因此作为短期性能定位手段是合理的。

### 6.2 为什么最终删除

提交 `6911ed40` 删除 dense entry 和 route-to-dense，使所有调用都进入真实 `is_varlen_q=True` kernel。原因包括：

- 产品 ABI 本身是 varlen；
- dense/varlen 两套 kernel 增加 compile cache 和维护成本；
- batch=1 不能代表以后不会使用 `seqused_q`、paged KV 或其他 varlen 元数据；
- 绕路掩盖了 varlen scheduler 和 epilogue 的真正性能问题；
- 两条路径可能在 mask、LSE、empty sequence 和边界处理上逐渐漂移。

### 6.3 注意事项

- fast-path 可以用于定位性能上限，但不应代替修复主路径。
- 删除 fast-path 前要确认 native varlen 已获得同等的 TMA O、scheduler 和寄存器配置。
- 应保留“native varlen 对 dense ceiling”的基准，以防后续 scheduler 修改重新引入固定开销。
- host 侧不要为了选择 kernel 读取 device `cu_seqlens`；这会引入同步点，通常比节省的 kernel 时间更糟。

## 7. 优化点三：依赖与源码收敛

### 7.1 早期实现的问题

原型最初依赖：

```text
/home/mudi/flash-attention
    + sys.path 注入
    + inspect.getsource
    + 字符串替换
    + exec 到 /tmp
```

这种方式适合快速试验，但不适合作为产品实现：

- 换机器或 clean environment 即失效；
- 本地 working tree 的未提交修改会静默影响 Atrex；
- wheel 中不包含真实执行源码；
- traceback、source mapping 和 cache key 难以追踪；
- 文本替换对上游格式变化极其脆弱；
- 很难证明 benchmark 测的是安装包而不是本地路径。

### 7.2 两次收敛

第一次收敛是 `06a506be`：

- 把 FA4 作为 pinned submodule；
- 把 ragged-O 修改保存为 tracked patch；
- build 时以幂等方式应用 patch；
- wheel/sdist 显式携带所需 Python 源码。

第二次收敛是 `80774d20`：

- 只 vendor 上游没有的 HD256 1CTA kernel；
- interface 从三千多行裁剪到目标 forward 路径；
- 通用 helper 从安装的 `flash-attn-4` wheel 导入；
- benchmark 从 `/tmp`、清空 `PYTHONPATH` 后运行安装 wheel。

后续 `be1c2f0c` 又把产品路径收敛为 SM103 2CTA-only，并固定 CUTLASS DSL 4.5.1 兼容组合。

### 7.3 注意事项

- `flash-attn-4` 使用 namespace package；系统中若同时存在带根 `__init__.py` 的 FA2，可能遮蔽 `flash_attn/cute`。
- 当前 Atrex `setup.cfg` 不直接解析安装 `flash-attn-4`，部署环境需要单独安装验证过的 b20，并避免其陈旧依赖元数据覆盖 CUTLASS DSL 版本。
- CUTLASS DSL、Quack、FA4 helper 和 vendored kernel 是一个版本组合，不能只升级其中一个包后依赖单测侥幸通过。
- compile cache 可能掩盖源码/依赖变化；版本升级验证要清理 JIT cache。
- wheel 验证必须从源码树之外运行，并确认实际 import path。

### 7.4 推荐部署检查

```text
1. 从 clean venv 安装 wheel
2. cwd 切到 /tmp
3. 清空 PYTHONPATH
4. 打印 atrex、flash_attn.cute、cutlass 的实际 __file__
5. 清理 JIT cache 后编译
6. 跑至少一个 contiguous 和一个 shuffled paged case
7. 用 NCU 确认 kernel class、block size 和 cluster size
```

## 8. 优化点四：varlen CLC+LPT 与寄存器拓扑

### 8.1 static varlen scheduler 的问题

causal attention 的 Q tiles 工作量呈三角形：越靠后的 Q tile 能看到越多 K/V blocks。如果按自然顺序启动：

```text
轻 tile 先启动
重 tile 留在 grid 尾部
大部分 SM 提前空闲
少量重 tile 拉长 kernel tail
```

varlen 又会叠加 batch 间长度不均，静态调度更容易出现 long tail。

### 8.2 CLC+LPT 的分工

```text
LPT：把重 causal tile 映射到更早的 work ID
CLC：resident worker 完成当前任务后接管尚未启动的 grid work
```

CLC 不知道哪个 tile 更重，LPT 也不会动态偷任务。两者必须同时存在，但不能混为一个机制。

提交 `45d1161b` 将 native varlen 路径从 base static scheduler 切到 dense 路径已经验证过的 CLC+LPT 调度思想，并调整 warp 寄存器：

```text
correction warp: 80 → 72
other warp:      48 → 56
```

目的不是减少总寄存器，而是消除 varlen scheduler 在低配 warp 上的 stack spill。

### 8.3 收益

代表性 8K、HQ32/HKV2 case：

```text
static varlen: 514.85 us
CLC+LPT:       463.62 us
TRTLLM:        466.56 us
dense ceiling: 463.01 us
```

说明 native varlen 可以在保留真实 varlen 语义的同时达到 dense 1CTA 上限附近。

### 8.4 注意事项

- 寄存器必须按 warp role 分析；只看 kernel 总 registers/thread 不够。
- scheduler 增加的临时变量可能只让某类 warp spill，平均指标会掩盖问题。
- CLC 可能改善负载均衡但破坏 K/V L2 locality；不能仅凭 causal/varlen 就默认会快。
- CLC response valid 与 varlen work valid 是两层概念，取消到 padding grid ID 时仍需安全映射为 invalid work。
- multi-CTA 下 scheduler worker 是整个 cluster，不是单个 CTA。
- 后续固定为 CLC-only 后，不能继续保留未验证的 static fallback，否则编译组合和测试矩阵会翻倍。

### 8.5 推荐分析指标

```text
kernel duration
tail wave / grid utilization
eligible warps per scheduler
barrier stall
long scoreboard
L2 hit rate
local load/store
每种 warp role 的 register allocation
```

## 9. 优化点五：HD256 1CTA 的 K ping-pong

### 9.1 背景

HD256 1CTA 使用 `q_stage=1`，TMEM 刚好容纳：

```text
S: two 128-column slots
O: 256 columns
```

虽然只有一个 Q tile/O accumulator，但可以沿 K-block 方向对 S/P 做双缓冲：

```text
QK(i+1) 与 softmax/PV(i) overlap
```

这就是 K ping-pong；它与 `q_stage` 是两个不同维度：

```text
q_stage：并行保留多少个 Q/O tile
s_pp：沿 K 方向保留多少个 S/P slot
```

### 9.2 注意事项

- S/P slot 的选择必须基于全局 K iteration parity，而不是每个局部循环从 0 重新开始。
- softmax、P producer、PV consumer 和 correction warp 必须对同一 slot/phase 达成一致。
- split-P 会让一个 P slot 产生两次到达信号，barrier expected count 必须匹配。
- causal mask 前几次 iteration、无 mask 主循环和最后一次 iteration 经常使用不同代码段，三段都要检查 parity。
- 单 work-tile 正确不代表 persistent/CLC 下正确；后续 2CTA 路径正是在跨 work-tile 时暴露 phase bug。
- KPP 会增加 barrier 状态和代码复杂度，必须用 NCU 确认 overlap 真正减少 scoreboard stall，而不是只增加同步。

## 10. 优化点六：Paged-KV multi-tile-per-page TMA

### 10.1 原始问题

Paged KV 同时存在三种单位：

```text
n_block       kernel 的逻辑 KV tile 编号
logical page  page_table 的列编号
physical page page_table 中保存的实际页号
```

当 `tile_n=128` 时：

```text
page128: 1 page = 1 tile
page256: 1 page = 2 tiles
page64:  2 pages = 1 tile
page32:  4 pages = 1 tile
page16:  8 pages = 1 tile
```

早期实现隐含假设 `page_size == tile_n`：

- page-table 直接用 `n_block` 索引；
- 页内 tile 固定取 index 0；
- page256 的第二个 tile 会重复读取第一页内的前 128 tokens；
- page16/32/64 只能退回通用 gather/cp.async 路径。

### 10.2 page_size 大于等于 tile_n

提交 `fe0a5c5c` 对 page128/256 使用：

```text
tiles_per_page = page_size / tile_n
logical_page   = n_block / tiles_per_page
tile_in_page   = n_block % tiles_per_page
```

然后使用 page table 选择 physical page，再用 `tile_in_page` 选择页内 tile。

### 10.3 收益

32K prefix 的代表性数据：

```text
LSU traffic: 168.9M → 13.0M
tensor pipe: 67.7%  → 76.3%
latency:     3200us → 2980us
```

这说明原路径并非 HBM 带宽饱和，而是 gather 指令、地址计算和 load latency 限制。

### 10.4 注意事项

- page table 的单位永远是 page，不是 tile。
- TMA tensor 的页内维度可能包含多个 tile，不能固定取页内 index 0。
- page size 必须进入 compile key，否则不同 TMA descriptor 会错误复用同一个 CUBIN/cache entry。
- shuffled physical page table 是必要测试；identity table 会让忽略 page table 的错误实现也通过。
- page128 和 page256 都要测，因为前者无法覆盖 `tile_in_page` 错误。
- Q/K 不对称的 prefix-cache case 比 equal-Q/K 更容易暴露页边界问题。

## 11. 优化点七：small-page Paged-KV TMA

### 11.1 gather 路径的瓶颈

page16/32/64 早期走 `PagedKVManager + LDGSTS/cp.async`。V0 NCU 的代表性症状：

```text
global sectors/request ≈ 13.9，理想值约 4
L1 hit ≈ 1.13%
L2 hit ≈ 78.4%
DRAM read SOL ≈ 2.2%
long-scoreboard 为主要 stall
```

这不是带宽问题，而是大量不连续小 load、page-table/index 工作和低 locality 导致的 latency 问题。

### 11.2 最终方案

提交 `526b741e` 为 small page 建立 page-size-specialized TMA：

```text
FP8 tile N=128:
  page16 → 聚合 8 页
  page32 → 聚合 4 页
  page64 → 聚合 2 页

BF16 tile N=64:
  page16 → 聚合 4 页
  page32 → 聚合 2 页
  page64 → 原生单页 TMA
```

2CTA 内的工作分工保持为：

```text
K load：按 KV token/page 范围拆给 CTA0/CTA1
V load：按输出 D 范围拆给 CTA0/CTA1
QK/PV：仍由 UTCQMMA.2CTA 完成完整 HD256 运算
```

每个 CTA 使用自己的 TMA barrier；non-leader 完成后用 cluster async store 通知 leader。

### 11.3 收益

最终 FP8 page16/32/64 矩阵：

```text
60/60 finite
60/60 correct
60/60 faster than nightly TRTLLM-gen
speedup: 1.090x–1.338x
```

### 11.4 注意事项

- “small page”不应自动等价为“只能 gather”；只要 page size 是 compile-time specialization，就可以组织多个 TMA transaction。
- K 和 V 在 2CTA 中的分工不同，不能复用同一套 `source_tile` 公式。
- pages-per-tile 必须能被 CTA 分工正确覆盖；增加新 page size 前要验证 divisibility。
- 每个 CTA 的 barrier transaction bytes 必须与实际 TMA 数量一致。
- cluster remote notification 必须发生在本 CTA 的全部 TMA 完成之后。
- `seqlen_k` 尾部的部分页不能读取未初始化物理页，详见第 17 节。
- `seqlen < 2048` 固定开销占比较大，历史验收只保证功能正确，不保证一定超过 TRT。

## 12. 优化点八：sQ/sO shared-memory 生命周期复用

### 12.1 原理

1CTA HD256 kernel 中，Q 在主循环开始后很快被消费完；O 只在 epilogue 阶段写入 shared memory。两者生命周期不重叠，因此可令 sQ 和 sO 使用同一块 shared memory：

```text
时间轴：
load/use Q ─────────┐
                    └──── 空闲 ──── write/store O

空间：
[          sQ / sO alias region          ]
```

提交 `083776d8` 释放约 32KB shared memory，使 KV pipeline 从 4 stage 增加到 5 stage。

### 12.2 收益

三个 B=1 prefix case：

```text
250.2 / 801.7 / 2980 us
  ↓
237.1 / 776.1 / 2950 us

改善约 5.2% / 3.2% / 1.0%
```

NCU 显示 long-scoreboard 下降、tensor pipe active 上升。

### 12.3 注意事项

- 必须证明所有 warp 已不再读取 Q，不能只看 MMA warp 的控制流。
- epilogue 提前、persistent loop 或 aux output 都可能延长 sQ/sO 生命周期。
- shared-memory alias 会改变 layout address 和 barrier storage 排布，要重新检查对齐。
- 该优化在 1CTA 路径有效，但当前 2CTA CLC kernel 显式禁止 `overlap_sO_sQ`；原因是 CLC/persistent 生命周期和 cluster pipeline 不满足同样的不重叠证明。
- 这是典型的“历史最优优化无法直接叠加到新拓扑”的案例。

## 13. 优化点九：KV pipeline stage 调优

### 13.1 实验结果

2CTA CLC 路径在 shared-memory 上限允许的情况下，将 KV stage 从 4 增至 5：

```text
代表性改善：0.8%–2.4%
tensor pipe active 上升
eligible warps 上升
barrier stall 下降
occupancy 不变，仍为每 SM 一个 CTA
```

其他 stage 数并不更好：

```text
3 stage：2K/16K/64K 约退化 8%/26%/24%
6 stage：B=1 基本无改善，B=4 略慢于 5 stage
```

### 13.2 注意事项

- 更多 stage 只有在不降低 residency 且确实存在 load latency 时才可能有效。
- shared memory 增长但 occupancy 不变，不代表一定有收益；barrier、producer distance 和 cache pressure 仍可能恶化。
- 单一长序列和 irregular B=4 对 stage depth 的敏感度不同，两类都要测。
- 必须确认测试 kernel 的动态 shared-memory bytes 真正发生变化，避免 JIT cache 复用旧 specialization。
- stage 调优应以 NCU 的 long-scoreboard、barrier、tensor active 为依据，而不是仅看理论 pipeline depth。

## 14. 优化点十：FP8 softmax 的 P scaling 与动态范围

### 14.1 为什么要缩放 P

softmax 概率 P 写入 FP8 E4M3 前容易下溢。kernel 会在 cast 前乘以 `2^max_offset`，PV 之后再做对应反缩放。

但 online softmax 的 running row max 可能落后真实 max。设：

```text
max_offset         P 的主动放大指数
rescale_threshold  running max 允许滞后的 log2 阈值
```

为了不超过 E4M3 最大有限值 448，需要保持近似约束：

```text
max_offset + rescale_threshold <= 8
```

当前 FP8 配置为：

```text
max_offset = 4
rescale_threshold = 4
```

早期使用更大的 offset 时，P 可能达到约 `2^12`，发生 E4M3 saturation。

### 14.2 注意事项

- softmax P 路径和最终 O normalization 必须使用同一个 offset；只改一处会产生整体 scale 错误。
- saturation 不一定产生 NaN，可能只是精度静默下降，因此必须比较 rel-L1/cosine，而不只是 finite。
- underflow 与 saturation 需要同时看；减小 offset 能防 saturation，但会增加小概率下溢。
- correction threshold 与 barrier 频率耦合，数值 knob 同时也是性能 knob。
- TRTLLM 的 FP8 output 在某些模型尺度输入上会接近全零，不能作为正确性 golden；历史报告使用 BF16 output 做正确性门槛。

### 14.3 失败实验：减少 correction 次数

曾尝试把 `(max_offset, threshold)` 从 `(4,4)` 改为 `(2,6)`，并只在 `should_rescale=True` 时等待 O-full barrier。

理论上这能降低 correction 同步频率，但实际中 `should_rescale` 是 warp-wide 条件：32 行中只要有一行需要 rescale，整个 warp 就执行 correction。结果它几乎每轮都为真。

实测改善只有：

```text
shape 0: -0.13%
shape 3: -0.02%
shape 4: -0.01%
```

属于性能中性，最终回退。

经验：优化 branch predicate 前必须统计 predicate 的实际命中率，不能只看源码中存在 `if` 就假设大多数轮次能跳过。

## 15. 优化点十一：split-P arrival 点

### 15.1 原理

softmax warp 不必等整个 P tile 写完才通知 PV MMA。可以先写一部分列并发出第一次 arrival，让 PV 提前启动，再写剩余 P：

```text
write P[0:split]
signal partial P ready
PV begins
write P[split:end]
signal full P ready
```

### 15.2 实验结论

- 1CTA B300 特定形状曾测试 50% split，部分区间中性到略快；
- 2CTA 最终复测中，75%→50% 只有约 `-0.3%～+0.1%`；
- 25% 导致 SM launch failure；
- 当前 2CTA 产品值保持 75%。

### 15.3 注意事项

- split 太早时，PV 可能拿不到足够连续工作，反而增加等待和 barrier 压力。
- expected arrival count、P slot phase、last-split barrier 必须同时修改。
- 1CTA 的最优 split 不能直接移植到 2CTA；pair-UMMA 和 cluster barrier 改变消费者节奏。
- launch failure 表明该 knob 不只是性能参数，还可能触发硬件/DSL pipeline 的非法同步组合。
- 必须覆盖短 K、长 K、奇数 block count 和 causal mask 边界。

## 16. 优化点十二：2CTA causal bounds 和逻辑 tile 几何

### 16.1 核心错误

2CTA kernel 中每个 CTA 拥有 128 个 Q rows，但一个 scheduler work tile 由 CTA pair 共同处理 256 rows：

```text
CTA0: Q rows [0, 128)
CTA1: Q rows [128, 256)
logical cluster tile: [0, 256)
```

早期 `BlockInfo` 使用单 CTA 的 `cta_tiler[0]` 计算 causal/local K 上界，导致 CTA1 后半 Q rows 需要的较晚 K blocks 被静默裁掉。

正确逻辑是：

```text
logical_m_tile = cta_tiler[0] * cta_group_size
```

提交 `1a10ace9` 修复该问题。

### 16.2 注意事项

- 必须明确区分 `cta_tiler`、`mma_tiler_qk`、scheduler tile 和 cluster tile。
- causal bounds、local window、grid count、epilogue row ownership不一定使用同一个 M 尺寸。
- equal-Q/K 的整 tile 测试可能不暴露错误；partial final cluster 和非整 256 Q 长度更关键。
- CTA0 正确不代表 CTA1 正确，应针对两个 CTA 的 row range 分别比较。
- 2CTA 数学上不是两个独立 HD128 attention；两个 CTA 共同完成完整 HD256 reduction。

## 17. 优化点十三：1CTA/2CTA dispatch 与最终收敛

### 17.1 为什么简单长度阈值不够

最早按 `max_seqlen >= 64K` 选择 2CTA。后续测量发现性能还取决于：

- batch size；
- total Q，而非只有 max Q；
- average sequence length；
- Q/K 是否对称；
- prefix-cache hit ratio；
- page size；
- HQ/HKV 的绝对数量；
- PackGQA 后的 cluster 数；
- 是否正好跨过一个 GPU wave。

因此 `0a60e703` 一度引入复杂的 host-known shape policy。

### 17.2 为什么后来删除 shape dispatch

`be1c2f0c` 后产品范围收敛为 SM103 2CTA-only，后续又固定 CLC。这样做的好处：

- 不读取 device sequence lengths；
- 不引入 host sync；
- 减少 compile variants；
- 避免 1CTA/2CTA 两套 kernel 漂移；
- 让 vLLM/CUDA Graph 路径固定；
- 所有优化集中到一个可验证的产品 topology。

代价是部分 short-Q/long-K shape 上 2CTA 并非绝对最优。产品选择的是稳定支持边界和整体矩阵，而不是逐 shape 理论最优。

### 17.3 注意事项

- 如果未来重新引入 dispatch，只能使用 host 已知静态元数据，不能同步读取 `cu_seqlens`。
- kernel variant 必须进入 compile key，防止 1CTA/2CTA CUBIN 别名。
- dispatch benchmark 要将 compile/plan 排除在 timed region 外。
- 不能只测 B=1 equal-Q/K；prefix-cache 和 irregular B=4 是最容易改变结论的 workload。
- 复杂 shape policy 很快会过拟合当前 benchmark；应优先寻找可解释的 work-geometry 指标。

## 18. 优化点十四：PackGQA 与 cluster 波次

### 18.1 原始问题

GQA16 若不 pack，每个 Q head 独立调度，会重复扫描相同 KV head，并按 head 分别向 tile 边界取整。

B300 有 148 SM。2CTA cluster 同时占两个 SM，因此一波最多驻留约：

```text
148 / 2 = 74 clusters
```

历史真实 shapes 中出现过：

```text
unpacked: 80 clusters
```

虽然只超过 74 六个 cluster，却必须启动第二波；第二波利用率很低，尾部成本很大。PackGQA 后可降为约 68 clusters，重新落入单波。

### 18.2 PackGQA 的收益来源

- 同一 KV head 的多个 Q heads 折入 packed M；
- 降低 scheduler work 数；
- 减少每个 Q head 单独向 tile 边界取整的 padding；
- 在 CTA 内复用 K/V；
- 改善 cluster wave 几何。

### 18.3 与 KPP 的组合坑

早期 KPP predicate 中包含 `not self.pack_gqa`，导致启用 PackGQA 后虽然 grid 变小，却退回旧的 S/P/O pipeline。结果某些 shapes work 数减少但 kernel 反而变慢。

最终方案删除匹配路径中的 PackGQA 排除，使 PackGQA 与 KPP 同时成立。

### 18.4 注意事项

- PackGQA 的收益不能只按 K/V 复用解释，cluster wave cliff 往往更重要。
- 必须确认启用 PackGQA 后没有静默切换到慢 pipeline。
- packed M 会改变 causal indexing、Q row 到 head/token 的映射和 epilogue layout。
- HQ16/HKV1 与 HQ32/HKV2 的 GQA ratio 相同，但绝对 head 数会改变并行度和 wave 数，性能可完全不同。
- 需要记录物理 CTA 数、cluster 数和相对 74-cluster 容量的波次数，而不只是 tensor shape。

## 19. 优化点十五：2CTA CLC 的物理 cluster 描述

### 19.1 错误表现

上游 varlen scheduler 会把 flattened CTA index 映射回 cluster tile，但其 CLC problem descriptor 最初仍声明：

```text
cluster_shape_m = 1
```

实际 launch 却是：

```text
physical cluster_shape_m = 2
```

结果 CLC 可能给 CTA0 和 CTA1 分配不同 dynamic work。两个 CTA 随后仍通过 2CTA UMMA/barrier 协作，导致 partial final Q cluster 上的数据错乱或 hang。

### 19.2 修复

提交 `45e9bda1` 增加 2CTA 专用 varlen scheduler descriptor：

```text
CLC problem cluster shape = (params.cluster_shape_m, 1, 1)
```

使 CLC 每次发放一个 CTA pair 共享的 work item。继承的 coordinate map 再按 `cluster_shape_m` 将 CTA index 归一为逻辑 cluster tile。

### 19.3 注意事项

- launch cluster shape、CLC problem shape、scheduler coordinate divisor 必须三者一致。
- CLC 的最小 worker 是 cluster，不是 CTA。
- 两个 CTA 必须消费同一个 CLC response；不能各自发请求、各自 advance scheduler。
- 只测完整 cluster 数无法覆盖该问题，必须测试 partial final cluster。
- 推荐 Q 长度包含 `128±1`、`256±1`、`1408` 等不整除 cluster tile 的值。
- 加入断言：`cluster_shape_m == cta_group_size`，比依赖调用方约定更安全。

## 20. 优化点十六：CLC 下 K ping-pong 的跨 work-tile phase

### 20.1 为什么 static 正确、CLC 错误

static kernel 中一个 cluster 通常只处理一个 work tile，pipeline state 随 kernel block 结束销毁。

CLC persistent cluster 会连续处理多个 work tile：

```text
work A: K blocks = odd
    ↓
work B: K blocks = ...
```

若每个 work 都假设 S/P ping-pong 从 slot 0、phase 0 开始，work A 的奇数 K-block 会让 work B 与真实 barrier generation 错一位。

### 20.2 第一阶段处理

`45e9bda1` 为保证正确性，曾在 CLC 下暂时关闭 KPP，回到已验证的单 stage S/P/O pipeline。这是合理的修复顺序：

```text
先隔离正确性问题
确认 CLC geometry 和 paged index 正确
再单独恢复性能 fast path
```

### 20.3 最终修复

`60e36fc3` 恢复 CLC KPP，关键措施：

1. `kpp_iter_global` 跨 work-tile 保留；
2. phase 显式限制为一位：`(iter / slots) & 1`；
3. P phase 使用对应异或关系；
4. 当前 work 的 K-block 数为奇数时，补一个 dummy slot handshake；
5. dummy 只闭合 barrier epoch，不发新的 QK/PV MMA；
6. 下一 work 总从完整的双 slot barrier 周期开始。

### 20.4 注意事项

- pipeline phase 是代数状态，不是“每个 for-loop 从零开始”的局部变量。
- producer 和 consumer 都要对 dummy slot 做匹配的 commit/wait/release。
- 只修 MMA 侧或只修 softmax 侧都会产生死锁。
- 奇偶 K-block 是必要测试维度；仅测试 2K、4K 等整齐长度可能全是偶数。
- CLC 必须覆盖同一 resident cluster 连续取得多个 work 的情况；单 work grid 无法验证。
- hang、CUDA 912、随机精度错都可能是 phase bug 的表现，不能只按数值 bug 排查。

## 21. 优化点十七：page-table index 缓存

### 21.1 NCU 证据

page64 真实 shape 的对比显示：

```text
Atrex DRAM bytes ≈ TRT DRAM bytes
Atrex global-load instructions: 228,720
TRT global-load instructions:     2,640
```

两者读取的 K/V payload 相近，差异主要来自 page-table/address bookkeeping，而不是多读了 HBM payload。

### 21.2 重复来源

一个 KV128 tile 包含两个 page64：

```text
K load：CTA0/CTA1 各取一个 page ID，共 2 次
V load：两个 CTA 再取 page0/page1，共 4 次
合计：每 cluster、每 KV tile 约 6 次 page-table load
```

K 和 V 使用相同 physical page IDs，没有必要重复查询。

### 21.3 优化方案

`4f1d748a` 将当前逻辑 KV tile 的 page IDs 先读入 load-warp registers，并在 K/V issue 间复用：

```text
load page ID tuple once
    ├─ issue K TMA
    └─ issue V TMA
```

最终 global loads 下降约 32.8%，使最差真实 shape 进入不慢于 TRT 5% 的验收范围。

### 21.4 注意事项

- tuple 长度取决于 page size，必须是 compile-time specialization。
- 不要为了共享 page IDs 新增 cluster shared-memory 同步；register-local 复用成本更低。
- K/V 的 page ID 相同，但 source tile/CTA 分工不同；只缓存索引，不缓存完整 address 计算结果更稳妥。
- partial tail 的无效 tuple entry 仍必须指向安全的有限页，不能默认 physical page 0 安全。
- 指令数减少不保证 wall time 下降；历史 V3 明确要求同时观察 latency，否则应回退。

## 22. 正确性风险：Paged-KV tail 不能指向未初始化 page 0

### 22.1 问题机制

Paged-KV 的最后一个逻辑 tile/页经常不完整。TMA 或 CTA-local copy 可能仍会为被 mask 的列发出物理 load。

早期无效 entry 使用：

```text
physical_page = 0
```

但 physical page 0 不一定初始化，也可能包含 FP8 NaN。风险链路：

```text
masked tail 指向 page 0
    ↓
load 出 NaN
    ↓
进入 MMA accumulator
    ↓
后续 score mask 无法可靠消除 NaN 污染
```

对于 page64/tile128，CTA1 的 local predicate 还可能没有包含自身 64-token offset，使其在 partial half 上继续发 load。

### 22.2 side-branch 修复

提交 `0d56d09f` 和 `8bbeb01d` 的策略是：

```text
invalid tail entry
    → 指向最后一个有效 physical page
    → 再依赖 seqlen/causal mask 丢弃这些列
```

最后有效页至少属于本请求的已初始化 KV 数据，避免 NaN 源。

### 22.3 当前状态警告

截至 HEAD `0381108a`，上述两个 fix 位于 side branches，并未进入当前主线。当前 kernel 中仍可看到：

```python
page = page_table[page_idx] if is_valid else 0
physical_page = Int32(0)
```

因此这不是纯历史问题，而是需要继续确认和合入的已知风险。除非上层 allocator 能严格保证 physical page 0 始终初始化为有限值，否则不应依赖当前行为。

### 22.4 必测用例

- 把 physical page 0 主动填成 FP8 NaN；
- page table 使用随机排列，且有效序列不引用 page 0；
- page16/32/64/128/256；
- `seqlen_k` 分别落在 `page_size±1`、`tile_n±1`；
- CTA1-only partial half；
- contiguous 结果与 paged 结果 bit-exact 或达到严格容差；
- 连续多次运行检查非确定性。

## 23. 优化点十八：prepared launcher，把 dispatch 移出热路径

### 23.1 原始问题

普通 Python wrapper 每次调用可能重复执行：

- shape/dtype/layout validation；
- causal/local 参数归一；
- kernel variant 选择；
- compile key 组装与字典查询；
- fake tensor/JIT compile 检查；
- 输出和 aux metadata 准备。

单次 kernel 较短时，这些 CPU 工作会污染端到端延迟，并阻碍 CUDA Graph capture。

### 23.2 最终接口

`1eab0af9` 增加 `FlashAttentionHd256Prefill`：

```text
prepare()/plan():
    validation
    static dispatch
    compilation/cache lookup
    固化 callable 和参数结构

run():
    接收本次 tensor 指针/长度 tensor
    直接启动已准备 kernel
    不把 sequence lengths 读回 host
```

后续 `0381108a` 将 page128 vLLM path 做成 graph-safe native kernel 调用。

### 23.3 注意事项

- prepare 固化的必须是静态属性；动态 `seqused_k`、`cu_seqlens` 的值仍由 device tensor 提供。
- run 不能基于动态长度重新选择 CUBIN，否则破坏 graph safety。
- 输入 tensor 的 dtype、layout、page size、head 数变化时必须重新 prepare。
- benchmark 必须明确是否把 prepare/compile 计时；kernel 对比通常排除，服务冷启动评估则应单独报告。
- prepared callable 的输出 buffer 和 workspace 地址约束需要与 CUDA Graph 生命周期一致。
- compile key 必须包含所有影响 descriptor/layout 的属性，尤其是 page size、HKV、PackGQA 和 dtype。

## 24. 优化点十九：固定 CLC，删除环境变量调度分叉

### 24.1 历史状态

早期使用：

```text
FA_CLC=0/1
FA_HD256_2CTA=0/1
FA_HD256_USE_MAIN=0/1
```

这对 A/B 测试有用，但产品中会产生：

- 环境依赖的隐式行为；
- static/CLC 两套 compile variants；
- 测试可能误跑错误 kernel；
- wheel 与本地工作树结果难以对齐；
- CUDA Graph capture 前后环境变量改变却不生效。

### 24.2 最终方案

`68c6014e` 将目标 HD256 prefill 固定为 CLC，删除 interface 中的动态开关和 static fallback 测试。当前构造函数直接断言：

```text
SM103
2CTA
varlen Q
CLC
```

### 24.3 注意事项

- 调优开关可以保留在独立实验脚本/branch，不应长期留在生产 dispatch。
- 固定 CLC 前必须确认所有支持 shapes 的 correctness，而不只是 CLC 快的 shapes。
- 删除 fallback 后，错误输入应尽早 assert，而不是静默进入 generic kernel。
- 文档、测试名和 profiler kernel identity 都应同步更新，避免仍声称支持 1CTA/static。

## 25. 没有收益或被回退的实验

失败实验与成功实验同样重要。下面这些方向不能简单认定“永远无效”，但在当前 B300 HD256 2CTA CLC 路径和已测 workload 上没有形成产品收益。

| 实验 | 预期 | 实测 | 结论/注意事项 |
| --- | --- | --- | --- |
| single-seq route-to-dense | 绕过 varlen 开销 | 短期有效 | 只作为定位 ceiling，最终应修 native varlen |
| KV stage 3 | 降低 shared memory/barrier | 退化约 8%–26% | load latency 不能被充分隐藏 |
| KV stage 6 | 更深预取 | B=1 基本无收益，B=4 略慢 | 5 stage 是当前平衡点，不是越多越好 |
| split-P 75%→50% | 更早启动 PV | `-0.3%～+0.1%` | 基本中性，保留 75% |
| split-P 25% | 更激进 overlap | SM launch failure | barrier/consumer 条件可能非法 |
| conditional O-full wait | 无 rescale 时跳过等待 | 仅 `0.01%–0.13%` | warp-wide predicate 几乎始终为真 |
| threshold 4→6/8 | 减少 correction | boundary case 无改善 | 数值范围与 warp-wide rescale 抵消收益 |
| 只启用 PackGQA | 减少 cluster 数 | 某 shape 反而退化 | 因 KPP predicate 排除了 PackGQA，切到慢 pipeline |
| CLC 下直接复用 static KPP | 保留 overlap | odd K-block 后 phase 错 | 必须维护跨 work 的 global parity |
| invalid tail 指向 page 0 | 省掉边界寻址 | 可能读未初始化 FP8 NaN | 必须指向有限的有效页或显式 zero-fill |
| 仅按 64K 阈值选 2CTA | 简化 dispatch | 多个 batch/prefix shape 误判 | 性能取决于 work geometry，不只是 max length |
| 复杂 shape policy | 每 shape 取最优 CTA | 规则快速膨胀 | 容易过拟合，产品最终收敛为固定 2CTA |

## 26. 性能测量规范

### 26.1 必须区分三种时间

```text
compile time   CuTe DSL/JIT 生成 CUBIN
plan time      validation、dispatch、descriptor/callable 准备
run time       真正 kernel launch 与 GPU execution
```

kernel 优化比较应排除 compile 和 plan；服务冷启动评估应另表报告三者，不能混成一个数字。

### 26.2 NCU 与 do_bench 的职责

```text
triton.testing.do_bench:
    快速 sweep
    观察端到端调用 p50
    找候选和异常 shape

NCU gpu__time_duration.sum:
    最终 kernel duration
    确认 kernel identity
    分析 stall、pipe、cache、指令和 launch geometry
```

最终性能结论优先使用 NCU，并至少报告：

- kernel class/name；
- grid block 数；
- threads per block；
- cluster dimension；
- dynamic shared memory；
- registers/thread；
- capture 次数和中位数；
- GPU 和频率状态。

### 26.3 时钟与 GPU 选择

历史 wheel reproduction 中，GPU0 短 kernel 没有及时升频，产生明显异常值；最终改用稳定 2032MHz 的 GPU2 重测。

注意：

- 同一轮 before/after 必须使用同一物理 GPU；
- 短 kernel 要先 warm up；
- 检查每次 capture 的离散度；
- 出现 3%以上离群值应配对重测；
- 不要把显示名 `L20D` 误认为硬件不是 B300，应以 compute capability/SM103 为准。

### 26.4 baseline 必须对齐

至少对齐：

```text
Q/K/V dtype
output dtype
causal 语义
Q/K 实际长度
page size 和 layout
GQA ratio 与绝对 head 数
prefix-cache hit ratio
num_splits
plan/compaction 是否在 timed region
```

历史上 TRTLLM-gen 的 FP8 output 更快，但某些模型尺度下严重下溢，不能作为正确性 reference。正确性使用 BF16 output；FP8 output 最多作为 speed-only lower bound。

### 26.5 不要只报告最好 case

推荐至少分组报告：

```text
B=1 equal Q/K
B=1 prefix-cache 20%/30%/50%/70%
B=4 irregular
B=4 long-tail
HQ16/HKV1
HQ32/HKV2
page16/32/64/128/256
short K < 2K
medium 2K–64K
long 64K–1M
```

对于 2CTA，还要报告每个 shape 的 cluster 数和 wave 数。

## 27. 正确性验证矩阵

### 27.1 基础数值门槛

FP8 attention 不适合只用 `torch.testing.assert_close` 的默认逐元素阈值。历史常用组合：

```text
finite output
cosine > 0.99（实际稳定结果通常 > 0.999）
relative L1 < 0.08（实际产品结果远小于该上限）
relative L2 <= 0.05
max abs 作为辅助观察
```

同一 kernel 的 contiguous/paged 或 1CTA/2CTA 等价路径若理论运算顺序一致，可以要求 bit-exact；与独立 baseline 比较则使用统计阈值。

### 27.2 varlen 必测

- batch=1；
- batch=4 equal；
- batch=4 irregular；
- long-tail `[短, 短, 中, 超长]`；
- 非 8/16/128 对齐的 `cu_seqlens` segment start；
- Q 长度 0/1 的支持边界，若不支持则应明确 assert；
- max_seqlen 大于真实长度；
- `cu_seqlens` 与 `seqused` 两种元数据形式。

### 27.3 causal/cluster 必测

- Q rows 覆盖 CTA0 与 CTA1；
- partial final 2CTA cluster；
- K-block 数分别为奇数和偶数；
- 同一 persistent cluster 连续取得多个 CLC work；
- equal Q/K；
- Q<K 的 prefix-cache 右下角 causal 对齐；
- very short Q + very long K。

### 27.4 Paged-KV 必测

- page16/32/64/128/256；
- identity page table；
- shuffled page table；
- physical page 0 填 NaN；
- page size 小于、等于、大于 tile_n；
- partial page；
- partial tile；
- 一个 physical page 含多个 tiles；
- 一个 tile 聚合多个 physical pages；
- HQ16/HKV1 与 HQ32/HKV2；
- contiguous 与 paged output 比较。

### 27.5 pipeline 必测

- KPP 首轮、主循环、末轮；
- causal mask loop 和 no-mask loop；
- split-P first/full arrival；
- rescale 发生和不发生；
- odd K-block work 后跟另一个 CLC work；
- PackGQA on/off 的实验路径；
- 清理 JIT cache 后重编译。

## 28. NCU 诊断顺序

建议按证据逐层定位，避免直接改最显眼的源码循环。

### 28.1 第一步：确认测到目标 kernel

```text
kernel name = FlashAttentionForwardHd256_2CTA_Sm103
cluster X   = 2
SM arch     = 103
input dtype = FP8 E4M3
output      = BF16
```

如果 identity 不匹配，后续所有 counter 都没有意义。

### 28.2 第二步：看 launch/work geometry

```text
physical CTAs
clusters = CTAs / 2
waves = clusters / 74
PackGQA 前后 cluster 数
Q-tail padding
```

如果刚刚超过一波，优先解决 tile/PackGQA geometry，而不是先调指令级 pipeline。

### 28.3 第三步：判断主要瓶颈

| 症状 | 可能原因 | 优先方向 |
| --- | --- | --- |
| long scoreboard 高、DRAM SOL 低 | 小 load、地址依赖、预取不足 | Paged TMA、stage、page index cache |
| barrier 高 | 2CTA/KPP/correction 同步 | phase、arrival、减少不必要同步 |
| local ld/st 非零 | warp role 寄存器不足 | 重分寄存器、减少 scheduler 临时变量 |
| tensor active 低、grid 不足一波 | 并行度不足 | PackGQA geometry、CTA topology |
| L2 hit 低、payload 相同 | 工作集/locality 或 page bookkeeping | swizzle、page cache、调度顺序 |
| DRAM SOL 高 | 真正带宽受限 | 减少 payload、提高 K/V 复用 |

### 28.4 第四步：用 SASS/SourceCounters 定位

历史有效案例：

- `STG.E`→`UTMASTG` 证明 ragged O TMA 生效；
- `LDGSTS` 数量揭示 small-page gather；
- `UTCQMMA.2CTA` 确认 2CTA UMMA 未丢失；
- `STL/LDL` 确认寄存器 spill；
- barrier stall 聚集到 O-full trywait，定位 correction 同步；
- page-table global loads 远多于 payload，定位重复索引。

## 29. 后续修改的合入门槛

每个 kernel 优化提交至少应回答下面问题。

### 29.1 正确性

- 改变了哪些支持组合？
- 是否覆盖 ragged、partial cluster、odd K-block 和 shuffled pages？
- 是否主动测试 page0 NaN？
- contiguous/paged 是否一致？
- 精度阈值和最差 case 是什么？

### 29.2 性能

- 最强 baseline 是谁？
- 使用 do_bench 还是 NCU？
- compile/plan 是否排除？
- 提升覆盖多少 shape，最差退化多少？
- 是否报告 grid/cluster/wave？
- counter 是否支持提出的瓶颈解释？

### 29.3 可部署性

- wheel clean install 是否通过？
- import 是否来自预期路径？
- CUTLASS DSL/Quack/FA4 版本组合是什么？
- 是否新增 compile-key 维度？
- prepared/CUDA Graph 路径是否仍有效？
- 是否引入环境变量决定产品行为？

### 29.4 可维护性

- 是否能删除旧 fallback，而不是永久保留双路径？
- 关键 cluster/tile/phase invariant 是否有 assert？
- 注释是否解释“为什么”，而不只是重复代码？
- 是否留下可复现 benchmark 和 raw shape catalog？

## 30. 当前已知风险与建议优先级

### P0：合入或重新实现 safe tail-page fix

当前 HEAD 仍可能让无效 Paged-KV tail load 指向 physical page 0。建议：

1. 将 `0d56d09f`/`8bbeb01d` 的思路适配到当前 HEAD；
2. 增加 page0=NaN 回归；
3. 覆盖 small-page TMA 与 non-TMA 两条 load path；
4. 确认 last valid page 在 `seqlen_k=0` 的行为，若不支持空 KV 应提前 assert。

### P1：把 profiler shape catalog 和报告纳入可追踪资产

当前大量关键证据位于 `output/` 和独立优化目录，未必进入产品 Git 历史。建议至少版本化：

- shape catalog；
- benchmark 脚本；
- 环境版本；
- 汇总 CSV/Markdown；
- 不必提交大型 `.ncu-rep`，但应记录生成命令和摘要指标。

### P1：固化依赖安装

当前需要独立安装 `flash-attn-4 b20` 并避免其依赖覆盖 CUTLASS DSL 4.5.1。建议提供一个明确、可验证的安装入口或 lockfile，并在启动时输出兼容性错误而不是静默 `has_fa4_hd256=False`。

### P2：真实服务 workload 回归

prepared/graph-safe kernel 已接入，但仍应长期监控：

- vLLM prefix-cache 命中率分布；
- Q/K 比例；
- page allocator 是否保证页初始化；
- CUDA Graph replay 下 tensor 地址和长度 tensor 更新；
- cold compile 和 warm run 分离指标。

## 31. 最终经验总结

1. **先把 varlen 做真，再谈 fast path。** dense 绕路能给性能上限，但不能成为产品语义。
2. **2CTA 的调度单位永远是 cluster。** causal bounds、CLC descriptor 和 worker state 都必须使用逻辑 cluster 几何。
3. **persistent scheduler 会让状态跨 work 存活。** barrier phase、ping-pong parity 和寄存器状态不能在局部循环随意归零。
4. **Paged-KV 的 page、tile、页内 offset 必须分层。** page256 和 page16 分别覆盖“一页多 tile”和“一 tile 多页”两个方向。
5. **masked load 仍必须安全。** 无效列不能读取未初始化 FP8 NaN，再指望后续 mask 修复。
6. **FP8 数值 knob 也是同步 knob。** offset、threshold、correction 和 barrier 必须联合调优。
7. **更多 pipeline stage 不一定更快。** 只有 NCU 能判断是预取不足还是同步/资源已经占主导。
8. **work geometry 常比单条指令更重要。** 74→75 个 2CTA clusters 的 wave cliff 足以压过局部优化。
9. **优化不会自动组合。** PackGQA、KPP、CLC、sQ/sO alias 各自正确，不代表组合后仍走同一 fast path。
10. **产品路径应尽量单一。** 环境变量和 fallback 适合实验，最终应收敛到明确支持范围和固定 topology。
11. **benchmark 口径是实现的一部分。** GPU 频率、输出 dtype、plan 范围和实际 import path 都能改变结论。
12. **失败实验要保留原因。** “为什么没用”能防止下一轮在相同假设上重复消耗时间。

## 32. 主要证据位置

### Git 提交

```text
5a292fab  初始 Blackwell FP8 HD256
487d3f3e  ragged O TMA-store
06a506be  vendor FA4 / clean build
6911ed40  native varlen-only
45d1161b  varlen CLC+LPT + register topology
80774d20  minimal vendored 1CTA
fe0a5c5c  multi-tile-per-page TMA
083776d8  sQ/sO alias + KV stage 5
1a10ace9  corrected 2CTA causal bounds
8afe6eb6  SM103 2CTA CLC+LPT
45e9bda1  2CTA CLC/prefix-cache correctness
60e36fc3  restore CLC KPP with phase fix
be1c2f0c  CUTLASS DSL 4.5.1 / 2CTA-only
526b741e  small-page TMA
1eab0af9  prepared launcher
68c6014e  CLC-only
4f1d748a  PackGQA + cached page IDs
0d56d09f  safe tail-page fix side branch
8bbeb01d  safe tail-page fix side branch
0381108a  graph-safe page128 vLLM kernels
```

### Atrex 源码入口

```text
python/atrex/api/flash_attn_hd256_cute.py
src/cutedsl/interface.py
src/cutedsl/flash_fwd_hd256_2cta_sm103.py
op_test/test_flash_attn_hd256_cute.py
```

### 本地性能记录

```text
output/fa4_hd256_2cta_prefix_cache_30pct/final_results.md
output/fa4_hd256_2cta_sm103_wheel_ncu/final_reproduction.md
output/fa4_prefill_ncu_20260717/summary.md
output/fa4_hd256_small_pages_20260720/summary.md
output/test_csv_fa4_20260721/report.md
output/test_csv_fa4_20260721/report_hq32_hkv2_ncu.md
output/page128_fa4_history_compare_20260722/report.md
kernel_opt_fa4_hd256_fp8_small_pages/profiles/
```

这些本地报告中部分不属于 Atrex 当前提交历史。引用性能数字时，应同时保存对应 commit、wheel、环境和原始测量命令。

