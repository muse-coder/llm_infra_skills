# FlashInfer Blackwell Decode Attention：实现、Flash-Decoding、GQA 与 Paged KV

> 范围：2026-07-17 的本地 FlashInfer checkout。重点讨论标准 MHA/GQA decode，
> 并补充 MLA。本文严格区分“源码在仓库中”“host 选择逻辑可见”和“核心 kernel
> 仅以 cubin 发布”。Paged KV、NHD/HND 和 prefix cache 的系统侧背景见
> [`vLLM_Qwen35_PagedKV_NHD_HND_Prefill_Decode.md`](vLLM_Qwen35_PagedKV_NHD_HND_Prefill_Decode.md)。

## 目录

1. [核心结论](#1-核心结论)
2. [Decode 为什么缺少并行度](#2-decode-为什么缺少并行度)
3. [Flash-Decoding 与 KV split](#3-flash-decoding-与-kv-split)
4. [GQA 下如何拆分](#4-gqa-下如何拆分)
5. [FlashInfer decode 实现谱系](#5-flashinfer-decode-实现谱系)
6. [CuTe DSL GQA Decode](#6-cute-dsl-gqa-decode)
7. [TRTLLM-Gen Decode](#7-trtllm-gen-decode)
8. [CuTe DSL 与 TRTLLM-Gen 对比](#8-cute-dsl-与-trtllm-gen-对比)
9. [Page size 的含义与影响](#9-page-size-的含义与影响)
10. [源码索引](#10-源码索引)

---

## 1. 核心结论

1. **Flash-Decoding 不是某个特定 kernel 名。** 它是一类 decode 并行策略：Q 很短时，
   沿长 KV reduction 维切分，让多个 CTA 合作计算同一输出。
2. **`kv_splits` 是每条 request、每个 KV head 的有效 KV token 范围被切成的份数。**
   Page 是存储单位，split 是计算单位，两者正交。
3. **GQA 中多个 Q heads 共享一个 KV head，但 softmax 状态仍按 Q head 独立。** 高效
   kernel 可加载一次 K/V tile，同时计算一组 Q heads。
4. **CuTe DSL GQA 与 TRTLLM-Gen 都使用 KV 维并行，但实现不同。** 前者叫
   `kv_splits`，后者叫 `numCtasPerSeqKv` / `MultiCtasKvMode`。
5. **CuTe DSL GQA kernel 源码完整可见。** TRTLLM-Gen 的 launcher、heuristic、TMA
   descriptor 和 reduction 选择可见，但主 attention kernel 通过预编译 cubin 提供。
6. **B200 是 SM100，B300 是 SM103。** TRTLLM-Gen 明确选择 SM100/SM103 cubin；
   CuTe DSL 面向 Blackwell CC 10.x，标准 GQA 大量配置以 SM100 为基准，MLA 中存在
   明确的 SM103 分支。

---

## 2. Decode 为什么缺少并行度

Attention：

```text
S = Q K^T
P = softmax(S)
O = P V

Q: [M, D]
K: [N, D]
V: [N, Dv]
```

### Prefill

Prompt prefill 中 `M` 较大，例如：

```text
M = 2048, N = 8192
S: [2048, 8192]
```

不同 Q rows 的输出独立，可以把 Q 切成很多 tile：

```text
Q[0:128]     -> CTA 0
Q[128:256]   -> CTA 1
Q[256:384]   -> CTA 2
...
```

### Decode

普通 autoregressive decode 每个 request 只有一个新 token：

```text
M = 1, N = 8192
S: [1, 8192]
```

Q 维没有足够多的独立 tile。若 `batch=1, Hkv=8`，并且一个 CTA 融合处理一个 KV
head 对应的 grouped Q heads，基础 grid 可能只有约 8 个任务；B200 有 148 个 SM，
大量 SM 会空闲。

“Prefill 沿 Q 并行、Decode 改成沿 KV 并行”只是便于记忆的缩写。更准确的说法是：

> Prefill 依靠大量独立 Q tiles 获得 grid-level 并行；decode 缺少 Q tiles，因此额外
> 切分 attention 的 KV reduction 维度。

---

## 3. Flash-Decoding 与 KV split

设：

```text
kv_len = 8192
kv_splits = 8
```

同一个 Q 被交给 8 个 CTA，每个 CTA 扫描约 1024 个 KV tokens：

```text
Q
├─ CTA 0: KV [0,    1024)
├─ CTA 1: KV [1024, 2048)
├─ CTA 2: KV [2048, 3072)
├─ ...
└─ CTA 7: KV [7168, 8192)
```

### 3.1 每个 split 的局部状态

对 split `i`：

```text
m_i = max(score_j)
l_i = sum_j exp(score_j - m_i)
u_i = sum_j exp(score_j - m_i) * V_j
```

`m_i` 是局部最大值，`l_i` 是局部分母，`u_i` 是未除分母的局部输出。

### 3.2 数值稳定地合并

```text
m = max_i(m_i)
l = sum_i exp(m_i - m) * l_i
u = sum_i exp(m_i - m) * u_i
O = u / l
```

因此 partial output 不能直接相加，必须同时合并 max、sum 和 output。

### 3.3 并行度与代价

粗略 grid 大小：

```text
CTA count ~= batch * Hkv * grouped_head_tiles * prediction_tiles * kv_splits
```

增加 `kv_splits`：

- 增加 CTA 数，提高小 batch/少 KV heads 时的 SM 利用率；
- 缩短每个 CTA 的 KV mainloop；
- 但增加 Q 重复读取、partial workspace、reduction、同步和 page-table 开销。

所以 `kv_splits` 不是越大越好。目标是填满 GPU，同时保证每个 split 有足够工作量。

---

## 4. GQA 下如何拆分

设：

```text
Hq = 32
Hkv = 8
group_size = Hq / Hkv = 4
```

映射：

```text
KV head 0 <- Q heads  0, 1, 2, 3
KV head 1 <- Q heads  4, 5, 6, 7
...
KV head 7 <- Q heads 28,29,30,31
```

对 Q head `hq`：

```text
hk = floor(hq / group_size)
O[hq] = softmax(Q[hq] K[hk]^T) V[hk]
```

同组 Q heads 共用 K/V，但 score、softmax 和 output 各自独立。一个高效 CTA 可以加载
一次 KV tile，同时计算多个 grouped Q heads；每个 KV split 仍需为每个 Q head 保存独立：

```text
m[split, q_head]
l[split, q_head]
u[split, q_head, Dv]
```

例如 `grouped_head_tile=4, prediction_tile=1, batch=1, Hkv=8, kv_splits=8`：

```text
grouped_head_tiles = ceil(4 / 4) = 1
CTA count ~= 8 * 1 * 8 * 1 * 1 = 64
```

不切 KV 时只有约 8 个 CTA。GQA/MQA 的 KV heads 少于 Q heads，因此比 MHA 更容易
缺少基础 grid 并行度，也更依赖 KV split；但 group 很大时单 CTA 工作量也更大，最终
split 数仍需结合 batch、KV 长度和 tile 配置选择。

---

## 5. FlashInfer decode 实现谱系

| 实现 | 核心源码 | B200/SM100 | B300/SM103 | KV 维并行名称 | 开源边界 |
|---|---|---:|---:|---|---|
| 原生 CUDA-core decode | `include/flashinfer/attention/decode.cuh` | 支持 | 支持 | `partition_kv` | 完整源码 |
| FA2 tensor-core decode | 复用 FA2 prefill，Q len=1 | 支持 | 支持 | split-KV | 完整源码 |
| FA3 decode | Hopper kernel | 否 | 否 | split-KV | 完整源码，SM90a |
| CuTe DSL GQA decode | `flashinfer/cute_dsl/attention/gqa_decode*.py` | 原生 | 支持 | `kv_splits` | 完整源码 |
| XQA | `csrc/xqa/` | 支持 | 支持 | multi-block/scratch | 完整源码，非 CC10 专项 |
| TRTLLM-Gen | cubin + 开源 host 选择逻辑 | 原生 | 原生 | `MultiCtasKvMode` | 主 kernel 为 cubin |
| cuDNN SDPA decode | cuDNN/SM100 cubin | SM100 | 未确认专用 cubin | Lean Attention 风格 | 主 kernel 不可见 |

MLA 另有独立家族：旧 CUDA-core/SM80 版本、CuTe DSL Blackwell 版本、TRTLLM-Gen
MLA 和 SM120 XQA/sparse MLA。CuTe DSL MLA 中存在 SM100 与 SM103 的 TMEM load/reduce
分支，是当前仓库里 B300 架构差异最明确的开放源码之一。

---

## 6. CuTe DSL GQA Decode

### 6.1 Kernel 与编译

核心源码：

```text
flashinfer/cute_dsl/attention/gqa_decode.py
flashinfer/cute_dsl/attention/gqa_decode_paged.py
flashinfer/cute_dsl/attention/wrappers/batch_decode.py
```

它针对当前 GPU JIT 编译，TMA、TMEM、MMA、warp 分工、online softmax、correction 和
reduction 都可阅读修改。

### 6.2 KV layout

Kernel 统一消费逻辑 NHD：

```text
K/V logical shape = [num_pages, page_size, Hkv, D]
```

公开 wrapper 接受：

```text
NHD: [P, N, Hkv, D]
HND: [P, Hkv, N, D]
```

HND 通过 `transpose(-3, -2)` 形成逻辑 `[P,N,Hkv,D]` view，不复制数据。前三维
stride 动态，唯一硬要求是最后的 `D` 连续。

合并 K/V tensor：

```text
NHD: [P, 2, N, Hkv, D]
HND: [P, 2, Hkv, N, D]
```

### 6.3 Page table

```text
indptr:   [batch + 1]
indices:  [total logical pages]
seq_lens: [batch]
```

```text
table_offset = indptr[batch]
physical_page = indices[table_offset + logical_page]
```

### 6.4 Page size

硬限制：

```text
page_size in {8, 16, 32, 64}
head_dim % 64 == 0
Q/K/V dtype 相同
```

### 6.5 Split 与 reduction

Auto split 近似：

```text
base_grid = batch * Hkv * grouped_head_tiles * prediction_tiles
kv_splits ~= SM_count / base_grid
```

同时以约 256-token 粒度限制 split 数。Reduction：

```text
none:   kv_splits == 1，直接写输出
atomic: 支持的少量 split，原子/cluster 协作
kernel: 写 partial workspace，再运行 reduction kernel
```

---

## 7. TRTLLM-Gen Decode

### 7.1 开源边界

FlashInfer 仓库中可见：

- Python API 与 FFI launcher；
- SM100/SM103 compatibility 和 cubin metadata 选择；
- TMA descriptor 构造；
- `numCtasPerSeqKv` heuristic；
- static/persistent scheduler 选择；
- global/CGA/separate reduction 模式选择；
- 部分 reduction kernel。

不可见：

- `SwapsMmaAbForGeneration` 主 attention kernel；
- `KeepsMmaAbForGeneration` 主 attention kernel；
- 主 kernel 内部 MMA/TMA pipeline 和 warp specialization。

这些通过运行时下载的 cubin 提供。因此准确描述是“host 调度部分开源，核心 GPU
attention kernel 在当前仓库中不开源”，不能仅凭 host 源码断言 cubin 内每条指令。

### 7.2 SM100/SM103 选择

Runner 只接受 SM100/SM103；兼容规则：

```text
SM100 GPU -> SM100 或 SM100f kernel
SM103 GPU -> SM103 或 SM100f kernel
```

### 7.3 Multi-CTA KV

`MultiCtasKvMode` 属于 TRTLLM-Gen Generation kernels，包括标准 MHA/GQA decode、
speculative decode、MLA 和 sparse MLA；context/prefill 明确关闭该模式。

Generation 入口先允许 multi-block/static scheduler，随后计算：

```text
numCtasPerSeqKv
```

它受 KV 长度、attention window、KV tile、基础 CTA 数和 SM 数影响：

```text
numCtasPerSeqKv > 1 -> 多 CTA 扫描同一 KV sequence
numCtasPerSeqKv == 1 -> 关闭 multi-CTA，切 persistent scheduler
```

因此选择 `backend="trtllm-gen"` 不代表每次都会实际启动多个 KV CTAs。

### 7.4 Kernel variants 与 reduction

主要 generation variants：

```text
SwapsMmaAbForGeneration
KeepsMmaAbForGeneration
```

Reduction modes：

```text
GmemReduction
GmemReductionWithSeparateKernel
CgaSmemReduction
```

当前只有 `SwapsMmaAbForGeneration` 支持 CGA shared-memory reduction。大 `Dv` 或部分
MLA/KeepsMmaAb 场景使用 global-memory + separate reduction。

### 7.5 CGA

CGA（Cooperative Grid Array，在这里对应 CTA cluster）让 cluster 内 CTAs 使用
Distributed Shared Memory 和 cluster barrier 协作。对 KV split：

```text
CTA 0 partial m/l/u ┐
CTA 1 partial m/l/u ├─ cluster DSM reduction -> final O
CTA 2 partial m/l/u ┘
```

相比先写 global memory 再 reduction，可减少中间流量和额外 kernel；代价是 cluster
规模/shared-memory/occupancy 约束。当前 heuristic 只在 CTA 数不超过 16、variant 和
head dimension 等条件满足时选择 CGA，否则使用 global-memory 路径。

### 7.6 Tile heuristic

标准 GQA generation 比较 `tileSizeQ={128,64,32,16,8}`，模型包含：

```text
mainloop cost + reduction cost，之后乘以 wave 数
```

这是手写 cost model，不是实际运行所有候选 kernel 的 autotune。

### 7.7 Page size

基本要求：

```text
page_size 是 2 的幂
```

当前测试明确覆盖：

```text
fixed-page:       16, 32, 64
dynamic GQA page: 128, 256, 512, 1024
```

Dynamic page 路径要求 paged KV、非 sparse MLA、GQA ratio > 1、`Dqk == Dv`、
`page_size >= 128`。Host 接受 2 的幂不等于一定存在对应 cubin；找不到匹配 kernel
会报 `Missing TRTLLM-GEN kernel`。

---

## 8. CuTe DSL 与 TRTLLM-Gen 对比

| 维度 | CuTe DSL GQA | TRTLLM-Gen |
|---|---|---|
| 高层算法 | KV split / Flash-Decoding | KV split / Flash-Decoding |
| 核心参数 | `kv_splits` | `numCtasPerSeqKv` / `MultiCtasKvMode` |
| Kernel | CuTe DSL JIT | SM100/SM103 cubin |
| 可修改性 | 主 kernel 完整可改 | 只能直接改 host 调度与开放 reduction |
| Reduction | none/atomic/kernel | CGA/gmem/separate kernel |
| Layout | 逻辑 NHD，HND stride view | HND/TMA 更自然；wrapper 可转换 NHD |
| Page size | 8/16/32/64 | power-of-two；测试确认 16~1024 的不同路径 |
| Dtype | 当前 Q/K/V 通常同 dtype | mixed Q/KV、FP8、NVFP4 覆盖更完整 |
| 架构选择 | Blackwell JIT；标准 GQA 主要按 SM100 配置 | 显式 SM100/SM103/SM100f cubin 匹配 |

研究、修改或验证新 B300 kernel 机制时优先使用 CuTe DSL；追求现有功能覆盖和成熟
heuristic 时可比较 TRTLLM-Gen，但不要把 cubin 路径称为“完整开源 kernel”。

---

## 9. Page size 的含义与影响

Page size 是每个物理 KV page 容纳的 token 数：

```text
num_pages(request) = ceil(kv_len / page_size)
```

### 9.1 Page-table 开销

小 page：

- page table 更大；
- logical-to-physical lookup 更多；
- KV tile 更容易跨 page；
- TMA/copy warp 处理更多 page boundaries。

### 9.2 TMA 搬运

TRTLLM-Gen descriptor 构造使用：

```text
numKeysPerTile = min(page_size, tileSizeKv)
```

小 page 可能限制单次从当前 page 取得的连续 tokens。CuTe DSL 中，一个 256-token
sequence tile 在 page size 8/16/64 时分别跨 32/16/4 pages。

### 9.3 内存碎片

每个 request 的最后一页可能未填满：

```text
最坏浪费 = page_size - 1 tokens/request
平均近似 = page_size / 2 tokens/request
```

大 page 减少 page-table/TMA 边界开销，但提高短序列和动态 batching 的内部碎片。

### 9.4 与 KV split 的关系

Page 和 split 不相同：

```text
page:  KV 在哪里、如何寻址
split: 哪些 CTA 负责哪些 KV token 范围
```

同样的 1024-token split：

```text
page_size=16 -> 64 pages
page_size=64 -> 16 pages
```

Page size 不改变 attention 数学结果，有效 token 仍由 `seq_lens` 控制；它改变内存
利用率、page-table 开销、TMA 连续性和 kernel availability。

---

## 10. 源码索引

以下路径相对本地 FlashInfer 仓库根目录：

### 公共 API 与 backend dispatch

```text
flashinfer/decode.py
```

重点：

- `BatchDecodeWithPagedKVCacheWrapper`
- `trtllm_batch_decode_with_kv_cache`
- `xqa_batch_decode_with_kv_cache`
- CuTe DSL 的 HND -> logical NHD view

### 原生 CUDA-core decode

```text
include/flashinfer/attention/decode.cuh
include/flashinfer/attention/scheduler.cuh
csrc/batch_decode.cu
```

搜索：

```text
partition_kv
kv_chunk_size
BatchDecodeWithPagedKVCacheDevice
```

### CuTe DSL GQA decode

```text
flashinfer/cute_dsl/attention/gqa_decode.py
flashinfer/cute_dsl/attention/gqa_decode_paged.py
flashinfer/cute_dsl/attention/wrappers/batch_decode.py
```

搜索：

```text
kv_splits
_compute_kv_splits
_resolve_reduction
GroupedQueryAttentionDecodePaged
```

### TRTLLM-Gen host 逻辑

```text
csrc/trtllm_fmha_kernel_launcher.cu
include/flashinfer/trtllm/fmha/fmhaRunner.cuh
include/flashinfer/trtllm/fmha/fmhaRunnerParams.h
include/flashinfer/trtllm/fmha/fmhaKernels.cuh
include/flashinfer/trtllm/fmha/kernelParams.h
flashinfer/jit/attention/modules.py
```

搜索：

```text
MultiCtasKvMode
numCtasPerSeqKv
CgaSmemReduction
selectTileSizeQForGqaGeneration
isSMCompatible
useDynamicNumTokensPerPage
gen_trtllm_gen_fmha_module
```

### 测试覆盖

```text
tests/attention/test_cute_dsl_decode.py
tests/attention/test_trtllm_gen_attention_decode.py
tests/attention/test_tensor_cores_decode.py
tests/attention/test_batch_decode_kernels.py
```

阅读时继续区分：

```text
测试覆盖到某配置
host 参数允许某配置
cubin metadata 中实际存在该 kernel
主 GPU kernel 源码是否可见
```

这四件事不能互相替代。
