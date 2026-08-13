# FlashInfer 的 GDN 优化方案（SM100 / SM103）

> 定位：**CuTe DSL 全融合实现。12 warp 专用化 + 双 MMA 发射流 + TMEM 常驻状态，以及一条纯代数的精确并行扫描。**
> 前置阅读：[`GDN_Algorithm.md`](GDN_Algorithm.md)，本文全程使用其中的符号与"四个数学杠杆"编号。
> **范围：数据中心 Blackwell（SM100 / SM103）。** SM120 不在范围内；SM90 只在 §11 作为对照。
> 代码基准：`flashinfer` @ `e9fb62b7`。路径相对于仓库根。

---

## 目录

1. [先纠正几个常见误解](#1-先纠正几个常见误解)
2. [文件地图与分派](#2-文件地图与分派)
3. [非 CP 路径实现的数学](#3-非-cp-路径实现的数学)
4. [12 warp 的角色划分与双 MMA 发射流](#4-12-warp-的角色划分与双-mma-发射流)
5. [片上资源：SMEM 226 KB 与 TMEM 恰好 512 列](#5-片上资源smem-226-kb-与-tmem-恰好-512-列)
6. [两 chunk 成对：真正的理由是求逆并行](#6-两-chunk-成对真正的理由是求逆并行)
7. [三角求逆：8×8 起步、3 层合并、走 warp 级 MMA](#7-三角求逆88-起步3-层合并走-warp-级-mma)
8. [零初始状态的编译期剥离](#8-零初始状态的编译期剥离)
9. [varlen：有界 TMA descriptor 让尾块与 padding 全免费](#9-varlenr有界-tma-descriptor-让尾块与-padding-全免费)
10. [CP 路径：四阶段精确并行扫描](#10-cp-路径四阶段精确并行扫描)
11. [推理侧设计：分页状态池与状态量化](#11-推理侧设计分页状态池与状态量化)
12. [tile scheduler](#12-tile-scheduler)
13. [精度与测试](#13-精度与测试)
14. [SM90 对照：同一个不变式，不同的落点](#14-sm90-对照同一个不变式不同的落点)
15. [性能数据（仓库里没有）](#15-性能数据仓库里没有)
16. [汇总](#16-汇总)

---

## 1. 先纠正几个常见误解

| 说法 | 实际 |
|---|---|
| "`gated_delta_net_chunked.py` 的文件头 docstring 有完整的算法推导，照它读就行" | **那 55 行 docstring（`:29-84`）几乎每一行表格都是过期的。** SMEM 级数、TMEM buffer 大小、warp 角色、`W_qkv = T·beta·W_qk`、`log`/`exp`（实际是 `log2`/`exp2`）、"state / decay_v" SMEM buffer（不存在）、"warp 8 发全部 7 个 GEMM"（实际两个发射流）、"warp 10 是 TMA gate warp"（实际是第二个 MMA 发射流）——全错。真正的推导散落在 `_setup_attributes` 的注释、各方法 docstring 和 `tests/gdn/reference_delta_rule.py` 里。 |
| "FlashInfer 把门乘进矩阵再求逆" | **只有非 CP 路径是这样。CP 路径反过来——求逆 gate-free，门在求逆之后以"门夹心"的形式施加。** 这是整个代码库里最有意思的数学对比，见 §3.2 与 §10.5。 |
| "SM103 有专门的处理" | **`gdn_kernels/` 里 `103` 零命中。** 分派只看 `_arch_major == 10`。非 CP kernel 甚至把 `arch = "sm_100"` 硬编码用于 TMEM 查询。唯一的差异是 **CP 路径编译时带 `cute.GPUArch(f"sm_{major}{minor}a")`**（B300 上就是 `sm_103a`），非 CP 路径完全不传 `GPUArch`。 |
| "求逆用 `collective_inverse_hmma.py`" | **分裂答案。** 非 CP：**不用**，有自己的 fork（`chunked.py:3207-3932`）。CP 阶段 1（T 预计算）：**用**。CP 阶段 4（主 prefill）：**根本不求逆**，TMA 载入预计算好的 T。 |
| "K 缓冲 2 级" | **4 级。** 而且 gate 和 beta 各 5 级。共 9 个分级 SMEM buffer + 6 个分级 TMEM 区域。 |
| "运行时谓词避免 prologue peeling，理由是 icache" | 注释说的是 *"keep one pair body in SASS instead of peeling a complete first-pair copy"*——**代码体积**论证。`gdn_kernels/` 里 `icache` / `instruction cache` **零命中**。而且传给 `compute_group_0_pair` 的谓词现在是**未使用的**（`_, _ = work_args`），机制已经退化成"body 本身与首末无关"。 |

另有约 600 行**死代码**（`mma_issuer_warp` `:2423` 与 `mma_warp_chunk` `:2558`，单发射流变体，`kernel()` 里没人引用），而**几条最好的设计理由注释恰好在这段死代码里**。本文引用它们时会明确标注。

---

## 2. 文件地图与分派

### 2.1 文件

| 文件 | 行数 | 角色 |
|---|---|---|
| `flashinfer/gdn_prefill.py` | 581 | 公开 `chunk_gated_delta_rule`、架构分派、CP 路由 |
| `gdn_kernels/blackwell/gdn_prefill.py` | 347 | SM100 非 CP 的 host adapter + 编译缓存 |
| `gdn_kernels/blackwell/gated_delta_net_chunked.py` | **4755** | **SM100 非 CP 主 kernel** |
| `gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py` | 263 | tile scheduler |
| `gdn_kernels/blackwell/gdn_cp_prefill.py` | 1128 | SM100 CP 的 host 编排（4 次 launch） |
| `gdn_kernels/blackwell/gated_delta_net_cp.py` | 2229 | CP 的 MN 预计算 + UTCMMA fixup kernel |
| `gdn_kernels/blackwell/gated_delta_net_cp_prefill.py` | 3257 | CP 的主 prefill kernel |
| `gdn_kernels/delta_rule_dsl/varlen_helper.py` | 228 | CP 的解析模型 / 启发式（host + device 共享） |
| `gdn_kernels/delta_rule_dsl/collective_inverse_hmma.py` | 404 | $64\times64$ 分治求逆（**仅 CP 的 T 预计算用**） |

### 2.2 分派逻辑

`gdn_prefill.py:354-364`：

```python
_sm_count = get_device_sm_count(device)
_device_capability = get_compute_capability(device)
_arch_major = _device_capability[0]
cp_heuristic_matches = _arch_major in (9, 10, 12) and should_use_cp_host(
    num_seqs * num_sab_heads, _sm_count, _device_name,
    device_capability=_device_capability,
)
will_use_cp = use_cp is True or (use_cp == "auto" and cp_heuristic_matches)
```

顺序是 **CP 优先**，再按 `_arch_major` 走非 CP。SM100 的门槛（`:461-471`）：`_cuda_major >= 13`、`head_size == 128`。

### 2.3 编译期 vs 运行时 —— serving 的关键

**编译缓存 key**（`blackwell/gdn_prefill.py:50-63`，`@functools.cache`）：

```python
io_dtype_str, state_dtype_str, HQ, HV, is_GQA,
use_initial_state, store_final_state, enable_checkpoints, use_state_indices
```

| 编译期常量 | 运行时动态值 |
|---|---|
| `io_dtype`（fp16/bf16）、`state_dtype`（5 种） | `total_tokens`（`mark_compact_shape_dynamic(mode=0, divisibility=1)`，`:233-255`） |
| `b_t = 64` 硬编码（`chunked.py:434`） | `batch_size`（= `cu_seqlens.shape[0]-1`） |
| `DK = DV = 128` 断言 | 每序列 `seqlen`（各 warp 角色从 `cu_seqlens[b]` 读） |
| 4 个 `mma_tiler` 形状（`can_implement` 拒绝其他） | `scale`（`cutlass.Float32` kernel 参数） |
| `is_GQA = HQ >= HV`（选 head reshape 分支） | `checkpoint_every_n_tokens` |
| `use_initial_state`（驱动寄存器切分与首 chunk 剥离） | 状态池 `N_pool` 与其**被 padding 的第 0 维 stride** |
| `HQ`, `HV`（baked in） | `num_sm` / grid |
| `is_persistent = True` | |

> **`divisibility=1` 是刻意的**：它禁止编译器假设 token 数有任何对齐，所以 ragged batch 永远不会掉出快速路径。
> **serving 后果**：对固定模型 + 固定 TP，`(io_dtype, state_dtype, HQ, HV, flags)` 是一个常量元组 ⟹ **整个 serving 生命周期只有一份 cubin**，覆盖所有 batch size、所有序列长度组合、所有 varlen 形状。代价是 head 数在 key 里，改 TP 度要重编译。

CP 路径用另一套机制：`KeyedCompileMixin` + `manual_cache_key(...)`（如 `gated_delta_net_cp.py:71-82`）加 `custom_compile_cache.py` 的 `cached_compile`。**`cp_chunk_len` 是运行时 `cutlass.Int32`**（`gdn_cp_prefill.py:879`），所以启发式可以逐调用换段长而不重编译。

---

## 3. 非 CP 路径实现的数学

### 3.1 kernel 自己的记号

每 chunk `c` 覆盖 token `[64c, 64(c+1))`，逐 (batch, head)：

| 变量 | 含义 | 位置 |
|---|---|---|
| `sCumsumlog[t]` | $\log_2(\alpha_t + 10^{-10})$ 的**含端**前缀和。**注意是 base-2** | `:2074-2089` |
| `sCumprod[t]` | $\exp_2(\text{sCumsumlog}[t]) = \gamma_t$ | `:2091` |
| `tGrCumsumlog_0/1[i,j]` | $\Lambda_{ij} = \exp_2(G_i - G_j)$，$i\ge j$（**含对角**），上三角为 0 | `:3038-3048` |
| `tGrBeta_0` | 按**行**索引 ⟹ per-row 标量 $\beta_i$ | `:3061` |
| `tGrCumprod[k]` | $\gamma_t$（按 token 索引） | `:4484-4486` |
| `tGrDecayScale[k]` | $\exp_2(G_{63} - G_t) = \gamma_{63}/\gamma_t$，用 `cute.arch.add_packed_f32x2`（Blackwell 打包 fp32 加，一条指令两元素）算 | `:4487-4501` |
| `cumprod_total` | `sCumprod[63]` = $\gamma_{\mathrm{end}}$ | `:4413` |

> **全程 base-2 对数。** 与 FLA 的 `exp2` 化（FLA 文档 §5.1）是同一个技巧，但 FlashInfer 是把 $\log_2$ 直接用在门上（`cute.math.log2(x + 1e-10)`），而 FLA 是把 $\log_2 e$ 预乘进 cumsum。**两者存在 HBM 里的 `g` 语义都不是标准 $\log\alpha$ 的 cumsum，但不是同一种偏差。**

另外：**FlashInfer 的公开 `g` 参数是线性空间的 $\alpha\in(0,1)$**（全 1 = 不衰减），传 log-space 值直接 NaN。这在 benchmark 里有明确注释（`benchmarks/bench_gdn_prefill.py:114-116`），而 FLA 收的是 $\log\alpha$。跨库替换时这是第一个坑（算法文档 §10.2）。

### 3.2 门在求逆之前（非 CP 路径）—— 已确认

`compute_group_0_pair`，`chunked.py:3089-3091`：

```python
tKKrKK[k, 0, sub] = (
    tKKrKK[k, 0, sub] * tGrCumsumlog_0[k, 0, sub] * tGrBeta_0
)
```

`tKKrKK` 是从 TMEM 读出的 $KK^{\top}$ 累加器。所以送进求逆的矩阵是

$$
M[i,j] = \beta_i \cdot \Lambda_{ij}\cdot (KK^{\top})_{ij},\quad i\ge j;\qquad 0,\ i<j
$$

对角线随后在求逆内部被强制置 1（`_invert_diagonal_NxN`，`:3413-3414`：`row[i] = 1.0 if tidx_in_group == i else row[i]`），实现 $(I + \mathrm{strictlower}(M))^{-1}$。

**求逆之后没有任何 Λ 的 Hadamard 乘。** 唯一的后处理是**按列**的 $\beta$ 缩放（`_finish_pair_inverse`，`:3319-3327`，注意 `coord[1]` 是列索引）。所以发布出去的操作数是

$$
A_{\mathrm{inv}} = \bigl(I + \mathrm{strictlower}(\mathrm{diag}(\beta)\,\Lambda\circ KK^{\top})\bigr)^{-1}\mathrm{diag}(\beta)
$$

$\beta$ 一半在求逆前折成行缩放、一半在求逆后折成列缩放。**这与 FLA 的选择相同（门在内），与 FlashQLA 相反（门在外）。**

> **但 CP 路径做了相反的选择**（§10.5）：CP 的 T 预计算求逆的是 **gate-free** 矩阵，门在求逆之后以两侧夹心的形式施加。合法性来自算法文档 §7.1 的恒等式：
> $$\bigl(I + \mathrm{tril}(\mathrm{diag}(\beta)\Lambda\circ KK^{\top},-1)\bigr)^{-1} = \mathrm{diag}(\gamma)\bigl(I + \mathrm{tril}(\mathrm{diag}(\beta)KK^{\top},-1)\bigr)^{-1}\mathrm{diag}(\gamma)^{-1}$$
> **为什么同一个库两条路走不同方向？** 非 CP 路径把 Λ 折在求逆前，是因为它一趟算完、$T$ 没有复用需求；CP 路径把 Λ 提到求逆后，是为了让 $T$ **与门无关**，从而**每 64-token 块只算一次、被后面两个 kernel 复用**。
> **这是本知识库里最能说明"同一个恒等式在不同复用需求下导出不同实现"的例子。**

### 3.3 w/u 已合并

没有 `w`，也没有独立的 `u`。`compute_group_1_chunk`（`:4518-4537`）：

```python
if valid_state:
    ks_handle = cg1_shared_acc_consumer.wait_and_advance()
    cute.copy(tiled_ks_t2r, tTR_tCtKS[...], tTR_rKS)
    for k in ...: tTR_rKS[k] = tTR_rKS[k] * tGrCumprod[k]                  # × γ_t
    ks_handle.release()
    for k in ...: tRT_rV[k] = tRT_rV[k] - tTR_rKS[k].to(self.io_dtype)     # V − γ·K S_prev
cute.copy(tiled_vks_r2t, tRT_rV, tRT_tCtVKS_inp[None, None, None, 0])
```

随后 GEMM5 一次算完 `NV = A_inv ⊗ VKS`。即算法文档 §3.8 的合并形式

$$
NV = A_{\mathrm{inv}}\bigl(V - \mathrm{diag}(\gamma_t)\,K S_{\mathrm{prev}}\bigr)
$$

**没有 $w = T\mathrm{diag}(\gamma)K$ 这个 GEMM，也没有两项式的状态更新。** $w$ 项被吸收，因为状态贡献在三角求解**之前**就已经从 $V$ 里减掉了。

### 3.4 方向约定（读懂形状必须知道）

$V$ 之后的一切都是 **value-major / token-minor**。`chunked.py:458-464`（GQA 分支）把 `v` reshape 成 `(DV, tokens, heads)`，`o` 同理（`:504-510`），状态 `[N,H,V,K]` reshape 成 `(V, K, heads, N)`（`:511-536`）。所以片上状态是 $S^{\top}\in\mathbb{R}^{DV\times DK}$，value/output 块是 $\mathbb{R}^{DV\times BT}$。

### 3.5 七个 GEMM，两个发射流

**发射流 A = warp 8**（`mma_cg0_pair`，`:2126-2222`），每 **pair** 调一次，4 个 GEMM，固定顺序 **KK0 → KK1 → QK0 → QK1**：

| # | 名 | A 操作数 | B 操作数 | tiled_mma | 形状 (M,N,K) |
|---|---|---|---|---|---|
| 1 | KK0 | `sK` **SMEM** | `sK` **SMEM** | `tiled_mma_qk` | (64,64,128) |
| 2 | KK1 | `sK` stage k1 | `sK` stage k1 | `tiled_mma_qk` | (64,64,128) |
| 3 | QK0 | `sQ` **SMEM** | `sK` stage k0 | `tiled_mma_qk` | (64,64,128) |
| 4 | QK1 | `sQ` stage q1 | `sK` stage k1 | `tiled_mma_qk` | (64,64,128) |

每个内部 8 次 `cute.gemm`（`num_kphases = 128/16 = 8`），配 `tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)`。

**发射流 B = warp 10**（`mma_cg1_chunk`，`:2224-2420`），每 **chunk** 调一次，5 个 GEMM，顺序 **KS → QS → NV → QKV → KV**：

| # | 名 | 累加器 | A 操作数 | B 操作数 | 形状 | 备注 |
|---|---|---|---|---|---|---|
| 3 | KS | `cg1_shared_acc` | `tCtStateInp` **TMEM** io_dtype | `sK` SMEM | (128,64,128) | `!valid_state` 时跳过 |
| 4 | QS | `tCtQState`（**独占**） | `tCtStateInp` **TMEM**（同操作数复用） | `sQ` SMEM | (128,64,128) | `!valid_state` 时跳过 |
| 5 | NV | `cg1_shared_acc` | VKS **TMEM** io_dtype | `sAinv` SMEM | (128,64,64) | |
| 6 | QKV | `tCtQState`（**叠在 QS 上**） | NV **TMEM** | `sQk` SMEM | (128,64,64) | `ACCUMULATE = valid_state or kphase!=0` |
| 7 | KV | `tCtState`（**递归状态**） | decayV **TMEM** | `sK_trans` SMEM，**MN-major** | (128,128,64) | 同上 |

操作数来源声明（`chunked.py:559-616`）：

```python
tiled_mma_qk  : OperandSource.SMEM, a_major=K,  b_major=K    # GEMM 1,2
tiled_mma_qs  : OperandSource.TMEM, a_major=K,  b_major=K    # GEMM 3,4
tiled_mma_qkv : OperandSource.TMEM, a_major=K,  b_major=K    # GEMM 5,6
tiled_mma_kv  : OperandSource.TMEM, a_major=K,  b_major=MN   # GEMM 7
```

> **一句话概括整个 SM100 设计：凡是链式中间量（状态、VKS、NV、decayV）当 A 操作数的，都住在 TMEM；凡是来自 HBM（Q/K/V）或来自 CG0（A_inv、带门的 QK）的，都住在 SMEM。**
> 这就是算法文档 §5 那句"链上中间量只被下一步用一次，没有理由落 SMEM 再读回来"在 Blackwell 上的落点。

⚠️ 命名陷阱：`tCrNv_B` 是从 `sQk` 构造的——装带门 QK 分数的 SMEM buffer **就是** GEMM6 的 B 操作数，而 `tCtSharedInp[...,0]` 那时装的是 NV。**变量名与数学是反的**，别被误导。

### 3.6 Epilogue

- **QS 原地重缩放**（`:4539-4554`）：`tTR_rQS[k] *= tGrCumprod[k] * scale` ⟹ $O_{\mathrm{inter}}[\cdot,t] = \gamma_t\cdot scale\cdot(q_t S_{\mathrm{prev}})$
- GEMM6 把 $O_{\mathrm{intra}} = (\Lambda\circ QK^{\top}\cdot scale)\cdot NV$ **叠加在同一个累加器上**
- **$\beta$ 不出现在 QK 路径上**（`:3151-3153` 只乘 $\Lambda$ 和 `scale`）——docstring 里的 `W_qkv = T*beta*W_qk` 是错的，$\beta$ 完全住在 `A_inv` 里
- decayV（`:4571-4574`）：`tTR_rDv[k] = NV[k] * tGrDecayScale[k]`
- **状态更新**（`:4470-4476`）：TMEM 里的 fp32 状态累加器被**原地**乘 `cumprod_total`，然后 GEMM7 把 $dS$ 累加进去：

  ```python
  for k in ...: tTR_rState[k] = tTR_rState[k] * cumprod_total
  cute.copy(tiled_state_r2t, tTR_rState[None,0,None], tRT_tCtState[None,0,None, kv_prev_handle.index])
  ```

  ⟹ $S_{\mathrm{next}} = \gamma_{\mathrm{end}}S_{\mathrm{prev}} + K^{\top}(\Lambda_{\mathrm{end}}\circ NV)$。**状态在 chunk 之间从不离开 TMEM。**
- O 路径：TMEM → 寄存器（`Ld16x256b`）→ fp16/bf16 → `StMatrix8x8x16b(transpose=True)` → `sO` → warp 11 的 TMA S2G

---

## 4. 12 warp 的角色划分与双 MMA 发射流

### 4.1 角色表（384 线程）

`chunked.py:231-271`：

```python
self.compute_group_0_warp_ids = [0, 1, 2, 3]
self.compute_group_1_warp_ids = [4, 5, 6, 7]
self.mma_warp_id = 8
self.tma_qkv_warp_id = 9
# The second issuer owns the five state/output GEMMs.
self.mma_cg1_warp_id = 10
self.epilogue_warp_id = 11
# The lightly loaded O epilogue warp also prefetches gate/beta.
self.load_gate_beta_warp_id = self.epilogue_warp_id
```

| Warp | 角色 | 依赖状态？ | 入口 |
|---|---|---|---|
| **0–3（CG0，128 线程）** | Λ 构造、KK epilogue ×2、**分层 $64\times64$ 求逆 ×2**、QK epilogue ×2 | ❌ | `compute_group_0_pair` `:2917` |
| **4–7（CG1，128 线程）** | 状态 TMEM 管理、KS 缩放 + $V-\gamma KS$、QS 重缩放、NV/decayV 发布、O epilogue、状态 GMEM I/O、checkpoint、**TMEM allocator 所有者** | ✅ | `compute_group_1_chunk` `:4149` |
| **8** | **MMA 发射流 A**：KK0/KK1/QK0/QK1 | ❌ | `mma_cg0_pair` `:2126` |
| 9 | TMA 载 Q/K/V + Q/K/V 的 tensormap 打补丁 | — | `tma_qkv_warp` `:1835` |
| **10** | **MMA 发射流 B**：KS/QS/NV/QKV/KV | ✅ | `mma_cg1_chunk` `:2224` |
| 11 | TMA 存 O + **gate/beta 标量载入与前缀扫描** + O 的 tensormap 打补丁 | — | `epilogue_warp` `:4637`，`load_gate_beta_warp` `:1979` |

> **这个分组的界线就是算法文档 §5 那张依赖表的界线**：CG0 + warp 8 负责环节 ①②（不依赖状态），CG1 + warp 10 负责 ③④⑤（被串行链锁住）。
> **与 FlashQLA 的分组原则不同**：FlashQLA 按"谁拥有哪个量"分组（状态 / $V_{\mathrm{new}}$ / 输出各一个 warpgroup），FlashInfer 按"是否依赖状态"分组。**两种切法都能追回同一张依赖图，但 FlashInfer 这种切法直接产生了"两条独立发射流"这个结构。**

### 4.2 两条独立 MMA 发射流：已确认

warp 8（`:1446`）与 warp 10（`:1482`）是两个独立的 `elif` 分支，各有：

- 自己的 tile scheduler 循环
- 自己的一整套生产者 pipeline（`cg0_shared_acc_producer` vs `cg1_shared_acc_producer`/`q_state_acc_producer`/`kv_acc_producer`）
- **不相交的 TMEM 累加器 ring**（§5.2）

唯一共享的资源是 K/Q 的载入 pipeline：`cg_mma_both = _cg(len([self.mma_warp_id, self.mma_cg1_warp_id]))`（`:1074`）——一个 2 线程的 consumer group，所以两个发射流都到达之后 K/Q stage 才释放（`:1090-1106`）。

**这个设计买到什么**：`tcgen05` 的 MMA 是单线程发射、异步执行的，所以"谁来发指令"可以自由安排。**把不依赖状态的 GEMM 交给一个独立发射 warp，意味着它可以任意超前**——只要 TMA 把 K、Q 搬进来，$KK^{\top}$ 就能发，完全不用等状态链。而 warp 10 那条流只能一步一步走。

配套的关键决定是**两个计算组各有独立的 TMEM 累加器 ring**（§5.2），而不是共用。多占 TMEM，换来把"这块累加器现在归谁"的所有权交接从关键路径上彻底拿掉。死代码里那句注释说得最清楚（`:2582-2587`，标注为"前单发射流变体的设计理由"）：

> *"The next KK0/KK1 are issued after current NV0 so their MMA latency overlaps current QKV0/KV0 and chunk-1 work. CG0 and CG1 accumulators use disjoint two-stage rings, so the lookahead cannot alias KS/NV."*

**注意 `smem_k_stages = 4` 和 `smem_ainv_stages = 3` 是为这个前瞻量身定的，所以缓冲深度活得比激励它的代码还长。**

### 4.3 轻活合并

写出 $O$ 是轻负载，于是 gate / beta 的预取挂在同一个 warp 上（`load_gate_beta_warp_id = epilogue_warp_id`），不再单独占一个 warp。这也解释了 `smem_o_stages = 2` 的注释（`:325-327`）：

> *"Gate/beta work now shares the epilogue warp, so O uses two stages to avoid back-pressuring CG1 while the warp publishes the next gate."*

**这是一条典型的"资源分配连锁"**：合并两个角色 ⟹ 该 warp 会被 gate 工作短暂占住 ⟹ O 必须双缓冲否则回压 CG1。

### 4.4 寄存器切分

`chunked.py:247-259`：

```python
self.num_regs_compute_group_0 = 224
self.num_regs_compute_group_1 = 256
self.num_regs_other = 24
if not self.use_initial_state:
    # The peeled zero-state MMA carries more pipeline cursors.  Transfer
    # twenty-four registers from each CG1 warp to each lightweight warp while
    # retaining the same 64,512-register CTA allocation.
    self.num_regs_compute_group_1 = 232
    self.num_regs_other = 48
```

两种配置都恰好 **64,512** 寄存器：

- 带初始状态：$4\cdot32\cdot224 + 4\cdot32\cdot256 + 4\cdot32\cdot24 = 28672+32768+3072 = 64512$
- 零初始状态：$4\cdot32\cdot224 + 4\cdot32\cdot232 + 4\cdot32\cdot48 = 28672+29696+6144 = 64512$

（SM 寄存器文件 65,536；`occupancy = 1`，`min_blocks_per_mp = 1`，`use_2cta_instrs = False`，`cluster_shape_mnk = (1,1,1)`。）

---

## 5. 片上资源：SMEM 226 KB 与 TMEM 恰好 512 列

### 5.1 SMEM 分配表

级数设置（`:308-332`），**注释本身就是理由，逐条抄录**：

```python
self.smem_q_stages = 2
# Four K stages let TMA make K3 available while K0 remains live for
# KV0, so the next pair's KK0/KK1 can be issued back-to-back.
self.smem_k_stages = 4
# Three V stages both break the double-KK lookahead dependency cycle
# and let TMA stay ahead while CG1 consumes the current pair.
self.smem_v_stages = 3
# Mid-pair KK lookahead overlaps next-pair inverse preparation with the
# current second chunk.  NV0 has released current Ainv0 by then, so the
# live set is current Ainv1 plus next Ainv0/1: three stages total.
self.smem_ainv_stages = 3
self.smem_qk_stages = 2
self.smem_o_stages = 2
# Five resident stages preserve four chunks of gate/beta lookahead.
self.smem_gate_stages = 5
# Let the scalar producer enter the next pair after CG0 releases beta0
# instead of waiting for both current-pair beta stages.
self.smem_beta_stages = 5
```

| Buffer | tile / stage | B/stage | 级数 | 总 B |
|---|---|---|---|---|
| mbarrier（16 个 pipeline）+ `tmem_holding_buf` | — | — | — | 564（被 `sQ` 的 `Align[1024]` 垫到 1024） |
| `sQ` | 64×128 fp16 | 16 384 | 2 | **32 768** |
| `sK` | 64×128 fp16 | 16 384 | 4 | **65 536** |
| `sK_trans` | 128×64 fp16 | 16 384 | 4 | **0 —— 别名 `sK`** |
| `sV` | 128×64 fp16 | 16 384 | 3 | **49 152** |
| `sAinv` | 64×64 fp16 | 8 192 | 3 | **24 576** |
| `sAinvCal` | — | — | — | **0（别名 `sAinv`）** |
| `sQk` | 64×64 fp16 | 8 192 | 2 | **16 384** |
| `sO` | 128×64 fp16 | 16 384 | 2 | **32 768** |
| `cumsumlog` | 64 fp32 | 256 | 5 | 1 280 |
| `cumprod` | 64 fp32 | 256 | 5 | 1 280 |
| `beta` | 64 fp32 | 256 | 5 | 1 280 |
| **合计** | | | | **226 048 B ≈ 220.8 KiB** |

SM100 最大动态 SMEM 是 232 448 B（227 KiB）⟹ 剩约 6.2 KiB。所有 buffer 都 `Align[1024]` 且尺寸都是 1024 的倍数，没有内部 padding。

**两处别名是最大的省法**：

1. **`sK` / `sK_trans` 双 descriptor 别名，省 64 KB。** `:633-638` 用两个不同的 `tiled_mma` 生成两套布局，`:1014-1019` 让两个 view 指向**同一块 `storage.sK`**。原因是 major mode 不同：`tiled_mma_qk` 要 K 作 **K-major** 的 (64,128) B 操作数（算 $KK^{\top}$、$QK^{\top}$），`tiled_mma_kv` 要 K 作 **MN-major** 的 (128,64) B 操作数（算 $K^{\top}V_{\mathrm{new}}$，沿 token 维收缩）。**两个 descriptor 恰好描述同一块 16 384 B/stage 的 swizzled tile，所以一份数据两种视图。** 而这个转置需求**是 chunk 公式 ② 本身带来的**（算法文档 §3.5，状态更新式里 $K$ 就是转置的），不是实现随意选的。
2. **`sAinvCal` 在 `inverse_dtype == io_dtype` 时别名 `sAinv`**（`:656-660, 1027-1033`），注释：*"Default inverse_dtype == io_dtype aliases sAinv and allocates no extra SMEM."* 因为整个分层求逆是在行主序 SMEM buffer 上**原地**做的，工作 buffer 和发布的操作数可以是同一块内存。

第三处：**`_transform_to_position_independent_layout`**（`:4703-4712`）让 CG0/CG1 重新基准化 `sQk`/`sAinv`/`sV`/`sO` 的 swizzle，使寄存器↔SMEM 拷贝能位置无关地寻址 swizzled buffer。

### 5.2 TMEM 分配表：恰好 512 列

`chunked.py:280`：`self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")` = 512。CG1 的 warp 4 拥有分配权（`:1318`），注释（`:1056-1057`）：*"CG1 owns allocation and is the last group to release TMEM state."*

| 列 | 区域 | 装什么 | dtype | 级数 | 为什么在这 |
|---|---|---|---|---|---|
| **0–127** | `tmem_state_offset` | **递归状态 $S^{\top}$（DV×DK）** = GEMM7 累加器 | fp32 | 1 | $128\times128$ fp32 = 128 列。**跨 chunk 从不离开 TMEM**；CG1 原地乘 $\gamma_{\mathrm{end}}$，GEMM7 把 $dS$ 累加上去 |
| **128–191** | `tmem_q_state_offset` | **QS / O 累加器（DV×BT）** | fp32 | 1 | **独占，因为它的生命期跨两个 GEMM 加两次 CG1 访问**，见 §5.3 |
| **192–255** | `tmem_state_inp_offset` | $S_{\mathrm{prev}}$ 降精度到 io_dtype，GEMM3/4 的 A 操作数 | fp16/bf16 | 1 | (128,128) 16-bit A-frag = 64 列 |
| **256–383** | `tmem_cg0_shared_acc_offset` | **CG0 的 ring**：KK0/KK1/QK0/QK1 累加器 | fp32 | **2** | 2 级让 warp 8 比 CG0 的读出**超前两个 GEMM**；与 CG1 的 ring 不相交，所以前瞻"不可能与 KS/NV 别名" |
| **384–447** | `tmem_cg1_shared_acc_offset` | **CG1 的 ring**：先 KS 后 NV | fp32 | 1 | KS 被消费并释放之后 NV 才发，1 级够用 |
| **448–511** | `tmem_shared_inp_offset` | slot 0 = VKS **然后** NV；slot 1 = decayV | fp16/bf16 | **2** | (128,64) 16-bit A-frag = 32 列 × 2 |
| **512** | | | | | **恰好用满** |

注释（`:341-343`）：

> *"CG0 owns KK/QK and CG1 owns KS/NV. Separate rings remove the cross-group ownership handoff from the shared-acc critical path."*

**slot 0 的双重身份值得单说**：CG1 先往 448–479 写 VKS（`:4531-4535`）给 GEMM5 用，等把 NV 从 CG1 的 acc 里读出来之后，**用同一个 slot 0 覆写成 NV**（`:4581-4586`）给 GEMM6 用，同时把 decayV 写到 slot 1。这是**固定槽位上的 TMEM 时间复用**，由三个独立的 1 级 `PipelineAsyncUmma` "ready-only" barrier 定序（`:1202-1225`），注释（`:725-726`）：

> *"CG1 -> MMA warp: fixed-slot TMEM inputs. The empty halves are unused because downstream accumulator-full signals prove reuse."*

### 5.3 QS 为什么独占一块累加器

看输出式 ③：$O = \mathrm{diag}(e^{g})QS_{\mathrm{prev}} + (\Lambda\circ QK^{\top})V_{\mathrm{new}}$。

QS 的生命期在**一个 chunk 内跨三个生产/消费回合**，而且是单级 barrier：

1. warp 10 `q_state_acc_producer.acquire_and_advance()` → GEMM4 写 QS → `commit()`（`:2333-2343`）
2. CG1 `wait_and_advance()` → t2r、`*= γ_t · scale`、r2t、`release()`（`:4540-4554`）
3. warp 10 再 acquire → **GEMM6 把 $O_{\mathrm{intra}}$ 叠加上去**（`ACCUMULATE = valid_state or kphase!=0`）→ `commit()`（`:2374-2389`）
4. CG1 再 wait → t2r → fp16 → `sO` → `release()`（`:4602-4619`）

它不能与 CG1 的 shared-acc ring 共用，因为那个 ring 在同一个窗口内要在 KS 和 NV 之间回收。**给 QS 独立区域，是让 $O = O_{\mathrm{inter}} + O_{\mathrm{intra}}$ 成为"累加器内零额外流量的加法"**，而不是走寄存器或 SMEM 归约。

> **这是"公式结构直接决定硬件资源分配"最干净的一例**：inter 项与 intra 项在公式里相隔多个 GEMM ⟹ 它的累加器必须独占。

---

## 6. 两 chunk 成对：真正的理由是求逆并行

### 6.1 结构

- CG0 与 warp 8 按**成对的 chunk**工作；CG1 与 warp 10 按单 chunk 工作
- `num_pairs_b = ceil_div(seqlen_b, b_t * 2)`；`num_chunks_padded = num_pairs_b * 2`（`:1264, 1352-1353`）。**有效 chunk 数为奇数时向上补成偶数**——多出来那个 chunk 是全零 pass（有界 TMA descriptor 会把它零填充，`:1622`，见 §9）
- 发射顺序 **KK0 → KK1 → QK0 → QK1**（`:2161-2211`），然后四个 Q/K stage handle 一起释放（`:2213-2216`）

### 6.2 表层理由：Λ 复用与求逆重叠

`compute_group_0_pair` docstring（`:2933-2937`）：

> *"Warps 0-3: reuse one T-pairwise calculation across KK and QK. The first pair completes both inverses before QK. A steady pair consumes the prefetched KK0/KK1 from CG0's private ring, then reuses the same T registers for QK0/QK1 before publishing both inverses."*

以及 `:3065-3067`：

> *"Step 2: kk_epi0 + kk_epi1 — Consume both KK accumulators first so their inverse can overlap the following QK0/QK1 MMA issues in the two shared-acc stages."*

即：**先把两个 chunk 的 $KK^{\top}$ 都发出去**，让两个三角求逆尽早开始，两级 shared-acc ring 正好容纳；$QK^{\top}$ 排在后面，因为它只服务输出、不在状态链上。

### 6.3 更深的理由：求逆本身需要成对才能吃满 4 个 warp

这是原先容易漏掉的一层。`_partial_pair_inverse`（`:3207`）/ `_finish_pair_inverse`（`:3249`）用 4 个 warp **并发求逆两个 $64\times64$ 矩阵**：

```
inverse_group      = warp_id // 2      # 选哪个 chunk 的 A_inv stage
inverse_local_warp = warp_id % 2
```

| 求逆阶段 | 每矩阵需要的 warp 数 | 4 warp 能否吃满单矩阵 |
|---|---|---|
| 阶段 1（8 个 $8\times8$ 对角块，Gauss-Jordan） | 2（64 线程 = 8 块 × 8 行） | ❌ 一半空转 |
| 阶段 2（4 个 $16\times16$ 对角块） | 4 | ✅（所以它被**调用两次**，每矩阵一次，`:3240-3245`） |
| 阶段 3（2 个 $32\times32$） | 2 | ❌ |
| 阶段 4（1 个 $64\times64$） | 2 | ❌ |

> **不成对的话，阶段 1/3/4 会让半个 warp group 闲置。** 成对之后每个 warp 都有活干。
> **这是"算法的并行度上限反过来决定流水结构"的例子**：三角求逆的分治树在小尺寸层次上并行度不足（$8\times8$ 只需 64 线程），于是把两个独立的求逆任务并排放，才填满一个 warpgroup。

---

## 7. 三角求逆：8×8 起步、3 层合并、走 warp 级 MMA

对应算法文档 **杠杆二**。**注意非 CP 与 CP 用的是两套实现**（§1）。这里讲非 CP 的那套（`chunked.py:3207-3932`）。

### 7.1 结构

头部注释（`:3354-3362`）：

```
# Compute X = (I + M)^{-1} for a 64x64 unit lower-triangular matrix in-place
# on a row-major SMEM buffer.  4-stage algorithm:
#   Stage 1: Gauss-Jordan inversion of 8 diagonal 8x8 blocks (warp shuffle)
#   Stage 2: 8x8 -> 16x16 via warp MMA  (SM80_16x8x8)
#   Stage 3: 16x16 -> 32x32 via warp MMA (SM80_16x8x16)
#   Stage 4: 32x32 -> 64x64 via warp MMA, 2 warps per 64x64 tile
```

| 阶段 | 操作 | 方法 | MMA 指令 | warp/矩阵 |
|---|---|---|---|---|
| 1 | 8 个 $8\times8$ Gauss-Jordan，每线程一行，pivot 用 `shuffle_sync` 广播 | `_invert_diagonal_NxN` `:3400` | **无**（纯 SIMT） | 2 |
| 2 | 4 个 $16\times16$：$C \leftarrow -D^{-1}CA^{-1}$ | `_blockwise_diagonal_8x8_to_16x16` `:3430` | `warp.MmaF16BF16Op(..., (16,8,8))` | 4 |
| 3 | 2 个 $32\times32$ | `..._16x16_to_32x32` `:3622` | `warp.MmaF16BF16Op(..., (16,8,16))`，`permutation_mnk=(16,16,16)` | 2 |
| 4 | 1 个 $64\times64$ | `..._32x32_to_64x64` `:3775` | 同上，`permutation_mnk=(16,32,32)`，2 warp 各拥一个 16×32 切片 | 2 |

**起始粒度 $8\times8$，3 层合并（8→16→32→64）。** 每层是算法文档 §7.2 的 Schur 步，两次 GEMM：

```
Step 1: DC ← −D⁻¹ · C
Step 2: O  ←  DC  · A⁻¹
```

中间量 $-D^{-1}C$ **留在寄存器里**，靠 `_make_acc_tensor_into_a_view`（`:3364-3383`）把累加器 fragment 重解释成 A 操作数 fragment，docstring：

> *"Reinterpret accumulator tensor as an A-operand tensor for the next MMA. For SM80_16x8x8 (ratio=1) the layout is unchanged; for SM80_16x8x16 (ratio=2) the C-frag atom size differs from the A-frag atom size and requires a reshape."*

**16-bit 路径因此完全不走 SMEM 往返**；不可达的 fp32 分支就必须走 SMEM（没有对应的寄存器重解释）。

### 7.2 关键观察：求逆用的是另一套 Tensor Core

阶段 2–4 跑在**经典 warp 级 Tensor Core**（`cute.nvgpu.warp.MmaF16BF16Op` → `mma.sync.m16n8k8/k16`），**不是 `tcgen05`/UTCMMA**。数据搬运用 `ldmatrix`/`stmatrix`。阶段 1 是纯 SIMT + shuffle。

> **所以求逆与七个主 GEMM 在同一个 SM 上并发使用两条不同的 Tensor Core 路径**：求逆占 CG0 的 SIMT / warp-MMA 管线，warp 8 和 10 继续喂 `tcgen05` 单元。
> **这是把"三角求逆是异类"（算法文档 §6.3）这个缺点变成优点的做法**——正因为它用的是不同的执行单元，它才能真正与主 GEMM 重叠，而不是排队。

### 7.3 关键路径上的位置与三重掩盖

$$
KK0,KK1\ (\text{warp }8) \to KK\ \text{epilogue (CG0)} \to \textbf{求逆 (CG0)} \to \text{发布 } A_{\mathrm{inv}} \to NV\ (\text{warp }10) \to QKV \to KV
$$

三重掩盖：

1. **执行单元不同**（§7.2）
2. **warp 8 在 CG0 开始求逆之前，已经把 QK0/QK1 发进了 ring 的另两个 stage**（`:3065-3067`）
3. **两个 chunk 的求逆并发进行**（§6.3），把每 chunk 的串行求逆成本砍半

3 级 `smem_ainv_stages` 的存在正是为了让 $A_{\mathrm{inv}}$ 的生产跑在消费之前（`:319-322`）。

### 7.4 精度：`inverse_dtype == io_dtype` 的取舍

强制约束（`:388-391`）：

```python
if inverse_dtype != io_dtype:
    raise testing.CantImplementError(f"inverse_dtype={inverse_dtype} must match io_dtype={io_dtype}")
```

而 host adapter 刻意传 `io_dtype` 而非类默认的 `Float16`（`blackwell/gdn_prefill.py:210-213`）：

```python
# The kernel requires the triangular-inverse dtype to match io_dtype
# (asserted in its __init__); pass io_dtype so bf16 uses the same
# validated path as fp16 instead of the default Float16.
inverse_dtype=io_dtype,
```

**所以整个 4 阶段求逆跑在 bf16/fp16 操作数精度上**（每层内部 fp32 累加，层边界降精度）。fp32 的替代路径**代码里存在但公开 API 到不了**：每个 `_blockwise_diagonal_*` 都有完整的 `elem_type == cutlass.Float32` 分支，用 `MmaTF32Op((16,8,8))` + `cvt_f32_tf32`。

**为什么不用它**：

- 需要独立的 `sAinvCal`，$64\cdot64\cdot4\cdot3 = 49\,152$ B——**剩下的 ~6 KiB SMEM 装不下**，除非砍某个 stage 级数
- 失去 `ldmatrix`/`stmatrix`，退化成普通 32/64-bit 拷贝
- **而且它给的也只是 TF32（10 bit 尾数），不是真 fp32**

**敢这么换的依据**是算法文档 §3.4 最后一条：$\lVert k\rVert_2=1$、$\beta<1$ ⟹ 待求逆矩阵元素模长 $\le 1$，条件数有界。测试数据支持这一点（§13）。**别名换来 24–49 KB SMEM，也就是 K/V/A_inv 能开 4/3/3 级流水的空间。**

---

## 8. 零初始状态的编译期剥离

**算法依据**：$S_{\mathrm{prev}} = 0 \Rightarrow KS_{\mathrm{prev}} = QS_{\mathrm{prev}} = 0$。而这两个恰好是七个 GEMM 里最大的两个（各 $C\!\times\!d\!\times\!d$，算法文档 §3.6）。

**实现**，warp 10（`:1497-1538`）：

```python
run_cg1_mma = num_chunks > 0
if cutlass.const_expr(self.use_initial_state):
    run_cg1_mma = True
if run_cg1_mma:
    first_loop_chunk = 0
    if cutlass.const_expr(not self.use_initial_state):
        (...) = self.mma_cg1_chunk(..., True)      # 剥离出的 chunk 0
        first_loop_chunk = 1
    for chunk_idx in cutlass.range(first_loop_chunk, num_chunks):
        is_first_chunk = False
        if cutlass.const_expr(self.use_initial_state):
            is_first_chunk = chunk_idx == 0
        (...) = self.mma_cg1_chunk(..., is_first_chunk)
```

`use_initial_state == False` 时 `is_first_chunk` 在两处都是**编译期常量**，于是 `mma_cg1_chunk` 里（`:2315-2318`）`valid_state` 折叠成常量：

- 剥离体：`valid_state = False` ⟹ **KS 与 QS 两个 GEMM被完全消除**，GEMM6/7 拿到 `ACCUMULATE = kphase != 0`（写而非加），**连 QS 与状态累加器的零初始化都不需要**
- 稳态循环：`valid_state = True` ⟹ 五个 GEMM 全在

**寄存器预算后果**，逐字引用（`:253-258`）：

> *"The peeled zero-state MMA carries more pipeline cursors. Transfer twenty-four registers from each CG1 warp to each lightweight warp while retaining the same 64,512-register CTA allocation."*

即剥离让 warp 10 里活跃的 pipeline cursor 数翻倍（11 个 handle 的元组有两份），所以 `num_regs_other` 24 → 48、CG1 256 → 232。**"算法上省掉两个 GEMM"必须付的实现税。**

> ⚠️ **一处纠正**：剥离**没有**把 GEMM5 的 A 操作数从 TMEM-VKS 换成 SMEM-V。那是**死代码** `mma_warp_chunk` 干的事（`:2816-2825`）。活路径的 `mma_cg1_chunk` 两个分支代码**完全相同**（`:2351-2370`），都读 `tCtSharedInp[...,0]`。CG1 总是把 V 经 TMEM slot 0 路由，注释（`:4509-4511`）：
> *"Always publish V through fixed TMEM slot 0. The SMEM V ring cursor survives persistent work boundaries, so a new work item cannot assume that its first V tile resides in SMEM stage 0."*
> **这是 persistent kernel 带来的约束**——work tile 边界不重置 ring cursor，所以不能假设"第一个 V tile 在 stage 0"。

---

## 9. varlen：有界 TMA descriptor 让尾块与 padding 全免费

### 9.1 机制

四个 tensormap，每个 128 B，放在 per-CTA 的 GMEM workspace 里：

```python
# :192-195
bytes_per_tensormap = 128
num_tensormaps = 4
# :946-947  "TMA descriptor workspace in GMEM (one q/k/v/o descriptor set per CTA)
#            Slots: Q=0, K=1, V=2, O=3."
```

workspace 大小（`:4714-4728`）：persistent 时 $128\cdot4\cdot num\_sm$。管理器 `TensorMapManager(TensorMapUpdateMode.GMEM, 128)`（`:987-989`）。

**打补丁的是哪四个**：正好是四个走 TMA 的张量——**Q、K、V（warp 9）与 O（warp 11）**。

**改的是什么**：只改 token 维的**上界**，换成 `batch_end = cu_seqlens[batch_idx+1]`（`:1623-1650, 1742-1748`）。逐字理由（`:1622`）：

> *"Bounded descriptors zero-fill partial and padded chunks."*

```python
bounded_q = cute.make_tensor(mQ.iterator, cute.make_layout(
    (batch_end, mQ.shape[1], mQ.shape[2]), stride=(...)))
tensormap_manager.update_tensormap((bounded_q, bounded_k, bounded_v),
                                   (tma_q.atom, tma_k.atom, tma_v.atom),
                                   (tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr),
                                   self.tma_qkv_warp_id, (None, None, None))
```

> **这是本节的要点：TMA 自己在 `batch_end` 之外供零，所以 kernel 里 Q/K/V/O 上没有任何 epilogue 掩码。**
> 而且它顺带让 §6.1 那个"奇数 chunk 补成偶数"完全免费——多出来那个 chunk 读到的全是零。
> **对比 FLA**（FLA 文档 §12）：FLA 靠 `chunk_indices` 索引表在软件层规避越界，天然安全但每个 kernel 都要带一段 `if IS_VARLEN` 索引解码。**同一个正确性需求，一个用索引间接，一个给硬件描述符打补丁。**

### 9.2 哪些情况能省

- 整块工作（更新 + 所有循环）都在 `if num_chunks_b > 0:` 里（`:1621, 1740`）——**空序列完全跳过 descriptor 工作**
- **fence 被摊销**：`tma_qkv_warp` 里每个 fence 都由 `if chunk_idx == 0:` 保护（`:1905-1906, 1935-1936, 1964-1965`）——**每 (张量, work tile) 一次 fence，不是每 chunk 一次**
- gate/beta 与状态张量**不走 TMA**，所以不需要 descriptor：gate 走 `CopyUniversalOp` + 显式布尔谓词（`:2023-2028, 2055-2070`），beta 走 `cpasync.CopyG2SOp` 并**复用同一个谓词**（`:2031-2038, 2109-2117`），注释（`:1994-1996`）：
  > *"The last tile uses predicated copies: elements with linear index >= valid_tokens are out-of-bounds and receive neutral values (gate=1 -> ln=0, beta=0)."*

  **注意中性值的选择**：`gate=1 ⟹ ln=0`、`beta=0`。这正是算法文档 §3.7 那个 NaN 陷阱的第三种解法（FLA 用掩码进 `where`，FlashQLA 复制 `g[seq_end-1]`，FlashInfer 用中性值）。**三个库三种解法，都对。**
- 状态走 `cute.autovec_copy` 配 `CacheEvictionPriority.NO_ALLOCATE`

### 9.3 关于"避免 prologue peeling"

`chunked.py:1279-1302`：

```python
# Runtime first/last predicates keep one pair body in SASS instead
# of peeling a complete first-pair copy before the steady loop.
for pair_idx in cutlass.range(num_pairs_b):
    (...) = self.compute_group_0_pair(..., (pair_idx == 0, pair_idx < num_pairs_b - 1))
```

**两点纠正**：

1. 理由是**代码体积**（"one pair body in SASS"），`icache` 一词在 `gdn_kernels/` 里零命中。效果上是 icache 论证，但别把这个术语归给源码。
2. **谓词现在是残留的**：`compute_group_0_pair` 把它们解包成 `_, _ = work_args`（`:2948`）后从不使用。pair body 已经完全均匀。**设计意图（一份 body、不剥离）保留了，机制（谓词）被优化掉了。**

反过来，**warp 10 里那个状态相关的剥离是真的 `const_expr` 剥离**（§8）。所以准确描述是：**没有运行时剥离，但有一个按 `use_initial_state` 键控的编译期剥离。**

---

## 10. CP 路径：四阶段精确并行扫描

对应算法文档 **杠杆三**（仿射结合律）。**这是 FlashInfer 破解环节 ④ 串行链的方式，纯代数、无近似。**

### 10.1 数学

每个 CP shard（`cp_chunk_len` 个 token）**不用任何 shard 外的数据**就能算出一个仿射映射

$$
S_{\mathrm{out}} = M\,S_{\mathrm{in}} + N,\qquad M\in\mathbb{R}^{K\times K},\ N\in\mathbb{R}^{K\times V}
$$

参考实现的 docstring 把这点写得很明确（`tests/gdn/reference_delta_rule.py:368-380`）：

> *"The returned `(M, B)` satisfies `S_out = M @ S_in + B` for this chunk. This function intentionally receives no data outside the CP chunk."*

片上保持**转置形式**：$M^{\top}_{\mathrm{next}} = M^{\top}M_{\mathrm{block}}^{\top}$、$N^{\top}_{\mathrm{next}} = N^{\top}M_{\mathrm{block}}^{\top} + N_{\mathrm{block}}^{\top}$（参考 `:554-570`），fixup 应用 $S_i = S_{i-1}M_i + N_i$（`:707-716`）。

shard 内每 64-token 块的块级仿射映射（参考 `:660-670`）：

```python
block_state_t_HKV    = -block_gamma * (V_inv @ T) @ K
block_transfer_t_HKK =  block_gamma * I + block_gamma * ((Kᵗ @ T) @ K)
```

其中 `V_inv = Vᵗ·diag(1/γ)`，`T` 是带符号、折了 $\beta$、加了门夹心的三角求解矩阵。

### 10.2 四次 kernel launch

`gdn_cp_prefill.py:1064-1128`，`cp_delta_rule_dsl_sm100`，全在一个 `nvtx.range` 里：

| # | launch | grid | 干什么 |
|---|---|---|---|
| 1 | `cp_delta_rule_t_precompute_dsl_sm100`（`:70`） | `(H_sab · max_t_blocks_per_seq, num_seqs, 1)` | 每 **64-token 块**：$KK^{\top}$、构造 $I+\mathrm{tril}(\mathrm{diag}(\beta)KK^{\top},-1)$、用 `CollectiveInverse` 求逆、产出 $T := -(T_{\mathrm{clean}}\mathrm{diag}(\beta))^{\top}$ 到 `[total_t_blocks, H, 64, 64]` 的 **io_dtype** workspace |
| 2 | `cp_delta_rule_mn_precompute_dsl_sm100`（`:188`） | `(H_sab · max_cp_chunks_per_seq, num_seqs, 1)` | 每 **CP shard**：递推 64-token 块产出 $M^{\top}$ 与 $N^{\top}$，**两个递推都常驻 TMEM** |
| 3 | `cp_delta_rule_fixup_dsl_sm100`（`:412`） | `(num_seqs · num_heads · row_ctas, 1, 1)` | **唯一的串行阶段**：$S_i = S_{i-1}M_i + N_i$ 在每条序列的 shard 上扫；写出每个 shard 边界的 `fixed_state[i]`，并**可选写 `output_state`** |
| 4 | `cp_delta_rule_prefill_dsl_sm100`（`:662`） | `(h_r·h_qv·max_cp_chunks_per_seq, num_seqs, 1)`，**非 persistent** | 完全并行的主 pass：shard $j$ 从 `fixed_state[j-1]`（或 `initial_state` / 零）取起始状态，在自己的 64-token 块上跑普通的 7-GEMM 递推 |

**注意阶段 4 不写 final state**（`gdn_cp_prefill.py:1110-1112` 传 `None`）——序列的最终状态由**fixup** kernel 产出（`cp.py:1845-1854` 的 `store_acc(..., store_output=True)`）。这是个容易搞错的分工。

起始状态的选择（`cp_prefill.py:2192-2232`）：

```python
if cp_chunk_idx_in_seq > 0: _load_initial_state(mFixedState, ..., cp_chunk_idx - 1, ...)
else:                       _load_initial_state(mInitialState, ...) 或 _zero_initial_state(...)
```

矩形 grid 里越界的 CTA 通过 `valid_chunk_len = 0`（`cp_prefill.py:884-886`）变成 no-op，所有循环零轮。

### 10.3 M / N 的存储与显存代价

`gdn_cp_prefill.py:283-289`：

```python
workspace_shape = (total_cp_chunks, num_sab_heads, d, d)     # d = 128
transfer_t = _get_cp_workspace("gdn_cp_sm100_local_transfer", workspace_shape, torch.float32, device)
state_t    = _get_cp_workspace("gdn_cp_sm100_local_state",    workspace_shape, torch.float32, device)
```

外加第三个同形状的 fixup 输出（`:511-513`）。

- **dtype 全 fp32**（`:447-451` 断言 *"CPDeltaRuleFixupSm100 only supports float32 inputs"*）
- **为 TMA 重设 stride**：$M$ 重排成 **K-minor**、$N$ 重排成 **V-minor**（`:524-529`），以给 fixup 的 A/B 操作数所需的 major mode，配 `mark_layout_dynamic(leading_dim=0)` / `(leading_dim=1)`
- **显存代价**：$128\cdot128\cdot4\,\text{B} = 64\,\text{KiB}$ 每 (shard, head) 每数组 **× 3 个数组 = 192 KiB 每 (shard, head)**。外加 $T$ workspace：**8 KiB 每 (64-token 块, head)**

**算个实例**：$1\times64\text{k}$ token、$H=16$、`cp_chunk_len = 2048`：

$$
M/N/\text{fixed}:\ 32\ \text{shard}\times16\ \text{head}\times192\,\text{KiB} = \mathbf{96\ MiB}
\qquad
T:\ 1024\ \text{块}\times16\times8\,\text{KiB} = \mathbf{128\ MiB}
$$

> **这 224 MiB 就是精确 CP 的真实代价**，规模是 $\frac{T}{L_{cp}}H\cdot3\cdot64\,\text{KiB} + \frac{T}{64}H\cdot8\,\text{KiB}$。全部从 `_get_cache_buf` 持久池分配，跨调用复用。
> **对照 FlashQLA**：它的近似路径**根本不需要 $M$**，只在慢衰减 head 上才付 $M$ 的成本（FlashQLA 文档 §7.4，`mt` 是 bf16 且 32 KB 每 (段,head)）。**这就是"精确 vs 近似"在显存上的定价。**

### 10.4 启发式：真实的公式与常数

全部在 `gdn_kernels/delta_rule_dsl/varlen_helper.py`，**Blackwell CP 路径确实在用它**（`gdn_prefill.py:33` 导入 `should_use_cp_host`；`gdn_cp_prefill.py:23-28` 导入 `CP_CHUNK_LEN_GRANULARITY, choose_cp_chunk_len_host, max_num_chunks_host, workspace_num_chunks_host`）。

**常数**（`varlen_helper.py:8-19`）：

```python
BLK = 64
CP_CHUNK_LEN_GRANULARITY = 512
CP_SM100_PARALLELISM_THRESHOLD_DENOMINATOR = 4
CP_HBM_PARALLELISM_THRESHOLD_NUMERATOR = 1 ; ..._DENOMINATOR = 2
CP_GDDR_PARALLELISM_THRESHOLD_NUMERATOR = 1 ; ..._DENOMINATOR = 3
```

#### (a) 开不开 CP（`:90-106`）

```python
def should_use_cp_host(num_parallel_work, num_sms, device_name, device_capability=None):
    if device_capability is not None and device_capability[0] == 10:
        return num_parallel_work * CP_SM100_PARALLELISM_THRESHOLD_DENOMINATOR < num_sms
    threshold_num, threshold_den = cp_parallelism_threshold_host(device_name)
    return num_parallel_work * threshold_den < num_sms * threshold_num
```

`num_parallel_work = num_seqs · num_sab_heads`。所以 SM100/SM103 上：

$$
\boxed{\text{CP 开启} \iff B\cdot H_{\mathrm{sab}}\cdot 4 < num\_sm}
$$

B200/GB200（148 SM）⟹ $B\cdot H_{\mathrm{sab}} \le 36$ 才开。

> **这个式子就是算法文档 §6.2 并行度饥饿判据的直接编码**：`num_parallel_work < num_sm/4` 意思是"并行任务数不到 SM 数的四分之一才值得付 CP 的代价"。
> 注意 SM100 **忽略**了 HBM/GDDR 的设备名区分（那个 1/2 比例只给 SM90/SM120 用）。

#### (b) 段长（`:109-169`）—— docstring 给了完整推导

> *"Short sequences are dominated by the fixup recurrence and prefill recurrence. Balance S / C * F against C / BLK * P. S / C: Number of chunks per sequence; C / BLK: Number of prefill iterations per chunk; F: Fixup recurrence cost per iteration; P: Prefill recurrence cost per iteration. Then S / C * F = C / BLK * P => C = sqrt(S * BLK * F / P)"*

SM100 上 `ratio = (1,1)` 恒定（`(1,2)` 与 16-head 上限是 SM120 专属），所以走的是 **$\sqrt{S\cdot64}$ 规则**：

$$
\boxed{cp\_chunk\_len = \mathrm{round\_up}\bigl(\lceil\sqrt{max\_seqlen\cdot 64}\,\rceil,\ 64\bigr)}
$$

| `max_seqlen` | $\sqrt{S\cdot64}$ | `cp_chunk_len` |
|---|---|---|
| 4096 | 512 | **512** |
| 16384 | 1024 | **1024** |
| 65536 | 2048 | **2048** |

段长按 $\sqrt{S}$ 增长，正好平衡串行 fixup（$S/C$ 轮）与并行 prefill（每段 $C/64$ 轮）。

> **与 FlashQLA 的段长模型对照**：FlashQLA 是 $L_{cp}^*\propto\sqrt{BHL_c/P}$（含 SM 数与 head 数），FlashInfer 是 $\sqrt{S\cdot 64}$（只含序列长度）。**两者都是"平衡串行链与并行波"，但 FlashInfer 把 SM 数与 head 数的作用放进了 §10.4(a) 的开关判据，FlashQLA 把它们放进了段长公式本身。** 数学上是同一个优化问题的两种参数化。

另有一条"一波"目标的分支（`approx_ctas >= num_sms/2` 时走），用二分搜索找最小的 512 对齐段长使 CTA 数不超过一波。

`cp_chunk_len` 必须是 64 的倍数（`:978-979`）且是**运行时**参数，改它不重编译。

#### (c) fixup kernel 变体选择（`gdn_cp_prefill.py:544-556`）

```python
# SIMT4 sustains two CTAs/SM. UTC64 and UTC128 launch D/rows_per_cta
# CTAs per state, so UTC64 switches out when its two-CTA grid exceeds one wave.
simt_row4_one_wave_states = num_sms * 2 // (d // 4)     # = num_sms // 16
utcmma64_one_wave_states  = num_sms // (d // 64)        # = num_sms // 2
if   num_parallel_states <= simt_row4_one_wave_states: _kernel_kind = "simt_row4"
elif num_parallel_states <= utcmma64_one_wave_states:  _kernel_kind = "utcmma64"
else:                                                  _kernel_kind = "utcmma128"
```

B200（148 SM）：$B\!\cdot\!H \le 18$ 走 `simt_row4`，$\le 74$ 走 `utcmma64`，否则 `utcmma128`。

**注意低并行度的两档是 fp32 SIMT 变体**（`CPDeltaRuleFixupSimtSm120`，128 线程、`min_blocks_per_mp=2`、256 寄存器/线程），高并行度那档是 `tcgen05` 的 **TF32 UTCMMA** 变体。**低并行度时精度更高、用 SIMT；高并行度时用 TF32 换吞吐。** 这是一个隐性的精度/性能耦合，见 §13。

#### (d) 其他 CP 拒绝条件（`gdn_prefill.py:49-115`）

SM100 要求 `cuda_major >= 13`；**CP 完全不支持 state checkpointing**（*"CP delta rule does not support state checkpointing yet"*）；要求 `head_size == 128`、fp16/bf16 且 q/k/v/o dtype 一致、g/beta 为 fp32 连续、q/k/v/o 连续、`initial_state` 内层 `[H,V,K]` 连续。`use_cp is True` 时任一不满足 ⟹ `ValueError`；`use_cp == "auto"` ⟹ `RuntimeWarning` + 静默回落非 CP（`:402-410`）。

### 10.5 CP 的求逆/衰减顺序：与非 CP 相反

**这是整个代码库里最有意思的数学对比。**

阶段 1 求逆的是 **gate-free** 矩阵（参考 `reference_delta_rule.py:497-551`）：

```python
IKK = identity_add_strict_lower_diagonal(beta_HS1 * (k_HSK @ k_HSKᵗ))    # 没有 Gamma
t_clean = torch.inverse(IKK) * beta_HS1ᵗ
t_HSS[blk] = (-t_clean.transpose(-2,-1))                                 # 带符号 + 转置
```

门由**消费方** kernel 以两侧夹心的形式施加（`gated_delta_net_cp_prefill.py:1865-1877`）：

```python
for i in cutlass.range_constexpr(cute.size(tTrT)):
    t, s = tTcT[i]
    pred = s >= t
    if is_final_block:
        pred = pred and s < valid_tokens and t < valid_tokens
    gamma = cutlass.Float32(0.0)
    if pred:
        gamma = cute.math.exp2(sCumsumlog[s, ...] - sCumsumlog[t, ...], fastmath=True)
    tTrT[i] = self.io_dtype(-gamma * cutlass.Float32(tTrT[i]))
```

CP prefill kernel 的头部 docstring（`cp_prefill.py:30-35`）：

> *"Each CTA processes one CP chunk as a recurrence of 64-token blocks. It follows the optimized non-CP SM100 state pipeline, but replaces the KK and hierarchical inverse path with a TMA load of the signed, beta-folded inverse produced by CP preprocessing. CG0 applies the gate sandwich and publishes the same A-inverse operand contract consumed by the common recurrence."*

**为什么两条路选相反的方向**：

| | 非 CP | CP |
|---|---|---|
| Λ 的位置 | 求逆**前**（折进矩阵） | 求逆**后**（门夹心） |
| $T$ 的复用需求 | 无——一趟算完就丢 | **每 64-token 块算一次，被阶段 2（MN 预计算）和阶段 4（主 prefill）两个 kernel 复用** |
| 后果 | 少一次逐元素 pass | $T$ **与门无关** ⟹ 可跨 kernel 复用 |

> **这与 FlashQLA 让 gate-free `A` 服务三个 pass 是同一个动机**（FlashQLA 文档 §3.3）。算法文档 §7.1 说"$A_{\mathrm{raw}}$ 门无关带来的复用收益主要对训练有意义"——**这里给出了一个纯推理场景的反例：跨 kernel 复用同样需要门无关。**

CP prefill kernel 的资源后果：

- `num_tensormaps = 5`（Q=0, K=1, V=2, **T=3**, O=4，`cp_prefill.py:156-157, 983`）
- `smem_k_stages = 3` 而非 4，注释（`:294`）：*"CP does not issue KK, so three K stages cover the QK/CG1 consumers."*
- 新增 `smem_t_stages = 2`
- `tmem_cg0_shared_acc` 只装 QK，注释（`:316-317`）：*"CG0 owns QK and CG1 owns KS/NV."*
- **TMEM 偏移表与其他所有 stage 深度与非 CP kernel 逐字节相同**（`cp_prefill.py:312-337` vs `chunked.py:337-361`）
- $T$ tile 以 fp16/bf16 的 `[64,64]` TMA 载入，CG0 用 `LdMatrix(transpose=False)` → 乘门 → `StMatrix(transpose=True)` 变成 `A_inv` 操作数（`:1844-1896`）

### 10.6 "精确"到什么程度

**不是逐位精确——只在代数意义上精确，数值上在浮点容差内验证。**

仿射复合在数学上是精确的（$S_{\mathrm{out}} = MS_{\mathrm{in}}+N$ 满足结合律，所以 fixup 恢复的是真正的 chunk 边界状态），但实现上与非 CP 不逐位相同，原因有三：

1. **阶段 3 用 TF32 MMA 操作数**（`gated_delta_net_cp.py:1541` `self.mma_dtype = cutlass.TFloat32`，`:1905-1913` 的 `tcgen05.MmaTF32Op`），fp32 TMEM 累加器。`convert_acc_to_opd`（`:1661-1709`）每轮显式 fp32 → TF32 降精度。
2. $M/N$ 的乘积是非 CP 从不执行的额外 fp32 矩阵乘。
3. $T$ 在 io_dtype 里预计算并重新载入，而非就地算出。

测试如实反映（§13）：fixup 阶段 `atol=rtol=2e-3`；CP vs 非 CP 端到端 `4e-2`~`5e-2`。**唯一断言逐位相等的地方是 CP 的确定性 / wrapper 等价性**（`test_prefill_cp_delta_rule.py:959-960` 的默认容差 `assert_close`）。**全仓库没有任何 CP↔非 CP 逐位相等的声明。**

参考实现的 docstring（`reference_delta_rule.py:773-778`）：

> *"Each chunk independently computes an affine transfer `S_out = M S_in + B`. The fixup pass applies those chunk transfers to recover true global chunk-boundary states. The final fixed state should match `delta_rule`."*

> **所以准确表述是**：算法文档 §7.3 说这条路"精确、无失效条件"——数学上对；实现上它是"**代数精确 + 浮点误差**"，误差量级 $\sim 10^{-3}\!\sim\!10^{-2}$（因为 TF32 fixup 与额外的矩阵乘）。
> **与 FlashQLA 的近似（$4.5\times10^{-5}$，算法文档 §8.2）相比，FlashInfer CP 的实际数值误差反而更大。** 这是个反直觉但重要的结论：**"精确算法"的浮点实现不一定比"近似算法"的浮点实现更准。**

---

## 11. 推理侧设计：分页状态池与状态量化

### 11.1 分页状态池就地读写

公开 API 文档（`gdn_prefill.py:223-239`，逐字节选）：

> *"`state_indices`: Int32 tensor of shape `[num_seqs]` (SM100/SM103 only). When provided, `initial_state` and `output_state` are treated as a state pool whose first dimension is indexed by these slot ids rather than laid out in sequence order: sequence `i` reads its initial state from row `state_indices[i]` and writes its final state back to the same row (in place when `output_state is initial_state`). This lets callers that keep a paged/indexed state pool avoid gathering the active rows into a packed buffer and scattering the result back. The pool may be non-compact (padded first-dimension stride). ... The ids **must be unique**: ... Uniqueness is a caller precondition (not checked at launch, to avoid a per-call host sync)."*

**只有 SM100 支持**（`:366-385` 对其他架构 `raise NotImplementedError`）。

**非紧凑 stride 的处理**是 adapter 里最值得引用的一段注释（`blackwell/gdn_prefill.py:95-124`）：

> *"Pool/indexed mode: the caller passes its real SSM state pool `[N_pool, H, V, K]` whose dim-0 (slot) stride is padded by the mamba conv+ssm cache packing, i.e. `stride[0] > H*V*K` -> the layout is NON-COMPACT. `mark_compact_shape_dynamic` asserts a compact layout and raises `RuntimeError`. `mark_layout_dynamic()` alone pins the single stride-1 dim (mode 3 = K) to stride 1 and carries every other stride (including the padded dim-0) through as a dynamic runtime value. ... cutlass-dsl offers no way to attach a divisibility hint without also requiring compactness, so this path drops the `divisibility=DK` hint; the stride-1 K dim is retained, so the 128x128 state autovec copy still vectorizes (possibly a narrower vector). That copy is a negligible fraction of the kernel, so correctness is kept with no meaningful perf cost."*

```python
if use_state_indices:
    s_cute.mark_layout_dynamic()
else:
    s_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
        mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
```

device 侧的间接只是一次 load，出现在三处（`:3980-3983, 4033-4036, 4131-4134`）：

```python
if cutlass.const_expr(mS_indices is not None):
    state_row = mS_indices[batch_idx]
else:
    state_row = batch_idx
```

**就地安全**是因为一个 work tile 在 prologue 把 `S_init[row]` 读进 TMEM、在 epilogue 写 `S_out[row]`；slot id 唯一 ⟹ 没有两个 tile 碰同一行。

**零长序列**走 `_store_empty_final_state`（`:4020-4044`），逐元素把 `mS_init[row]` 拷到 `mS_out[row]`，**保住那一行而不是清掉它**。

> **收益**：免掉"先 gather 到连续 buffer、算完再 scatter 回去"这两次完整的状态拷贝。
> **代价**：非紧凑布局无法给最内维加整除性提示，状态拷贝的向量化宽度可能变窄——但状态拷贝在整个 kernel 里占比极小。
> **这是三个库里唯一支持分页状态池的实现**（FlashQLA 文档 §10 明确没有）。对接 vLLM/SGLang 那种 mamba 风格状态池时这是决定性差异。

### 11.2 状态 I/O 量化：五种 dtype

`blackwell/gdn_prefill.py:77-92`：

```python
torch.float32       -> cutlass.Float32
torch.bfloat16      -> cutlass.BFloat16
torch.float16       -> cutlass.Float16
torch.float8_e4m3fn -> cutlass.Float8E4M3FN
torch.float8_e5m2   -> cutlass.Float8E5M2
```

**两种 fp8 都支持。** 从 `initial_state.dtype` 自动推断（`:176-183`），进编译 key。

片上精度**无条件 fp32**（`acc_dtype = Float32`，`can_implement:384` 强制）。转换：

- 载入（`chunked.py:3990-4009`）：GMEM `state_dtype` → RMEM → `.to(acc_dtype)` → TMEM fp32（`St32x32bOp(Repetition(32))`）
- 写回（`:4104-4144`）：TMEM fp32 → `.to(state_dtype)` → GMEM
- `if cutlass.const_expr(self.acc_dtype != self.state_dtype)` 让 fp32 情形成为零指令的直通

> **算法依据是算法文档 §7.4 用途 2**：GDN 有遗忘门，量化误差进入状态后随 $e^{\sum g}$ 指数衰减，**有界且自愈**，不会像纯累加型线性注意力那样无限累积。**这是"I/O 量化，不是计算量化"。**
> 显存与带宽减到 1/4（fp8 vs fp32），代价见 §13 的 `atol_o = 1e-1`。

### 11.3 一个不那么显眼的量化

**GEMM3/GEMM4 的状态操作数被降到 `io_dtype`，而不是 `state_dtype`**（`:2300, 4429-4436`）：

```python
tCtState_inp = cute.make_tensor(
    cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype), ...)
tRT_rState_inp[None,0,None].store(tTR_rState[None,0,None].load().to(self.io_dtype))
```

**所以即使 `state_dtype=float32`，$KS_{\mathrm{prev}}$ 与 $QS_{\mathrm{prev}}$ 也是从 fp32 状态的 bf16/fp16 副本算出来的。** fp32 累加器只负责跨 chunk 携带，16-bit 副本只是 MMA 操作数。

> **这是 chunk 间项的主要精度限制**，也解释了 §13 里输出容差为什么相对宽松。**注意这不是可选项——`tcgen05` 的 fp16/bf16 MMA 只接受 16-bit 操作数**，任何走 Tensor Core 的实现都必须付这笔（FlashQLA 的 `h_shared` 是 bf16，同理）。算法文档 §8.5 那张表把"操作数 bf16"列为一般原则，这里是它的具体后果。

---

## 12. tile scheduler

`gated_delta_net_tile_scheduler.py`，头部 docstring（`:29-38`）：

> *"Each tile = one (batch, head) pair. The assigned CTA loops over all chunks for that tile sequentially, which is required because the recurrent state S must be propagated chunk-by-chunk.*
> *Persistent grid shape: `(min(B * H, max_active_clusters), 1, 1)`*
> *Non-persistent grid shape: `(B, H, 1)`"*

- **工作分解**：tile = `(batch, head)`。**序列方向没有任何切分**——chunk 循环天生串行。`total_tiles = num_seqs · num_o_heads`
- **persistent？** 非 CP 路径**是**（`blackwell/gdn_prefill.py:226`），grid `(min(total_tiles, num_sm), 1, 1)`。CP prefill **不是**（`gdn_cp_prefill.py:657`），而且它压根不用这个 scheduler——直接从 `blockIdx` 推工作
- **CLC（cluster launch control）？没有。** `clc|cluster_launch_control|StaticPersistentTileScheduler` 全部零命中。分发是普通的跨步计数器：
  ```python
  self._current_work_linear_idx += Int32(advance_count) * Int32(self.num_persistent_ctas)
  ```
  `cluster_shape_mnk = (1,1,1)`、`CtaGroup.ONE`、`use_2cta_instrs = False` ⟹ **没有 cluster，CLC 也就无从分发**
- **负载均衡只有两个机制**：
  1. **tile 顺序是 head-major**（`:231-237`），用 host 侧预计算的 `cute.FastDivmodDivisor` 解码（两次乘移位而非两次整除）：
     ```python
     # Tile ordering: head-major, i.e. linear_idx = batch * num_o_heads + head.
     remain_work_idx, head_idx = divmod(linear_idx, self.params.num_heads_fdd)
     _, batch_idx = divmod(remain_work_idx, self.params.num_seqs_fdd)
     ```
     **head-major 意味着连续编号的 CTA 拿到的是同一条序列的不同 head ⟹ chunk 数完全相同 ⟹ 首波天然完美均衡，与 varlen 组合无关。** 这是一个很便宜的好设计。
  2. **round-robin 回绕**：`total_tiles > num_sm` 时每个 CTA 处理 `bidx, bidx+G, bidx+2G, …`
- **但没有 work stealing、没有动态队列、没有按长度排序。** 一条 64k token 序列加一堆 64 token 序列，长 tile 的那个 CTA 会在尾部独自跑。**这个长尾正是 CP 路径要解决的东西。**
- **没有 shard 维度**：`WorkTileInfo` 返回 `(batch_idx, head_idx, 1)`（`:236`），第三个坐标是常量 1，所有消费者都丢弃。**非 CP scheduler 里不存在 (batch, head, shard) 均衡；分片只存在于 CP 路径，作为 grid 的 x/y 维。**
- **每个 warp 角色各建一个 scheduler 实例**，独立走同一条确定性序列（`chunked.py:1255, 1322, 1450, 1486, 1593, 1716`）。kernel docstring（`:956-962`）：
  > *"Warp specialization is the outermost control flow: each warp role owns its own persistent tile-scheduler loop."*

  **没有跨 warp 的工作广播。** 这也是为什么空序列的守卫必须在每个角色里都是同一粒度——注释（`:1620`）：*"All warp roles skip empty workloads at the same granularity."*

---

## 13. 精度与测试

### 13.1 累加器精度全表（非 CP）

| 量 | 位置 | 精度 |
|---|---|---|
| 7 个 tcgen05 GEMM 累加器 | TMEM | **fp32**（强制） |
| tcgen05 GEMM **操作数** | SMEM / TMEM | **io_dtype = bf16/fp16** |
| 门：`log2` / 前缀扫描 / `exp2` | 寄存器 | **fp32**（`:2050-2091`） |
| Λ、β 乘到 KK / QK 上 | 寄存器 | **fp32**，再 `.to(io_dtype)` |
| 求逆阶段 1 | 寄存器 | io_dtype → **fp32** 消元 → io_dtype |
| 求逆阶段 2–4 | 寄存器 | 操作数 io_dtype，**fp32 累加**，逐层降精度 |
| 递归状态跨 chunk | TMEM | **fp32**，原地乘 $\gamma_{\mathrm{end}}$ |
| **$S_{\mathrm{prev}}$ 作 GEMM3/4 操作数** | TMEM | **降到 io_dtype** ← **主要误差来源**（§11.3） |
| $V-\gamma KS$ | 寄存器 | KS fp32 → γ 乘 fp32 → `.to(io_dtype)` |
| QS 重缩放 | 寄存器 | **fp32**，写回 fp32 TMEM |
| 衰减差 | 寄存器 | fp32，走 `cute.arch.add_packed_f32x2(ftz=False, rnd="rn")` |
| O 输出 | 寄存器 → SMEM | fp32 → io_dtype |
| 状态 GMEM I/O | GMEM | `state_dtype` 5 选 1，片上 fp32 |

超越函数用 fast-math：`cute.math.log2(x + 1e-10, fastmath=True)`、`cute.math.exp2(..., fastmath=True)`。

完全符合算法文档 §8.5。

### 13.2 测试容差

`tests/gdn/test_prefill_delta_rule.py` 主判据（`:164-178`），参考是 fp32 的 `blockwise_delta_rule`：

```python
if dtype == torch.bfloat16:
    atol_o, rtol_o = 1e-2, 1e-2 ;  atol_kv, rtol_kv = 5e-3, 1e-3
else:  # float16
    atol_o, rtol_o = 2e-3, 1e-3 ;  atol_kv, rtol_kv = 1e-3, 1e-4
```

> **bf16 的输出容差比 fp16 宽 5 倍**——与"bf16 三角求逆 + bf16 状态操作数"一致。

其他：

| 测试 | 容差 |
|---|---|
| `test_prefill_block_end_decay`（近 1 的门，$\alpha = 0.99+0.01r$） | `o: 2e-3/1e-3`，`state: 1e-3/1e-4` |
| varlen / 空序列等价性 | `2e-2` |
| **零长序列的状态行逐位不变** | `atol=0, rtol=0` |
| 状态往返 | `1e-3/1e-4` |
| CP vs 非 CP wrapper 等价 | **默认容差** `assert_close` |
| **`test_prefill_kernel_state_dtype`**（bf16/fp16/**fp8e4m3/fp8e5m2** 状态） | `atol_o = 1e-1`（fp32 时 `5e-2`），`rtol_o = 5e-2`，`atol_kv = 1e-1` |

**fp8 状态那条要小心读**：参考实现本身也用同一个 `state_dtype` 跑（`:1155-1165`），所以它验证的是**量化方案**，不是 fp32 保真度。`atol_o = 1e-1` 是 bf16 状态容差的 10 倍——**fp8 状态是一个真实的精度代价，不是免费的。**

CP 测试（`tests/gdn/test_prefill_cp_delta_rule.py`）：

| 项 | 容差 |
|---|---|
| fixup（TF32） | `FIXUP_TF32_ATOL = FIXUP_TF32_RTOL = 2e-3`（`:77-78`） |
| T 预计算 vs 参考 | `5e-3`；varlen 尾部**恰好为零** `atol=0, rtol=0` |
| MN 预计算 | bf16 `5e-3/2e-3`；fp16 `1e-3/5e-4` |
| **CP prefill vs 非 CP prefill** | `4e-2` |
| CP 链路 long/small-BH、含初始状态的 e2e | `5e-2` |
| CP e2e vs 参考 | bf16 `o: 2e-2/2e-2, state: 1e-2/5e-3`；fp16 `o: 5e-3/5e-3, state: 1e-3/1e-3` |

`tests/gdn/test_prefill_state_indices.py` 是**唯一有硬性精确断言的文件**，全部对着**同一 kernel 的 packed 非索引基线**：

```python
:119-120  assert not pool.is_contiguous()             # dim-0 stride 被 padding
          assert pool.stride()[1:] == (D*D, D, 1)     # 内层 [H,V,K] 连续
:165      assert len(set(perm)) == num_seqs           # slot 互不相同
:172      assert torch.equal(output_a, output_b)
:174      assert torch.equal(final_a, pool[perm])
:178      assert torch.equal(pool[untouched], torch.zeros_like(pool[untouched]))
```

> **分页池路径被断言为与 packed 路径逐位相同，包括"未触碰的池行绝不被写"。** 这是很强的保证。

**解读所有容差时必须知道的前提**：测试里 **K 恒被 L2 归一化**（`:113, 198` 等，注释 *"l2 norm k to avoid numerical instability"*），且 `alpha=False and beta=False` 的配置被 **skip**，理由 *"large diff due to output value amplitude explosion along token dimension"*（`:231-234`）。**所有容差都以"门表现良好 + key 单位范数"为条件。** 这恰好是算法文档 §3.4 与 §7.4 依赖的同一组假设。

---

## 14. SM90 对照：同一个不变式，不同的落点

`gdn_kernels/delta_rule_dsl/delta_rule_sm90.py`（2467 行），类 `_FullyFusedDeltaRuleSm90`。

**一句话点题** —— `delta_rule_sm90.py:716`，状态参数上的注释：

```python
# Running KV state (D×D fp32, in registers across all blocks)
tKVrKV: cute.Tensor,
```

**SM90 上那个 $128\times128$ fp32 递归状态住在寄存器文件里，铺在两个协作的数学 warpgroup 上。SM100 上它住在 TMEM 的 0–127 列。**

这一个放置决定向下传播到每一个链式 GEMM。SM90 上四个状态/输出 GEMM 全部从**寄存器**取 A 操作数，走 `wgmma` 的 RS 形式（`:730-767`）：

```python
mma_atom_o1   = warpgroup.MmaF16BF16Op(dtype, acc, (64, blk_q,  16), warpgroup.OperandSource.RMEM, K, K)
mma_atom_sk   = warpgroup.MmaF16BF16Op(dtype, acc, (64, blk_kv, 16), warpgroup.OperandSource.RMEM, K, K)
mma_atom_newv = warpgroup.MmaF16BF16Op(dtype, acc, (64, blk_kv, 16), warpgroup.OperandSource.RMEM, K, K)
# "O1/O2/SK/NewV: two state warpgroups cooperate as in the C++ Hopper GMMA path."
o1_tiled_mma = cute.make_tiled_mma(mma_atom_o1, cute.make_layout((2, 1, 1)))
```

而 KK 与 QK——唯一操作数直接来自 HBM 的两个 GEMM——用 `OperandSource.SMEM`（`:1587-1604`）。**对比 `chunked.py:559-616`：SM100 上出现的是同一个切分，只是"寄存器"被换成了"TMEM"。**

| | SM90 | SM100 |
|---|---|---|
| 链式中间量（S, KS, NV, decayV） | **寄存器**，`wgmma` A-from-RMEM | **TMEM**，`tcgen05.mma` A-from-TMEM |
| 线程 / warp | **512**（4 warpgroup） | **384**（12 warp） |
| warp 角色 | `{LDST, MATH_STATE0, MATH_STATE1, MATH_AUX}` + 子角色 | 2 个计算组 + **2 个独立单线程 MMA 发射流** + TMA + epilogue |
| 谁发 MMA | 数学 warpgroup 自己发 `wgmma`（数据路径与 MMA 融在同 128 线程） | 专用发射 warp 8/10；计算组只碰 TMEM/寄存器 |
| 状态 GEMM 的 warpgroup 协作 | `make_tiled_mma(atom, (2,1,1))`——**两个** warpgroup 共享一条 MMA，状态被**切在 256 线程上** | 一条 `CtaGroup.ONE` 的 tcgen05 指令，状态是一整块 TMEM |
| 寄存器压力 | **支配整个设计**：`get_register_requirements`（`:60-84`）算 `load_registers = 40 - 2*granularity`、`aux = 128 - load`，状态 warpgroup 走 `round_down(...)`，上限 `min(248, ...)` | 寄存器近乎充裕（CG0 224 / CG1 256 / 其他 24）；**约束变成 512 列 TMEM** |
| 三角求逆 | `CollectiveInverse`，1 warpgroup，单个 $64\times64$，硬 fp16 | 自己的 fork，4 warp，**两个 $64\times64$ 并行**，io_dtype |
| grid | 每 tile 一个 CTA，非 persistent | **persistent** |
| chunk 成对 | 无 | 有 |

> **叙事**：算法完全相同（同样 7 个 GEMM、同样的合并 NV 形式、同样的分层求逆），**不变式也完全相同——链式中间量绝不能往返 HBM，理想情况连 SMEM 都不走**。
> Hopper 用 `wgmma` 的 A-from-registers 满足这个不变式，代价是状态必须切在 256 线程上，于是**寄存器分配成了核心设计问题**（所以有 512 线程 4 warpgroup 布局和那个算术化的寄存器预算函数）。
> Blackwell 用 TMEM 满足**同一个**不变式，于是寄存器解放、一个 warp 就能为整个 $128\times128$ 累加器发 MMA，**预算问题搬到了 512 列 TMEM 上**——而这反过来才使"双独立发射流 + 双计算组"这种专用化在仅 384 线程下成为可能。
> **同一个性质，不同的落点，不同的约束在起作用。**

---

## 15. 性能数据（仓库里没有）

**`flashinfer` 仓库里没有任何绝对性能数字、TFLOPS 数据或加速比声明**——注释、docstring、`docs/`、所有 markdown 里都没有。存在的只有**测量基础设施**。

`benchmarks/bench_gdn_prefill.py`：

- 头部（`:17-26`）：*"Compares FlashInfer GDN prefill against FLA baseline across Qwen3.5 family model configurations."*
- 基线是 `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_fwd`（Triton）
- `HEAD_CONFIGS`（`:49-64`）就是 Qwen3.5 的形状表（算法文档 §2.2 那张表的来源）
- `SEQ_CONFIGS`（`:66-84`）：15 种 varlen 形状，`1x2048` 到 `8192x32`，含刻意倾斜的组合（`6144+2048`、`1024+7168`）
- 方法学细节：**旋转输入 buffer 到约 4 GB 以击穿 L2 复用**（`:93-98, 122-126`）、`bench_gpu_time(..., enable_cupti=, use_cuda_graph=)`、默认 `warmup=5, iters=20`、测量间隔 0.1 s 冷却

> ⚠️ **FLOP 模型严重低估**（`:87-90`）：
> ```python
> def _gdn_tflops(total_tokens, h_v, d, time_ms):
>     """Calculate TFLOPS: 2 GEMMs (kv outer product + q@state) per token per head."""
>     flops = 2 * 2 * total_tokens * h_v * d * d
> ```
> `benchmarks/routines/gdn.py:386-393` 同一模型，docstring 说 *"Intra-chunk attention terms are excluded for consistency with that convention."*
> **它只数了 7 个 GEMM 里的 2 个。用这个脚本产出的任何 TFLOPS 数字都比实际执行量低估约 3–4 倍。** 引用时必须说明。

`benchmarks/README.md:86` 是唯一的散文陈述，而且**已过期**：它说 SM90 是 C++ 实现，实际上 `gdn_prefill.py:546-574` 分派到 `delta_rule_dsl/delta_rule_sm90.py`，也是 CuTe DSL（`:2465` 的 `cute.GPUArch("sm_90a")`）。**两个架构都是 CuTe DSL。**

**唯一可引用的第三方实测**在 FlashQLA 仓库里（FlashQLA 文档 §12，GB200，flashinfer 0.6.14）：低 head 数 + 长序列时 FlashInfer 明显落后 FlashQLA（TP8 32k：1.600 ms vs 0.275 ms），高 head 数 + 多序列时 FlashInfer 反超（`8192x4`：0.414 ms vs 0.513 ms）。

> **对这组数据要谨慎**：那是 flashinfer **0.6.14**，而本文分析的 HEAD 是 `e9fb62b7`，CP 路径的成熟度可能不同。而且按 §10.4(a) 的判据，TP8（$B\!\cdot\!H_{\mathrm{sab}} = 8$，$8\cdot4=32 < 148$）本该开启 CP。**结论只能说：那个版本在那个配置下 CP 未生效或收益未兑现。** 想下结论必须在目标硬件上重测，并确认 `will_use_cp` 的实际取值。

---

## 16. 汇总

### 16.1 优化清单

| 优化 | 算法依据 | 换来什么 | 付出什么 |
|---|---|---|---|
| 依赖图 → warp 分组 + **双 MMA 发射流** | ①② 不依赖状态，③④⑤ 依赖 | 无依赖的 GEMM 可任意超前发射 | 组间同步逻辑复杂；两套完整 pipeline handle |
| CG0 / CG1 **独立** TMEM 累加器 ring | 同上 | 关键路径上没有累加器所有权交接 | 多占 TMEM（2×64 + 1×64 列） |
| 链式中间量常驻 TMEM，全程不经 SMEM | 链上中间量只被下一步用一次 | 5 个 GEMM 免掉 SMEM 往返 | 数学式要按操作数位置改写；TMEM 成为新瓶颈 |
| **QS 独占累加器** | 输出式 ③ 里 inter 与 intra 相隔多个 GEMM | $O=O_{\mathrm{inter}}+O_{\mathrm{intra}}$ 是累加器内零流量加法 | 64 列 TMEM |
| TMEM slot 0 时间复用（VKS → NV） | 生命期不重叠 | 省 32 列 | 三个 ready-only barrier 定序 |
| 两 chunk 成对流水 | ①求逆延迟长在关键路径；②**求逆分治树小尺寸层并行度不足** | 求逆延迟被邻居掩盖 + 4 warp 吃满 | 活跃集变大 ⟹ K 4 级、A_inv 3 级 |
| Λ 一次算两处用（跨 pair 两个 chunk 一起构造） | Λ 在公式里出现两次 | `exp2` 次数减半 | 两处后处理必须在同一 warp 组 |
| **`sK` / `sK_trans` 双 descriptor 别名** | 公式 ② 需要 $K^{\top}$ 视图 | **省 64 KB SMEM** | 布局约束求解复杂 |
| `sAinvCal` 别名 `sAinv`（求逆精度 = I/O 精度） | $\lVert k\rVert=1,\beta<1$ ⟹ 元素有界 | 省 24–49 KB SMEM，换来 4/3/3 级流水 | 求逆尾数降到 8/11 bit |
| 求逆走 **warp 级 MMA**（非 tcgen05） | 求逆是"异类"、粒度小 | **与主 GEMM 使用不同执行单元 ⟹ 真重叠** | 两条 Tensor Core 路径共存，代码复杂 |
| 零状态首 chunk 编译期剥离 | $S=0\Rightarrow KS=QS=0$ | 省掉最大的两个 GEMM + 免累加器零初始化 | 多一套 pipeline cursor，寄存器重分配 24→48 |
| **有界 TMA descriptor 运行时打补丁** | — | varlen 尾块与奇偶 padding **全免费**，Q/K/V/O 零掩码 | 每 CTA 4 个 descriptor 槽位；fence 摊到每 tile |
| 运行时谓词代替 peeling（现已退化） | — | SASS 里只有一份 pair body | 现在是残留代码 |
| head-major tile 顺序 + FastDivmod | 同序列不同 head 的 chunk 数相同 | 首波天然完美均衡 | 无 work stealing，长尾暴露 |
| **分页状态池就地读写 + 非紧凑 stride** | — | 免两次全状态 gather/scatter | 丢掉整除性提示，向量化可能变窄 |
| **状态 fp8/bf16/fp16 I/O 量化** | 杠杆四用途 2（遗忘门让噪声自愈） | 显存与带宽减到 1/4 | 输出容差从 1e-2 放到 1e-1 |
| **CP：四阶段精确并行扫描** | **杠杆三**（仿射结合律） | 精确破解串行链，并行度 $\to B H N_{\mathrm{shard}}$ | **每 (shard,head) 192 KiB + 每块 8 KiB 的 workspace**；4 次 launch；fixup 用 TF32 引入 $\sim2\times10^{-3}$ 误差 |
| **CP 的 gate-free 求逆 + 门夹心** | 杠杆一 | $T$ 门无关 ⟹ **跨两个 kernel 复用** | 消费方多一次逐元素 pass |

### 16.2 两条路的分工

| | 非 CP（`gated_delta_net_chunked.py`） | CP（4 个 kernel） |
|---|---|---|
| 何时启用 | $B\cdot H_{\mathrm{sab}}\cdot4 \ge num\_sm$ | $B\cdot H_{\mathrm{sab}}\cdot4 < num\_sm$ |
| 破解环节 ④ | **不破解**（每 tile 一个 CTA 串行走 chunk） | **用杠杆三精确破解** |
| Λ 位置 | 求逆**前** | 求逆**后**（门夹心） |
| 求逆实现 | 自己的 fork，8×8 起步，成对 | `CollectiveInverse`（阶段 1）；阶段 4 不求逆 |
| 状态 checkpoint | 支持 | **不支持** |
| 分页状态池 | 支持 | 由 host 编排 |
| 编译目标 | 不传 `GPUArch` | `sm_{major}{minor}a`（B300 上即 `sm_103a`） |
| 额外 workspace | 4 个 tensormap × num_sm | **192 KiB/(shard,head) + 8 KiB/(块,head)** |

### 16.3 一句话

> FlashInfer 的 SM100 GDN prefill 是**两条互补的路**。
> **非 CP 路径**是一个 384 线程、12 warp、7 个 GEMM 的全融合巨核，把算法文档 §5 那张依赖表**原封不动画成了 warp 分组的边界**：不依赖状态的工作交给 CG0 + 独立发射流 warp 8，依赖状态的交给 CG1 + warp 10，两组各有不相交的 TMEM 累加器 ring。所有链式中间量常驻 TMEM，**TMEM 512 列被恰好用满**，SMEM 226 KB 里靠两处别名（`sK`/`sK_trans` 省 64 KB、`sAinvCal` 省 24–49 KB）挤出 4/3/3 级流水。它**不解决并行度饥饿**——每 tile 一个 CTA 串行走 chunk。
> **CP 路径**用杠杆三（仿射结合律）**精确**破解串行链：T 预计算 → MN 预计算 → 串行 fixup → 并行 prefill，四次 launch。开关判据 $B\cdot H_{\mathrm{sab}}\cdot4 < num\_sm$ 就是并行度饥饿的直接编码，段长 $\sqrt{S\cdot64}$ 平衡串行 fixup 与并行 prefill。
> **它与 FlashQLA 的根本差别是"精确 vs 近似"的定价**：FlashInfer 必须显式构造并存储 $M$（每 (shard,head) 192 KiB 加 $T$ 的 8 KiB/块，64k token 下约 224 MiB），FlashQLA 的近似路径根本不构造 $M$。**但反直觉的是**，FlashInfer CP 的实际浮点误差（fixup 走 TF32，$\sim2\times10^{-3}$）比 FlashQLA 近似路径的理论误差（$4.5\times10^{-5}$）**更大**——"精确算法"的浮点实现不一定更准。
> 独有的两项工程能力值得单记：**分页状态池就地读写**（免两次全状态拷贝，还处理被 padding 的非紧凑 stride）与**状态 fp8 I/O 量化**——三个库里只有它做，两者都是 serving 集成的决定性差异。
> 最后一条实用提醒：**这个仓库里没有任何性能数字**，而它自带的 benchmark FLOP 模型只数了 7 个 GEMM 里的 2 个，**TFLOPS 低估约 3–4 倍**。

---

## 相关文档

- [`GDN_Algorithm.md`](GDN_Algorithm.md)：算法推导、依赖结构、四个数学杠杆（本文用了杠杆一、二、三、四）
- [`FLA_Triton_Baseline.md`](FLA_Triton_Baseline.md)：FlashInfer benchmark 的对照基线
- [`FlashQLA_GDN_Blackwell.md`](FlashQLA_GDN_Blackwell.md)：用杠杆四近似破解串行链的另一条路
