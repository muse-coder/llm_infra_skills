# FlashQLA 的 GDN 优化方案（SM100 / SM103）

> 定位：**Qwen 官方 TileLang 实现。三段式调度 + 门驱动的近似上下文并行。**
> 前置阅读：[`GDN_Algorithm.md`](GDN_Algorithm.md)，本文全程使用其中的符号与"四个数学杠杆"编号。
> **范围：数据中心 Blackwell（SM100 / SM103）。** `blackwell_sm120/`（消费级）不在范围内。`hopper/` 只作为对照。
> 代码基准：`FlashQLA` v0.1.2，`tilelang==0.1.9`（`setup.py:24`）。路径相对于仓库根。

---

## 目录

1. [先纠正四个常见误解](#1-先纠正四个常见误解)
2. [顶层调度：无 CP 三个 kernel，有 CP 六个](#2-顶层调度无-cp-三个-kernel有-cp-六个)
3. [优化一：gate-free 三角求逆](#3-优化一gate-free-三角求逆)
4. [优化二：w / u 塌缩](#4-优化二w--u-塌缩)
5. [fused_fwd：SM100 上的四 warpgroup 结构](#5-fused_fwdsm100-上的四-warpgroup-结构)
6. [优化三（核心）：门驱动的近似上下文并行](#6-优化三核心门驱动的近似上下文并行)
7. [prepare_h 与转移矩阵 M](#7-prepare_h-与转移矩阵-m)
8. [优化四：把标量衰减从矩阵乘积里提出来](#8-优化四把标量衰减从矩阵乘积里提出来)
9. [TileLang 用到了什么，各买到什么](#9-tilelang-用到了什么各买到什么)
10. [varlen 与推理集成](#10-varlen-与推理集成)
11. [精度策略与测试](#11-精度策略与测试)
12. [性能数据（GB200 实测）](#12-性能数据gb200-实测)
13. [反向传播（简述）](#13-反向传播简述)
14. [hopper → blackwell 的移植改了什么](#14-hopper--blackwell-的移植改了什么)
15. [汇总](#15-汇总)

---

## 1. 先纠正四个常见误解

这四条与流传的说法不符，都经 `diff` / grep 验证：

| 说法 | 实际 |
|---|---|
| "cumsum → kkt_solve → fused_fwd 三段式" | **只在 CP 关闭时成立。CP 开启（默认）时前向是 6 次 launch。** |
| "blackwell/ 目录是为 SM100 重写的" | **`blackwell/kkt_solve.py`、`cp_fwd.py`、`cp_bwd.py`、`__init__.py` 与 `hopper/` 逐字节相同**（`diff` 退出码 0）。只有 `fused_fwd.py`、`fused_bwd.py`、`prepare_h.py` 真的移植到了 tcgen05 / TMEM。 |
| "分治求逆从 $8\times8$ 起步" | **$16\times16$ 起步，2 层合并（16→32→64）**，且只有第二层用 MMA。 |
| "TileLang 带 autotune，所以 tile 是调出来的" | **全仓库零 autotune**（无 `autotune` / `AutoTuner`）。所有 tile 都是手写字面量，唯一的运行时选择是按 SM 数二选一的 `block_DV`。 |

另外：**SM103 是 SM100 的严格别名**。`"10.3"` 只在 `elif ... in ["10.0", "10.3"]` 里与 `"10.0"` 并列出现，没有任何一个 kernel 常量、阈值或代码路径不同。加 SM103 支持的那个 commit（`1563fbd`）只改了 5 个文件的分派字符串和测试标记，+16/−10 行。

---

## 2. 顶层调度：无 CP 三个 kernel，有 CP 六个

### 2.1 后端选择

`flash_qla/ops/gated_delta_rule/chunk/__init__.py:10-27`：

```python
if tilelang.contrib.nvcc.get_target_compute_version() == "9.0":
    from .hopper import ...
    CHUNK_SIZE = 64
elif tilelang.contrib.nvcc.get_target_compute_version() in ["10.0", "10.3"]:
    from .blackwell import fused_gdr_fwd, fused_gdr_bwd, fused_gdr_h, kkt_solve
    from .blackwell import get_warmup_chunks, get_warmup_chunks_bidi, correct_initial_states, correct_terminal_states
    CHUNK_SIZE = 64
```

**导入期分派**，且用的是 `tilelang.contrib.nvcc.get_target_compute_version()` 返回的字符串，不是 `torch.cuda.get_device_capability`。`chunk/cp_context.py:12-29` 有第二份独立的同样分派，额外设置 `ARCH ∈ {"SM90","SM100","SM103","SM120"}`——这是全仓库唯一把 `"SM103"` 当独立字符串的地方，而它的用途只是喂给 `elif ARCH in ["SM100","SM103"]`（`cp_context.py:110`）。

### 2.2 前向的 kernel 序列

`chunk_gated_delta_rule_fwd`，`chunk/__init__.py:33-88`：

| # | Python 入口 | grid | threads | 输入 → 输出 | 环节 |
|---|---|---|---|---|---|
| 1 | `chunk_local_cumsum`（`ops/utils/cumsum.py:137`） | `num_chunks` | 128 | `g[B,T,H]` → `g_cumsum` | **①** |
| 2 | `kkt_solve`（`blackwell/kkt_solve.py:284`） | `num_chunks·H` | 128 | `k, beta` → `A[B,T,H,64]` bf16 | **②** |
| **3a** | `get_warmup_chunks[_bidi]`（`blackwell/cp_fwd.py:81/224`） | `cp_batch_size` | `ceil(H/32)·32` | `g, ht_mask, cp_cu_seqlens` → `num_warmup_chunks[cp_B,H]`, `fallback_mask` | CP 决策 |
| **3b** | `fused_gdr_h`（`blackwell/prepare_h.py:669`） | `cp_batch_size·H` | 512 | k,v,A,g,b,`num_warmup_chunks` → `ht` bf16 + `mt[cp_B,H,128,128]` bf16 | CP 预热 + $M$ |
| **3c** | `correct_initial_states`（`blackwell/cp_fwd.py:483`） | `4·H·raw_B` | 128 | `raw_h0,ht,mt,fallback_mask` → `cp_h0` | CP 段间扫描 |
| 4 | `fused_gdr_fwd`（`blackwell/fused_fwd.py:775`） | `ceil(DV/block_DV)·B·H` | 512 | q,k,v,A,g,b,h0 → `o`, 可选 `h`, `ht` fp32 | **③④⑤** |

3a–3c 只在 `auto_cp=True` **且** `_calc_cp_seqs` 返回 `use_cp=True`（`cp_context.py:171-179`）时发射。

### 2.3 与 FLA 的 kernel 数对照

这个对照在 profile 脚本里直接可见：

- FLA 前向 5 个（`profile/profile_gdr.py:210-214`）：`chunk_local_cumsum_scalar_kernel`、`kkt_solve_kernel`、`recompute_w_u_fwd_kernel`、`fwd_kernel_h_blockdim64`、`chunk_fwd_kernel_o`
- FlashQLA 前向 3 个（`profile/profile_gdr.py:217-219`）：`csum`、`solve`、`gdr`

> **FlashQLA 把 FLA 的 `wu` + `h` + `o` 三个 kernel 合成了一个。**

### 2.4 为什么是三段，而不是一段或五段

`README.md:22` 把理由写明了：

> *"Rather than following the step-by-step decomposition into independent kernels, nor fusing the entire computation flow into a single kernel, we take CP and backward requirements into account."*

具体来说，`A` 之所以是一个物化点，是因为 **CP 的 `prepare_h` 和反向传播都要读它**；`prepare_h` 之所以是独立 kernel，是因为它同时是 **CP 的预热状态 pass** 和**反向的 `h` 重算 pass**。

这是一个值得单独记住的设计原则：**融合边界不是由"能不能融"决定的，而是由"哪些中间量被多个 pass 共享"决定的。** 而 §3 的 gate-free 求逆正是让 `A` 能被共享的前提。

---

## 3. 优化一：gate-free 三角求逆

对应算法文档 **杠杆一**（门与求逆可交换）。这是 FlashQLA 与 FLA 的第一处明确分歧。

### 3.1 Λ 确实不在被求逆的矩阵里

`kkt_solve.py` **甚至没有把 `g` 作为参数**（`kkt_solve.py:207-212, 284-289`）。构造的矩阵是（`kkt_solve.py:105-121`）：

```python
T.gemm(k_shared, k_shared, a64_fragment, transpose_B=True, clear_accum=True)   # A = K @ Kᵀ
for j_s, j_t in T.Parallel(block_S, block_S):
    a64_fragment[j_s, j_t] *= b_shared[j_s]                                    # A = β·A
for j_s, j_t in T.Parallel(block_S, block_S):
    if   j_s <  j_t: a64_fragment[j_s, j_t] = 0                                # A = I + StrictLower(A)
    elif j_s == j_t: a64_fragment[j_s, j_t] = 1
```

即严格的 $M = I + \mathrm{tril}(\mathrm{diag}(\beta)KK^{\top},-1)$，**无 Λ**。输出 `A` = $A_{\mathrm{raw}} = M^{-1}$，bf16，`[B,T,H,64]`（`:304-306`）。

**注意一个跨库对齐的陷阱**：参考实现 `tests/ref_gdr.py:100-107` **是**把 `decay_mask = exp(g_i − g_j)` 乘进 $KK^{\top}$ 再求逆的。所以 **FlashQLA 的 `A` 张量与 FLA 的 `A`、与参考实现的 `A` 不是同一个对象**。逐张量比中间结果时会直接对不上。

### 3.2 Λ 在哪里施加

在 `fused_fwd` 的 CONSUMER_O warpgroup 里（`blackwell/fused_fwd.py:413-433`）：

```python
for j_s, j_t in T.Parallel(block_S, block_S):
    g_fragment[j_s, j_t] = g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
for j_s, j_t in T.Parallel(block_S, block_S):
    if j_s >= j_t: g_fragment[j_s, j_t] = T.exp2(g_fragment[j_s, j_t] * 1.442695)
    else:          g_fragment[j_s, j_t] = 0
for j_s, j_t in T.Parallel(block_S, block_S):
    a_fragment[j_s, j_t]  = a_shared[i_s % 2, j_s, j_t]
    a_fragment[j_s, j_t] *= g_fragment[j_s, j_t]        # Λ ⊙ A_raw
    a_fragment[j_s, j_t] *= b_shared[i_s % 2, j_t]      # ⊙ β（按列）
```

得到 $A_g = \Lambda \circ (A_{\mathrm{raw}}\,\mathrm{diag}(\beta))$。

**这 8 行里其实藏了三个优化**：

1. **Λ 只在 `fused_fwd` / `fused_bwd` 出现，从不进 `kkt_solve`。**
2. **$\beta$ 施加在 $64\times64$ 的 `A` 上（按列），而不是分别施加在 $64\times128$ 的 $K$ 和 $64\times128$ 的 $V$ 上。** 逐元素乘法量从 $2\times64\times128$ 降到 $64\times64$，省一半。
3. **同一个 `g_fragment` 两行之后被复用去缩放 `P`**（`:441`）。所以 $64\times64$ 的 `exp2` **每 chunk 只付一次**，摊给 `Ag` 和 `Pg` 两个消费者。这正是算法文档 §6.3 提到的"Λ 在公式里出现两次"，FlashQLA 用寄存器复用消掉了一半——**FLA 在两处各算一次，做不到这点**（见 FLA 文档 §5.2）。

### 3.3 gate-free 换来了什么

除了算法文档 §7.1 列的"$\exp$ 移出关键路径 + 数值范围最小"，在 FlashQLA 这里还有一个**结构性**收益：

> **$A_{\mathrm{raw}}$ 与 $g$ 无关 ⟹ 一份 `A` 同时服务前向、反向和 CP 的 `prepare_h`。**

`chunk_gated_delta_rule_bwd` 直接把 `A` 当输入接收（`chunk/__init__.py:96`），**反向从不重算求逆**。对比 FLA：它存的 `A` 是 gate-full 的，反向要为了拿 $dg$ 再重算一遍 $KK^{\top}$（FLA 文档 §13 第 6 步）。

**所以 §2.4 那个"三段式而非一段式"的决定，与本节的 gate-free 决定是绑在一起的**：正因为 `A` 门无关、可以被三个 pass 共享，把它物化出来才划算。

### 3.4 分治结构：$16\times16$ 起步，2 层合并

grid `num_chunks·H`，**每个 $64\times64$ 求逆只用 128 线程（一个 warpgroup）**（`kkt_solve.py:214, 252`）。

**基例——4 个独立的 $16\times16$ 块，16 行串行前代，四块齐步走**（`kkt_solve.py:132-146`）：

```python
T.clear(a16i_row)
for k_s in T.unroll(1, 16):
    for j_s, k_t in T.Parallel(4, 16):
        if k_t < k_s: a16i_row[j_s, k_t] = a16i_shared[j_s, k_s, k_t]
    T.clear(a16i_sum)
    for k_r in T.unroll(k_s):
        for j_s, k_t in T.Parallel(4, 16):
            a16i_sum[j_s, k_t] -= a16i_shared[j_s, k_r, k_t] * a16i_row[j_s, k_r]
```

纯 CUDA core FFMA、fp32、全展开（`T.unroll`）、SMEM 原地。**4 个块并行**正好把 16 行的串行长度摊掉，这是"$4\times16$ 而不是 $2\times32$ 或 $8\times8$"的实际动机——它同时匹配 128 线程的形状。

**合并层 1（16→32）**，`kkt_solve.py:148-163`：两个 $16^3$ 乘积算 $M_{11}^{-1}(-L_{21})M_{00}^{-1}$，写成 `T.Parallel(2,16,16)` 的寄存器循环——**CUDA core，fp32，不上 Tensor Core**。

**合并层 2（32→64）**，`kkt_solve.py:165-184`：

```python
T.gemm(a32i1_shared, a32o_shared, a32o_fragment, clear_accum=True)
T.copy(a32o_fragment, a32o_shared)
T.gemm(a32o_shared, a32i0_shared, a32o_fragment, clear_accum=True)
```

**这是整棵合并树里唯一发 MMA 的地方**，且操作数声明为 `dtype=accum_dtype="float32"`（`kkt_solve.py:69-72`）。

> ⚠️ **未确定**：TileLang 0.1.9 对 SM100 上 fp32 操作数的 `T.gemm` 究竟发 TF32 MMA 还是回落到 FFMA，从本仓库无法判断。`:108` 处的 $KK^{\top}$ 那次 `T.gemm` 确定是 bf16 tensor core。

**布局细节值得一提**：`T.annotate_layout({a16i_shared: make_linear_layout(...), ...})`（`kkt_solve.py:76-81`）——小块上**主动禁掉 swizzle**，好让标量下标寻址能用；bank conflict 改由 `a16i_shared = T.alloc_shared((4, 17, 16), ...)` 的 **17 padding** 解决（`:64`）。这是"手写布局"的典型形态：不是无脑加 swizzle，而是按访问模式二选一。

---

## 4. 优化二：w / u 塌缩

对应算法文档 §3.8 的合并形式。FlashQLA 走得比"合并"更彻底。

`fused_fwd.py:366-378`（CONSUMER_V warpgroup）：

```python
T.barrier_wait(tcbar_1, i_s % 2)
T.copy(v_tmem, u_fragment)                  # u_fragment = U = K @ S_prev（从 TMEM 取）
T.sync_threads(101, 128)
for j_s, j_v in T.Parallel(block_S, block_DV):
    u_fragment[j_s, j_v] *= -g_exp_shared[j_s]      # −diag(e^g)·U
for j_s, j_v in T.Parallel(block_S, block_DV):
    u_fragment[j_s, j_v] += v_shared[i_s % 2, j_s, j_v]   # + V
for j_s, j_v in T.Parallel(block_S, block_DV):
    v_shared[i_s % 2, j_s, j_v] = u_fragment[j_s, j_v]    # 原地写回 v_shared
```

随后 `fused_fwd.py:515-522`：`# Vd = Ag @ W`。

即

$$
V_{\mathrm{new}} = A_g\bigl(V - \mathrm{diag}(e^{g})\,K\,S_{\mathrm{prev}}\bigr),
\qquad A_g = \Lambda\circ\bigl(A_{\mathrm{raw}}\mathrm{diag}(\beta)\bigr)
$$

**$w$ 和 $u$ 从不物化**，而且 `W` 是**原地覆写在 `v_shared` 上**的（同一块 SMEM）。相比标准 $w/u$ 形式，每 chunk 省掉一次 $64\times64\times128$ 的 GEMM 加整个 `w` 张量。

**为什么能塌缩，依赖的正是 gate-free 的 $A_{\mathrm{raw}}$**：$w$ 里的 $\mathrm{diag}(e^{g})$ 与 $A_{\mathrm{raw}}$ 隐含的 $D^{-1}$ 相消，而本该作用在 $64\times128$ 的 $K$ 上的行缩放 $\mathrm{diag}(e^{g_l-g})$ 迁移到了 $64\times block_{DV}$ 的 $V$/$Y$ 上。**两个优化互为前提。**

这就解释了为什么 FLA 需要一个独立的 `recompute_w_u_fwd_kernel` 而 FlashQLA 不需要（§2.3 的 profile 标签对照）。

---

## 5. fused_fwd：SM100 上的四 warpgroup 结构

### 5.1 tile 常量（全部字面量，无 autotune）

| 常量 | 值 | 位置 |
|---|---|---|
| `CHUNK_SIZE` / `block_S` | **64**，硬断言 | `chunk/__init__.py:19`；`fused_fwd.py:51,797`；`kkt_solve.py:293`；`prepare_h.py:686`；`cp_fwd.py:93` |
| `DK = DV` | **128**，硬断言 `assert K == V == 128` | `fused_fwd.py:796` |
| `block_DV` | **128 或 64**，运行时二选一 | `fused_fwd.py:860-864` |
| threads | **512**（`fused_fwd`/`prepare_h`/`fused_bwd`/`cp_bwd`）；128（`kkt_solve`/`cumsum`/`correct_h0`） | — |
| 流水深度 | **2，硬编码**（`(2, block_S, DK)` + `i_s % 2` 相位算术），不是 `T.Pipelined` | `fused_fwd.py:143-148` |
| 寄存器上限 | `PRODUCER=72, CONSUMER_V=128, CONSUMER_S=168, CONSUMER_O=128` | `fused_fwd.py:225-228` |
| CTA 光栅化 | `T.use_swizzle(10)` | `fused_fwd.py:221`；`prepare_h.py:200` |

`block_DV` 的选法是全库**唯一**的运行时自适应参数：

```python
grid_size = real_batch_size * H
if grid_size >= TARGET_NUM_CTAS:  block_DV = 128
else:                             block_DV = 64
# TARGET_NUM_CTAS = int(multi_processor_count * 0.7)     fused_fwd.py:11-12
```

**这个式子本身就是算法文档 §6.2 并行度饥饿的直接回应**：$B\!\cdot\!H_v$ 不够填满 SM 时，就沿 $V$ 维再切一刀换并行度，代价是 $KS$、$QS$ 要各算两遍。Hopper 版还有第三档 `block_DV=32`（`hopper/fused_fwd.py:731-736`），Blackwell 砍掉了——SMEM/TMEM 更大使 128 更常可用，而 32 会浪费 TMEM 列。

### 5.2 每 chunk 的 GEMM 序列：6–7 条 tcgen05 MMA

全部由**一个 32 线程的 warp**（`tx ∈ [384,416)`）发射，`fused_fwd.py:475-596`，每条带自己的 `mbar`，且 `use_2cta=False`：

| 序 | 行 | 注释 | 形状 | 目标 |
|---|---|---|---|---|
| 1 | `:481` | `P = Q Kᵀ` | (64×128)·(128×64) | `p_tmem` |
| 2 | `:493` | `U = K @ S` | (64×128)·(128×bDV) | `v_tmem` |
| 3 | `:503` | `O = Q @ S` | (64×128)·(128×bDV) | `o_tmem` |
| 4 | `:515` | `Vd = Ag @ W` | (64×64)·(64×bDV) | `v_tmem` **（复用槽位）** |
| 5 | `:526` | `O += Pg @ Vd` | (64×64)·(64×bDV) | `o_tmem`，`clear_accum=False` |
| 6/7 | `:537-596` | `S += Kᵀ @ V'` | (128×64)·(64×64)，`bDV=128` 时拆两半 | `h_tmem_L`, `h_tmem_R` |

正是算法文档 §3.6 的七个矩阵乘（$KK^{\top}$ 在 `kkt_solve` 里，所以这里是 6 条，状态更新拆半时 7 条）。**GEMM 4 复用 GEMM 2 的 TMEM 槽位**（`v_tmem` 既装 `U` 又装 `Vd`）——刻意的 TMEM 压力管理，见 §5.4。

### 5.3 warp 专用化：4 个 warpgroup，producer 再拆 4 warp

| 线程范围 | 角色 | 职责 |
|---|---|---|
| `tx < 128` | **CONSUMER_S** | 拥有递归状态。`h_fragment_L/R` 是 `T.alloc_fragment((DK,64), accum_dtype)` = **fp32 寄存器**（`:163-174`）。每 chunk：转 bf16 到 `h_shared` 供 MMA 用（`:263-275`）→ 施加标量衰减 `h *= g_exp_shared[63]`（`:282-293`）→ push 到 `h_tmem_L/R` → 等 `tcbar_5a/5b` → 从 TMEM 拉回寄存器（`:316-320`）。末尾写 final state（`:323-335`）。 |
| `128 ≤ tx < 256` | **CONSUMER_V** | 算 `g_exp = exp2(g·log2e)` 和 `g_rev_exp = exp2((g_last−g)·log2e)`（`:349-362`，**带越界→0.0 的显式保护**）；构造 `W`（§4）；产出 `vd_shared` 与 `V' = diag(e^{g_l−g})·Vd → vn_shared`。 |
| `256 ≤ tx < 384` | **CONSUMER_O** | 构造 Λ 与 `Ag`（§3.2）；`Pg = scale·Λ⊙P`；`O *= scale·e^g`；staging `o_shared`。 |
| `tx ≥ 384` | **producer WG，再拆 4 个 32 线程 warp** | `384–415`：**MMA warp**，全部 `T.tcgen05_gemm`（`:475-596`）<br>`416–447`：**TMA warp**，Q/K/V/A 的 `T.tma_copy`（`:605-627`），尾块另走手写掩码标量路径（`:631-662`）<br>`448–479`：`beta` / `g` 的标量 load（`[T,H]` 跨步向量，不走 TMA）（`:664-691`）<br>`480–511`：**epilogue warp**，写 `o`，可选写 per-chunk 状态，`disable_tma=True`（`:702-770`） |

**这个分组与算法文档 §5 的依赖图对应关系是**：CONSUMER_S 持有环节 ④（串行链），CONSUMER_V 做环节 ③，CONSUMER_O 做环节 ⑤ 的 intra 部分和 Λ 构造。**注意与 FlashInfer 的分组原则不同**：FlashQLA 是**按公式里的量分组**（谁拥有状态、谁拥有 $V_{\mathrm{new}}$、谁拥有输出），FlashInfer 是**按"是否依赖状态"分组**。两种切法都能追回同一张依赖图，但落点不同。

同步全部是手写 `mbarrier` 加显式 `arrive_count`（`:199-219`）：`data_is_ready = [64]*2`（2 个 loader warp）、`data_is_free = [384]*2`（3 个 consumer WG）、`bar_0=448, bar_1=256, bar_3=256, bar_4=256, bar_5=288, bar_o=128`，外加 6–7 个单次到达的 `tcbar_*` MMA 完成 barrier。`README.md:22` 说的 *"manually implement warpgroup specialization"* 指的就是这个。

### 5.4 片上资源

**SMEM**（`:143-161`）：`q/k` 各 $2\times64\times128$ bf16（32 KB 每个）、`v` $2\times64\times bDV$、`a` $2\times64\times64$（16 KB）、`g/b` $2\times64$ fp32、`o_shared`、`h_shared` $128\times bDV$ **bf16**、`vd_shared`、`vn_shared`、`p_shared`、`g_exp_shared`、`g_rev_exp_shared`。$bDV=128$ 时约 **217 KB** ⟹ 每 SM 约 1 个 CTA，与 `TARGET_NUM_CTAS = 0.7·SM` 一致。

**TMEM**（`:183-197`）：

| 用途 | 列数（$bDV=128$） |
|---|---|
| `v_tmem`（$U$ 与 $V_d$ 共用） | 128 |
| `p_tmem`（$QK^{\top}$） | 64 |
| `o_tmem` | 128 |
| `h_tmem_L` | 64 |
| `h_tmem_R` | 64 |
| **合计** | **448 / 512** |

> **这 448/512 就是状态必须拆成两个 64 列半块（而不是一个 128 列 tile）的原因。** TMEM 只有 512 列，状态 $128\times128$ fp32 要占 128 列，如果再要双缓冲或整块搬运就超了。所以 `h_fragment_L/R`、`h_tmem_L/R`、`tcbar_5a/5b` 和拆半的 $S \mathrel{+}= K^{\top}V'$ GEMM 全是这个约束的下游产物。

**状态精度**：寄存器与 TMEM 里都是 fp32（`accum_dtype="float32"`，`:881`）；只有为了当 MMA 操作数而拷进 `h_shared` 时才转 bf16。输出的 per-chunk `h` 是 bf16（`:839`），`final_state`/`ht` 是 **fp32**（`:849,856`）。完全符合算法文档 §8.5。

---

## 6. 优化三（核心）：门驱动的近似上下文并行

对应算法文档 **杠杆四**。**这是三个库里独一份的东西**，也是 FlashQLA 全部性能优势的来源。

### 6.1 决策分三层

| 层 | 位置 | 决定什么 |
|---|---|---|
| **host Python** | `_calc_cp_seqs`，`cp_context.py:47-145`，带 `@tensor_cache` | 是否用 CP、段长、段边界 |
| **小 device kernel** | `tilelang_get_warmup_chunks[_bidi]`，`cp_fwd.py:10-78 / 119-221` | **逐 (段, head)** 决定预热几个 chunk、是否 fallback |
| **消费 kernel** | `prepare_h.py:110-114`、`cp_bwd.py:108-112` | 直接把 `num_warmup_chunks[bb,bh]` 当循环上界读 |

第一层调了 `cu_seqlens.tolist()`（`:56`），是**一次 D2H 同步**，靠 `@tensor_cache` 摊销（`flash_qla/utils/index.py:11-76`，其中为 CUDA graph capture 做了静态 pinning，commit `6ddc09a` "sglang-style tensor cache for cuda-graph"）。

第二层的 grid = `cp_batch_size`，threads = `ceil(H/32)·32`——**每段一个小 CTA，所有 head 在 `T.Parallel` 里并行**。全程在设备上，`num_warmup_chunks` 从不回 host。

### 6.2 预热规则：逐 (段, head)，阈值字面量 −10.0

`blackwell/cp_fwd.py:52-76`：

```python
for i_s in T.serial(num_iters):
    for i_h in T.Parallel(num_heads):
        g_fragment[i_h] = g[0, seq_end_idx - i_s * chunk_size - 1, i_h]
    for i_h in T.Parallel(num_heads):
        g_cumsum[i_h] += g_fragment[i_h]
    for i_h in T.Parallel(num_heads):
        if g_cumsum[i_h] < threshold and n_fragment[i_h] == num_iters:
            n_fragment[i_h] = i_s + 1
            f_fragment[i_h] = False
```

这里的 `g` 是**chunk 内 cumsum**，所以 `g[chunk_end-1]` 就是该 chunk 的总对数衰减。**从段末尾往前扫**，累加总对数衰减，第一次跌破 `threshold` 的 `i_s` 决定 `num_warmup = i_s + 1` 并清掉 `fallback`。

初始化是 `T.fill(n_fragment, num_iters); T.fill(f_fragment, True)`（`:49-50`）——**所以永远没跌破阈值时，`num_warmup = 整段长度` 且 `fallback = True`**。这个初始化就是"自我保护"机制：**衰减不够快自动退回精确路径**，不需要额外的判断逻辑。

阈值字面量：**`warmup_threshold: float = -10.0`**（`cp_context.py:156`），`cp_fwd.py:87` 与 `:230` 的默认值同样是 `-10.0`。$e^{-10}\approx 4.54\times10^{-5}$，对照算法文档 §8.3 的表：**低于 bf16 分辨率两个数量级**。

**不可覆盖**：任何公开 API 都不转发 threshold（`chunk/__init__.py:61-67` 调 `intra_card_cp_preprocess` 时不传）。**全包零 `os.environ` / `getenv`。**

### 6.3 "预热"的实际语义：是上一段的后缀，不是本段的前缀

这个实现细节值得记住。`prepare_h` 从**零状态**出发，跑**第 $j$ 段最后 `num_warmup` 个 chunk** 的递推（`prepare_h.py:110-122`）：

```python
num_iters = num_warmup_chunks[bb, bh] if is_cp else T.ceildiv(...)
calc_mt = is_cp and num_iters >= T.ceildiv(seq_end_idx - seq_start_idx, block_S)
if is_cp:
    if seq_end_idx - num_iters * block_S > seq_start_idx:
        seq_start_idx = seq_end_idx - num_iters * block_S
```

得到的 `ht[j]` 就是第 $j+1$ 段的初始状态。**功能上等价于给第 $j+1$ 段前置预热**，但这样写让一次 kernel launch 就能服务所有段，且每个 head 有自己的 trip count。

`ht_mask` 的语义（`cp_context.py:81-99`）：`ht_mask[j] = True` 表示第 $j$ 段是某条序列的**最后一段**（它拥有该序列真正的 final state）；`ht_mask_bwd[j]` 对应第一段。`ht_mask[bb] == True` 时直接短路 `num_warmup = 0`（`cp_fwd.py:33-35`），因为下游没人会读这段的 `ht`/`mt`/`fallback`。

**双向变体**（`cp_fwd.py:119-221`）一趟同时正扫和反扫，输出 `num_warmup_h = max(n_fwd, n_bwd)`（`:209-213`），**一次 `prepare_h` launch 同时满足前向和反向**——这就是 `enable_fwd_cp_cache` 买到的东西。

> 一处小不一致值得记一笔：单向 kernel 用 **floor** 除法 `num_iters = (seq_end − seq_start)//chunk_size`（`cp_fwd.py:42`），双向 kernel 用 `T.ceildiv`（`:149`）。

### 6.4 慢衰减 fallback：用 M 做精确串行修正

某 head 始终不跌破 −10 ⟹ `fallback = True` 且 `num_warmup = 整段` ⟹ `calc_mt = True`（`prepare_h.py:116-119`）⟹ `prepare_h` 额外为该段产出 $K\times K$ 的转移矩阵 `M`。

然后 `correct_initial_states`（`cp_fwd.py:296-374`）在**段维度上串行扫描**：

```python
for i_s in T.Pipelined(num_iters - 1, num_stages=2):
    idx = seq_start_idx + num_iters - 1 - i_s if reverse else seq_start_idx + i_s
    T.copy(h_fragment, cp_h0[idx, bh, 0:DK, DV_start:DV_end])      # 发布当前运行状态
    T.copy(ht_buffer[idx, bh, ...], h_shared)
    T.copy(mt_buffer[idx, bh, 0:DK, 0:DK], m_shared)
    if fallback_mask[idx, bh]:
        T.copy(h_fragment, hd_shared)                              # 暂存上一段状态
        T.fence_proxy_async()
    T.copy(h_shared, h_fragment)                                   # h = ht[idx]（近似路径）
    if fallback_mask[idx, bh]:
        T.gemm(m_shared, hd_shared, h_fragment, clear_accum=False)  # h += M · h_prev
```

即

$$
cp\_h0[j+1] = ht[j] + \bigl(\text{fallback}\ ?\ M_j \cdot cp\_h0[j]\ :\ 0\bigr)
$$

**`fallback` 为假时跨段项直接丢掉——这就是全部的近似。** 为真时修正是**精确的**，因为 `num_warmup` 已被强制成整段，所以 `ht[j]` 是精确的零初始状态、`M_j` 是精确的线性算子。

**四点值得注意**：

1. **`fallback_mask` 是逐 (段, head) 的**，所以同一个 grid 里快衰减 head 跳过 GEMM、慢衰减 head 拿到精确修正。混合情形天然支持。
2. **代价极小**：`correct_initial_states` 本来就无条件发射；段维度串行，但在 `raw_batch × H × ceil(128/32)=4` 个 DV 块上并行，128 线程（`cp_fwd.py:387-389`，`block_DV=32` 在 `:281`）。每段一次 $128\times128 \cdot 128\times32$ GEMM，相对主 pass 可忽略。
3. `T.Pipelined(num_iters-1, num_stages=2)`（`cp_fwd.py:324`）是**整个 Blackwell 路径里唯一一处 `T.Pipelined`**。
4. **反向是同一个模板**：`correct_terminal_states`（`cp_fwd.py:547-610`）用 `reverse=True, transpose_m=True`（`:582-583`）实例化同一个 `tilelang_correct_h0`，得到 $dh_j = M_j^{\top}dh_{j+1} + dht_j$，反向走段。

> **总结这个机制的结构**：算法文档 §7.4（近似）为主，§7.3（精确仿射结合律）兜底，**两条杠杆同时用**。这是它与 FlashInfer 只用 §7.3 的根本区别——近似路径不需要构造 $M$（省一大笔），只在必要时才付 $M$ 的成本。

### 6.5 段长选择：一个显式的延迟模型

`cp_context.py:63-72`，注释完整给出了推导：

```python
# Latency model: T = a·L_cp + b·(B·H·Lc/P) / L_cp + c
# Minimizing T yields the theoretical optimum: L_cp* ∝ √(B·H·Lc / P)
# Scaled by empirical factor (3) and aligned to the nearest power of 2.
max_local_chunks = 2 ** round(
    math.log2(math.sqrt(H * sum(num_chunks) / MULTI_PROCESSOR_COUNT) * 3)
)
max_local_chunks = max(max_local_chunks, 4)   # 保证 fused_gdr 能多级流水
```

模型很直观：第一项 $a L_{cp}$ 是**段内串行链**的延迟（段越长越慢），第二项 $b(BH L_c/P)/L_{cp}$ 是**波数**（段越长段数越少，但每波要跑更久）。求导得 $L_{cp}^*\propto\sqrt{BHL_c/P}$。

具体例子（GB200，148 SM，$H_v=8$，$T=32768$ ⟹ 512 chunk）：

$$
\sqrt{8\times512/148}\times3 = 3\times5.26 = 15.8
\ \Longrightarrow\ 2^{\mathrm{round}(\log_2 15.8)} = 16\ \text{chunk} = \mathbf{1024\ token/段}
$$

⟹ 32 段 × 8 head = **256 个 CTA**。对比不开 CP 时的 $B\times H_v = 8$——**并行度从 8 涨到 256**。这就是算法文档 §6.2 那个数量级问题的解。

下限 4 chunk 的理由写在注释里：保证 `fused_gdr` 的两级流水（§5.1）有东西可流。

### 6.6 CP 的开关阈值：SM100 上有个坑

`auto_cp` 默认 **`True`**（`chunk/__init__.py:44, 179, 283`）。但 `_calc_cp_seqs` 把 CP 卡得很死（`cp_context.py:102-124`，注释原文）：

```python
# Disable CP when sequences are too short or B * H naturally saturates SM occupancy.
# CP has fixed overhead (warmup + correct_initial_states) that only pays off
# when the longest sequence has enough chunks to amortize the cost.
Be = sum(num_chunks) / max(num_chunks)

if ARCH == "SM90" or ARCH == "SM120":
    use_cp = Be * H <= 40 or (Be * H <= 56 and max(num_chunks) >= 128)
elif ARCH in ["SM100", "SM103"]:
    if is_bwd:
        use_cp = Be * H <= 56 and max(num_chunks) >= 16
    else:
        use_cp = (Be * H <= 56 and max(num_chunks) >= 256) or (
                  Be * H <= 32 and max(num_chunks) >= 192)
```

`Be = total_chunks / max_chunks` 是"有效 batch size"。所以 **SM100 前向要求 $H_v \le 56/B_e$ 且至少 256 chunk = 16384 token**（或 ≤32 head 且 ≥192 chunk = 12288）。**反向宽松得多**（≥16 chunk = 1024 token）——注释解释了原因：反向 kernel 每 chunk 算术强度更高，欠占用出现得更早，而且它还要跑 `prepare_dh`，本身也吃 CP 并行。

另外 `intra_card_cp_preprocess` 对 `batch_size > 1` 直接放弃（`cp_context.py:165-166`）——**CP 只在 packed `B=1 + cu_seqlens` 布局下工作。**

**这个阈值差在实测数据里是可见的**：GB200、`1x8192`、$h_v=8$（128 chunk ⟹ SM100 上 CP 关闭）耗时 **0.246 ms**；H200 同配置（CP 开启，因为 $8\le40$）**0.119 ms**（`benchmark/benchmark_results_GB200.txt:12` vs `benchmark_results_H200.txt:12`）。

> **Blackwell 在这个点上比 Hopper 慢 2 倍，纯粹因为 256-chunk 的前向 CP 门槛。** 到 `1x32768`（512 chunk，CP 开）就恢复正常：GB200 0.275 ms vs H200 0.320 ms。
> 这是一条实用的排查线索：**在 SM100 上遇到 8k–12k token 的 GDN prefill 明显偏慢，先看是不是被 CP 阈值卡在了非 CP 路径上。**

---

## 7. prepare_h 与转移矩阵 M

`blackwell/prepare_h.py`，grid `batch_size·H`，threads 512，`num_stages=2`。**双用途 kernel**：

- **CP 预热状态 pass**：`cp_context.py:200-212` 调用（`output_final_state=True, output_h=False, num_warmup_chunks=...`）
- **反向的 `h` 重算 pass**：`chunk/__init__.py:133-140` 调用（`output_h=True`，无 warmup）

### 7.1 输出

| 张量 | 形状 | dtype |
|---|---|---|
| `h`（per-chunk 状态，给 bwd） | `[B,N,H,128,128]` | bf16 |
| `ht`（段末状态） | `[B,H,128,128]` | **CP 时 bf16**，否则 fp32（`prepare_h.py:730`） |
| `mt`（转移矩阵） | `[cp_B,H,128,128]` | **bf16**（`prepare_h.py:91`；Hopper 版是 `ht_dtype`，这是少数几处真实的 Blackwell 差异） |

### 7.2 M 是什么

代码把状态递推因式分解成 $S_{\mathrm{new}} = e^{g_l}S + X^{\top}Y$，其中

- $X = -\mathrm{diag}(\beta)A_{\mathrm{raw}}^{\top}K$（`:471-478` 的 `# X = A^T @ K` 走 tcgen05 到 `x_tmem`，然后 `:315-316` 的 `x_fragment *= -b_shared`）
- $Y = e^{g_l}(KS) - \mathrm{diag}(e^{g_l-g})V$（`:401-410` 的 `# Y = g_last * U - g_last/g * V`）

展开得

$$
S_{\mathrm{new}} = e^{g_l}\bigl(I - K^{\top}A_{\mathrm{raw}}\mathrm{diag}(\beta)K\bigr)S + K^{\top}A_{\mathrm{raw}}\mathrm{diag}(\beta)\mathrm{diag}(e^{g_l-g})V
$$

$$
\boxed{\;M = e^{g_l}\bigl(I - K^{\top}A_{\mathrm{raw}}\,\mathrm{diag}(\beta)\,K\bigr) \in \mathbb{R}^{128\times128}\;}
$$

与算法文档 §7.3 的 $M$ 一致（左右乘约定不同）。**又一次是 gate-free $A_{\mathrm{raw}}$ 的红利**：本该作用在 $64\times128$ 的 $K$ 上的 $\mathrm{diag}(e^{g_l-g})$ 塌到了 $Y$ 上，$w$ 里的 $\mathrm{diag}(e^{g})$ 直接相消——每 chunk 少一次 $64\times128$ 的 $\exp$ 缩放（`README.md:20` 声称的 SFU 节省）。

### 7.3 M 怎么累积

`:322-351`（右半）/ `:416-445`（左半）：

```python
g_prod_X[0] += g_shared[i_s % num_stages, block_S - 1]
T.copy(m_fragment_R, m_shared_R); T.fence_proxy_async(); T.barrier_arrive(bar_3)
T.barrier_wait(bar_3, i_s % 2)
T.gemm(k_shared[i_s % num_stages, :, :], m_shared_R, z_fragment_R, clear_accum=True)   # Z = K @ M
T.copy(z_fragment_R, z_shared_R); T.sync_threads(105, 128); T.fence_proxy_async()
T.gemm(x_shared, z_shared_R, m_fragment_R, transpose_A=True, clear_accum=False)        # M += Xᵀ @ Z
```

即 $M \leftarrow (I - K^{\top}A_{\mathrm{raw}}\mathrm{diag}(\beta)K)M$，初值为单位阵（`:293-297` 右半在 `j_k == j_v + DK/2` 处置 1，`:372-376` 左半在 `j_k == j_v` 处）。

$M$ 被**按列拆成两个 $128\times64$ 半块**，由两个"闲置"的 consumer warpgroup 负责（CONSUMER_X 管 64:128 列，CONSUMER_Y 管 0:64 列），所以 **$M$ 的乘积是搭着状态递推顺路算出来的**。

> **一处遗留**：两次 $M$ 的 GEMM 用的是 `T.gemm` 而非 `T.tcgen05_gemm`，`:331` 和 `:425` 都写着 `# TODO: calc M on tcgen05`，对应的 `m_tmem_L/R` 分配被注释掉了（`:179-180`）。**$M$ 是 Blackwell 前向路径里唯一一个还没走 tcgen05 的 GEMM。**

### 7.4 显存与共享

`mt` = $cp\_B \times H \times 128\times128\times 2\,\text{B}$ = **每 (段, head) 32 KB**。被跳过的段写 **0**（`:363-366, 457-460`），所以 buffer 恒为稠密。

`mt` 的消费方：`correct_initial_states`（前向）、`correct_terminal_states`（反向，`transpose_m=True`，且 `mt_buffer.float()` 在 `cp_context.py:316` 把 bf16 的 `mt` 升精度）、以及 `cp_cache = (cp_h0, mt, fallback_mask_bwd, num_warmup_chunks_bwd)`（`cp_context.py:224`）通过 `ctx.save_for_backward`（`chunk/__init__.py:203-209`）传给反向，**让反向完全跳过 `get_warmup_chunks_bidi` + `prepare_h` + `correct_initial_states`**（`cp_context.py:272-274`）。

---

## 8. 优化四：把标量衰减从矩阵乘积里提出来

一个小而漂亮的优化，值得单列。

$M = \prod_i e^{g_{l,i}}(I - \cdots)$ 里的标量因子 $e^{g_l}$ **不逐 chunk 施加**，而是把 $g_l$ 累加到 `g_prod_X` / `g_prod_Y`，最后一次性 `exp2(Σg_last · log2e)`（`prepare_h.py:355-362, 449-456`）。

$$
\prod_i e^{g_{l,i}} X_i = \exp\!\Bigl(\sum_i g_{l,i}\Bigr)\prod_i X_i
$$

**把 $n$ 次 $128\times128$ 的整矩阵缩放变成一次。** 寄存器常驻的矩阵乘积过程中完全不需要重新缩放。

同类思路在 `fused_fwd` 里也有：状态的衰减是一次标量乘 `h *= g_exp_shared[63]`（`:282-293`），而不是构造对角阵去乘。**标量门（算法文档 §2.1 强调过 $\alpha,\beta$ 是标量而非向量）在这里直接兑现成了实现上的便宜。** 如果 GDN 的门是逐通道的（像 Mamba2），这个优化就不存在。

---

## 9. TileLang 用到了什么，各买到什么

### 9.1 Blackwell 专属原语（`hopper/` 里没有）

| 原语 | 位置 | 买到什么 |
|---|---|---|
| `T.alloc_tmem` | `fused_fwd.py:183-197`；`prepare_h.py:167-178`；`fused_bwd.py:189-205` | **TileLang 确实暴露了 SM100 的 TMEM，FlashQLA 用得很重。** 既当 MMA 累加器，也当通用 fp32 scratch / 跨 warpgroup buffer（`mask_tmem`）。把 $64\times128$ fp32 累加器搬出寄存器文件，才使 `CONSUMER_S_NREG=168` 成为可能。 |
| `T.tcgen05_gemm(A,B,C_tmem, transpose_A/B, clear_accum, mbar, use_2cta)` | `fused_fwd.py:475-596`；`prepare_h.py:471-531`；`fused_bwd.py:874-1147` | 第 5 代 Tensor Core。`mbar=` 给每条 GEMM 独立的异步完成 barrier，所以**一个 warp 就能流水 6–22 条 MMA**，三个 consumer warpgroup 同时干 CUDA core 的活。`use_2cta=False` **处处如此**——2-CTA / pair-SM MMA 模式从未启用。 |
| `T.copy(tmem ↔ fragment)` | `fused_fwd.py:317-320, 367, 382, 438, 450-455, 464` | TMEM↔RF 搬运。`o_tmem → RF → 缩放 → o_tmem` 的往返（`:450-455`）是**在同一个 TMEM tile 上的两条累加 MMA 之间插入标量缩放**的唯一办法。 |
| `T.copy(..., disable_tma=True)` | `fused_fwd.py:716, 722, 749, 755` | **Blackwell 专属改动**：per-chunk `h` 的 store 刻意绕过 TMA。 |

### 9.2 通用原语

| 原语 | 位置 | 买到什么 |
|---|---|---|
| `T.alloc_barrier(arrive_count=…)` + `T.barrier_arrive/wait(bar, phase)` | `fused_fwd.py:199-219` 及各处 | 手写 mbarrier warp 专用化。**仓库里没有 `T.ws` / producer-consumer 语法糖**——全靠手数 arrive count 和 phase bit。 |
| `T.set_max_nreg(n, is_consumer)` | `fused_fwd.py:231, 338, 403, 473` | `setmaxnreg` 寄存器再分配。全库无 `T.no_set_max_nreg`。 |
| `T.fence_proxy_async()` | ~25 处，commit `c4d7234` "Add fence.proxy.async to SM90 and SM100 kernels" 批量加入 | RF→SMEM 写之后、被 TMA/MMA 消费之前的 generic↔async proxy 定序。**这是一个专门的 bugfix commit，说明它是踩坑踩出来的，不是一开始就写对的。** |
| `T.sync_threads(id, count)` | `fused_fwd.py:294, 368, 383, 439, 451, 454, 465` 等 | 128 线程 warpgroup 内的命名局部 barrier。**之所以必需，是因为 `TL_DISABLE_THREAD_STORAGE_SYNC=True`（`fused_fwd.py:19`）关掉了 TileLang 自动插同步。** |
| `T.tma_copy(src, dst, barrier=)` | `fused_fwd.py:605-627`；`prepare_h.py:565-595` | 对齐的大张量（q/k/v/A/do/h）走 TMA。**尾块回落到手写掩码标量循环**（`fused_fwd.py:631-662`）——不依赖 TMA 的越界钳位。 |
| `T.use_swizzle(10)` | `fused_fwd.py:221`；`prepare_h.py:200` | L2 友好的 CTA 光栅化。`fused_bwd.py:279` 显式**注释掉**了。 |
| `T.annotate_layout` | `fused_bwd.py:248-277`（9 个 SMEM buffer 上 `make_swizzled_layout` + 自定义 `mask_tmem_layout`）；`kkt_solve.py:76-81`（`make_linear_layout`） | MMA 操作数 SMEM 上加 swizzle；$16\times16$ 小逆块上**反向操作**——用线性布局禁掉 swizzle 好让标量下标能用。`fused_fwd.py` 里完全没有。 |
| `T.exp2(x * 1.442695)` | 所有门指数运算 | SFU 的 `ex2.approx.f32`。搭配每个 `@tilelang.jit` 上的 `TL_ENABLE_FAST_MATH: True`。**与 FLA 的 `exp2` 化是同一个技巧，但 FlashQLA 把 $\log_2 e$ 写在乘法里而不是预乘进 `g`——所以它的 `g_cumsum` 是标准语义**（跨库对齐时的差异点）。 |
| `T.cumsum(g_fragment, dim=1, reverse=)` | `ops/utils/cumsum.py:62` | 内建 tile 内 scan。周围那圈 SMEM 转置（`:57-67`）是为了把布局换成 `[H,64]` 好扫。 |
| `T.reinterpret` / `T.vectorized` | `fused_bwd.py:773-818, 437-441` | `T.reinterpret` 支撑 §11 的 hi/lo fp32 转置；`T.vectorized(2)` 强制打包 `fmul2`。`fused_bwd.py:376-377` 的注释明说：*"Referencing the stable shared scalar directly enables TileLang packed fmul2 lowering for contiguous fragment lanes."* |
| `T.dynamic(...)` | 每个 kernel 的 shape 序言 | 动态 `batch_size`/`num_tokens`/`num_chunks` ⟹ **一份编译产物服务所有 shape**（serving 的必要条件）。 |

**未使用**：atomics（`atomic` 零命中）、`T.ws` 语法糖、`T.no_set_max_nreg`、2-CTA MMA、fp8 / 任何状态量化、**autotune**。

### 9.3 TileLang 到底改变了什么（诚实评估）

**明显受益的地方**：

1. **逐元素代数几乎免费**，所以设计敢重度依赖它：$A_g = \Lambda\circ A_{\mathrm{raw}}\circ\beta$、$W = V - \mathrm{diag}(e^g)U$、$V' = \mathrm{diag}(e^{g_l-g})V_d$、$O \mathrel{*}= scale\cdot e^g$ 各只要 2–4 行 `T.Parallel`。CuTe 作者必须手工映射到 fragment lane。**§3.2、§4、§8 那几个代数重排在 TileLang 里"试一下"的成本极低——这大概解释了为什么它们是在这里被发现的。**
2. **$64\times64$ 三角求逆写成了嵌套 `T.Parallel` / `T.unroll`**，配 `make_linear_layout` 主动禁 swizzle。$16\times16$（而非 $8\times8$ 或 $32\times32$）的基例更像是**为了循环嵌套的可读性和"128 线程 × 4 块"的形状**而选，不是对应任何硬件 fragment 尺寸。
3. **布局义务是声明式的**：`fused_bwd.py:248-277` 一次 `T.annotate_layout` 调用搞定 9 个 buffer 的 swizzle。
4. **编译期特化用 Python `if`**：`is_varlen`、`is_cp`、`state_v_first`、`use_initial_state`、`store_h`、`store_o`、`block_DV` 从同一份源码产出不同二进制（`fused_fwd.py:22-46`）。

**抽象泄漏的地方**：

- **零 autotune。** Hopper→Blackwell 的移植全是**手工**改动：`T.gemm` → `T.tcgen05_gemm` 加显式 `mbar`、TMEM 分配与 RF↔TMEM 往返、手工重数 barrier `arrive_count`（416→448、128→256、416→288……）、手工重调 `set_max_nreg`、手工重划 warp 分工。
- `TL_DISABLE_THREAD_STORAGE_SYNC: True`（`fused_fwd.py:19`）关掉自动同步插入，换来 ~12 处手写 `T.sync_threads` 和 ~25 处手写 `T.fence_proxy_async`（后者还是一个专门的 bugfix commit 补的）。

> **结论**：TileLang 的生产力红利**在 tile 代数层面是真实的，在异步流水层面基本不存在**。这正好解释了 FlashQLA 的优化分布——算法层的重排（§3、§4、§6、§8）很出色，硬件层的调度则和手写 CuTe 一样辛苦。

---

## 10. varlen 与推理集成

- **varlen = packed 布局**：给 `cu_seqlens` 就要求 `q.shape[0] == 1`（`chunk/__init__.py:373-379`）。`is_varlen` 是**编译期**模板参数，产出独立特化（`fused_fwd.py:53-65`）。`seqlen_dtype = cu_seqlens.dtype`（`:816`），int32/int64 都行。
- **两套索引**（host 侧纯 torch + `@tensor_cache`，`flash_qla/utils/index.py:84-107`）：`prepare_chunk_indices` → `[num_chunks,2]` 的 `(batch_idx, chunk_idx)`，给扁平 chunk grid（`cumsum`、`kkt_solve`）；`prepare_chunk_offsets` → `[N+1]` 偏移，给 per-sequence grid（`fused_fwd`、`prepare_h`、`fused_bwd`）。**与 FLA 的机制完全同构。**
- **尾块显式处理，不 padding**：`num_iters = ceildiv(len,64)`、`num_unmasked_iters = len // 64`（`fused_fwd.py:140-141`），另配完整的掩码 epilogue（`:631-662, 725-756, 758-770`）。最后一条序列 `seq_end` 之后的 `o` 清零（`:767-770`）；**`g` 的 padding 复制 `g[seq_end-1]` 而不是置零**（`:689`），保证衰减算式有限——这是算法文档 §3.7 那个 NaN 陷阱的另一种解法。
- **状态 I/O**：`initial_state` 是 `[N,H,K,V]`，或 `state_v_first=True` 时 `[N,H,V,K]`（编译期分支贯穿每个 kernel，`fused_fwd.py:79-88`）。`final_state` **恒为 fp32**（`:849,856`），与输入 dtype 无关。per-chunk `h` 是 bf16。
- **CP 感知的 final state 路由**（packed batch 下的关键细节）：`cp_seq_map` 把 CP 段映射回原序列，只有拥有序列真正结尾的那段写 `ht`（`fused_fwd.py:127-136`）：
  ```python
  raw_batch_idx      = cp_seq_map[bb] if is_cp else bb
  raw_seq_end_idx    = raw_cu_seqlens[raw_batch_idx + 1] if is_cp else seq_end_idx
  need_store_final_state = store_final_state & (raw_seq_end_idx == seq_end_idx)
  ```
- **CUDA graph**：`tensor_cache` 在 capture 期间把条目标为 static，并在 capture 中途 cache miss 时硬断言（`flash_qla/utils/index.py:33-59`）——否则 `_calc_cp_seqs` 里的 host `.tolist()` 会破坏图捕获。`chunk_gated_delta_rule` 带 `@torch.compiler.disable`（`chunk/__init__.py:269`）。
- **无分页 / 池化状态**：`page`、`slot_idx`、`state_indices`、`conv_state`、`ssm_state` 零命中。状态是稠密 `[N,H,K,V]` 张量。
- **无状态量化**：`fp8|float8|e4m3|e5m2|quant` 零命中。输入必须 bf16 或 fp16（`chunk/__init__.py:365-367`）。

> **§10 最后两条是 FlashQLA 与 FlashInfer 在 serving 集成上的明确差距**：算法文档 §7.4 用途 2（遗忘门让量化噪声自愈）FlashQLA 没有利用，分页状态池也没有支持。如果对接的是 vLLM/SGLang 那种 mamba 风格的状态池，FlashQLA 的契约是稠密张量加一个 `state_v_first` 布局开关，需要外部做 gather/scatter。

另有一条 API 文档里的注意事项（`chunk/__init__.py:330-333`）：*"The TVM host code does not accept `strides == nullptr` even for compact tensors. You must explicitly set `strides` to a valid array when constructing the DLTensor."*

---

## 11. 精度策略与测试

### 11.1 精度

- **每个** kernel 的 `accum_dtype = "float32"`（`fused_fwd.py:881`、`kkt_solve.py:316`、`prepare_h.py:755`、`cp_fwd.py:99,512`、`cumsum.py:163`、`group_reduce.py:75`）。
- **三角求逆全程 fp32**：$KK^{\top}$ 是 bf16 tensor core → fp32 累加器；前代与两层合并都在 fp32 fragment/SMEM 上；只有最终 `A` 存成 bf16（`kkt_solve.py:74`）。**这一点与 FLA 不同——FLA 的 GDN 快速路径用 tf32 做 Schur 合并**（FLA 文档 §4.3）。
- **状态**：fwd/bwd/prepare_h 全程寄存器与 TMEM 里都是 fp32；只有 MMA 操作数副本 `h_shared` 是 bf16。`final_state`/`dh0` fp32。
- **策略可概括为一句**：**凡是归约目标或跨 chunk 携带的量 → fp32；凡是 MMA 操作数 → bf16。** 为守住这条有两个专门的补丁：`mt` 的 dtype 改成 `qkva_dtype`（`prepare_h.py:91`），以及下面的 hi/lo 转置。
- `TL_ENABLE_FAST_MATH: True` + `exp2(x·1.442695)`。
- `dg` 在 Python 边界硬断言 fp32：`assert dg.dtype == torch.float32, "dg should be fp32"`（`chunk/__init__.py:157`）。

### 11.2 测试

`tests/test_gdr_unit.py`：

- **判据是相对 Frobenius 范数，不是逐元素 `allclose`**（`:192-203`），`RTOL = 0.02`（`:23`）：
  ```python
  error     = torch.linalg.vector_norm(actual.double() - expected.double()).item()
  reference = torch.linalg.vector_norm(expected.double()).item()
  assert error <= reference * RTOL
  ```
  同时断言 padding 区无 NaN。**参考实现是 float64**（`REF_DTYPE = torch.float64`，`:28`），数据 bf16。
  这个判据选择与算法文档 §8.4 的第 2 条局限完全对应——**范数界本来就只是全局界，测试也就只测全局界**。
- `test_fwd_auto_cp`（`:522-575`）：`T=16384, H=4`，以及 varlen `[0,4096,8192,12288,16384]`，两种 `state_v_first`，对 `auto_cp=True` 和 `False` 都跑。注意 16384 token = 256 chunk，**正好是 SM100 前向 CP 阈值**，所以这条确实覆盖了 CP 路径；而 varlen 那组（每段 4096 = 64 chunk）在 SM100 上 `max(num_chunks)=64 < 256`，**静默地只测了非 CP 路径**。
- `test_bwd_auto_cp`、`test_fwd/bwd_cp_cache`（`T=32768`）、`test_mixed_cp_control` 都用同一个 2% 相对 L2 门槛。
- **门分布是刻意设计成对抗性的**（`:106-114`）：
  ```python
  A = rand(H) * 16;  A[0] = 0;  A[-1] = 16
  g = -A * softplus(gate_input + 1)
  ```
  所以 **head 0 的 $g\equiv 0$（零衰减 ⟹ 必然 `fallback=True`，走精确 $M$ 路径）**，head $H-1$ 衰减最大（必然走快速近似路径）。**每个 CP 测试都同时覆盖两条分支。** 这是很讲究的测试设计。
- **确定性测试**：重跑 `DETERMINISM_ITERS = 1000` 次再断言（`:24, 438-515`）——专门抓手写 barrier 里的竞态非确定性。
- **没有任何测试单独隔离近似误差本身**（比如扫 −10 阈值）；唯一的界就是全局 2% 相对 L2。

---

## 12. 性能数据（GB200 实测）

`README.md:14` 的声明：**"2–3× forward speedup and 2× backward speedup over the FLA Triton kernel"**，并说 *"particularly pronounced in pretraining scenarios and edge-side agentic inference."*

随仓库提供的原始数据：`benchmark/benchmark_results_GB200.txt`（GB200；torch 2.9.1+cu130、fla 0.5.2、flashinfer 0.6.14、tilelang 0.1.9、cudagraph 后端、warmup 10 / 100 reps、$d=128$）。**单位是 ms，仓库里没有任何 TFLOPS 数据。** 前向节选：

| 配置 | $h_{qk}/h_v$ | FlashQLA | FlashInfer | FLA | vs FLA | vs FI |
|---|---|---|---|---|---|---|
| TP8 `1x32768` | 2/8 | **0.275** | 1.600 | 1.085 | 3.95× | 5.82× |
| TP8 `1x16384` | 2/8 | 0.183 | 0.809 | 0.550 | 3.01× | 4.42× |
| TP8 `1x8192` | 2/8 | 0.246 | 0.411 | 0.277 | 1.13× | 1.68× |
| TP4 `1x32768` | 4/16 | 0.463 | 1.572 | 1.324 | 2.86× | 3.39× |
| TP2 `1x32768` | 8/32 | 0.753 | 1.573 | 1.815 | 2.41× | 2.09× |
| TP1 `1x32768` | 16/64 | 1.299 | 1.592 | 2.997 | 2.31× | 1.23× |
| TP1 `16384+16384` | 16/64 | 1.008 | 0.807 | 3.007 | 2.98× | **0.80×** |
| TP2 `8192x4` | 8/32 | 0.513 | 0.414 | 1.499 | 2.92× | **0.81×** |

反向（GB200，文件尾部）：相对 FLA 1.48×–**4.04×**，最好的一档是 `hk2_hv4 / 32k_1seq`（0.666 ms vs 2.690 ms）。

**这张表的形状比数值更有信息量**：

- **相对 FLA 处处大胜**，与 FLA 文档 §8、§10 的分析一致（并行度饥饿 + 访存）。
- **相对 FlashInfer 在低 head 数 + 长序列上大胜（最多 5.8×）**——正是 CP 开火的区间。$h_v=8$、32k 时 FlashQLA 0.275 vs FlashInfer 1.600，差了近 6 倍。
- **在高 head 数 + 多序列时输给 FlashInfer（0.80–0.93×）**——那里 GPU 本来就饱和、CP 关闭，所以这是**纯 kernel 质量的比较**。

> **读法**：CP 开着的时候，比的是算法（$B\times H_v \to B\times H_v\times N_{\mathrm{shard}}$ 的并行度提升）；CP 关着的时候，比的是手艺（warp 分工、TMEM 利用、流水深度）。FlashQLA 在前者领先，在后者略逊。
> 再叠上 §6.6 那个 8k 反常（SM100 比 H200 慢 2 倍），可以得到一个实用结论：**FlashQLA 在 SM100 上的性能对"CP 有没有被启用"极度敏感，排查性能问题第一步就该确认这一点。**

benchmark 覆盖 $h_{qk}/h_v \in \{2/8, 4/16, 8/32, 16/64, 16/32, 8/24, 16/48, 16/16, 32/32\}$（`benchmark/bench_gated_delta_rule.py:70-79`），对应 Qwen3.5 家族 TP1–TP8。`profile/profile_gdr.py:205-337` 给出 per-kernel 分解标签（`csum/solve/gdr` + CP 时的 `cp-w/cp-h/cp-c`）——**排查时直接看这三到六个标签的占比。**

---

## 13. 反向传播（简述）

推理不用，但它解释了 §2.4 的融合边界，所以简述。

结构（`chunk/__init__.py:91-159`）：`intra_card_cp_preprocess_bwd` → `fused_gdr_h`（重算全部 `h`）→ `fused_gdr_bwd`（一个巨核）→ GVA 的 `group_reduce_vector`（`:153-156`）→ `dg` 的反向 `chunk_local_cumsum`（`:158`）。

- **保存**：`q, k, v, g_cumsum, beta, A, initial_state, cu_seqlens`（+ CP cache）。**重算**：全部 per-chunk 状态 `h`（`[B,N,H,128,128]` bf16，$T=32768, H_v=32$ 时 512 MB），以及 `fused_bwd` 内部的 `w/u/V'/U`。**`A` 被复用，从不重算，正因为 `kkt_solve` 是 gate-free 的。**
- `blackwell/fused_bwd.py`：grid `batch_size·H`（无 DV 拆分），threads 512，**逆序 chunk**，**单缓冲** SMEM 加逐张量的 just-in-time refill barrier（`:1149-1230`），**每 chunk ~22 条 tcgen05 MMA**（`:874-1147`），11 个 TMEM tile（`:189-205`）含显式别名 `dq_tmem = u_tmem`（`:195`）。SMEM 预算在注释里逐项标注（`:121,138,143`），合计约 216 KB——**几乎吃满 SM100 的 227 KB，所以放不下双缓冲。**
- 两个值得记的数值技巧：
  1. `mask_tmem` 是 Λ 掩码的 **TMEM scratchpad**，算一次读 4 次（`:613,634,681,695,708`），配手写布局 `tilelang.layout.Layout([64,64], lambda i,j: [i + (j//32)*64, j%32])`，注释：*"TCGEN05 Layout-E maps relative Consumer-A threads; do not add the warp group's absolute +256 thread offset"*（`:41-46`）。
  2. **通过 bf16 SMEM 做 fp32 精确转置**：把每个 fp32 拆成 hi/lo 两个 `uint16` 分别转置（`:772-805`）。这是 commit `6e5d2ea` "Fix dg accuracy by high-precision matrix transposition"。**别的实现不需要这个，因为别的实现不会把 fp32 从复用的 bf16 tile 里转置出来。**
- `T.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True`（`:16`），而 `TL_DISABLE_THREAD_STORAGE_SYNC` 在 bwd 里是**注释掉的**（`:15`），在 fwd/prepare_h 里是开的。
- CP 的反向额外加 `fused_gdr_dh`（= `blackwell/cp_bwd.py::fused_gdr_dh_ws`，与 Hopper 版相同）+ `correct_terminal_states`。

---

## 14. hopper → blackwell 的移植改了什么

`diff -u` 结果：

| 文件 | 状态 |
|---|---|
| `kkt_solve.py` | **完全相同** |
| `cp_fwd.py` | **完全相同** |
| `cp_bwd.py` | **完全相同** |
| `__init__.py` | **完全相同** |
| `prepare_h.py` | 移植到 tcgen05/TMEM |
| `fused_fwd.py` | 移植到 tcgen05/TMEM |
| `fused_bwd.py` | 移植（1218 → 1413 行） |

**移植只碰了三个状态递推 kernel。** `kkt_solve`（128 线程、小规模 fp32 工作）和 CP 决策/修正 kernel 被判定为不值得移植——这本身是个有信息量的判断：**这些 kernel 不在瓶颈上。**

`fused_fwd.py` 的具体改动：

1. **wgmma → tcgen05。** Hopper 在 consumer warpgroup **内部**发 `T.gemm`（wgmma 累加在该 warpgroup 的寄存器里）。Blackwell **把全部 6 条 GEMM 移到一个专用的 32 线程 MMA warp**（`tx ∈ [384,416)`），发 `T.tcgen05_gemm(..., mbar=tcbar_i)` 到 TMEM。consumer 退化成纯 CUDA core / SFU 工人，从 TMEM 取结果。**这是 tcgen05"warp 发射 + TMEM 目标"而非"warpgroup 发射 + RF 目标"的直接结构后果。**
2. **引入 TMEM**（fwd 5 个 tile，bwd 11 个），Hopper 只有寄存器 fragment。**因为 TMEM 只有 512 列，状态必须拆成两个 64 列半块**——于是有了 `h_fragment_L/R`、`h_tmem_L/R`、`tcbar_5a/5b` 和拆半的 $S\mathrel{+}=K^{\top}V'$（`fused_fwd.py:537-575`）。Hopper 保持一个连续 `h_fragment` 和一次 GEMM。`prepare_h` 也无条件拆了 L/R（`prepare_h.py:147-154`）。
3. **producer warp 角色重排。** Hopper：{Q+K loader, V+β loader, A+g loader, store}，`data_is_ready` arrive_count `[96]*2`（3 个 loader warp）。Blackwell：{**MMA**, Q+K+V+A TMA loader, β+g loader, store}，arrive_count `[64]*2`（2 个 loader warp）。**四次 TMA copy 被压进一个 warp，腾出一个 warp 专门发 MMA。**
4. **barrier arrive-count 重新配平**：`bar_0` 416→448、`bar_3` 128→256、`bar_4` 128→256、`bar_5` 416→288，外加 6–7 个新的单次到达 `tcbar_*`。
5. **寄存器预算向 producer 倾斜**：`PRODUCER_NREG` 32→**72**，`CONSUMER_S_NREG` 160→**168**。MMA warp 需要寄存器构造 descriptor；S consumer 因为多了 TMEM 往返也要更多。
6. **`TL_DISABLE_THREAD_STORAGE_SYNC: True` 在 Blackwell fwd 和 prepare_h 上启用**（Hopper 上注释掉），换成显式 `T.sync_threads(100/101/102, 128)`。
7. **per-chunk `h` store 上 `disable_tma=True`**（`fused_fwd.py:716-755`），Blackwell 专属。
8. **`block_DV` 档位 3 → 2**（砍掉 32）。SMEM/TMEM 更大使 128 更常可用，而 32 会浪费 TMEM 列。
9. **`mt` dtype `ht_dtype` → `qkva_dtype`**（`prepare_h.py:91`）——$M$ 恒为 bf16，buffer 减半且匹配 MMA 操作数 dtype。
10. **`fused_bwd` 改成单缓冲**，配逐张量 JIT refill barrier，22 条 tcgen05 MMA。TMEM 吸走 11 个累加器，但 SMEM（~216 KB）已经支撑不起双缓冲。
11. **CP 阈值按架构分叉**（`cp_context.py:108-124`），理由与后果见 §6.6。

**没变的**：`use_2cta=False` 处处如此——pair-SM / 2-CTA MMA 从未利用。无 fp8。无 cluster 级特性。chunk size 仍是 64。

---

## 15. 汇总

### 15.1 优化清单

| 优化 | 算法依据 | 换来什么 | 付出什么 |
|---|---|---|---|
| **门驱动近似 CP + $M$ 精确兜底** | **杠杆四为主 + 杠杆三兜底** | 并行度 $B\!H_v \to B\!H_v N_{\mathrm{shard}}$（例：8 → 256） | 一次 host D2H 同步；`mt` 每 (段,head) 32 KB；3 个额外 kernel；SM100 上被 256-chunk 阈值卡住 |
| gate-free 三角求逆 + Λ 后置 | 杠杆一 | $\exp$ 出关键路径；数值范围最小；**一份 `A` 服务 fwd/bwd/prepare_h** | `A` 与参考实现/FLA 的 `A` 不是同一对象 |
| Λ 寄存器复用（`Ag` 与 `Pg` 共用） | Λ 在公式里出现两次 | $64\times64$ 的 `exp2` 每 chunk 只付一次 | 要求两处后处理在同一 warpgroup |
| $\beta$ 施加在 $64\times64$ 的 `A` 上 | $\mathrm{diag}(\beta)$ 与 $A_{\mathrm{raw}}$ 可交换位置 | 逐元素乘从 $2\times64\times128$ 降到 $64\times64$ | — |
| $w/u$ 塌缩，`W` 原地覆写 `v_shared` | 算法文档 §3.8 合并形式 | 每 chunk 省一次 $64^2\times128$ GEMM + 整个 `w` 张量 | 反向要重算 |
| 标量衰减从矩阵乘积提出（log 域累加） | $\prod e^{g_i}X_i = e^{\sum g_i}\prod X_i$；**门是标量** | $n$ 次 $128^2$ 缩放 → 1 次 | — |
| 三段式（而非一段或五段）分解 | `A` 被 3 个 pass 共享 | CP 与 bwd 都能复用 `A` | `A` 一次 HBM 往返 |
| 四 warpgroup + producer 拆 4 warp | 依赖图（谁拥有状态 / $V_{\mathrm{new}}$ / 输出） | 一个 warp 流水 6 条 tcgen05 | 手数 arrive_count，手插 fence/sync |
| TMEM 448/512 列 + 状态拆两半 | TMEM 只有 512 列 | 状态累加器出寄存器文件，`CONSUMER_S_NREG=168` | 状态更新 GEMM 拆两条 |
| `block_DV ∈ {128,64}` 按 SM 数 | 并行度不足时沿 $V$ 再切 | 小 grid 时并行度翻倍 | $KS,QS$ 各算两遍 |
| $16\times16\times4$ 基例 + 2 层合并 | 杠杆二 | 依赖链 $64\to16+2$；4 块并行摊掉串行 | 只有第 2 层上 MMA |
| `T.annotate_layout` 双向使用 | 大 tile 要 swizzle，小 tile 要标量寻址 | 两者各得其所 | 手写布局 |
| bf16 SMEM 上 hi/lo fp32 精确转置 | fp32 = 两个 uint16 | `dg` 精度修复 | 两次转置 |

### 15.2 独有的东西

按算法依据的强度排：

1. **门驱动的近似上下文并行。** 三个库只有它做。FLA 与 FlashInfer 都把 chunk 间状态扫描当硬串行依赖（FlashInfer 用杠杆三精确破解，但要付构造 $M$ 的代价）。FlashQLA 直接利用**GDN 的门是收缩算子**这个数值事实：$\sum g < -10$ 之后入态影响 $< 4.5\times10^{-5}$，于是段可以**独立起跑**，用一段短预热后缀替代真正的 carry-in。理由是定量的、**逐 head 的**（Qwen3-Next 各 head 的 $A$ 尺度差异极大），而且**优雅降级**：衰减不够快的 head 走**精确**的 $M$ 修正而不是给出错答案。**这是一个由数值性质驱动的并行化，不是调度技巧。**
2. **gate-free 三角求逆 + Λ 后置。** 三个后果：求逆里没有 $\exp$；求逆与 $g$ 无关，所以**一份 `A` 同时服务前向和反向**；$\exp$ 的溢出不可能污染前代递推。
3. **$w/u$ 塌缩。** 这就是 FLA 需要独立 `recompute_w_u_fwd_kernel` 而 FlashQLA 不需要的原因，而它成立依赖第 2 条。
4. **刻意的三段式分解**，理由写在 README 里，融合边界由"哪些中间量被多 pass 共享"决定。
5. **标量衰减在 log 域累加、最后一次施加。**
6. **TMEM 当通用 fp32 scratchpad**（`mask_tmem`），不只当 MMA 累加器。
7. **bf16 SMEM 上的 hi/lo fp32 精确转置。**

### 15.3 一句话

> FlashQLA 是**三段式（CP 开启时六段）TileLang 实现**，把 FLA 的 `wu`+`h`+`o` 三个 kernel 合成一个 512 线程、4 warpgroup、6–7 条 tcgen05 MMA 的融合核。
> 它的**核心武器是门驱动的近似上下文并行**：逐 (段, head) 判断累计对数衰减是否跌破 −10，跌破就让该段从零状态独立起跑（只预热几个 chunk），没跌破就用精确的 $K\times K$ 转移矩阵 $M$ 做串行修正。**这是唯一利用了"GDN 状态会指数遗忘"这个模型性质的实现**，也是它在低 head 数 + 长序列上能比 FlashInfer 快 3–6 倍的全部原因。
> 与之绑定的是三处代数重排（gate-free 求逆、$w/u$ 塌缩、标量衰减提取），**它们互为前提**：正因为 $A_{\mathrm{raw}}$ 门无关，$w/u$ 才能塌缩、$M$ 才能省掉一次 $\exp$ 缩放、`A` 才能被三个 pass 共享。
> 短板有三个：**SM100 前向的 256-chunk CP 阈值**在 8k–12k token 区间留下一个比 Hopper 还慢 2 倍的坑；**CP 关闭时纯 kernel 质量略逊 FlashInfer**（0.80–0.93×）；**没有分页状态池、没有 fp8 状态 I/O**，serving 集成需要外部 gather/scatter。

---

## 相关文档

- [`GDN_Algorithm.md`](GDN_Algorithm.md)：算法推导、依赖结构、四个数学杠杆（本文用了杠杆一、二、三、四全部）
- [`FLA_Triton_Baseline.md`](FLA_Triton_Baseline.md)：被 FlashQLA 替换的那条多 kernel Triton 路径
- [`FlashInfer_GDN_Blackwell.md`](FlashInfer_GDN_Blackwell.md)：只用杠杆三、精确无近似的另一条路
