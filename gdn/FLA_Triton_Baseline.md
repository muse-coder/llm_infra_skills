# flash-linear-attention（FLA）的 GDN 优化方案

> 定位：**Triton 多 kernel 实现。可移植的参考基线、训练主力、以及所有特殊配置的唯一回退路径。**
> 前置阅读：[`GDN_Algorithm.md`](GDN_Algorithm.md)，本文全程使用其中的符号（$C$、$A_{\mathrm{raw}}$、$\Lambda$、$V_{\mathrm{new}}$、环节 ①–⑤）。
> 代码基准：`flash-linear-attention` @ `51a8c0a8`，路径相对于仓库根。

---

## 目录

1. [FLA 在三个库里的位置](#1-fla-在三个库里的位置)
2. [8 个 kernel 的流水线](#2-8-个-kernel-的流水线)
3. [优化一：kkt + solve_tril 寄存器内融合](#3-优化一kkt--solve_tril-寄存器内融合)
4. [优化二：两级三角求逆](#4-优化二两级三角求逆)
5. [优化三：`exp2` 化与 Λ 永不物化](#5-优化三exp2-化与-λ-永不物化)
6. [优化四：编译期特化作为主要抽象手段](#6-优化四编译期特化作为主要抽象手段)
7. [w / u 为什么保留并物化](#7-w--u-为什么保留并物化)
8. [状态递推：默认路径不解决串行问题](#8-状态递推默认路径不解决串行问题)
9. [intracard CP：FLA 确实有分段扫描，但门槛很高](#9-intracard-cpfla-确实有分段扫描但门槛很高)
10. [访存账：HBM 流量的完整核算](#10-访存账hbm-流量的完整核算)
11. [精度策略](#11-精度策略)
12. [varlen 机制](#12-varlen-机制)
13. [反向传播：以 A 为唯一检查点](#13-反向传播以-a-为唯一检查点)
14. [FLA 刻意不做的事](#14-fla-刻意不做的事)
15. [backend 分派：FLA 自己承认会被替换](#15-backend-分派fla-自己承认会被替换)
16. [汇总](#16-汇总)

---

## 1. FLA 在三个库里的位置

先把定位说清楚，否则后面所有"缺点"都会读成贬义。

FLA 的 GDN 实现要同时满足的约束比另两个库多得多：

| 需求 | FLA | FlashQLA | FlashInfer |
|---|---|---|---|
| 反向传播（训练） | ✅ 完整 | ✅ | ❌ 仅前向 |
| 任意 $K,V$（≤256） | ✅ | ❌ 仅 128/128 | ❌ 仅 128 |
| $C \in \{16,32,64\}$ | ✅ | ❌ 固定 | ❌ 固定 |
| 融合 gate / beta / l2norm 激活 | ✅ | ❌ | ❌ |
| `allow_neg_eigval`（$\beta$ 可 >1） | ✅ | ❌ | ❌ |
| 非 NVIDIA 硬件（Ascend NPU） | ✅ | ❌ | ❌ |
| GVA 且 $H_v > H_{qk}$ | ✅ | ✅ | ✅ |
| 跨卡 CP | ✅ | ✅ | ✅ |

所以 FLA 是**用 Triton 换覆盖面**：一份可读的 Python 代码，覆盖训练 + 推理 + 多硬件 + 多配置。它的性能上限被这个定位限死，而不是被作者的能力限死。

**FLA 的优化重心是"在 Triton 的表达能力之内尽可能省 HBM 往返"，而不是"填满机器"。** §8 会说明它完全没有触碰 GDN prefill 的头号问题（并行度饥饿）。

---

## 2. 8 个 kernel 的流水线

在 Qwen3-Next 层设置下（`fla/layers/gated_deltanet.py:309-325`：`use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, state_v_first=True`），一次 GDN chunked forward 发射 **8 个 Triton kernel**。入口 `chunk_gated_delta_rule_fwd`，`fla/ops/gated_delta_rule/chunk.py:33-123`。

| # | Kernel | 发射位置 | grid | 输入 | 输出（HBM） | 对应环节 |
|---|---|---|---|---|---|---|
| 1 | `l2norm_fwd_kernel` (q) | `chunk.py:282` | per-row | `q` | `q̂`, `q_rstd` | 前处理 |
| 2 | `l2norm_fwd_kernel` (k) | `chunk.py:283` | per-row | `k` | `k̂`, `k_rstd` | 前处理 |
| 3 | `fused_beta_sigmoid_fwd_kernel` | `chunk.py:287` | `(cdiv(numel,2048),)` | `b` | `beta` fp32 | 前处理 |
| 4 | `gdn_gate_chunk_cumsum_scalar_kernel` | `chunk.py:53-61`，kernel 在 `gate.py:180` | `(NT, B·HV)` | `a`, `A_log`, `dt_bias` | `g` **fp32** | **①** |
| 5 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | `chunk_fwd.py:383` | `(NT, B·HV)` | `k, g, beta` | `A` = $(I+\mathrm{tril}(\Lambda\circ\beta KK^{\top}))^{-1}$，bf16 | **②** |
| 6 | `recompute_w_u_fwd_kernel` | `wy_fast.py:271` | `(NT, B·HV)` | `k, v, beta, A, g` | `w [T,HV,K]`, `u [T,HV,V]` | **③ 的一半** |
| 7 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `chunk_delta_h.py:715` | `(cdiv(V,BV)·N·HV,)` | `k, u, w, g, h0` | `h [B,NT,HV,V,K]`, `v_new`, `final_state` | **③ 余下 + ④** |
| 8 | `chunk_fwd_kernel_o` | `chunk_o.py:554` | `(cdiv(V,BV), NT, B·HV)` | `q, k, v_new, h, g` | `o` | **⑤** |

**"核心 GDN 数学"是 #5–#8 四个 kernel。** #1–#4 是前处理，另两个库通常要求调用方在算子外做完（或折进投影 GEMM 的 epilogue）。

三点值得先记住：

1. **环节 ③ 被切成两半，跨越 kernel #6 和 #7。** $(\Lambda\circ A_{\mathrm{raw}})\mathrm{diag}(\beta)$ 乘 $K$ 和 $V$ 的部分在 #6，减 $KS_{\mathrm{prev}}$ 的部分在 #7。这是 $w/u$ 形式的直接后果。
2. **环节 ④ 与 ③ 在同一个 kernel（#7）**，因为它们必须共享跨 chunk 的状态。
3. **#7 与 #8 的 grid 分解方式不同**：#7 是 per-(sequence, v-block) 加 chunk 内串行循环；#8 是 per-(chunk, head, v-block) 完全并行。**这个差异是 `h` 必须落 HBM 的根本原因**——两个 kernel 对同一批状态的访问模式不兼容，除非引入 warp specialization 或分段扫描。

---

## 3. 优化一：kkt + solve_tril 寄存器内融合

这是 FLA 唯一的重量级融合优化，也是它相对早期版本最大的收益点。

`chunk_gated_delta_rule_fwd_kkt_solve_kernel`，`chunk_fwd.py:40-328`。docstring（`chunk_fwd.py:57-69`）把意图写得很清楚：

> *"This kernel fuses chunk_scaled_dot_kkt_fwd and solve_tril into a single kernel, avoiding the HBM round-trip for the intermediate A matrix."*

**做法**：把 $64\times64$ 的 chunk 切成 $4\times4$ 的 $16\times16$ 子块网格，**10 个下三角子块全部作为 fp32 累加器留在寄存器里**（`chunk_fwd.py:124-135`），在同一个 kernel 内依次完成：

```
step 1  沿 K 维分 BK 块循环，累出 10 个 [16,16] 的 KKᵀ 子块      chunk_fwd.py:137-169
step 2  逐元素乘 Λ 和 β，加严格下三角掩码                        chunk_fwd.py:171-212
step 3  4 个对角块各自做 16 步前代求逆                            chunk_fwd.py:214-251
step 4  12 次 tl.dot 做分块 Schur 补合并                          chunk_fwd.py:253-291
step 5  10 次掩码 store，写出 A（bf16）                           chunk_fwd.py:319-328
```

**收益**：$C\times C$ 中间矩阵 `A` 的 HBM 往返从 4 次降到 2 次。$T=32\text{k}, H_v=32$ 时是 640 MiB → 256 MiB（详见 §10）。

**代价与限制**：

- **只有 $C=64$ 有这条融合路径**（`chunk_fwd.py:378`）。$C\in\{16,32\}$ 落回两 kernel 的旧路径（`chunk_fwd.py:397-413`）。所以 $C=64$ 不只是默认值，是**唯一被优化的值**。
- autotune 空间只有 `num_warps ∈ {1,2,4}`，**没有 `num_stages`**（`chunk_fwd.py:30-38`）。这说明它是**寄存器/ILP 受限**，不是流水受限——寄存器已经被 10 个 $16\times16$ fp32 块 + 4 个逆块吃满，再开流水级数只会溢出。
- 尾块用嵌套的 `if i_tc1 < T:` 早退（`chunk_fwd.py:144,152,161`），避免算无效子块。

**这个 kernel 是理解 FLA 优化哲学的最佳样本**：Triton 给不了 TMA、warp 专用化、异步流水，但给了寄存器分块和跨阶段融合。FLA 就把这一条用到极致。

---

## 4. 优化二：两级三角求逆

对应算法文档 §7.2 的杠杆。FLA **不用**全 $64\times64$ 前代，而是两级方案。

### 4.1 第一级：$16\times16$ 块内前代

`solve_tril.py:249-268`（独立版）/ `chunk_fwd.py:227-246`（融合版）：

```python
for i in range(2, min(16, T - i_t * BT)):
    b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
    b_a_11 = tl.where(o_i < i, b_a_11, 0.)
    b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
    b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
b_Ai_11 += m_I                     # solve_tril.py:269
```

- 单位下三角 ⟹ **不需要除法**（对角恒为 1，掩码 `o_i[:,None] > o_i[None,:]`，`solve_tril.py:220`）
- 循环从 `i=2` 开始（第 0、1 行无需更新）
- 全部是 `tl.sum` 归约，**不是 `tl.dot`** ⟹ 不上 Tensor Core，纯向量 ALU，14 次串行迭代
- 融合版从**寄存器**里抽行而不是重新 load，注释明说（`chunk_fwd.py:214-220`）：
  > *"Same algorithm as solve_tril, but rows are extracted from in-register [BC,BC] tensor via `tl.sum(tl.where(mask, tensor, 0), 0)` instead of `tl.load` from HBM."*

### 4.2 第二级：分块 Schur 补合并

`solve_tril.py:295-317` / `chunk_fwd.py:257-291`：

```python
b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21), b_Ai_11)
b_Ai_32 = -tl.dot(tl.dot(b_Ai_33, b_A_32), b_Ai_22)
b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43), b_Ai_33)
b_Ai_31 = -tl.dot(b_Ai_33, tl.dot(b_A_31, b_Ai_11) + tl.dot(b_A_32, b_Ai_21))
b_Ai_42 = -tl.dot(b_Ai_44, tl.dot(b_A_42, b_Ai_22) + tl.dot(b_A_43, b_Ai_32))
b_Ai_41 = -tl.dot(b_Ai_44, tl.dot(b_A_41, b_Ai_11) + tl.dot(b_A_42, b_Ai_21) + tl.dot(b_A_43, b_Ai_31))
```

正是算法文档 §7.2 的一般公式，按**反对角波前**顺序求值（`21,32,43` → `31,42` → `41`），12 次 $16\times16\times16$ 的 `tl.dot`。

### 4.3 粒度与精度

- **粒度恒为 $16\times16$。** 三个独立 kernel：`solve_tril_16x16_kernel`（`solve_tril.py:38`）、`merge_16x16_to_32x32_inverse_kernel`（`:108`）、`merge_16x16_to_64x64_inverse_kernel`（`:198`），按 $C$ 分派（`:386-391`）。
- 累加恒为 **fp32 寄存器**（`solve_tril.py:231-234`）。
- **`tl.dot` 精度可调，且两条路径不一致——这是一处容易被忽略的差异**：

| 路径 | 合并步的 dot 精度 | 依据 |
|---|---|---|
| 独立 `solve_tril` | 默认 **`ieee`**（fp32 模拟） | `FLA_TRIL_PRECISION` 默认 `'ieee'`，`solve_tril.py:19-22`；$16\times16$ kernel 硬编码 `'ieee'`（`:30`） |
| **GDN $C=64$ 融合路径** | **`tf32`**（Ampere+） | `SOLVE_TRIL_DOT_PRECISION = 'tf32' if IS_TF32_SUPPORTED else 'ieee'`，`chunk_fwd.py:20-23`，用于全部 12 次合并 dot（`:258-291`） |

也就是说**在任何现代 NVIDIA GPU 上，GDN 的快速路径用 tf32 做 Schur 合并**，而通用 `solve_tril` 用 ieee。这是 GDN 专属的一次速度/精度交换，代价是尾数从 24 bit 降到 10 bit。之所以敢换，依据是算法文档 §3.4 的最后一条：待求逆矩阵元素模长 $\le 1$。

- 独立版的 store 显式指定 `fp_downcast_rounding="rtne"`（`solve_tril.py:89,175-177,330-339`），**融合版没有**（`chunk_fwd.py:319-328`）。属于源码里的不一致。

---

## 5. 优化三：`exp2` 化与 Λ 永不物化

### 5.1 全局 `exp2` 化

FLA 把 $\log_2 e$ 预乘进 cumsum 的输出：`scale = RCP_LN2 = 1.4426950216`（`chunk.py:57,65`，常量在 `fla/ops/utils/constant.py:10`），之后**所有指数运算一律用 `exp2`**：

$$
\exp(g) = \exp_2\!\left(g\cdot\log_2 e\right)
$$

`exp2` 直接映射到单条 `ex2.approx.ftz.f32` SFU 指令，而 `exp` 需要额外一次乘法。调用点遍布全库：`chunk_fwd.py:183-193`、`wy_fast.py:90,169,218`、`chunk_delta_h.py:241-242,255-278`、`chunk_o.py:123-129,290-295,516`。

代价：**存在 HBM 里的 `g` 不是标准 $\log\alpha$ 的 cumsum，而是它乘了 $\log_2 e$ 之后的值。** 跨库对比数值或替换 backend 时，这是第一个要检查的地方（见算法文档 §10.2）。

### 5.2 Λ 永不物化

$\Lambda_{ij}=\exp_2(g_i-g_j)$ **从不写入 HBM**，只以寄存器表达式 `tl.where(..., exp2(b_g_r[:,None] - b_g_c[None,:]), 0.)` 的形式在每个需要它的地方就地重算：

| 位置 | 代码 |
|---|---|
| 融合 kkt+solve | `chunk_fwd.py:183-193`（10 个 $16\times16$ 的 Λ 子块） |
| 独立 kkt | `chunk_scaled_dot_kkt.py:73-74` |
| 输出 kernel | `chunk_o.py:124` |
| `dv_local`（bwd） | `chunk_o.py:516` |
| `dqkwg`（bwd） | `chunk_o.py:295` |
| wy bwd | `wy_fast.py:218` |

这是正确的取舍：Λ 是 $C\times C=4096$ 个元素、纯由两个标量之差生成，重算比存取便宜得多。

**但注意与 FlashInfer 的对照**：FlashInfer 也不物化到 HBM，但它把 Λ 算在寄存器里之后**在两处消费者之间复用同一批寄存器**，$\exp$ 次数直接减半；FLA 每处都重算。这是 Triton 表达能力的边界——跨 `tl.dot` 阶段精确控制寄存器生命期，Triton 里做不到。

### 5.3 一个必须记住的 NaN 陷阱

`chunk_fwd.py:176-178` 记录了一个真实修复：越界的 `g` 读成 0，而 `exp2(0 - g_inbounds)` 可能溢出成 $+\infty$，随后 `0 * inf = NaN`。**解法是把边界掩码 AND 进 `tl.where` 的条件里**，在乘法之前置零，而不是乘完再掩：

```python
b_A10 *= tl.where(m_tc1[:, None] & m_tc0[None, :],
                  exp2(b_g1[:, None] - b_g0[None, :]), 0.)   # chunk_fwd.py:188
```

Ascend 后端换了另一种解法——事后 NaN 过滤 `b_dA = tl.where(b_prod == b_prod, b_prod, 0.0)`（`backends/triton_ascend/wy_fast.py:464`）。这正是算法文档 §3.7 提到的坑。

---

## 6. 优化四：编译期特化作为主要抽象手段

FLA 处理配置组合爆炸的方式是 `@triton.heuristics` + constexpr 分支，而不是运行时谓词或模板。

**特化标志清单**：`USE_G`、`USE_GK`、`IS_VARLEN`、`USE_INITIAL_STATE`、`STORE_FINAL_STATE`、`SAVE_NEW_VALUE`、`STATE_V_FIRST`、`USE_DW`、`USE_A`、`USE_G_GAMMA`、`HAS_BIAS`、`HAS_SCALE`、`USE_FINAL_STATE_GRADIENT`
（声明于 `chunk_fwd.py:26-29`、`wy_fast.py:25-28,105-108`、`chunk_delta_h.py:31-38,343-349`、`chunk_o.py:143-147,332-335,436-441`、`gate.py:49-53,109-111,230-232`、`solve_tril.py:25-27,94-96,184-186`）

配套的一个关键细节：**所有 kernel 都用 `@triton.jit(do_not_specialize=['T'])`**，这样序列长度变化不触发重编译——serving 场景的必要条件。

**最激进的一处**：状态递推 kernel 用编译期 `if K > 64 / 128 / 192` 把 $K$ 展开成最多 4 个 64 宽的寄存器 tile `b_h1..b_h4`（`chunk_delta_h.py:91-106`），并断言 `K <= 256`（`:704`）。kernel 名字里的 `blockdim64` 就是指这个。

代价很实在：**代码复制约 4 倍，`STATE_V_FIRST` 再翻倍 ⟹ 每个操作大约 8 个近乎相同的分支**（`chunk_delta_h.py:91-106,140-163,177-200,208-228,244-249,287-307,317-340`）。可读性和 icache 都为此付费。这是"用编译期特化换零动态索引"的典型代价。

**架构条件化的 autotune 空间**也属于这一类，且大多是在绕编译器 bug：

| 位置 | 限制 | 原因 |
|---|---|---|
| `chunk_delta_h.py:25-28` | Blackwell 上 `num_warps` 钉死 `[2]` | `tl.dot` 递推竞态 |
| `wy_fast.py:18-22` | Blackwell 上 `num_warps=[2], num_stages=[4]` | autotune 不稳定（issue #913） |
| `chunk_o.py:704-709` | Hopper + `3.4.0 ≤ Triton < 3.7.1` 直接 `RuntimeError` | 已知误编译（issue #640），提示用户装 TileLang |
| `op.py:40-61` | `safe_dot` 用 `mov.f32` 内联汇编围栏 | 阻止 `TritonGPUHoistTMEMAlloc` 融合 add/dot（issue #638）；**GDN 路径未使用** |
| `chunk_o.py:304-306` | 反向 cumsum 单独一个 kernel | *"strange triton compiler issue"* |

这串 workaround 本身就是一条重要信息：**把 SMEM 布局、swizzle、寄存器分配全交给编译器，代价是要不断为编译器的具体版本打补丁。** 另两个库手写这些，问题换成了"移植到新架构要重写"。

---

## 7. w / u 为什么保留并物化

算法文档 §3.8 说过合并形式在推理上更省。FLA 仍然保留 $w/u$，有两个可在代码里验证的理由。

### 7.1 代码

`wy_fast.py:60-102`：

```python
b_A = tl.load(p_A, ...)                                            # (Λ∘A_raw)，wy_fast.py:75-76
for i_v in range(tl.cdiv(V, BV)):                                  # u 循环
    b_u = tl.dot(b_A, (b_v * b_b[:, None]), allow_tf32=False)      # wy_fast.py:85
if USE_G:
    b_g = exp2(tl.load(p_g, ...))                                  # wy_fast.py:88-90
for i_k in range(tl.cdiv(K, BK)):                                  # w 循环
    b_kb = b_k * b_b[:, None]
    if USE_G: b_kb *= b_g[:, None]                                 # wy_fast.py:100
    b_w = tl.dot(b_A, b_kb.to(b_k.dtype))                          # wy_fast.py:101
```

即 $u = A_{\mathrm{gated}}(\beta\odot V)$、$w = A_{\mathrm{gated}}(\beta\odot e^{g}\odot K)$，与算法文档 §3.8 一致。`BK=BV=64` 硬编码（`wy_fast.py:262-263`）。

（一处源码不一致：$u$ 的 dot 用 `allow_tf32=False`，$w$ 的没用。两边操作数都已转 bf16，Ampere+ 上影响不大，但确实不对称。）

### 7.2 理由一：反向传播复用（可验证）

forward **只把 `A` 存进 `ctx`**（`chunk.py:310-326`），不存 `w/u/h/v_new`。backward 第一步就重算 $w,u$：

```python
def chunk_gated_delta_rule_bwd(...):
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g, ...)   # chunk.py:147-155
```

之后 $w$ 被消费**三次**：`bwd_dhu`（`chunk.py:202-216`）、`bwd_dqkwg`（`:217-232`）、以及它的梯度 $dw$ 送进 `prepare_wy_repr_bwd`（`:233-243`）。

**物化 $w$ 一次、读三次，比重算 $A_{\mathrm{gated}}(\beta k)$ 三次便宜。** 函数名 `recompute_w_u_fwd`（别名 `fwd_recompute_w_u`，`wy_fast.py:346`）本身就编码了这个策略：**`A` 是检查点，$w/u$ 是可重算量**。

### 7.3 理由二：递推 kernel 已经没有寄存器了

`chunk_delta_h.py:91-106` 已经持有最多 $4\times[64,BV]$ 的 fp32 状态 tile 加 `b_v` 累加器。再在 kernel 内做一次 $[C,C]\times[C,\cdot]$ 的 inverse-apply 会直接溢出——证据是它的 `BV` autotune 在非 Ada 硬件上已经被压到 `{32}`（`chunk_delta_h.py:44`）。

### 7.4 结论

$w/u$ 的保留**不是疏忽，是训练定位的必然结果**。但它对推理是纯损失：多两个 $C\times d$ 张量的 HBM 往返（§10 里的 1.0 GiB），多两次矩阵乘，依赖链更长。**这是 FLA 与另两个库在推理性能上的第一处结构性差距**，且无法在保持训练支持的前提下消除。

---

## 8. 状态递推：默认路径不解决串行问题

这一节是全文最重要的部分。

### 8.1 grid 与循环结构

`chunk_delta_h.py:714`：

```python
def grid(meta): return (triton.cdiv(V, meta['BV']) * N * HV, )
```

kernel 内分解（`chunk_delta_h.py:77-89`）：`i_v = pid % NV`，`i_n, i_h = (pid // NV) // HV, (pid // NV) % HV`。

主循环（`chunk_delta_h.py:166`）：

```python
for i_t in range(NT):
    # 1. 把当前 b_h1..b_h4（本 chunk 之前的状态）store 到 h[i_t]      :170-200
    # 2. b_v = Σ_j dot(w_j, h_j)                                      :202-228
    # 3. b_v = u - b_v；可选 store v_new                              :230-234
    # 4. b_v *= exp2(g_last - g)；b_h *= exp2(g_last)                 :236-249
    # 5. b_h += dot(kᵀ, b_v)                                          :281-307
# 循环后 store final_state                                            :309-340
```

### 8.2 后果

**并行度 $= N_V \times B \times H_v$，chunk 轴严格串行，fp32 状态常驻寄存器。没有 `NUM_CHUNKS` 拆分，没有两趟扫描，没有任何序列方向的并行。**

代入 Qwen3.5-35B TP1（$B=1, T=32\text{k}, H_v=32, K=V=128, BV=64$）：

$$
\text{grid} = 2\times1\times32 = \mathbf{64}\ \text{个 program}，
\qquad
\text{每个做 } NT = 512\ \text{次串行依赖迭代}
$$

H100 有 132 个 SM，B200 更多。**64 个 program 对 132 个 SM，欠占用约 2 倍**；而且关键路径是 512 组依赖的矩阵乘。TP8 场景（$H_v=8$）grid 只有 16，欠占用 8 倍以上。

这正是算法文档 §6.2 描述的并行度饥饿，**FLA 的默认路径原封不动地承受了它**。这是 FLA 与另两个库最根本的差距——不是常数因子，是数量级。

反向的 `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` 是镜像结构：同样的 grid（`chunk_delta_h.py:774`），反向串行循环 `for i_t in range(NT-1, -1, -1)`（`:480`）。

### 8.3 顺便：per-chunk 衰减的应用顺序有个细节

`chunk_delta_h.py:236-249`：

```python
last_idx = min((i_t + 1) * BT, T) - 1
b_g_last = tl.load(g + ... + last_idx*HV + i_h)
b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
b_g_last = exp2(b_g_last)
b_h1 *= b_g_last
```

`b_v` 的衰减是在 `v_new` **store 之后**才施加的（`:232-234` 先 store）。所以**落到 HBM 的 `v_new` 是未衰减版本**，与 `chunk_fwd_kernel_o` 的预期一致（输出式 ③ 用的是未衰减的 $V_{\mathrm{new}}$，衰减 $\mathrm{diag}(e^{g_C-g})$ 只属于状态更新式 ②）。跨库对齐中间结果时这是个必查点。

---

## 9. intracard CP：FLA 确实有分段扫描，但门槛很高

FLA **有**分段扫描实现，只是不在默认路径上。这一点常被误读，值得写清楚。

### 9.1 算法

`fla/ops/common/intracard_cp.py:435-624`，用的正是算法文档 §7.3 的仿射结合律。

1. **选段长**（`intracard_cp.py:168-171`）：
   ```python
   target_splits = max(4, num_sms // (NUM_V_BLOCKS * num_heads))    # NUM_V_BLOCKS = 2
   ```
   下限 `MIN_SUBSEQ_CHUNKS = 128` 个 chunk = **8192 token**（`:177-180`）。**注意这个式子的形式**：它直接用 `num_sms // (2 · num_heads)` 算需要切几段——这就是算法文档 §6.2 并行度饥饿公式的逆运算，$H_v$ 越小切得越多。
2. **拆分阈值** `3 * subseq_len`（`:204`）⟹ **短于 24576 token 的序列永不拆分**；且若所有 `seq_lens < 2*subseq_len` 直接早退（`:466`）。
3. **Pass 1 预扫描**（`:253-291`）：`pre_process_fwd_kernel_merged`（`fla/ops/cp/chunk_delta_h.py:41`），grid `(cdiv(V,BLOCK)+cdiv(K,BLOCK), HV, S_split)`，每段产出一个融合 buffer `hm [S_split, HV, K, V+K]` fp32，**同时装本段的状态贡献 $N$ 和本段的转移算子 $M$**（把 $[K,V]$ 和 $[K,K]$ 拼在一个张量里，最后一维 `V+K`）。
4. **Merge**（`:294-359`）：`merge_fwd_bwd_kernel`（`fla/ops/cp/chunk_delta_h.py:332`）以 `INTRACARD_MODE=True` 串起各段状态，得到每段精确的初始状态。
5. **Pass 2**：**复用普通的** `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`，只是喂一个合成的 `cu_seqlens_subseq_gpu`，让每段冒充一条独立序列（`:604-617`），最后 `final_state = final_state_subseq[last_subseq_indices]`（`:620`）。

第 5 步的设计很经济：**不写新 kernel，用 varlen 机制把"段"伪装成"序列"。** 这是很典型的 FLA 风格解法。

### 9.2 为什么在默认路径外

`fla/ops/common/backends/intracard.py`：

- `default_enable = False`（`:35`），需要 `FLA_INTRACARD_CP` 环境变量
- verifier 要求 `torch.is_inference_mode_enabled()` **且** `cu_seqlens is not None`（`:59-66`）
- `FLA_INTRACARD_MAX_SPLITS` 默认 32（`:26`），注释：*"Limits merge chain depth to control precision loss."*

即：**仅推理、仅 varlen、需显式开启、且序列 ≥24k。** 训练 prefill 和短序列推理永远拿不到它。

### 9.3 另外两条

- **跨卡 CP**：`cp_context: FLACPContext` 贯穿 `chunk.py:82-93,108-109,157-158,183-200`，同样的预扫描 + merge 数学，跨 rank。与 `initial_state`/`output_final_state` 互斥（`chunk.py:538-541`）。
- **死代码**：`fla/ops/common/chunk_h_split.py`（619 行的通用 split-scan）和 `chunk_h_parallel.py` **全仓库无引用**。看到它们不要以为 GDN 用了。

---

## 10. 访存账：HBM 流量的完整核算

取 Qwen3.5-35B 风格配置 $H_{qk}=16, H_v=32, K=V=128$，$B=1, T=32768$，bf16，$C=64$ ⟹ $NT=512$。

### 10.1 物化的张量

| 张量 | 形状 | dtype | 分配位置 | 存给 bwd？ | 大小 |
|---|---|---|---|---|---|
| `g` | `[B,T,HV]` | **fp32** | `gate.py:179` | ✅ | 4.0 MiB |
| `beta` | `[B,T,HV]` | fp32 | `common/gate.py:59` | ✅ | 4.0 MiB |
| `A` = $\Lambda\circ A_{\mathrm{raw}}$ | `[B,T,HV,64]` | bf16 | `chunk_fwd.py:382` | ✅ | **128 MiB** |
| `w` | `[B,T,HV,K]` | bf16 | `wy_fast.py:269` | ❌ 重算 | 256 MiB |
| `u` | `[B,T,HV,V]` | bf16 | `wy_fast.py:270` | ❌ 重算 | 256 MiB |
| `h`（全部 chunk 状态） | `[B,NT,HV,V,K]` | bf16 | `chunk_delta_h.py:707/710` | ❌ 重算 | **512 MiB** |
| `v_new` | `[B,T,HV,V]` | bf16 | `chunk_delta_h.py:713` | ❌ 重算 | 256 MiB |
| `final_state` | `[N,HV,V,K]` | **fp32** | `chunk_delta_h.py:708/711` | 返回 | 小 |
| `o` | `[B,T,HV,V]` | bf16 | `chunk_o.py:552` | — | 256 MiB |

**前向峰值活跃占用 ≈ 1.63 GiB / 层。** 跨 fwd→bwd 持久化 ≈ 128 MiB (`A`) + 8 MiB (`g`,`beta`) + $\hat q,\hat k,v$（512 MiB）；`h`/`w`/`u`/`v_new`（1.28 GiB）释放后重算。

**`h` 是最大的单一中间量**，大小 $= \frac{T}{C}H_vKV$。注意它与 $C$ **反比**：$C$ 减半，`h` 翻倍。这是 $C=64$ 成为实际上限的又一个理由——算法文档 §4 只谈了计算冗余随 $C$ 上升，这里是存储侧反方向的对称约束。

### 10.2 HBM 往返次数

| 张量 | 写 | 读 | 流量 |
|---|---|---|---|
| `A`（融合 $C=64$ 路径） | 1（`chunk_fwd.py:319-328`） | 1（`wy_fast.py:76`） | 256 MiB |
| `A`（$C\in\{16,32\}$ 旧路径） | 2（fp32 + bf16） | 2 | **≈640 MiB** |
| `w` | 1（`wy_fast.py:102`） | 1（`chunk_delta_h.py:203`） | 512 MiB |
| `u` | 1（`wy_fast.py:86`） | 1（`chunk_delta_h.py:230`） | 512 MiB |
| `v_new` | 1（`chunk_delta_h.py:234`） | 1（`chunk_o.py:136`） | 512 MiB |
| `h` | 1（`chunk_delta_h.py:176-200`） | 1（`chunk_o.py:109`） | 1024 MiB |
| `k` | — | **4 次**（kkt / w_u / fwd_h / fwd_o） | 512 MiB |

**前向总 HBM 流量 ≈ 3.9 GiB / 层**，其中 **≈2.8 GiB 是纯中间量搬运**。

### 10.3 这意味着什么

算法文档 §6.1 算出 GDN prefill 的理论算术强度约 213 FLOP/Byte（算力受限）。实际计算量是

$$
T \cdot H_v \cdot (4Cd + 3d^2) \cdot 2 \approx 32768 \times 32 \times (4\!\cdot\!64\!\cdot\!128 + 3\!\cdot\!128^2)\times 2 \approx 137\ \text{GFLOP}
$$

对 3.9 GiB 流量，实际算术强度 $\approx 137\times10^9 / 4.2\times10^9 \approx \mathbf{33}$ FLOP/Byte。

> **一个理论上算力受限的公式，被实现成了带宽受限的 kernel——算术强度掉了 6 倍多。**

一个完全融合的 mega-kernel 可以把 `A`、`w`、`u`、`v_new` 全部留在 SMEM/寄存器，消掉约 2.3 GiB（**约 60% 的前向流量**）。

**但 `h` 消不掉**，理由已在 §2 结尾说过：`fwd_h` 是 per-sequence 分解（chunk 轴串行），`fwd_o` 是 per-chunk 分解（完全并行），两者对状态的访问模式不兼容。要消掉 `h` 必须**同时**解决这个问题——办法只有两个：warp specialization（在一个 kernel 里同时跑两种分解，即 FlashInfer / FlashQLA 的路），或者分段扫描（让 `h` 只在段内存在）。

**所以 §8 的并行度问题和 §10 的访存问题实际上是同一个问题的两面**：都源于"状态递推必须独占一种 grid 分解"。这也解释了为什么另两个库的核心优化都落在环节 ④ 上——解决了它，融合问题自动跟着解决。

---

## 11. 精度策略

| 量 | dtype | 位置 |
|---|---|---|
| `g`（$\log\alpha$ cumsum，含 $\log_2 e$ 预乘） | **fp32** on HBM | `gate.py:171,179`; `cumsum.py:269,281` |
| `beta`（sigmoid 后） | **fp32** on HBM | `common/gate.py:59` |
| 门内部（softplus / exp） | fp32，包装函数强转 | `fla/ops/utils/op.py:28-37` |
| $KK^{\top}$ 累加器 | **fp32** 寄存器 | `chunk_fwd.py:124-135` |
| `A` on HBM（融合路径） | **bf16** | `chunk_fwd.py:382` |
| 三角求逆工作精度 | **fp32** 寄存器（dot 本身 tf32，见 §4.3） | `chunk_fwd.py:222-225` |
| `w`, `u` | bf16 | `wy_fast.py:269-270` |
| 递推状态 `b_h1..b_h4` | **fp32 寄存器，贯穿全部 512 步** | `chunk_delta_h.py:91-106` |
| `h`（per-chunk 快照 on HBM） | bf16 | `chunk_delta_h.py:707,710` |
| `final_state` / `dh0` | **fp32** | `chunk_delta_h.py:708,711,771` |
| `o` 累加器 | fp32 寄存器 → bf16 store | `chunk_o.py:88,140` |

完全符合算法文档 §8.5 的一般原则：**门与状态累加 fp32，Tensor Core 操作数 bf16。**

**tf32 相关**：
- `IS_TF32_SUPPORTED = IS_NVIDIA and capability[0] >= 8`（`fla/utils/_device.py:139`）；pre-Ampere 上全局强制 `TRITON_F32_DEFAULT='ieee'`（`:151-154`）
- 显式 `allow_tf32=False` 只在 `wy_fast.py:85`（$u$ 的 dot）；Ascend 后端**每个** dot 都关
- 其余 kernel 不给精度提示——输入是 bf16，mma 本来就是 bf16×bf16→fp32

**其他**：`FLA_USE_FAST_OPS=1` 换 `libdevice.fast_*` 近似（`op.py:16-26`，默认关）；`softplus` 是手写内联 PTX 带 `x>20` 快速路径（`fla/ops/utils/softplus.py:19-27,69-81`）。

**没有 fp8，没有状态 I/O 量化。** 这是与 FlashInfer 的一处明确差距——算法文档 §7.4 用途 2 那条论证（遗忘门让量化噪声自愈）FLA 没有利用。

---

## 12. varlen 机制

**核心：`cu_seqlens` + 预计算的 `chunk_indices` 查表 + 编译期 `IS_VARLEN` 特化。**

1. **host 侧，每次调用一次**（`chunk.py:289-291`）：
   ```python
   chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
   ```
   `prepare_chunk_indices`（`fla/ops/utils/index.py:155-164`）建一张 `[NT_total, 2]` 的 int 表 `(seq_id, intra_chunk_idx)`，$NT_{\mathrm{total}} = \sum_n \lceil T_n/C\rceil$。带 `@tensor_cache`，且可以从 CPU 副本 `cu_seqlens_cpu` 构建以**避免 D2H 同步**（`:161`）——`_segmented_arange`（`:113-134`）的注释专门说明了 `repeat_interleave` 在 CUDA counts 上的同步隐患。

2. **device 侧统一惯用法**（几乎每个 kernel 逐字相同）：
   ```python
   if IS_VARLEN:
       i_n, i_t = tl.load(chunk_indices + i_t*2), tl.load(chunk_indices + i_t*2 + 1)
       bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
       T = eos - bos
   else:
       bos, eos = i_b * T, i_b * T + T
   ```
   于是 `i_t` 这个 program id 索引的是**跨所有序列的扁平 chunk 列表**，grid 变成 `NT = len(chunk_indices)`（`chunk_fwd.py:381`、`wy_fast.py:267`、`chunk_o.py:548`）。**天然负载均衡**——每个 program 恰好一个 chunk。

3. **递推 kernel 是例外**：它是 per-sequence 而非 per-chunk 的 grid，所以用 `prepare_chunk_offsets`（`index.py:167-172`，per-sequence chunk 数的 exclusive cumsum），kernel 内 `boh = tl.load(chunk_offsets + i_n)` 找本序列的 `h` 切片起点（`chunk_delta_h.py:85`）。**这里没有负载均衡**——长序列的 program 跑 512 轮，短序列的跑几轮就退出，长尾直接暴露。

4. **约束**：给了 `cu_seqlens` 就必须 $B=1$（`chunk.py:546-551`）；`initial_state.shape[0] == len(cu_seqlens)-1`（`:552-556`）。

5. **尾块**：掩码 `m_t = o_t < T` 遍布各处，加上融合 kkt kernel 的早退 `if i_t*BT >= T: return`（`chunk_fwd.py:80-81`）和 `last_idx = min((i_t+1)*BT, T) - 1`（`chunk_delta_h.py:236`）。

6. **层级**：`gated_deltanet.py:243-244` 用 `unpad_hidden_states` 把 padded `[B,T,D]` + 2-D mask 转成 packed `[1,ΣT,D]` + `cu_seqlens`，末尾 `repad_hidden_states`（`:362`）。

**与另两个库对照**：FLA 靠索引表在软件层解决越界，天然安全；FlashInfer 走 TMA，必须在运行时改 descriptor 的 `global_dim` 才能防止跨序列越界（见其文档）。**同一个正确性需求，一个用索引间接，一个给硬件描述符打补丁。**

---

## 13. 反向传播：以 A 为唯一检查点

`chunk_gated_delta_rule_bwd`，`chunk.py:126-250`。推理不用，但它解释了前向为何长成这样，所以简述。

| # | 步骤 | Kernel | 备注 |
|---|---|---|---|
| 1 | 从 `A` 重算 `w,u` | `recompute_w_u_fwd_kernel` | **重算** |
| 2 | 重算 `h, v_new` | 与 fwd 同一个 kernel | **重算**，`output_final_state=False` |
| 3 | `dv`（intra 部分） | `chunk_bwd_kernel_dv_local` | 调用时**不传 `A=`** ⟹ `USE_A=False` ⟹ **重算 $qk^{\top}$ 与门矩阵**（`chunk_o.py:505-516`） |
| 4 | `dh, dh0, dv`（inter） | `bwd_kernel_dhu_blockdim64` | 反向串行扫描，物化 `dh [B,NT,HV,K,V]`（又 512 MiB） |
| 5 | `dq, dk, dw, dg` | `chunk_bwd_kernel_dqkwg` | `dg` 是 `[NK,B,T,HV]` fp32 partial，host 侧 `.sum(0)` |
| 6 | `dk2, dv, db, dg2` | `prepare_wy_repr_bwd_kernel` | **再重算一次 $KK^{\top}$**（`wy_fast.py:233`） |
| 7 | `dg` 反向 cumsum | `chunk_local_cumsum_scalar_kernel(reverse=True)` | 单独一个 kernel，注释：*"strange triton compiler issue"* |
| 8 | 门反向 | `gdn_gate_bwd_kernel` | `dA` host 侧 `.sum(0)` |

**检查点策略**：只存 `A`；`w,u,h,v_new` 全部重算，$qk^{\top}$ 重算两次，$KK^{\top}$ 再算一次。

$KK^{\top}$ 之所以要重算，是因为**存的 `A` 是求逆之后的，逆前的矩阵没留**。想省这次重算就得多存一份 $C\times C$——又是一次显存 / 算力交换。

**注意跨程序归约全部走 HBM partial + host `.sum()`，没有 atomics**（`chunk_o.py:728,765`；`gate.py:209,224`）。全库 `atomic` 出现 0 次。

---

## 14. FLA 刻意不做的事

以下全部经源码验证为**缺席**。这些不是 bug，是 Triton 定位的必然结果。

1. **阶段之间没有异步流水。** 4 个核心 kernel 之间是**设备级 barrier + 完整 HBM 往返**。CuTe kernel 可以把 chunk $i+1$ 的 $KK^{\top}$ 与 chunk $i$ 的三角求解重叠，FLA 结构上不可能。
2. **GDN 路径上零 TMA。** 唯一的 TMA 代码在 `solve_tril`（`solve_tril.py:74-75,142-143,236-237`），且 `IS_TMA_SUPPORTED` 需要显式 `FLA_USE_TMA=1`（`fla/utils/_device.py:141-149`，**默认 `'0'`**）；而 $C=64$ 的 GDN 快速路径**根本绕过 `solve_tril`**。所以实践中 GDN prefill 一次 TMA 都不用。
3. **零 warp specialization。** 没有 `tl.async_task`，没有 warp 角色划分。每个 warp 执行同一条指令流，没有专用 DMA warp、没有独立的 MMA 发射流。
4. **零 cluster / DSMEM**（`clusterlaunchcontrol`、`mapa`）。没有任何 SM90+ 专属机制。
5. **无 persistent kernel / 无 megakernel。** 每个阶段都是新 grid。（只有 **Ascend** 后端用了 persistent 模式：`kernel[(num_core,)]` + `for task_id in tl.range(core_id, task_num, num_core)`，`backends/triton_ascend/wy_fast.py:118-120,155`。）
6. **中间量全物化**（§10 已量化）。
7. **默认递推路径无任何序列并行**（§8）。
8. **无 fp8、无 MXFP、无状态 I/O 量化。**
9. **无手写 SMEM 布局 / swizzle。** bank conflict 规避、`ldmatrix` 布局、寄存器分配全交给 Triton 编译器——代价就是 §6 那一串编译器 workaround。
10. **跨程序归约走 HBM + host `.sum()`**，不用 atomics 或 cooperative group。
11. **反向重复重算 $KK^{\top}$ 与 $qk^{\top}$**（§13）。

---

## 15. backend 分派：FLA 自己承认会被替换

`fla/ops/gated_delta_rule/backends/flash_qla.py`，注册于 `backends/__init__.py:17`。

**它是整算子替换，不是某个阶段的替换。** `FlashQLABackend.chunk_gated_delta_rule`（`flash_qla.py:98-132`）直接 `import flash_qla` 并转发，带 `auto_cp=True`。类 docstring（`flash_qla.py:33-43`）措辞很直白：

> *"Fused TileLang forward and backward with intra-card CP (**replaces the multi-kernel Triton path**)."*

**FLA 自己把它的 Triton 实现称作 "the multi-kernel Triton path"，并说 FlashQLA 是来替换它的。**

**选中条件**（`chunk_gated_delta_rule_verifier`，`flash_qla.py:51-96`），全部必须成立：

- `IS_NVIDIA_HOPPER or IS_NVIDIA_SM100 or IS_NVIDIA_SM120`（`:70-71`）
- dtype ∈ {fp16, bf16} 且三者一致（`:74-77`）
- **$K = 128$ 且 $V = 128$ 严格相等**（`:82-85`）
- 不使用 `use_gate_in_kernel`（`:86-87`）、`use_beta_sigmoid_in_kernel`（`:88-89`）、`allow_neg_eigval`（`:90-91`）、`transpose_state_layout`（`:92-93`）、跨卡 `cp_context`（`:94-95`）

**优先级**：`priority = 3`，`default_enable = True`，`env_var = "FLA_FLASH_QLA"`（`:45-49`）。数字越小优先级越高；GDN 注册表里是 `TritonAscendGDNBackend`(0) → `FlashQLABackend`(3)，基类 5，原生 Triton 实现是兜底。

**但有一处关键的自相矛盾值得记住**：verifier 拒绝 `use_gate_in_kernel` 和 `use_beta_sigmoid_in_kernel`，而 `fla/layers/gated_deltanet.py:320-322` **把两者都设成了 `True`**。所以**一个标准的 `GatedDeltaNet` layer 永远会被 FlashQLA 拒绝、落回 Triton**。FlashQLA 只有在调用方（vLLM 或自定义 Qwen3-Next modeling 文件）直接调 `chunk_gated_delta_rule` 并传入已激活的 `g`、`beta` 时才真正生效。

**这套分派机制说明了 FLA 对自己的定位**：

- 三条 backend 都能抢占 GDN 相关算子：FlashQLA（整算子）、TileLang（**只抢 `chunk_bwd_dqkwg`，Hopper + Triton≥3.4.0 时默认启用**，为的是绕开已知误编译，`fla/ops/common/backends/tilelang/__init__.py:36-44`）、TritonAscend（NPU）。
- 分派层（`fla/ops/backends/__init__.py:160-220`）是通用的、per-function 的、verifier 驱动的，还会打日志说明哪个 backend 胜出——**架构上就是在承认高端硬件上参考实现会被替换。**
- `chunk_o.py:704-709` 甚至直接抛错让用户去装 TileLang。
- 而 Triton 路径是**唯一**支持任意 $K/V$、$C\in\{16,32\}$、融合门激活、`allow_neg_eigval`、跨卡 CP、`state_v_first`、以及非 NVIDIA 硬件的实现。

---

## 16. 汇总

### 16.1 优化清单

| 优化 | 算法依据 | 换来什么 | 付出什么 |
|---|---|---|---|
| kkt + solve_tril 寄存器内融合 | 两阶段之间的 $C\times C$ 中间量只用一次 | `A` 的 HBM 往返 4→2 次 | 只在 $C=64$ 可用；寄存器吃满，无法再开流水级数 |
| 两级三角求逆（$16\times16$ 前代 + Schur 合并） | 杠杆二（工作量换深度） | 依赖链 $64 \to 16 + 3$ 层；合并上 Tensor Core | 总 FLOP 增加 |
| GDN 路径合并步用 tf32 | 待求逆矩阵元素模长 $\le1$ | 合并更快 | 尾数 24→10 bit；与通用 `solve_tril` 精度不一致 |
| 全局 `exp2` + $\log_2 e$ 预乘 | $\exp(x)=\exp_2(x\log_2 e)$ | 每次指数省一次乘法，直落 SFU | HBM 里的 `g` 语义非标准，跨库对齐要换算 |
| Λ 永不物化 | Λ 由两个标量之差生成，重算极便宜 | 省 $C\times C$ 张量 | 每个消费点重算一次（FlashInfer 能复用寄存器，FLA 不能） |
| 编译期特化（13 个 flag）+ `do_not_specialize=['T']` | — | 一份代码覆盖全部配置；变长不重编译 | 代码复制约 8 倍；一串编译器 workaround |
| 输出 kernel 单趟双累加 | $q$、$k$ 同时服务 inter 与 intra | `q`,`k` 各读一次 | — |
| `A` 作为唯一 bwd 检查点 | $w,u,h,v_{\mathrm{new}}$ 皆可由 `A` 重算 | 持久化显存 1.63 GiB → 128 MiB | bwd 重算 $KK^{\top}$、$qk^{\top}$ |
| varlen 扁平 chunk 索引表 | — | per-chunk kernel 天然负载均衡 | 递推 kernel 仍是 per-sequence，长尾暴露 |
| intracard CP（分段扫描） | 杠杆三（仿射结合律） | 序列方向并行，复用现有 kernel | 仅推理 + 仅 varlen + 需开关 + ≥24k token |

### 16.2 一句话

> FLA 的 GDN prefill 是一条 **8 kernel、中间量全物化、以编译期特化为主要抽象手段的 Triton 流水线**。
> 它唯一的重量级优化是 $C=64$ 下 kkt + solve_tril 的寄存器内融合。
> 它的性能天花板由两件事锁定：**中间量 HBM 流量**（把 213 FLOP/Byte 的公式跑成 33 FLOP/Byte）和**递推 kernel 只有 $N_V\!\times\!B\!\times\!H_v$ 个 program 的串行扫描**（TP8 下 16 个 program 对上百个 SM）。
> 而这两件事**是同一个根因的两面**——状态递推独占一种 grid 分解。
> 分段扫描的代码 FLA 其实有（`intracard_cp.py`，用的正是仿射结合律），但被限制在"仅推理 + 仅 varlen + 需显式开启 + ≥24k token"的窄门里。
> **FLA 的价值不在性能，在覆盖面**：它是唯一支持训练、任意 head dim、$C\in\{16,32\}$、融合门激活和非 NVIDIA 硬件的实现，也是另两个库的正确性基准。它的 backend 分派层本身就写明了这个定位。

---

## 相关文档

- [`GDN_Algorithm.md`](GDN_Algorithm.md)：算法推导、依赖结构、四个数学杠杆
- [`FlashInfer_GDN_Blackwell.md`](FlashInfer_GDN_Blackwell.md)：用杠杆三精确消除串行链
- [`FlashQLA_GDN_Blackwell.md`](FlashQLA_GDN_Blackwell.md)：用杠杆四近似消除串行链
