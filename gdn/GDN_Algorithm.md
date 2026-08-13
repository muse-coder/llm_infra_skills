# GDN（Gated DeltaNet）算法

本文先讲数学，再给实现文档提供统一符号。默认状态布局为 $S\in\mathbb{R}^{d_k\times d_v}$；有些 kernel 在显存中保存它的转置，但不改变算法。

## 1. 从线性注意力到 Gated Delta Rule

### 1.1 线性注意力是一块固定大小的关联记忆

最简单的线性注意力维护

$$
S_t=S_{t-1}+k_tv_t^\top,
\qquad
o_t=S_t^\top q_t.
$$

$k_t$ 是“地址”，$v_t$ 是写入该地址的“内容”。读取时

$$
S_t^\top q_t=\sum_{i\le t}v_i(k_i^\top q_t).
$$

状态大小与序列长度无关，但相近的 key 会不断把内容累加到同一方向，无法覆盖旧值。

### 1.2 Delta rule 先读旧值，再写残差

令 $\beta_t$ 为写入强度，delta rule 为

$$
r_t=\beta_t\left(v_t-S_{t-1}^\top k_t\right),
\qquad
S_t=S_{t-1}+k_tr_t^\top.
$$

等价地，

$$
S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
$$

当 $\lVert k_t\rVert_2=1$ 且 $\beta_t=1$ 时，旧状态在 $k_t$ 方向的内容被完全擦除，再写入 $v_t$；$\beta_t<1$ 时是软更新。

### 1.3 GDN 在擦写前加入遗忘门

GDN 每步先衰减旧状态：

$$
\bar S_t=\alpha_tS_{t-1},
$$

$$
r_t=\beta_t\left(v_t-\bar S_t^\top k_t\right),
$$

$$
\boxed{S_t=\bar S_t+k_tr_t^\top},
\qquad
o_t=s\,S_t^\top q_t.
$$

合并后得到常见写法：

$$
\boxed{
S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}
    +\beta_tk_tv_t^\top
}.
$$

这里 $s$ 通常是 $1/\sqrt{d_k}$。实现常传入 $\ell_t=\log\alpha_t\le 0$，再用 `exp`/`exp2` 得到 $\alpha_t$。

Qwen/FLA 的常见门参数化是

$$
\ell_t=-\exp(A_{\log})\operatorname{softplus}(a_t+\mathrm{dt\_bias}),
\qquad
\beta_t=\sigma(b_t),
$$

并对 $q_t,k_t$ 做 L2 norm。注意：这是模型层参数化，不是 GDN 算子的唯一合法接口；三个库对 `g` 参数的语义并不相同。

## 2. GVA：为什么 q/k head 和状态 head 数量不同

Qwen 使用 Grouped Value Attention（GVA）时，$q,k$ 的 head 数为 $H_{qk}$，$v$ 和状态的 head 数为 $H_v$，且 $H_v$ 是 $H_{qk}$ 的整数倍。同一组中的多个 value/state head 复用一个 q/k head：

$$
h_{qk}=\left\lfloor h_v/(H_v/H_{qk})\right\rfloor.
$$

真正决定 recurrent state 数量和默认 kernel 并行度的是 $H_v$，不是 $H_{qk}$。

## 3. 为什么 prefill 要 chunk 化

逐 token 递推每步只有矩阵-向量乘和 rank-1 update，无法充分使用 Tensor Core。把连续 $C$ 个 token 合成一个 chunk 后，可以把大部分工作改写为矩阵乘。三个实现最常用 $C=64$。

对一个 chunk，令

$$
Q,K\in\mathbb{R}^{C\times d_k},
\qquad
V\in\mathbb{R}^{C\times d_v},
$$

第 $i$ 行分别是 $q_i^\top,k_i^\top,v_i^\top$。定义 chunk 内累计 log decay

$$
\gamma_i=\sum_{j=1}^{i}\log\alpha_j,
$$

以及因果 pairwise decay

$$
G_{ij}=
\begin{cases}
\exp(\gamma_i-\gamma_j), & i\ge j,\\
0, & i<j.
\end{cases}
$$

### 3.1 单位下三角求解

先忽略 gate，构造

$$
A_0=\left[I+\operatorname{strictLower}
\left(\operatorname{diag}(\beta)KK^\top\right)\right]^{-1}.
$$

被求逆矩阵是单位下三角矩阵，因此逆一定存在。它编码了 chunk 内“前一个 token 的写入会改变后一个 token 读到的状态”这一因果关系。

gate 可以通过对角相似变换加回去：

$$
\boxed{A=G\odot A_0}.
$$

原因是令 $D=\operatorname{diag}(e^\gamma)$ 后，gated 的严格下三角部分等于 $DLD^{-1}$，所以

$$
[I+DLD^{-1}]^{-1}=D(I+L)^{-1}D^{-1}.
$$

这解释了实现上的两种等价选择：

- FLA 和 FlashInfer 非 CP 路径可以把 pairwise gate 放进 KKT/solve。
- FlashQLA 与 FlashInfer CP 路径可以先求 gate-free 的 $A_0$，消费时再乘 $G$。

### 3.2 修正 value、更新状态、生成输出

令 $S_0$ 为 chunk 入口状态。定义

$$
V_{\mathrm{new}}
=A\operatorname{diag}(\beta)
\left(V-\operatorname{diag}(e^\gamma)KS_0\right).
$$

然后

$$
\boxed{
S_C=e^{\gamma_C}S_0
 +\left[\operatorname{diag}(e^{\gamma_C-\gamma})K\right]^\top
 V_{\mathrm{new}}
},
$$

$$
\boxed{
O=s\left[
\operatorname{diag}(e^\gamma)QS_0
 +\left(G\odot\operatorname{Lower}(QK^\top)\right)V_{\mathrm{new}}
\right]
}.
$$

`Lower` 包含对角线，保证 token 能读到自己刚完成的更新。

FLA 常把 $V_{\mathrm{new}}$ 拆成 WY 中间量：

$$
U=A\operatorname{diag}(\beta)V,
\qquad
W=A\operatorname{diag}(\beta)\operatorname{diag}(e^\gamma)K,
$$

$$
V_{\mathrm{new}}=U-WS_0.
$$

两种形式数学等价。显式 $W/U$ 便于反向重算和复用；推理专用融合 kernel 更倾向在片上直接形成 $V_{\mathrm{new}}$，避免把它们都写回 HBM。

## 4. 依赖图与真正的瓶颈

一个 chunk 的前向可概括为：

```text
log gate cumsum ─┐
K K^T ───────────┴─> triangular solve ─┐
K S0 / V ──────────────────────────────┴─> V_new
Q K^T / Q S0 / V_new ────────────────────> O
K / V_new / S0 ───────────────────────────> S_C
```

chunk 内的 KKT、solve、value 修正和输出都能并行；但 chunk 边界满足

$$
S_{c+1}=F_c(S_c),
$$

普通实现必须先得到 $S_c$ 才能算下一块。因此一个 `(sequence, state-head, value-tile)` CTA 往往要串行遍历所有 chunk。

当 $B\times H_v$ 很小时，任务数远少于 GPU SM 数量。此时把单个 kernel 再融合只能改善常数，不能解决 SM 空闲。长序列、高 tensor parallel、单请求 prefill 正是最容易触发这一问题的组合。

## 5. 两条跨 chunk 并行化路线

### 5.1 精确路线：把一段视为仿射变换

任意一段 token 对入口状态的作用都能写成

$$
S_{\mathrm{out}}=M S_{\mathrm{in}}+N.
$$

相邻两段的复合仍是仿射变换：

$$
(M_2,N_2)\circ(M_1,N_1)
=\left(M_2M_1,\;M_2N_1+N_2\right).
$$

因此可以先并行计算每段的 $(M,N)$，再 scan/fixup 出每段真实入口状态，最后并行重放各段。代数上完全精确，但要额外保存 $d_k\times d_k$ 的转移矩阵、执行额外 GEMM，并产生多次 kernel launch。FlashInfer 的 CP 主路径采用这条路线。

### 5.2 近似路线：利用 gate 对旧状态的遗忘

设两条递推只在入口状态上有误差 $\Delta S$。一步后

$$
\Delta S_t
=\alpha_t(I-\beta_tk_tk_t^\top)\Delta S_{t-1}.
$$

若 $\lVert k_t\rVert_2=1$ 且 $0\le\beta_t\le2$，$I-\beta_tk_tk_t^\top$ 的谱范数为 1，于是

$$
\lVert\Delta S_t\rVert
\le\left(\prod_i\alpha_i\right)\lVert\Delta S_0\rVert
=\exp\left(\sum_i\log\alpha_i\right)\lVert\Delta S_0\rVert.
$$

如果某段开始前回看若干 chunk，直到累计 log decay 小于阈值，未回看的更早状态对该段入口的影响就有明确的绝对误差界。FlashQLA 使用阈值 `-10`，对应衰减因子 $e^{-10}\approx4.54\times10^{-5}$；若某个 head 衰减太慢，则用转移矩阵做精确修正。

这个界是绝对误差界，不自动等于输出的逐元素相对误差界；实际正确性仍必须通过端到端数值测试确认。

## 6. 三个实现的优化空间

从上述公式出发，GDN prefill 只有四类主要优化空间：

| 公式块 | 原始代价 | 可用优化 |
| --- | --- | --- |
| $G$ 与 $A$ | pairwise exp + $C^2d_k$ KKT + 三角求逆 | gate-free 相似变换、分块求逆、KKT/solve 融合 |
| $V_{new}$ | 依赖 $S_{prev}$，包含 $KS$ 和 inverse apply | WY 表示、合并中间量、片上复用 |
| $S_{next}$ | chunk 间严格递推 | 状态片上驻留；仿射精确扫描；利用遗忘做有限 warmup |
| $O$ | $QS$、$QK^\top$、$V_{new}$ 的组合 | 与 state/value 阶段融合，或给定 boundary state 后按 chunk 并行 |

三个库的根本区别不是用了哪种 kernel DSL，而是如何选择这些数学边界：

- FlashInfer：面向前向推理，把四个公式块尽量放进同一片上流水；并行度不足时使用仿射变换的精确分段扫描。
- FLA：保留显式 WY 表示，把 intra-chunk、state recurrence 和 output 分开，方便训练反向、shape 覆盖与后端复用。
- FlashQLA：先做 gate-free 三角逆，再用融合 kernel 计算其余公式；长序列低并行度时利用 gate 遗忘决定 warmup 长度，慢衰减 head 用转移矩阵精确修正。

## 7. 跨库最容易混淆的约定

| 项目 | FlashInfer | FLA | FlashQLA |
| --- | --- | --- | --- |
| `g` 的公开语义 | $\alpha$，线性空间，默认全 1 | 默认 $\log\alpha$；可选传 raw gate 并在内部激活 | $\log\alpha$ |
| q/k norm | 可选 kernel 内执行 | 可选内部执行 | 可选内部执行 |
| beta | 期望已经是线性空间 | 可选内部 sigmoid；还能启用 `2*sigmoid` | 期望已经是线性空间 |
| 默认状态布局 | API 说明为 `[N,H,V,K]` | `[N,H,K,V]`，可选 V-first | `[N,H,K,V]`，可选 V-first |
| 主要 head 关系 | GQA 与 GVA | GVA | GVA |

对照数值前必须先统一 gate 空间、q/k norm、scale、head 映射、状态转置和尾 chunk padding。多数“kernel 算错”的初步现象来自这些接口语义不一致。

## 源码入口

- FLA token/chunk 参考：`fla/ops/gated_delta_rule/naive.py`
- FlashQLA PyTorch 参考：`tests/ref_gdr.py`

继续阅读：[`FlashInfer_GDN_Blackwell.md`](FlashInfer_GDN_Blackwell.md)。
