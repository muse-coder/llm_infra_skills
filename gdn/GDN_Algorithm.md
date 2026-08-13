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

### 1.4 从逐 token 递推到 chunk 公式

chunk 公式没有改变 1.3 的算法，只是把一个 state head、一个 chunk 内按顺序计算的 $C$ 个 residual $r_i$ 改写成一次下三角求解。下面逐步建立对应关系。

对一个 chunk，把入口状态记为 $S_0$，chunk 内 token 编号为 $i=1,\ldots,C$。令

$$
\gamma_i=\sum_{m=1}^{i}\log\alpha_m,
\qquad
G_{ij}=\begin{cases}
e^{\gamma_i-\gamma_j},&i\ge j,\\
0,&i<j.
\end{cases}
$$

$e^{\gamma_i}$ 表示入口状态 $S_0$ 到 token $i$ 一共经历的衰减；$G_{ij}$ 表示 token $j$ 写入的内容传播到 token $i$ 时经历的衰减。

先把 1.3 的状态更新展开。token $i$ 写入之前的状态是

$$
\boxed{
\bar S_i=e^{\gamma_i}S_0+
\sum_{j<i}G_{ij}k_jr_j^\top
}.
$$

第一项是衰减后的 chunk 入口状态；第二项是 chunk 内所有更早 token 的写入。将它代回逐 token residual

$$
r_i=\beta_i(v_i-\bar S_i^\top k_i)
$$

得到

$$
r_i^\top=
\beta_i\left[
v_i^\top-e^{\gamma_i}k_i^\top S_0
-\sum_{j<i}G_{ij}(k_i^\top k_j)r_j^\top
\right].
$$

这一步是从 token 公式到 chunk 公式的关键：当前 residual $r_i$ 依赖所有 $j<i$ 的 residual，但不依赖未来 token，所以所有依赖组成一个单位下三角系统。

例如 chunk 只有两个 token 时，先算出的 $r_1$ 会改变 token 2 写入前读到的状态：

$$
\begin{aligned}
r_1^\top&=\beta_1\left(v_1^\top-e^{\gamma_1}k_1^\top S_0\right),\\
r_2^\top&=\beta_2\left(v_2^\top-e^{\gamma_2}k_2^\top S_0-G_{21}(k_2^\top k_1)r_1^\top\right).
\end{aligned}
$$

把包含 $r_1$ 的项移到左边，就得到一个 $2\times2$ 单位下三角方程；$C$ 个 token 只是把这个结构扩展为 $C\times C$。

把 $q_i^\top,k_i^\top,v_i^\top,r_i^\top$ 分别堆成 $Q,K,V,R$ 的第 $i$ 行，其中 $Q,K\in\mathbb{R}^{C\times d_k}$、$V,R\in\mathbb{R}^{C\times d_v}$，则上式可以一次写成

$$
\left[
I+\operatorname{strictLower}
\left(G\odot\operatorname{diag}(\beta)KK^\top\right)
\right]R
=\operatorname{diag}(\beta)
\left(V-\operatorname{diag}(e^\gamma)KS_0\right).
$$

因此定义

$$
\boxed{
A=\left[
I+\operatorname{strictLower}
\left(G\odot\operatorname{diag}(\beta)KK^\top\right)
\right]^{-1}
},
$$

就有

$$
\boxed{
R=A\operatorname{diag}(\beta)
\left(V-\operatorname{diag}(e^\gamma)KS_0\right)
}.
$$

后面的 kernel 文档把 $R$ 记为 $V_{new}$。它不是原始 $V$ 的另一种投影，而是把逐 token 的写入 residual $r_1,\ldots,r_C$ 堆起来：

$$
\boxed{V_{new}=R}.
$$

求得所有 residual 后，只需把逐 token 的 state update 和 output read 展开。chunk 结束状态为

$$
\boxed{
S_C=e^{\gamma_C}S_0+
\left[\operatorname{diag}(e^{\gamma_C-\gamma})K\right]^\top V_{new}
},
$$

其中第 $j$ 行的系数 $e^{\gamma_C-\gamma_j}$ 表示第 $j$ 个写入传播到 chunk 末尾的衰减。所有 token 的输出为

$$
\boxed{
O=s\left[
\operatorname{diag}(e^\gamma)QS_0+
\left(G\odot\operatorname{Lower}(QK^\top)\right)V_{new}
\right]
}.
$$

第一项读取衰减后的入口状态；第二项读取 chunk 内截至当前 token 已发生的写入。`Lower` 包含对角线，对应 1.3 中先更新 $S_i$、再计算 $o_i=sS_i^\top q_i$。

gate 也可以从三角求逆中拆出来。令

$$
A_0=\left[I+\operatorname{strictLower}
\left(\operatorname{diag}(\beta)KK^\top\right)\right]^{-1},
\qquad
D=\operatorname{diag}(e^\gamma),
$$

则 gated 系统是 $I+DLD^{-1}$，所以

$$
A=D A_0D^{-1}=G\odot A_0.
$$

这就是 FlashQLA 先求 gate-free $A_0$、使用时再乘 $G$ 的依据；FLA 和 FlashInfer non-CP 也可以直接构造 gated $A$。两种写法都精确对应同一组逐 token residual。

| 1.3 的逐 token 量 | chunk 量 | 含义 |
| --- | --- | --- |
| $\alpha_i$ | $\gamma_i$、$G_{ij}$ | 入口状态和历史写入的累计衰减 |
| $k_i^\top\bar S_i$ | $e^{\gamma_i}(KS_0)_i+\sum_{j<i}G_{ij}(KK^\top)_{ij}r_j^\top$ | token $i$ 写入前读到的旧内容（按行表示） |
| $r_i$ | $V_{new}$ 的第 $i$ 行 | 实际写入 state 的 residual value |
| 顺序计算 $r_1,\ldots,r_C$ | 下三角求解 $A$ | 一次求出 chunk 内所有因果 residual |
| $S_i=\bar S_i+k_ir_i^\top$ | $S_C$ 公式 | 汇总入口状态和全部写入得到 chunk 末状态 |
| $o_i=sS_i^\top q_i$ | $O$ 公式 | 入口状态读取与 chunk 内因果读取之和 |

## 2. GVA：为什么 q/k head 和状态 head 数量不同

Qwen 使用 Grouped Value Attention（GVA）时，$q,k$ 的 head 数为 $H_{qk}$，$v$ 和状态的 head 数为 $H_v$，且 $H_v$ 是 $H_{qk}$ 的整数倍。同一组中的多个 value/state head 复用一个 q/k head：

$$
h_{qk}=\left\lfloor h_v/(H_v/H_{qk})\right\rfloor.
$$

真正决定 recurrent state 数量和默认 kernel 并行度的是 $H_v$，不是 $H_{qk}$。

## 3. GDN 层的完整数据流

前面的递推公式只描述 GDN recurrent core。一个完整的 GDN 层还包含输入投影、局部卷积、门控参数化和输出投影。以常见的 GatedDeltaNet 层为例，输入 hidden states 为 $X\in\mathbb{R}^{B\times T\times d_{model}}$，完整前向数据流是：

```text
X
├─ Wq -> short causal conv + SiLU -> reshape heads -> L2 norm -> Q
├─ Wk -> short causal conv + SiLU -> reshape heads -> L2 norm -> K
├─ Wv -> short causal conv + SiLU -> reshape value heads         -> V
├─ Wa -> a -> -exp(A_log) * softplus(a + dt_bias)                -> log(alpha)
├─ Wb -> b -> sigmoid(b) [or 2 * sigmoid(b)]                     -> beta
└─ Wz -> z                                                       -> output gate

(Q, K, V, log(alpha), beta, S_initial, sequence metadata)
                         -> GDN recurrent core
                         -> O_core, S_final

O_core -> per-head RMSNorm -> multiply by SiLU(z) -> merge heads -> Wo -> Y
S_final ---------------------------------------------------------------> state cache
```

这里的 short causal convolution 和输出门可以按模型配置关闭；residual connection 通常位于包住该 mixing 层的 block 中，不属于 GDN recurrent core。FlashInfer、FLA 和 FlashQLA 的 GDN kernel 都从图中的 $Q,K,V,\alpha/\log\alpha,\beta,S_{initial}$ 开始，并不负责 $W_q/W_k/W_v$、输出 RMSNorm 或 $W_o$。

### 3.1 从 hidden states 生成 core 输入

1. $X$ 经过不同的线性投影得到 q/k/v 分支、遗忘门原始值 $a$、写入门原始值 $b$ 和可选输出门 $z$。
2. q/k/v 分支通常先经过短因果卷积和 SiLU，以注入局部时序信息；随后 reshape 成 head 形式。q/k 使用 $H_{qk}$ 个 head，v 使用 $H_v$ 个 head。
3. 对每个 q/k 向量做 L2 norm，使 delta update 的擦写方向具有稳定尺度。
4. $a$ 与每个 state head 的 $A_{log}$、`dt_bias` 生成 $\log\alpha\le0$；$b$ 经 sigmoid 生成 $\beta$。这两个标量门控制“旧状态保留多少”和“当前 token 写入多少”。
5. 从 cache 读取每条 sequence、每个 value/state head 的入口状态 $S_{initial}\in\mathbb{R}^{d_k\times d_v}$；没有历史状态时使用零矩阵。

### 3.2 Recurrent core 内每个 token 的读写顺序

对一个 state head，token $t$ 的数据依赖必须按下面的顺序理解：

```text
S_(t-1) -- alpha_t --> S_bar_t --------------------------┐
K_t -----------------> v_hat_t = S_bar_t^T K_t           |
V_t - v_hat_t -- beta_t --> residual r_t                 |
S_bar_t + K_t r_t^T -------------------------> S_t -------┤
Q_t -----------------------------------------> O_t = s S_t^T Q_t
```

对应步骤为：

1. **遗忘：** $\bar S_t=\alpha_tS_{t-1}$。
2. **读旧值：** $\hat v_t=\bar S_t^\top k_t$，得到当前 key 在衰减后状态中已经关联的内容。
3. **计算写入残差：** $r_t=\beta_t(v_t-\hat v_t)$；如果旧内容已经等于 $v_t$，就不重复写入。
4. **更新状态：** $S_t=\bar S_t+k_tr_t^\top$。
5. **读取输出：** $o_t=sS_t^\top q_t$。输出读取的是更新后的 $S_t$，所以当前 token 可以看到自己刚写入的 value。

同一个 q/k head 可以服务多个 value/state head；每个 value head 分别读取自己的 $v_t,\alpha_t,\beta_t,S_{t-1}$ 并产生自己的 $o_t,S_t$。因此 recurrent core 的主状态链数量是 $B\times H_v$。

### 3.3 Core 输出如何回到模型维度

recurrent core 输出 $O_{core}\in\mathbb{R}^{B\times T\times H_v\times d_v}$ 和可选的 $S_{final}$。常见层实现先对每个 value head 的 $O_{core}$ 做 RMSNorm，再乘以 $\operatorname{SiLU}(z)$，合并所有 value heads，最后通过 $W_o$ 投影回 $d_{model}$ 得到层输出 $Y$。$S_{final}$ 不参与当前层的输出投影，而是写入 cache，作为该层下一段输入的 $S_{initial}$。

### 3.4 用给定 Qwen MoE 配置理解 shape 与瓶颈

这个模型的 92 层中有 69 个 `linear_attention`（GDN）层和 23 个 `full_attention` 层。`num_attention_heads=64`、`num_key_value_heads=4`、`head_dim=256` 只描述 full-attention 层；GDN 由 `linear_*` 字段决定：

$$
H_{qk}=16,\quad d_k=128,\quad H_v=128,\quad d_v=128.
$$

因此 q/k 总宽度为 $16\times128=2048$，value 和 output-gate 总宽度均为 $128\times128=16384$，一个 q/k head 服务 8 个 value/state head。下表中 $B$ 是 batch size，$T$ 是本次 token 数，$N$ 是实际 sequence 数。

| 计算部分 | 相关 config | shape |
| --- | --- | --- |
| 层输入 | `hidden_size=8192` | $X:[B,T,8192]$ |
| q/k 投影与卷积 | 16 个 key head、head dim 128、卷积宽度 4 | $Q,K:[B,T,16,128]$ |
| value 与输出门 | 128 个 value head、head dim 128 | $V,Z:[B,T,128,128]$ |
| 遗忘门与写入门 | 每个 value/state head 一个标量 | $\log\alpha,\beta:[B,T,128]$ |
| GDN state | $H_v\times d_k\times d_v$，fp32 | $S:[N,128,128,128]$ |
| core 输出 | 与 value head 相同 | $O_{core}:[B,T,128,128]$ |
| 合并与输出投影 | 合并 value heads 后投影到 `hidden_size` | $[B,T,16384]\to Y:[B,T,8192]$ |

Qwen 实现通常把投影合并成 `in_proj_qkvz:[8192\to36864]` 和 `in_proj_ba:[8192\to256]`；q/k/v 经过宽度为 4 的 depthwise causal convolution，$Z$ 用于 core 输出后的 swish gate。支持原生 GVA 的 kernel 保留 16 个 q/k head；若集成层先把 q/k 重复成 128 个 head，会增加 activation 流量，但不改变算法。

一份 fp32 recurrent state 的大小为

$$
128\times128\times128\times4\ \text{bytes}=8\ \text{MiB}.
$$

即每条 sequence、每个 GDN 层 8 MiB，69 个 GDN 层合计 552 MiB；按 head 做 TP 时，每卡大致再除以 TP。state 大小不随序列长度增长，但计算链会随序列增长：`max_position_embeddings=262144` 在 $C=64$ 时对应 4096 个连续 chunk。

这个配置下最重要的瓶颈有三个：

1. **value/state 侧很宽：** $H_v/H_{qk}=8$，因此 $KS$、$QS$、state update 和输出相关计算比 q/k 侧更重。
2. **chunk 间有串行状态依赖：** $S_{c+1}$ 必须等待 $S_c$。长序列、低 batch 或较大 TP 会减少每卡可并行的 state head，容易出现 GPU 欠占用。
3. **中间 state 是否落 HBM：** 若每个 chunk 都保存 boundary state，流量会随 chunk 数线性增长；让 state 跨 chunk 片上驻留，或用 CP 把长状态链拆成多个 segment，是三个优化实现的核心差别。

## 4. 为什么 prefill 要 chunk 化

逐 token 递推每步只有矩阵-向量乘和 rank-1 update，难以充分使用 Tensor Core。1.4 已经证明：把连续 $C$ 个 token 堆成矩阵后，可以用 $KK^\top$、$KS_0$、$QK^\top$、$QS_0$ 和 $K^\top V_{new}$ 这些矩阵乘，一次处理整个 chunk；三个实现最常用 $C=64$。

chunk 化只改变执行方式，不改变因果顺序：chunk 内用下三角求解一次得到全部 residual $V_{new}$，chunk 之间仍通过 $S_C$ 串行连接。其计算顺序可以概括为：

```text
log(alpha) -> gamma, G
K, beta, G -> K K^T -> triangular solve A
K, V, S0, A -> residual writes V_new
Q, K, V_new, S0 -> O
K, V_new, S0 -> S_C
```

FLA 常把 $V_{\mathrm{new}}$ 拆成 WY 中间量：

$$
U=A\operatorname{diag}(\beta)V,
\qquad
W=A\operatorname{diag}(\beta)\operatorname{diag}(e^\gamma)K,
$$

$$
V_{\mathrm{new}}=U-WS_0.
$$

这只是 1.4 中 $V_{new}$ 公式的分配律：$U$ 是与入口状态无关的当前 value 项，$WS_0$ 是需要从中扣除的旧状态读取。显式 $W/U$ 便于训练反向重算；推理专用 kernel 更倾向在片上直接形成 $V_{new}$。

## 5. Chunk 化后的数据流与真正的瓶颈

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

## 6. 两条跨 chunk 并行化路线

### 6.1 精确路线：把一段视为仿射变换

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

### 6.2 近似路线：利用 gate 对旧状态的遗忘

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

## 7. 三个实现的优化空间

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

从端到端数据流看，三者的 HBM 边界可以直接概括为：

| 实现 | 主数据流 | 跨 kernel 物化的核心中间量 |
| --- | --- | --- |
| FlashInfer non-CP | input $\to$ 单个持久 chunk kernel $\to O,S_{final}$ | 正常路径不物化 $A,V_{new},H$ |
| FlashInfer CP | input $\to T\to(M,N)\to$ fixup $\to$ segment replay | $T$、$(M,N)$、segment initial states |
| FLA | input $\to\gamma\to A/W/U\to H/V_{new}\to O$ | $\gamma,A,W,U,H,V_{new}$ |
| FlashQLA | input $\to\gamma,A_0\to$ fused state/output | $\gamma,A_0$；CP 时再加 segment states 和 fallback $M$ |

## 8. 跨库最容易混淆的约定

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

继续阅读：[`FLA_FlashInfer_GDN.md`](FLA_FlashInfer_GDN.md)。
