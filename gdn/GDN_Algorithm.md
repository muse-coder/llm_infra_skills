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

### 3.4 给定 Qwen3.5-MoE 配置如何决定 GDN shape

你给出的配置中，92 层按“三层 `linear_attention` + 一层 `full_attention`”重复，因此有 69 个 GDN/linear-attention 层和 23 个 full-attention 层。两类层使用不同的 head 配置，不能混用：

| config 字段 | 数值 | 作用 |
| --- | --: | --- |
| `hidden_size` | 8192 | GDN 层输入和最终输出宽度 $d_{model}$ |
| `linear_num_key_heads` | 16 | GDN 的逻辑 q/k head 数 $H_{qk}$ |
| `linear_key_head_dim` | 128 | GDN 的 key/query head dimension $d_k$ |
| `linear_num_value_heads` | 128 | GDN 的 value/state/output head 数 $H_v$ |
| `linear_value_head_dim` | 128 | GDN 的 value/output dimension $d_v$ |
| `linear_conv_kernel_dim` | 4 | q/k/v 短因果卷积的 kernel size |
| `mamba_ssm_dtype` | fp32 | recurrent state 的保存/累加 dtype |
| `attn_output_gate`、`output_gate_type` | true、swish | 启用 $z$ 分支，并在输出投影前执行 gated RMSNorm |
| `dtype` | bf16 | 投影权重和大多数 token activation 的 dtype |
| `rms_norm_eps` | $10^{-6}$ | 每个 value head 输出 RMSNorm 的 epsilon |
| `num_hidden_layers`、`full_attention_interval` | 92、4 | 每 4 层插入一个 full-attention 层 |
| `layer_types` | 69 linear + 23 full | 明确每层选择 GDN 还是 full attention |
| `use_cache` | true | 保存每个 GDN 层的 conv state 与 recurrent state |
| `max_position_embeddings` | 262144 | 最长序列会增加 chunk 数，但不会增大单份 GDN state |
| `num_attention_heads`、`num_key_value_heads`、`head_dim` | 64、4、256 | 只属于 full-attention 层，不决定 GDN shape |

由此得到

$$
d_Q=d_K=16\times128=2048,
\qquad
d_V=d_Z=128\times128=16384,
$$

$$
\frac{H_v}{H_{qk}}=\frac{128}{16}=8.
$$

也就是说，一个 q/k head 在逻辑上服务 8 个 value/state head。支持原生 GVA 的 kernel 可以只保存 16 个 q/k head 并在 kernel 内映射；某些模型集成会先 `repeat_interleave`，把 q/k 物化成 128 个 head，数学结果相同，但会增加 q/k activation 流量。

### 3.5 该配置下一层 GDN 的逐阶段 shape

下面的 $B$ 是 batch size，$T$ 是本次输入 token 数，$N$ 是实际 sequence 数。Qwen 风格实现通常把多个小投影合并成 `in_proj_qkvz` 和 `in_proj_ba`，但拆开看仍对应 q/k/v/z/beta/alpha 六条语义分支：

| 阶段 | 张量 shape | dtype/说明 |
| --- | --- | --- |
| 输入 | $X:[B,T,8192]$ | bf16 |
| `in_proj_qkvz` 输出 | $[B,T,36864]$ | $2048_Q+2048_K+16384_V+16384_Z$ |
| `in_proj_ba` 输出 | $[B,T,256]$ | $128_b+128_a$ |
| q/k/v 合并卷积输入 | $[B,T,20480]$ | $2048_Q+2048_K+16384_V$；depthwise causal conv，kernel size 4 |
| 逻辑 $Q$、$K$ | 各 $[B,T,16,128]$ | conv + SiLU 后再做 L2 norm |
| $V$、输出门 $Z$ | 各 $[B,T,128,128]$ | $V$ 经过 conv + SiLU，$Z$ 绕过 conv |
| $\log\alpha$、$\beta$ | 各 $[B,T,128]$ | gate 通常以 fp32 计算；$\beta=\sigma(b)$ |
| recurrent state | $[N,128,128,128]$ | 逻辑布局 $[N,H_v,d_k,d_v]$；某些 kernel 使用转置布局 |
| GDN core 输出 | $O_{core}:[B,T,128,128]$ | 与 $V/Z$ 同 head shape |
| norm + swish gate 后 | $[B,T,16384]$ | 128 个 value head 合并 |
| `out_proj` 输出 | $Y:[B,T,8192]$ | 回到 model hidden size |

`in_proj_qkvz` 的宽度来自 $2d_Q+2d_V=36864$，`in_proj_ba` 的宽度来自 $2H_v=256$。因此这个 GDN 层不是“8192 hidden 对应 8192 value”；它把 value/output-gate 分支扩展到 16384，再由 `out_proj` 压回 8192。

### 3.6 该配置下 state、activation 和中间量有多大

每个 state head 有 $128\times128=16384$ 个 fp32 元素，即 64 KiB；128 个 state head 合计

$$
128\times128\times128\times4\ \text{bytes}=8\ \text{MiB}
$$

每条 sequence、每个 GDN 层需要 8 MiB recurrent state。69 个 GDN 层合计 552 MiB/sequence；若 tensor parallel 按 head 均匀切分，则每卡约为 $552/TP$ MiB/sequence。这个 state 大小与 $T$ 无关，$T$ 只决定状态被更新多少次。

若 TP degree 为 $p$，并且 q/k head 与 value head 都按 head 均匀分片，则每卡的逻辑 core shape 为 $H_{qk}^{local}=16/p$、$H_v^{local}=128/p$，GVA group size 仍为 8，state 为 `[N,128/p,128,128]`，即 $8/p$ MiB/sequence/layer。以 TP=8 为例，每卡 $Q/K:[B,T,2,128]$、$V/O:[B,T,16,128]$、state `[N,16,128,128]`，state 为 1 MiB/sequence/layer，69 层合计 69 MiB/sequence；具体 projection GEMM 的切分与 collective 位置由模型并行实现决定。

以 bf16 activation 计，每个 token 的逻辑 Q、K 各 4 KiB，V、Z、$O_{core}$ 各 32 KiB。一个 64-token chunk 的 V、Z、$O_{core}$ 各为 2 MiB，而整层 state 是 8 MiB fp32。GDN core 的关键中间量在该配置下为：

| 每个 64-token chunk、全 128 个 state head | 元素数 | bf16 大小 | 是否算法必须落 HBM |
| --- | --: | --: | --- |
| $A$ 或 $A_0:[H_v,64,64]$ | 524,288 | 1 MiB | 否；取决于融合边界 |
| $W:[64,H_v,d_k]$ | 1,048,576 | 2 MiB | 否；FLA 显式保存 |
| $U$ 或 $V_{new}:[64,H_v,d_v]$ | 1,048,576 | 2 MiB | 否；FLA 显式保存/生成 |
| 一个 chunk boundary state | 2,097,152 | 4 MiB | 否；FLA 为独立 output kernel 保存 bf16 副本 |
| recurrent state cache | 2,097,152 | fp32 为 8 MiB | 启用 cache 时需要；只保存入口/最终状态，理想情况下跨 chunk 片上驻留 |

当 $T=262144$ 且 $C=64$ 时，一条 sequence 有 4096 个 chunk。若把 4 MiB 的 boundary state 对每个 chunk 全部物化，单层就是约 16 GiB；因此 FlashInfer/FlashQLA 将 state 与 output 融合、避免保存所有 boundary state，不只是小幅 kernel 调优，而是在消除一个随 chunk 数线性增长的巨大中间张量。

### 3.7 从这些 shape 看实际瓶颈

1. **层外投影很宽。** `in_proj_qkvz` 是 $8192\to36864$，`in_proj_ba` 是 $8192\to256$，`out_proj` 是 $16384\to8192$；这些投影单层约有 4.383 亿参数，bf16 权重约 836 MiB。端到端 GDN 层不能只看 recurrent kernel，长序列时投影 GEMM 也是主要计算量。
2. **core 的状态链具有严格依赖。** 每个 value head 的 64 KiB fp32 state 必须按 chunk 顺序更新。只要同一 CTA 能让 state tile 跨 chunk 驻留，HBM 流量就低；一旦拆成多个 kernel 并物化 boundary state，流量会迅速放大。
3. **GVA 让 value 侧远宽于 q/k 侧。** $H_v/H_{qk}=8$，所以 $KS$、$QS$、$K^\top V_{new}$、state update 和 output/value activation 通常比单纯的 $QK^\top$、$KK^\top$ 更重。
4. **长序列不增大 state，却拉长串行链。** 262144 token 对应 4096 个 chunk；低 batch 或较大 TP 会减少每卡的 local state heads，使默认 CTA 数下降，GPU 可能在一条很长的状态链上欠占用。这正是 FlashInfer 精确 CP 和 FlashQLA gate-driven CP 要解决的问题。
5. **TP 同时降低内存和并行度。** head sharding 下，每卡 state 约除以 TP，但每卡可并行的 $H_v$ 也从 128 降为 $128/TP$。例如 TP=8 时，每卡只有 16 个 value/state head；若一个 head 的 value dimension 再切成两个 tile，默认也只有约 $B\times16\times2$ 个持久 CTA，长序列时容易不足以覆盖所有 SM。

## 4. 为什么 prefill 要 chunk 化

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

### 4.1 单位下三角求解

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

### 4.2 修正 value、更新状态、生成输出

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

继续阅读：[`FlashInfer_GDN_Blackwell.md`](FlashInfer_GDN_Blackwell.md)。
