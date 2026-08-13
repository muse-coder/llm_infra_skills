# GDN（Gated DeltaNet）算法与模型层数据流

本文只解释逐 token GDN 算法、完整 GDN 层的计算链路，以及 model config 如何决定各张量 shape。逐 token GDN 如何精确演化为 chunk GDN，见 [`FLA_FlashInfer_GDN.md`](FLA_FlashInfer_GDN.md)；FlashQLA 的公式重排和 gate-driven CP，见 [`FlashQLA_GDN_Blackwell.md`](FlashQLA_GDN_Blackwell.md)。

默认状态布局为 $S\in\mathbb{R}^{d_k\times d_v}$。某些 kernel 在显存中保存 $S^\top$，但这只是物理布局变化。

## 1. 从线性注意力到 Gated Delta Rule

### 1.1 线性注意力是一块固定大小的关联记忆

最简单的线性注意力维护

$$
S_t=S_{t-1}+k_tv_t^\top,
\qquad
o_t=S_t^\top q_t.
$$

$k_t$ 是写入地址，$v_t$ 是写入内容。读取时

$$
S_t^\top q_t=\sum_{i\le t}v_i(k_i^\top q_t).
$$

状态大小与序列长度无关，但相近的 key 会不断向同一方向累加内容，无法主动覆盖旧值。

### 1.2 Delta rule 先读旧值，再写残差

令 $\beta_t$ 为写入强度。delta rule 先读取当前 key 已经关联的内容，再只写入差值：

$$
r_t=\beta_t\left(v_t-S_{t-1}^\top k_t\right),
\qquad
S_t=S_{t-1}+k_tr_t^\top.
$$

合并后为

$$
S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
$$

当 $\lVert k_t\rVert_2=1$ 且 $\beta_t=1$ 时，旧状态在 $k_t$ 方向的内容被擦除，再写入 $v_t$；$\beta_t<1$ 时是软更新。

### 1.3 GDN 在 delta rule 前加入遗忘门

GDN 每步按照固定顺序执行五件事：

$$
\boxed{\bar S_t=\alpha_tS_{t-1}}
\qquad\text{遗忘旧状态},
$$

$$
\boxed{\hat v_t=\bar S_t^\top k_t}
\qquad\text{读取当前 key 的旧内容},
$$

$$
\boxed{r_t=\beta_t(v_t-\hat v_t)}
\qquad\text{形成实际写入 residual},
$$

$$
\boxed{S_t=\bar S_t+k_tr_t^\top}
\qquad\text{更新状态},
$$

$$
\boxed{o_t=sS_t^\top q_t}
\qquad\text{从更新后的状态读取输出}.
$$

因此当前 token 可以读到自己刚完成的写入。状态更新也可以合并成

$$
\boxed{
S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}
+\beta_tk_tv_t^\top
}.
$$

$\alpha_t\in(0,1]$ 控制保留多少历史，$\beta_t$ 控制写入强度，$s$ 通常是 $1/\sqrt{d_k}$。实现常传入 $\ell_t=\log\alpha_t\le0$，再用 `exp` 或 `exp2` 得到 $\alpha_t$。

Qwen/FLA 常见的门参数化为

$$
\ell_t=-\exp(A_{\log})\operatorname{softplus}(a_t+\mathrm{dt\_bias}),
\qquad
\beta_t=\sigma(b_t),
$$

并对 $q_t,k_t$ 做 L2 norm。这里 $a_t,b_t$ 来自当前 hidden state 的线性投影，$A_{\log}$ 和 `dt_bias` 是每个 state head 的可训练参数。

## 2. GVA：q/k head 与 state head 的关系

Grouped Value Attention（GVA）允许 q/k 的 head 数 $H_{qk}$ 小于 value/state 的 head 数 $H_v$，并要求 $H_v$ 能被 $H_{qk}$ 整除。同一组的多个 value/state head 复用一个 q/k head：

$$
h_{qk}=\left\lfloor\frac{h_v}{H_v/H_{qk}}\right\rfloor.
$$

每个 value head 拥有独立的 $v_t,\alpha_t,\beta_t,S_t,o_t$，因此真正的 recurrent state 数量和主要并行维度由 $H_v$ 决定，而不是 $H_{qk}$。

## 3. 完整 GDN 层的计算链路

逐 token recurrence 只是 GDN recurrent core。完整模型层还包含输入投影、短因果卷积、门参数化、输出门、归一化和输出投影。

```text
hidden states X
├─ q projection -> short causal conv + SiLU -> reshape heads -> L2 norm -> Q
├─ k projection -> short causal conv + SiLU -> reshape heads -> L2 norm -> K
├─ v projection -> short causal conv + SiLU -> reshape heads             -> V
├─ a projection -> -exp(A_log) * softplus(a + dt_bias)                   -> log(alpha)
├─ b projection -> sigmoid(b)                                            -> beta
└─ z projection                                                            -> output gate Z

(Q, K, V, log(alpha), beta, S_initial)
                    -> GDN recurrent core
                    -> O_core, S_final

O_core -> per-head RMSNorm -> multiply by SiLU(Z) -> merge heads -> output projection -> Y
S_final -------------------------------------------------------------------------------> state cache
```

short convolution 和输出门是否启用由模型配置决定；包住 mixing 层的 residual connection 不属于 GDN recurrent core。底层 GDN 算子从 $Q,K,V,\alpha/\log\alpha,\beta,S_{initial}$ 开始，不负责输入投影和最终输出投影。

### 3.1 从 hidden states 生成 core 输入

1. $X$ 经过投影得到 q/k/v、遗忘门原始值 $a$、写入门原始值 $b$ 和输出门 $z$。
2. q/k/v 通常经过 depthwise short causal convolution 和 SiLU；分段处理输入时还需要缓存卷积 state。
3. q/k reshape 为 $H_{qk}$ 个 $d_k$ 维 head 并执行 L2 norm；v reshape 为 $H_v$ 个 $d_v$ 维 head。
4. $a,A_{\log},\mathrm{dt\_bias}$ 生成 $\log\alpha$，$b$ 经 sigmoid 生成 $\beta$。
5. 每条 sequence、每个 value/state head 从 cache 读取 $S_{initial}\in\mathbb{R}^{d_k\times d_v}$；没有历史时从零状态开始。

### 3.2 Core 内的状态与输出数据流

对每个 token、每个 value/state head，数据依赖为：

```text
S_(t-1) -- alpha_t --> S_bar_t --------------------------+
K_t -----------------> v_hat_t = S_bar_t^T K_t           |
V_t - v_hat_t -- beta_t --> residual r_t                 |
S_bar_t + K_t r_t^T -------------------------> S_t -------+
Q_t -----------------------------------------> O_t = s S_t^T Q_t
```

$S_t$ 一方面传给下一个 token，另一方面在当前 token 被 $q_t$ 读取。整段输入处理完后，$S_{final}$ 写回该模型层的 state cache，作为下一段输入的 $S_{initial}$。

### 3.3 Core 输出如何回到 hidden size

$O_{core}$ 的 head 结构与 $V$ 相同。模型通常先对每个 value head 做 RMSNorm，再乘 $\operatorname{SiLU}(Z)$，随后合并所有 value heads，通过 $W_o$ 投影回 $d_{model}$。$S_{final}$ 不进入输出投影。

## 4. 给定 Qwen MoE config 如何决定 shape

给定配置的 92 层中有 69 个 `linear_attention`（GDN）层和 23 个 `full_attention` 层。`num_attention_heads=64`、`num_key_value_heads=4`、`head_dim=256` 只描述 full-attention 层；GDN 使用独立的 `linear_*` 字段：

$$
H_{qk}=16,\qquad d_k=128,\qquad H_v=128,\qquad d_v=128.
$$

因此 q/k 总宽度为 $16\times128=2048$，value 和 output-gate 总宽度均为 $128\times128=16384$，一个 q/k head 服务 8 个 value/state head。下表中 $B$ 是 batch size，$T$ 是本次 token 数，$N$ 是实际 sequence 数。

| 计算部分 | 相关 config | shape |
| --- | --- | --- |
| 层输入 | `hidden_size=8192` | $X:[B,T,8192]$ |
| q/k | `linear_num_key_heads=16`、`linear_key_head_dim=128` | $Q,K:[B,T,16,128]$ |
| value 与输出门 | `linear_num_value_heads=128`、`linear_value_head_dim=128` | $V,Z:[B,T,128,128]$ |
| 遗忘门与写入门 | 每个 value/state head 一个标量 | $\log\alpha,\beta:[B,T,128]$ |
| recurrent state | $H_v\times d_k\times d_v$，`mamba_ssm_dtype=fp32` | $S:[N,128,128,128]$ |
| core 输出 | 与 value head 相同 | $O_{core}:[B,T,128,128]$ |
| 合并后的 core 输出 | $H_vd_v=16384$ | $[B,T,16384]$ |
| 层输出 | 投影回 `hidden_size` | $Y:[B,T,8192]$ |

Qwen 实现通常把投影合并成 `in_proj_qkvz:[8192\to36864]` 和 `in_proj_ba:[8192\to256]`。q/k/v 经过 `linear_conv_kernel_dim=4` 的 depthwise causal convolution；$Z$ 绕过卷积，用于 core 输出后的 swish gate。

一份 fp32 recurrent state 的大小为

$$
128\times128\times128\times4\ \text{bytes}=8\ \text{MiB}.
$$

所以每条 sequence、每个 GDN 层需要 8 MiB state，69 个 GDN 层合计 552 MiB；按 head 做 tensor parallel 时，每卡 state 大致再除以 TP degree。state 大小不随序列长度增长，序列长度只增加 recurrence 的执行步数。

## 阅读顺序

1. [`FLA_FlashInfer_GDN.md`](FLA_FlashInfer_GDN.md)：逐 token GDN 如何精确变成 chunk GDN，以及 FLA/FlashInfer 如何优化同一组 chunk 公式。
2. [`FlashQLA_GDN_Blackwell.md`](FlashQLA_GDN_Blackwell.md)：FlashQLA 如何继续做 gate-free 求逆、主 kernel 融合和 gate-driven CP。
