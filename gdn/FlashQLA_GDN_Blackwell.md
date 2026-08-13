# FlashQLA GDN prefill：公式重排、两段式融合与 gate-driven CP

前置阅读：[`GDN_Algorithm.md`](GDN_Algorithm.md)。FLA 与 FlashInfer 的精确 chunk GDN 推导见 [`FLA_FlashInfer_GDN.md`](FLA_FlashInfer_GDN.md)。本文以 [Qwen 官方 FlashQLA 文章](https://qwen.ai/blog?id=flashqla)和 `FlashQLA` commit `c18a4860ea9c` 为依据，重点解释 GDN chunked prefill 为什么演化成 FlashQLA、每个公式由哪个 kernel 负责，以及 Auto-CP 在哪里引入近似。

## 1. FlashQLA 要解决的不是一个，而是两个相互牵制的瓶颈

逐 token GDN 维护状态 $S_t\in\mathbb{R}^{K\times V}$：

$$
\bar S_t=\alpha_tS_{t-1},
\qquad
r_t=\beta_t(v_t-\bar S_t^\top k_t),
$$

$$
S_t=\bar S_t+k_tr_t^\top,
\qquad
o_t=sS_t^\top q_t.
$$

精确 chunk GDN 将 chunk 内的 token 递推改写成三角求解和矩阵乘，但 chunk 之间仍通过边界状态 $S_i$ 串行连接。FLA 的经典实现把一个 chunk 的前向拆成四组计算：求三角逆 $A_i$，生成 $W_i/U_i$，生成修正 value $V'_i$ 并更新状态，最后生成输出 $O_i$。

这产生两个方向相反的性能问题：

1. 多 kernel 流程会重复读取 $K/V$，并把 $W/U/V'$、chunk boundary state 等中间量写入 HBM 后再读回，大部分阶段容易受显存带宽和 kernel launch 开销限制。
2. 如果把所有阶段完全融合，一个 CTA 必须持有状态并串行消费多个 chunk；其 grid 主要只有 `batch × state_heads × value_tiles`，在小 batch、高 TP 或 state head 数较少时又没有足够 CTA 占满 GPU。

因此 FlashQLA 的核心不是“融合得越多越好”，而是在两个目标之间选择边界：用独立 kernel 并行求所有 chunk 的三角逆，在状态递推前保留可插入 CP preprocessing 的位置，再用一个主 fused kernel 合并修正 value、状态更新和输出计算。

```text
减少 HBM 流量                         增加状态链并行度
      \                                  /
       \                                /
        独立 solve -> optional CP -> fused state/output
```

## 2. 从 FLA chunk 公式到 FlashQLA 公式

### 2.1 统一 gate 记号

代码输入的 `g` 是 $\log\alpha_t$。对一个长度为 $C$ 的 chunk，定义累计 log gate、累计 gate 和 pairwise decay：

$$
\gamma_t=\sum_{u=1}^{t}\log\alpha_u,
\qquad
d_t=e^{\gamma_t},
\qquad
D=\operatorname{diag}(d),
$$

$$
\Gamma_{ij}=
\begin{cases}
e^{\gamma_i-\gamma_j},&i\ge j,\\
0,&i<j.
\end{cases}
$$

本文的 $\gamma$ 表示累计 log gate，$d=e^\gamma$ 才是累计乘积。代码中的 `chunk_local_cumsum(g)` 生成前者；消费侧再计算指数或 `exp2`。这一区分可以避免把官方公式中常用的累计 gate $\gamma$ 与代码张量 `gamma` 混为一谈。

### 2.2 原始精确 chunk 流程

令 $Q,K\in\mathbb{R}^{C\times K}$、$V\in\mathbb{R}^{C\times V}$，$\beta\in\mathbb{R}^{C}$。原始 gated 三角逆为

$$
A_g=\left[I+\operatorname{strictLower}
\left(\operatorname{diag}(\beta)(\Gamma\odot KK^\top)\right)\right]^{-1}.
$$

接下来依次计算

$$
W=A_g\operatorname{diag}(\beta)DK,
\qquad
U=A_g\operatorname{diag}(\beta)V,
$$

$$
V'=U-WS_i,
$$

$$
S_{i+1}=d_C S_i+K^\top\operatorname{diag}(d_C/d)V',
$$

$$
O=s\left[DQS_i+\left(\Gamma\odot\operatorname{Lower}(QK^\top)\right)V'\right].
$$

这里的 $V'$ 就是 chunk 内所有 token 完成 delta-rule 去重后的修正 value。它与 [`FLA_FlashInfer_GDN.md`](FLA_FlashInfer_GDN.md) 中从三角系统求出的 $V_{new}$ 是同一个算法量，只是不同实现使用了不同名字。

### 2.3 把 gate 从三角求逆中拿出来

定义 gate-free 三角逆

$$
A_0=\left[I+\operatorname{strictLower}
\left(\operatorname{diag}(\beta)KK^\top\right)\right]^{-1}.
$$

由于 pairwise decay 可以写成对角相似变换，严格下三角部分满足

$$
I+DLD^{-1}=D(I+L)D^{-1},
$$

所以

$$
A_g=D A_0D^{-1}=\Gamma\odot A_0.
$$

这是严格代数等价，不是近似。将它代回 $V'=U-WS_i$：

$$
R=V-DKS_i,
$$

$$
V'=(\Gamma\odot A_0)\operatorname{diag}(\beta)R.
$$

因此旧流程与 FlashQLA 的量一一对应：

| 原始 chunk 量 | FlashQLA 重排后的量 | 对应关系 |
| --- | --- | --- |
| gated inverse $A_g$ | gate-free $A_0$ 与现场生成的 $\Gamma$ | $A_g=\Gamma\odot A_0$ |
| $U=A_g\operatorname{diag}(\beta)V$ | 不单独生成 | 合并到 $V'$ |
| $W=A_g\operatorname{diag}(\beta)DK$ | 不单独生成 | 合并到 $V'$ |
| $V'=U-WS_i$ | $R=V-DKS_i$，再左乘 $(\Gamma\odot A_0)\operatorname{diag}(\beta)$ | 完全相同的修正 value |
| state update | fused kernel 内直接消费 $V'$ | 公式不变 |
| output | fused kernel 内直接消费 $S_i$ 和 $V'$ | 公式不变 |

这一步的价值是同时消除显式 $W/U$，并让三角求逆只依赖 $K,\beta$；pairwise gate 留到主 fused kernel 中按需生成。

## 3. FlashQLA 普通路径的完整数据流

公开入口 `flash_qla/ops/gated_delta_rule/chunk/__init__.py::chunk_gated_delta_rule` 接收：

```text
q:             [B, T, Hqk, K]
k:             [B, T, Hqk, K]
v:             [B, T, Hv,  V]
g:             [B, T, Hv]          # log(alpha)
beta:          [B, T, Hv]
initial_state: [N, Hv, K, V]
```

本文分析的主路径使用 $C=64$，Qwen GDN 的主要 shape 是 $K=V=128$。普通前向按以下顺序执行：

```text
log(alpha) ------------------------> chunk_local_cumsum -> gamma [HBM]
K, beta ---------------------------> kkt_solve          -> A0 [HBM]

initial_state S0
   + Q, K, V, gamma, beta, A0 -----> fused_gdr_fwd
                                      for chunk i = 0..Nc-1:
                                        P  = Q K^T
                                        U  = K S_i
                                        R  = V - D U
                                        V' = (Gamma * A0) diag(beta) R
                                        O  = cross-state output + local output
                                        S_(i+1) = decayed S_i + K^T scaled(V')
                                      -> O [HBM]
                                      -> final_state [optional HBM]
```

`kkt_solve` 以 `(chunk,state-head)` 为并行维度，一次处理所有 chunk，不依赖前一个 chunk 的状态。`fused_gdr_fwd` 以 `(sequence,state-head,value-tile)` 为主要并行维度，每个 CTA 将 state tile 留在片上并串行消费属于该序列的 chunk。

普通路径中跨 kernel 物化的主要中间量只有累计 gate `gamma` 和 $A_0$。$R/V'$、chunk boundary state 和 output partial 都在主 fused kernel 内产生并立即消费。普通路径从逐 token GDN 到最终输出完全精确，只改变代数表达和 kernel 边界。

## 4. 为什么 Exact CP 需要 $M$，又为什么它很贵

主 fused kernel 的 chunk 链是串行的。要让不同 segment 同时运行，必须先知道每个 segment 的入口状态。对第 $j$ 个 segment，其全部 token 对入口状态构成一个仿射映射：

$$
S_{j,\mathrm{out}}=M_jS_{j,\mathrm{in}}+N_j.
$$

其中：

- $N_j$ 是以零状态进入该 segment 后，由该段 token 自己写入的局部状态。
- $M_j$ 是入口状态穿过该 segment 后保留下来的线性转移。

Exact CP 先并行计算每段的 $(M_j,N_j)$，再按 segment 顺序组合

$$
S_{j+1,\mathrm{in}}=M_jS_{j,\mathrm{in}}+N_j,
$$

从而得到所有 segment 的真实入口状态，最后让各 segment 的主计算并行运行。这个方案不改变 GDN，但 $M_j\in\mathbb{R}^{K\times K}$；递推和保存完整 $M$ 的代价可能高于状态局部项 $N$，而且必须在主 fused kernel 之前完成。

这解释了 FlashQLA 为什么保留如下两段式边界：

```text
kkt_solve(A0)
      ↓
optional CP preprocess: 计算 segment initial states
      ↓
fused_gdr_fwd: 各 segment 并行计算 V'、S、O
```

完全融合虽然减少 HBM 流量，却没有地方插入 segment initial-state correction；完全依赖 Exact CP 又会付出昂贵的 $M$。FlashQLA 的 gate-driven Auto-CP 正是为了只在确有必要的 head 上计算完整 $M$。

## 5. Gate-driven Auto-CP：从精确修正到有界 warmup

### 5.1 它如何增加 GPU 并行度

不使用 CP 时，CTA 数量近似为

$$
B\times H_v\times\left\lceil V/B_V\right\rceil,
$$

而每个 CTA 内部仍需串行遍历 chunk。intra-card Auto-CP 在一张 GPU 内把 grid 扩展为

```text
(sequence, state_head, value_tile)
                 ↓
(sequence, segment, state_head, value_tile)
```

它不是把张量分到多张卡，而是把一条长状态链切成更多可并行的 segment CTA。这对长序列、小 batch、较少本地 head 或高 TP 场景尤其重要。

### 5.2 为什么有限 warmup 可以代替部分 $M$

设两个递推只有入口状态不同。按本文 $S\in\mathbb{R}^{K\times V}$ 的方向，误差满足

$$
\Delta S_t=\alpha_t(I-\beta_tk_tk_t^\top)\Delta S_{t-1}.
$$

当 $\lVert k_t\rVert_2=1$ 且 $0\le\beta_t\le2$ 时，$I-\beta_tk_tk_t^\top$ 的谱范数为 1，因此

$$
\lVert\Delta S_{\mathrm{out}}\rVert
\le e^{\sum_t\log\alpha_t}\lVert\Delta S_{\mathrm{in}}\rVert.
$$

对目标 segment 的起点，Auto-CP 从该点向前检查累计 gate。如果某个 suffix 已满足

$$
\sum_t\log\alpha_t<-10,
$$

就从这个 suffix 的起点以零状态开始重放 warmup，得到目标 segment 的近似入口状态。这里 $e^{-10}\approx4.54\times10^{-5}$ 是“入口状态误差还能保留多少”的相对衰减因子：

$$
\lVert\Delta S_{\mathrm{segment\ start}}\rVert
\le e^{-10}\lVert\Delta S_{\mathrm{warmup\ start}}\rVert.
$$

它不是与入口状态大小无关的绝对误差上界。数学上，这条 fast-decay 路径确实截断了更早历史，因此属于近似；官方实测中约 60%–80% 的 linear-attention head 具有明显衰减，通常 warmup 6–8 个 chunk 后误差已经进入数值噪声范围。

### 5.3 慢衰减 head 仍走精确 fallback

如果回看完整个候选范围仍未达到衰减阈值，说明该 head 不能安全截断。此时 preprocessing 计算完整 $(M_j,N_j)$，并按

$$
S_{j+1,\mathrm{in}}=M_jS_{j,\mathrm{in}}+N_j
$$

精确传播历史。因此 Auto-CP 是按 head 选择的混合策略：

| head 类型 | segment 入口状态如何得到 | 是否近似 | 代价 |
| --- | --- | --- | --- |
| fast-decay | 从零状态重放有限 suffix | 是；误差受累计 gate 衰减约束 | 不计算完整 $M$ |
| slow-decay | 计算并组合完整 $(M,N)$ | 否 | 支付完整转移矩阵代价 |

Auto-CP 数据流为：

```text
gamma + segment boundaries -----------> warmup length / fallback mask

fast-decay head:
K, V, A0, gamma, beta + zero state ---> suffix warmup -------> segment S0

slow-decay head:
K, V, A0, gamma, beta ----------------> exact local (M, N)
raw initial_state + all (M, N) --------> sequential correction -> segment S0

Q, K, V, A0, gamma, beta + segment S0
                                      -> fused_gdr_fwd over segments in parallel
                                      -> token O + original sequence final_state
```

切段还需要平衡段内串行时间和段间 correction 时间。若总共有 $N_c$ 个 chunk、每段含 $L$ 个 chunk，可用

$$
T(L)\approx aL+b\frac{N_c}{L}
$$

描述两部分代价，因此最优长度具有 $L^\star=\lambda\sqrt{N_c}$ 的形式；$\lambda$ 还取决于 batch、head 数和硬件。具体启用阈值是版本与架构相关的性能 heuristic，只决定是否值得切段，不改变 fast-decay 近似与 slow-decay 精确 fallback 的正确性策略。

## 6. 两个主 kernel 分别优化哪部分公式

### 6.1 `kkt_solve`：计算 gate-free $A_0$

Blackwell 实现位于 `flash_qla/ops/gated_delta_rule/chunk/blackwell/kkt_solve.py`。每个 CTA 负责一个 `(chunk,state-head)`：

1. Tensor Core 计算 $KK^\top$。
2. 每行乘对应的 $\beta_i$。
3. 保留严格下三角并将对角设为 1。
4. 求单位下三角逆 $A_0$。

对 $C=64$，实现先求四个 $16\times16$ 对角块，再进行 $16\to32\to64$ 的块合并。块合并使用

$$
\begin{bmatrix}A&0\\B&D\end{bmatrix}^{-1}
=\begin{bmatrix}A^{-1}&0\\-D^{-1}BA^{-1}&D^{-1}\end{bmatrix}.
$$

这样只有短小的块内前代保持串行，跨块工作转成矩阵乘。求逆不与主 kernel 融合，是因为 solve 希望以所有 `(chunk,head)` 形成完全并行的 grid，而主 kernel 必须以 `(sequence,head,value-tile)` 持有并递推状态；两者最合适的 grid、寄存器和片上存储需求不同。

### 6.2 `fused_gdr_fwd`：计算 $V'$、状态和输出

Blackwell SM100 实现位于 `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py`。每个 chunk 内执行：

1. $P=QK^\top$，$U=KS_i$，并计算跨 chunk 输出项 $QS_i$。
2. $R=V-DU$。
3. 现场构造 $\Gamma$，形成 $A_g=(\Gamma\odot A_0)\operatorname{diag}(\beta)$。
4. $V'=A_gR$。
5. $O=s[DQS_i+(\Gamma\odot\operatorname{Lower}(P))V']$。
6. $S_{i+1}=d_CS_i+K^\top\operatorname{diag}(d_C/d)V'$。

$R/V'$、输出 partial 和 chunk boundary state 都不作为跨 kernel 接口。Q/K/V/$A_0$/gate 使用流水化加载，状态以 fp32 fragment 或片上存储保留，使数据搬运、Tensor Core 计算和 CUDA Core 标量计算重叠。

FlashQLA 采用一个 producer warpgroup 和三个 consumer warpgroups，通过 shared memory、barrier 和 ping-pong buffer 交换数据。不同 consumer 分别推进修正 value、状态和输出依赖链，producer 负责异步搬运与矩阵运算发射；这不是任意线程划分，而是直接对应 $V'\to S/O$ 的公式依赖。

### 6.3 哪些结论是通用的，哪些只属于 Blackwell

以下设计在 Hopper 和 Blackwell 实现中共享：

- gate-free $A_0$ 与 fused state/output 的 kernel 边界。
- optional CP preprocessing。
- producer/consumer warp specialization 和片上流水。

`tcgen05`、TMEM 以及具体 warpgroup/线程布局属于 SM100 Blackwell 实现细节；Hopper SM90 使用对应架构可用的 MMA 和片上存储机制。当前 FlashQLA 仓库支持多个 SM 版本，但不能把某个 Blackwell 文件中的指令级实现直接当成所有架构的统一实现。

## 7. 与给定 Qwen config 的 shape 对应

给定配置中：

```text
linear_num_key_heads   = 16
linear_key_head_dim    = 128
linear_num_value_heads = 128
linear_value_head_dim  = 128
```

所以全局 GDN shape 为：

| 张量 | 逻辑 shape | $C=64$ 时每个 chunk/head 相关大小 |
| --- | --- | --- |
| $Q,K$ | `[B,T,16,128]` | 每个 QK head 的 $K$ 为 `[64,128]` |
| $V,O$ | `[B,T,128,128]` | 每个 value head 的 $V$ 为 `[64,128]` |
| $\gamma,\beta$ | `[B,T,128]` | 每个 chunk 为 `[64,128]` |
| $A_0$ | packed `[B,T,128,64]` | 等价于每个 chunk `[128,64,64]`，bf16 约 1 MiB |
| initial/final state | `[N,128,128,128]` | fp32 为 8 MiB/sequence/layer |
| CP segment state | `[num_segments,128,128,128]` | 每个 segment 一份入口状态 |
| fallback $M$ | 每个 fallback head `[128,128]` | fp32 为 64 KiB/head/segment |

这里采用 GVA，$H_v/H_{qk}=128/16=8$：每个 QK head 被 8 个 value/state heads 复用。$K$ 可以复用，但每个 value head 有自己独立的 $\beta$、$\gamma$ 和状态，因此 `kkt_solve` 最终仍需形成 128 份逻辑上的 $A_0$，不能只求 16 份后无条件广播。

TP 会进一步减少每张 GPU 的本地 head 数。例如均匀 TP8 后，本地约为 2 个 QK heads 和 16 个 value heads；此时 batch 1 的普通 fused grid 很容易并行度不足，而长序列又提供了很多 chunk，所以正是 intra-card Auto-CP 更容易获益的 shape。与此同时，$A_0$ 每个全局 chunk 约 1 MiB，说明 FlashQLA 虽然保留了 solve 输出的 HBM 接口，但消除了更大的 $W/U/V'$ 和 boundary-state 中间流量。

## 8. 反向为什么也影响前向的 kernel 边界

FlashQLA 不仅优化前向。训练反向会复用 CP preprocessing 重算所需的 chunk boundary state，并把原本分散的 `bwd_dv`、`bwd_dhu`、`bwd_dqkwg` 和 `bwd_wy` 等阶段融合。$A_0$ 作为前向输出保留，也为反向提供了可复用的三角求解结果。

这解释了为什么 FlashQLA 没有追求把前向全部塞进一个 kernel：独立的 $A_0$、明确的 CP 插入点和可重算的 boundary state 同时服务前向并行度与训练反向。本文不展开梯度公式，因为它们不改变 GDN 的前向算法。

## 9. 算法演化与近似边界总结

| 阶段 | 从 GDN 公式做了什么 | 优化目标 | 是否近似 |
| --- | --- | --- | --- |
| 精确 chunk GDN | 将逐 token recurrence 化为三角求解和矩阵乘 | chunk 内并行 | 否 |
| gate-free $A_0$ | 用 $A_g=\Gamma\odot A_0$ 将 gate 移出三角求逆 | 减少 solve 输入与指数计算 | 否 |
| 独立 `kkt_solve` | 对全部 `(chunk,head)` 并行求 $A_0$ | 使用适合三角逆的独立 grid | 否 |
| `fused_gdr_fwd` | 片上生成 $R/V'$，同时完成 state update 和 output | 消除 $W/U/V'$ 等 HBM 中间量 | 否 |
| Exact CP | 用完整 $(M,N)$ 生成所有 segment 入口状态 | 将长状态链拆成并行 segment | 否 |
| Auto-CP fast-decay | 从零状态重放达到阈值所需的有限 suffix | 避免计算完整 $M$ | 是；相对状态误差受累计 gate 约束 |
| Auto-CP fallback | 对慢衰减 head 计算并组合完整 $(M,N)$ | 无法安全截断时保留完整历史 | 否 |

完整演化链路是：

```text
逐 token GDN
  -> 精确 chunk GDN
  -> 发现多 kernel HBM 流量大
  -> 全融合又受 chunk 状态链和 CTA 数限制
  -> 用 A_g = Gamma * A0 重排公式并选择两段式融合
  -> 需要并行度时插入 Exact CP
  -> fast-decay head 用有限 warmup 省掉 M
  -> slow-decay head 保留完整 M 精确修正
```

因此，FlashQLA 的普通路径和 Exact CP 路径都与 GDN 严格等价；只有 Auto-CP 对 fast-decay head 的有限 warmup 在算法上截断了历史。这个截断有 gate 衰减界，并在官方实测中通常落入数值噪声范围，但仍应与严格精确路径明确区分。

## 源码地图

| 公式部分 | 路径/符号 |
| --- | --- |
| orchestration | `flash_qla/ops/gated_delta_rule/chunk/__init__.py` |
| gate-free $A_0$ | `chunk/blackwell/kkt_solve.py` |
| $V',S_{i+1},O$ | `chunk/blackwell/fused_fwd.py` |
| segment local $(M,N)$ | `chunk/blackwell/prepare_h.py` |
| warmup 与入口状态 correction | `chunk/blackwell/cp_fwd.py` |
| CP 切段 heuristic | `chunk/cp_context.py` |

## 参考资料

- [Qwen：FlashQLA: CP-/Bwd-Friendly Fused Linear Attention Kernels for GDN](https://qwen.ai/blog?id=flashqla)
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)
