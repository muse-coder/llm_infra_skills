# FlashQLA GDN prefill：公式重排、融合与 gate-driven CP

前置阅读：[`GDN_Algorithm.md`](GDN_Algorithm.md)。FLA 与 FlashInfer 的精确实现对照见 [`FLA_FlashInfer_GDN.md`](FLA_FlashInfer_GDN.md)。本文基于 `FlashQLA` commit `c18a4860ea9c`，只分析 GDN prefill 的前向算法与 kernel 优化。

## 1. 从逐 token GDN 到 FlashQLA

逐 token GDN 维护

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

精确 chunk GDN 将一个 chunk 的 residual 堆成 $V_{new}$，并通过 gated 三角逆

$$
A=\left[I+\operatorname{strictLower}
\left(G\odot\operatorname{diag}(\beta)KK^\top\right)\right]^{-1}
$$

一次求出。FlashQLA 在这个精确 chunk 公式上继续做两层演化：

```text
逐 token GDN
   -> 精确 chunk GDN
   -> A = G ⊙ A0：三角求逆与 gate 解耦
   -> kkt_solve(A0) + fused_gdr_fwd：重选 kernel 边界
   -> gate-driven Auto-CP：快速衰减 head 使用有限 warmup
                           慢速衰减 head 使用完整 M 精确修正
```

前三个箭头都是严格代数等价，不改变 GDN；最后一个 Auto-CP 箭头中，快速衰减 head 会截断足够早的历史，引入有绝对衰减界的近似，慢速衰减 head 仍走精确 fallback。因此 FlashQLA 的普通路径是精确 GDN，算法层调优只发生在启用 Auto-CP 且满足截断条件的 head 上。

## 2. 入口与主流程

公开入口是 `flash_qla/ops/gated_delta_rule/chunk/__init__.py::chunk_gated_delta_rule`。它接收 log-space gate：

```text
q:    [B, T, Hqk, K]
k:    [B, T, Hqk, K]
v:    [B, T, Hv,  V]
g:    [B, T, Hv]       # log(alpha)，不是 alpha
beta: [B, T, Hv]
state:[N, Hv, K, V]
```

SM90/SM100/SM103 的基本 chunk size 为 64，主优化 shape 是 $K=V=128$。前向首先执行：

```text
chunk_local_cumsum(g) -> gamma
kkt_solve(k, beta)    -> A0
optional auto-CP preprocess -> 每个分段的入口状态
fused_gdr_fwd(q, k, v, A0, gamma, beta, state) -> O, final_state
```

因此正常路径的数学主体是“gate cumsum + gate-free KKT/solve + fused state/output”三段，而不是把三角求逆也塞进主 fused kernel。

### 2.1 普通路径的数据如何流动

```text
Q, K ------------------------------> optional L2 norm
log(alpha) ------------------------> chunk_local_cumsum -> gamma [HBM]
K, beta ---------------------------> kkt_solve          -> A0 [HBM]

initial_state S0
   + Q, K, V, gamma, beta, A0 -----> fused_gdr_fwd
                                      for chunk c = 0..N-1:
                                        P = Q K^T
                                        U = K S_c
                                        R = V - exp(gamma) U
                                        V_d = (G * A0) diag(beta) R
                                        O_c = cross-state output + local output
                                        S_(c+1) = decayed S_c + K^T V_d
                                      -> O [HBM]
                                      -> final_state [optional HBM]
```

`kkt_solve` 对所有 `(chunk,state-head)` 并行，产生 gate-free $A_0$；`fused_gdr_fwd` 再按 `(sequence,state-head,value-tile)` 启动 CTA，每个 CTA 把 state tile 留在片上并串行消费多个 chunk。跨 kernel 必须物化的主要中间量只有 $\gamma$ 和 $A_0$，而 $R,V_d$、boundary state 与 output partial 不落 HBM。

### 2.2 Auto-CP 路径的数据如何流动

auto-CP 启用时，$\gamma$ 和 $A_0$ 仍先按普通路径生成；随后 preprocessing 把原 sequence 切成 segment，并为每段准备入口状态：

```text
gamma + segment boundaries ------------> warmup chunk count / fallback mask
K, V, A0, gamma, beta + zero state -----> suffix state; fallback heads get full N
slow-decay fallback heads --------------> full local transition M
local state, optional M, raw initial_state -> correct_initial_states
                                           -> initial_state[segment]

Q, K, V, A0, gamma, beta
   + initial_state[segment] --------------> fused_gdr_fwd over all segments in parallel
                                           -> token O + original sequence final_state
```

快衰减 head 的 segment initial state 来自有限 suffix warmup，是带 $e^{-10}$ 绝对衰减界的近似；慢衰减 head 使用完整 $M$ 沿 segment 修正，保持精确历史。preprocess 只改变传给 `fused_gdr_fwd` 的 `initial_state`、`cu_seqlens` 和 sequence mapping，不改变 token 的 q/k/v 排列或最终 output 顺序。

### 2.3 各阶段 shape

| 张量 | 逻辑 shape | 给定 Qwen 配置、$C=64$ 时的大小 |
| --- | --- | --- |
| $Q,K$ | `[B,T,Hqk,K]` | `[B,T,16,128]` |
| $V,O$ | `[B,T,Hv,V]` | `[B,T,128,128]` |
| $\gamma,\beta$ | `[B,T,Hv]` | 每个 chunk 各 `[64,128]` |
| $A_0$ | `[B,T,Hv,C]` | 每个 chunk `[128,64,64]`，bf16 为 1 MiB |
| ordinary initial/final state | `[N,Hv,K,V]` | fp32 为 8 MiB/sequence/layer |
| CP segment initial state | `[num_segments,Hv,K,V]` | 每段一份入口 state |
| fallback $M$ | 概念上 `[num_segments,Hv,K,K]` | 只为慢衰减 head 参与精确修正 |

普通路径每个 chunk 将约 1 MiB 的 $A_0$ 写到 HBM，再由 fused kernel 读取；$R,V_d$ 和 boundary state 不作为跨 kernel 接口。保留独立 $A_0$ 的目的是让求逆使用完全并行的 grid、为训练 backward 保存检查点，并降低主 fused kernel 的资源压力。

## 3. 从公式看 FlashQLA 优化了什么

精确 chunk GDN 的 gated triangular inverse 是

$$
A=G\odot A_0,
$$

其中

$$
A_0=\left[I+\operatorname{strictLower}
\left(\operatorname{diag}(\beta)KK^\top\right)\right]^{-1},
$$

$$
G_{ij}=\begin{cases}e^{\gamma_i-\gamma_j},&i\ge j,\\0,&i<j.\end{cases}
$$

由此可写出 FlashQLA kernel 实际采用的计算次序：

$$
R=V-\operatorname{diag}(e^\gamma)KS_{\mathrm{prev}},
$$

$$
V_d=(G\odot A_0)\operatorname{diag}(\beta)R,
$$

$$
S_{\mathrm{next}}=e^{\gamma_C}S_{\mathrm{prev}}
+\left[\operatorname{diag}(e^{\gamma_C-\gamma})K\right]^\top V_d,
$$

$$
O=s\left[\operatorname{diag}(e^\gamma)QS_{\mathrm{prev}}
+\left(G\odot\operatorname{Lower}(QK^\top)\right)V_d\right].
$$

公式到 kernel 的映射为：

| 公式部分 | FlashQLA kernel | 优化角度 |
| --- | --- | --- |
| $\gamma$ | `chunk_local_cumsum` | log 域累计，消费时用 `exp2` |
| $A_0$ | `kkt_solve` | gate-free 求逆；$16\to32\to64$ 分块合并 |
| $R,V_d$ | `fused_gdr_fwd` | 不生成显式 $W/U$，直接形成修正 value |
| $S_{next}$ | `fused_gdr_fwd` | 状态驻留 fp32 fragment/TMEM，逐 chunk 更新 |
| $O$ | `fused_gdr_fwd` | $QS$ 与 gated $QK^\top V_d$ 在同一流水内完成 |
| 长序列状态链 | CP preprocess + `fused_gdr_fwd` | gate 决定有限 warmup；慢衰减时用 $M$ 精确修正 |

FlashQLA 的优化主线有三条：

1. 用相似变换把 gate 从三角求逆中拿出来，简化独立的 solve kernel。
2. 把 $V_{new}$、state update 和 output 合进一个 warp-specialized kernel，消除显式 $W/U/V_{new}/H$ 的大部分全局中间流量。
3. 当 chunk 间串行导致并行度不足时，利用 gate 的指数遗忘缩短状态依赖；不能截断的 head 才计算完整转移矩阵修正。

## 4. `kkt_solve`：先求 gate-free 的 $A_0$

实现位于 `flash_qla/ops/gated_delta_rule/chunk/blackwell/kkt_solve.py`。每个 CTA 对一个 `(chunk, state-head)` 工作，使用 128 threads。

### 4.1 为什么求逆里没有 gate

kernel 只读取 $K$ 和 $\beta$：

1. Tensor Core 计算 $KK^\top$。
2. 每一行乘 $\beta_i$。
3. 保留严格下三角并把对角设为 1。
4. 求单位下三角逆 $A_0$。

pairwise gate $G$ 留到 `fused_gdr_fwd` 中逐元素乘到 $A_0$。这由

$$
[I+DLD^{-1}]^{-1}=D(I+L)^{-1}D^{-1}
$$

保证，而不是近似。这样 `kkt_solve` 不需要加载 gate、计算 $C^2$ 个指数，也使 $A_0$ 能作为干净的 backward 检查点。

### 4.2 分块求逆

$64\times64$ 的求逆分三层：

1. 四个 $16\times16$ 对角块做短前代。
2. 两组 $16\to32$ 合并。
3. 一次 $32\to64$ 合并。

合并使用

$$
\begin{bmatrix}A&0\\B&D\end{bmatrix}^{-1}
=\begin{bmatrix}A^{-1}&0\\-D^{-1}BA^{-1}&D^{-1}\end{bmatrix}.
$$

块内串行工作只剩 16 步，跨块部分转成矩阵乘。最终 $A_0$ 写成输入 dtype，形状为 `[B,T,Hv,64]`；主 fused kernel 再加载它。

求逆没有与主 kernel 融合，是有意的边界：求逆按 `(chunk,head)` 完全并行，而主 kernel 按 `(sequence,head,value-tile)` 持有并递推状态。两者的最佳 grid 和片上资源需求不同。

## 5. `fused_gdr_fwd`：融合其余四个公式块

SM100 实现位于 `flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py`。一个 CTA 使用 512 threads，按四个 warpgroup 分工。

### 5.1 公式执行顺序

每个 chunk 内的主要矩阵运算是：

1. $P=QK^\top$。
2. $U=KS_{prev}$。
3. $O_{inter}=QS_{prev}$。
4. $R=V-e^\gamma U$。
5. 现场构造 $G$，并形成 $A_g=(G\odot A_0)\operatorname{diag}(\beta)$。
6. $V_d=A_gR$。
7. $O=e^\gamma sO_{inter}+s(G\odot P)V_d$。
8. $S\leftarrow e^{\gamma_C}S+[\operatorname{diag}(e^{\gamma_C-\gamma})K]^\top V_d$。

FlashQLA 没有独立的 `recompute_w_u`、state-scan 和 output kernel，也不把 $W/U/V_d$ 作为跨 kernel 接口。这是它减少 HBM 流量的主要来源。

### 5.2 为什么仍保留 `kkt_solve`

若连 $KK^\top$ 和三角求逆也融合，单 CTA 同时需要：状态矩阵、Q/K/V、两个 $64\times64$ score、三角逆、value/output accumulator。资源和同步链会过长，并且求逆阶段无法像独立 grid 那样按所有 chunk 并行。

因此 FlashQLA 的融合边界是：

```text
完全并行且资源形态特殊的 triangular solve | 依赖状态的主流水
```

这个边界把资源形态特殊、完全并行的求逆阶段，与需要持有 recurrent state 的主流水分开。

### 5.3 Warp specialization 如何服务公式依赖

四个 warpgroup 分别持有不同的数据：

| threads | 主要职责                                   |
| ------- | ------------------------------------------ |
| 0–127   | 持有并更新 $S$                             |
| 128–255 | 形成 residual、$V_d$ 和 state-update value |
| 256–383 | 构造 $G/A_g$，组合 output                  |
| 384–511 | 发射 tcgen05 MMA、TMA load 和 store        |

Q/K/V/$A_0$/gate 使用双缓冲；状态以 fp32 fragment/TMEM 保存。对 $V=128$，状态拆成左右两个 64-column tile，以适配 TMEM 和 MMA tile。这里的硬件分工来自公式依赖：state owner 串行维护 $S$，value owner 处理 $KS\to R\to V_d$，output owner 处理 $QK^\top$、pairwise gate 和 $O$；producer 只负责异步搬运/MMA 发射。

## 6. Gate-driven intra-card CP

默认 fused kernel 的 CTA 数近似为

$$
B\times H_v\times\left\lceil V/B_V\right\rceil,
$$

每个 CTA 内部仍串行遍历 chunk。长序列、小 batch、少 head 时，这一数量不足以占满 GPU。FlashQLA 的 `auto_cp` 把一条序列切成多个 segment，让 segment 也进入并行维度。

### 6.1 为什么可以只 warmup 一段有限历史

设两次递推只有入口状态不同，误差为 $\Delta S$。由逐 token GDN recurrence 可得

$$
\Delta S_t=\alpha_t(I-\beta_tk_tk_t^\top)\Delta S_{t-1}.
$$

当 $\lVert k_t\rVert_2=1$ 且 $0\le\beta_t\le2$ 时，$I-\beta_tk_tk_t^\top$ 的谱范数为 1，因此

$$
\lVert\Delta S_{out}\rVert
\le e^{\sum\log\alpha}\lVert\Delta S_{in}\rVert.
$$

`get_warmup_chunks` 从一个 segment 的尾部向前累计 chunk-end log gate，并对每个 head 找到第一个

$$
\sum\log\alpha<-10
$$

的位置。对该 head，只从零状态重放这几个 suffix chunk，就能近似得到 segment 结束状态；它将作为下一 segment 的入口状态。被截断的更早历史最多剩 $e^{-10}\approx4.54\times10^{-5}$ 的绝对影响。

这是 gate 驱动而不是固定窗口：衰减快的 head 回看少，衰减慢的 head 回看多。

### 6.2 慢衰减 head 的精确 fallback

如果扫描完整个 segment 仍未达到阈值，`fallback_mask=True`。此时 `prepare_h` 运行整个 segment，既得到零初态下的精确局部项 $N$，也计算该段对入口状态的转移矩阵 $M$：

$$
S_{out}=MS_{in}+N.
$$

`correct_initial_states` 沿 segment 顺序执行

$$
S_{i+1}=M_iS_i+N_i
$$

来得到真实入口。对于已达到阈值的 head，不再乘完整 $M$，直接采用 suffix warmup 得到的近似 outgoing state；对于慢衰减 head，使用完整的 $(M,N)$ 精确传递历史。

所以 FlashQLA CP 是一个混合策略：

- 快衰减 head：利用模型性质近似截断，减少转移矩阵 GEMM。
- 慢衰减 head：仿射转移精确修正，不强行截断。

### 6.3 分段长度与启用条件

`_calc_cp_seqs` 用

$$
L_{segment}\propto
\sqrt{H_v\times\text{num\_chunks}/\text{num\_SM}}
$$

的延迟模型选择每段 chunk 数，再对齐到 2 的幂并保证至少 4 chunk。随后还按架构、有效 batch-head 数和最长序列 chunk 数判断 CP 固定开销是否值得。

这一启发式不改变算法正确性策略；它只决定何时用“更多短状态链 + correction”替代“更少的长状态链”。

## 7. FlashQLA 的算法与 kernel 演化总结

| 阶段 | 从 GDN 公式做了什么 | 优化目标 | 是否近似 |
| --- | --- | --- | --- |
| chunk-local cumsum | 将逐 token $\log\alpha$ 变为 $\gamma$ 和 pairwise decay $G$ | 批量生成 chunk 内衰减 | 否 |
| gate-free $A_0$ | 用 $A=G\odot A_0$ 将 gate 从三角求逆移到消费侧 | 减少 solve 的指数与输入依赖 | 否 |
| 独立 `kkt_solve` | 对所有 `(chunk,head)` 并行求 $A_0$ | 使用适合三角逆的独立 grid | 否 |
| `fused_gdr_fwd` | 片上形成 $R,V_d$，同时计算 state update 与 output | 消除 $W/U/V_d/H$ 的 HBM 接口 | 否 |
| Auto-CP fast-decay | 只重放达到 decay 阈值所需的 suffix chunk | 缩短长序列状态依赖 | 是，误差受累计 gate 衰减约束 |
| Auto-CP fallback | 对慢衰减 head 保存并应用完整 $(M,N)$ | 不可安全截断时传递完整历史 | 否 |

因此 FlashQLA 的普通路径首先是一个精确、训练友好的 fused GDN 实现；gate-driven Auto-CP 才是在低并行度长序列场景中额外加入的算法级调优。

## 源码地图

| 公式部分              | 路径/符号                                          |
| --------------------- | -------------------------------------------------- |
| orchestration         | `flash_qla/ops/gated_delta_rule/chunk/__init__.py` |
| gate-free $A_0$       | `chunk/blackwell/kkt_solve.py`                     |
| $V_d,S_{next},O$      | `chunk/blackwell/fused_fwd.py`                     |
| segment local $(M,N)$ | `chunk/blackwell/prepare_h.py`                     |
| warmup 与 correction  | `chunk/blackwell/cp_fwd.py`                        |
| CP 切段 heuristic     | `chunk/cp_context.py`                              |
