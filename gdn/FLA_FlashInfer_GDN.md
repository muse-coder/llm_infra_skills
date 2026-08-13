# FLA 与 FlashInfer GDN：同一算法的两种 kernel 设计

前置阅读：[`GDN_Algorithm.md`](GDN_Algorithm.md)。本文基于 `flash-linear-attention` commit `ab181c671576` 和 `flashinfer` commit `53a1c3bd7a3f`，只分析 GDN prefill 的前向算法、数据流与 kernel 优化。

## 1. 从逐 token GDN 到精确 chunk GDN

逐 token GDN 对每个 token 执行

$$
\bar S_i=\alpha_iS_{i-1},
\qquad
r_i=\beta_i(v_i-\bar S_i^\top k_i),
$$

$$
S_i=\bar S_i+k_ir_i^\top,
\qquad
o_i=sS_i^\top q_i.
$$

这种写法每步都是矩阵-向量乘和 rank-1 update，难以充分使用 Tensor Core。chunk GDN 不改变递推，只把一个 state head、连续 $C$ 个 token 的因果依赖改写成矩阵乘和一次下三角求解。

### 1.1 用累计 decay 表示历史传播

把某个 chunk 的入口状态记为 $S_0$，token 编号为 $i=1,\ldots,C$。将 $q_i^\top,k_i^\top,v_i^\top$ 堆成

$$
Q,K\in\mathbb{R}^{C\times d_k},
\qquad
V\in\mathbb{R}^{C\times d_v}.
$$

定义 chunk 内累计 log decay 和 pairwise decay：

$$
\gamma_i=\sum_{m=1}^{i}\log\alpha_m,
\qquad
G_{ij}=\begin{cases}
e^{\gamma_i-\gamma_j},&i\ge j,\\
0,&i<j.
\end{cases}
$$

$e^{\gamma_i}$ 是入口状态传播到 token $i$ 的总衰减；$G_{ij}$ 是 token $j$ 的写入传播到 token $i$ 时经历的衰减。

### 1.2 展开写入前状态

将前面 token 的 state update 逐步代入，可以得到 token $i$ 写入之前的状态：

$$
\boxed{
\bar S_i=e^{\gamma_i}S_0+
\sum_{j<i}G_{ij}k_jr_j^\top
}.
$$

第一项是衰减后的 chunk 入口状态，第二项是 chunk 内所有更早 token 的 residual 写入。代回 $r_i=\beta_i(v_i-\bar S_i^\top k_i)$ 得到

$$
r_i^\top=\beta_i\left[
v_i^\top-e^{\gamma_i}k_i^\top S_0
-\sum_{j<i}G_{ij}(k_i^\top k_j)r_j^\top
\right].
$$

当前 residual 只依赖 $j<i$ 的 residual，不依赖未来 token，因此依赖矩阵一定是下三角。

例如 $C=2$ 时：

$$
\begin{aligned}
r_1^\top&=\beta_1(v_1^\top-e^{\gamma_1}k_1^\top S_0),\\
r_2^\top&=\beta_2(v_2^\top-e^{\gamma_2}k_2^\top S_0-G_{21}(k_2^\top k_1)r_1^\top).
\end{aligned}
$$

把包含 $r_1$ 的项移到左边，就是一个 $2\times2$ 单位下三角方程；一般的 $C$ 只是扩展为 $C\times C$。

### 1.3 一次下三角求解得到所有 residual

将 $r_i^\top$ 堆成 $R\in\mathbb{R}^{C\times d_v}$ 的第 $i$ 行，上式变为

$$
\left[I+\operatorname{strictLower}
\left(G\odot\operatorname{diag}(\beta)KK^\top\right)\right]R
=\operatorname{diag}(\beta)
\left(V-\operatorname{diag}(e^\gamma)KS_0\right).
$$

定义

$$
\boxed{
A=\left[I+\operatorname{strictLower}
\left(G\odot\operatorname{diag}(\beta)KK^\top\right)\right]^{-1}
},
$$

则

$$
\boxed{
V_{new}=R=A\operatorname{diag}(\beta)
\left(V-\operatorname{diag}(e^\gamma)KS_0\right)
}.
$$

`V_new` 不是新的模型投影，而是逐 token residual $r_1,\ldots,r_C$ 的堆叠。下三角求解只是一次得到这些因果 residual，数学上与按顺序计算完全等价。

### 1.4 展开 state update 和 output

求得全部 $V_{new}$ 后，chunk 结束状态为

$$
\boxed{
S_C=e^{\gamma_C}S_0+
\left[\operatorname{diag}(e^{\gamma_C-\gamma})K\right]^\top V_{new}
}.
$$

第 $j$ 个 residual 乘 $e^{\gamma_C-\gamma_j}$，表示该写入从 token $j$ 传播到 chunk 末尾的衰减。所有 token 输出为

$$
\boxed{
O=s\left[
\operatorname{diag}(e^\gamma)QS_0+
\left(G\odot\operatorname{Lower}(QK^\top)\right)V_{new}
\right]
}.
$$

第一项读取衰减后的入口状态，第二项读取当前 chunk 已发生的 residual 写入。`Lower` 包含对角线，对应逐 token GDN 中先更新 $S_i$、再计算 $o_i$。

### 1.5 Chunk 内并行，chunk 间仍然串行

精确 chunk GDN 的数据流是：

```text
log(alpha) -> gamma, G -----------------------------+
K, beta, G -> K K^T -> triangular solve A ---------+
K, V, S_c, A -> residual writes V_new -------------+-> S_(c+1)
Q, K, V_new, S_c -----------------------------------+-> O_c
```

$KK^\top$、$KS_c$、$QK^\top$、$QS_c$ 和 $K^\top V_{new}$ 都能转换为矩阵乘；token 因果性被编码进 $A$。但是不同 chunk 仍满足

$$
S_{c+1}=F_c(S_c),
$$

所以默认执行必须先得到 $S_c$，才能处理下一个 chunk。这是后面 state 驻留、kernel 融合和精确 CP 要优化的共同瓶颈。

## 2. 同一组 chunk 公式，两种 kernel 边界

FLA 和 FlashInfer 都精确执行第 1 节的 chunk GDN，没有截断历史。区别只在这些公式由哪些 kernel 执行、哪些中间量落 HBM，以及低并行度时如何调度 chunk 间状态链。

| 问题 | FLA | FlashInfer non-CP | FlashInfer CP |
| --- | --- | --- | --- |
| chunk 数学 | 精确 GDN | 精确 GDN | 精确 GDN |
| $V_{new}$ | 显式拆成 $U-WS_c$ | 片上直接形成 | segment replay 时形成 |
| state/output | 两个 kernel | 同一持久 kernel | 各 segment 并行重放 |
| 主要全局中间量 | $\gamma,A,W,U,H,V_{new}$ | 不物化 $A,V_{new},H$ | `T`、$(M,N)$、segment initial states |
| chunk 间调度 | 默认串行 state scan | 持久 CTA 内串行 | 仿射 fixup 后 segment 并行 |
| 是否近似 | 否 | 否 | 否 |
| 首要目标 | 训练反向、通用 shape/backend | 推理融合与片上驻留 | 低并行度下增加 CTA |

## 3. 接口语义必须先对齐

两者的公开参数名称相似，但 `g` 和 state layout 不能直接互传。

| 项目 | FLA | FlashInfer |
| --- | --- | --- |
| q/k/v 布局 | `[B,T,H,D]` | varlen `[total_tokens,H,D]` |
| `g` | 默认是 $\log\alpha$；也可传 raw gate 并在内部激活 | 是线性空间 $\alpha\in(0,1]$；`None` 表示全 1 |
| `beta` | 可在内部执行 sigmoid 或 `2*sigmoid` | 期望已经是线性空间；`None` 表示全 1 |
| q/k norm | 可在 wrapper 内执行 | 可在 kernel 内执行 |
| head 关系 | 主要支持 GVA | 同时支持 GQA 与 GVA |
| state | 默认 `[N,H,K,V]`，可切为 V-first | API 物理布局 `[N,H,V,K]` |
| varlen | 可选 `cu_seqlens` | `cu_seqlens` 必填 |

以算法文档中的 Qwen 配置为例，逻辑 shape 为 $Q,K:[B,T,16,128]$、$V,O:[B,T,128,128]$、state `[N,128,128,128]`。每条 sequence、每个 GDN 层的 fp32 state 为 8 MiB；两个库可能采用相反的最后两维物理布局，但表达的是同一个线性映射。

## 4. FLA：用显式 WY 表示连接多个通用 kernel

公开入口是 `fla/ops/gated_delta_rule/chunk.py::chunk_gated_delta_rule`。默认 chunk size 为 64，也支持 16 和 32。

### 4.1 完整前向数据流

```text
Q, K -----------------------> optional L2 norm --------------------> Qn, Kn
raw a, A_log, dt_bias ------> log(alpha) ----+
log(alpha) ----------------------------------+-> chunk cumsum -----> gamma(log2)
raw b ---------------------------------------> optional sigmoid ----> beta

Kn, beta, gamma -----------------------------> KKT + solve ---------> A
A, beta, V ----------------------------------> WY value branch ------> U
A, beta, gamma, Kn --------------------------> WY key branch --------> W

initial state S0
   + Kn, W, U, gamma ------------------------> serial state scan
                                                ├─ H[c] = S_c
                                                ├─ V_new[c] = U[c] - W[c] S_c
                                                └─ final_state

Qn, Kn, gamma, H[c], V_new[c] --------------> parallel output ------> O
```

前半段对所有 chunk 并行；state scan 沿 sequence 的 chunk 方向串行；得到每个 boundary state 后，output kernel 再按 chunk 并行。FLA 用 HBM 中间张量连接三种不同的 grid。

公式到 kernel 的映射为：

| 公式部分 | FLA kernel | 跨 kernel 输出 |
| --- | --- | --- |
| $\gamma$ | gate/cumsum kernel | cumulative gate $\gamma$ |
| $A$ | KKT + `solve_tril` | $A$ |
| $U,W$ | `recompute_w_u_fwd` | $U,W$ |
| $V_{new},S_{c+1}$ | `chunk_gated_delta_rule_fwd_h` | $V_{new}$、boundary states $H$、final state |
| $O$ | `chunk_fwd_o` | token output |

### 4.2 Gate cumsum

FLA 先计算 chunk-local

$$
\gamma_i=\sum_{j\le i}\log\alpha_j.
$$

写回时乘 $\log_2e$，后续用 `exp2` 计算 $e^x=2^{x\log_2e}$。因此内部名为 `g` 的中间张量已经是累计 log decay，并且使用底数 2 的指数坐标；pairwise decay $G_{ij}$ 在消费时由两行 $\gamma$ 的差现场生成。

### 4.3 KKT 与三角求逆

入口是 `fla/ops/gated_delta_rule/chunk_fwd.py::chunk_gated_delta_rule_fwd_intra`。非 Intel GPU 的 $C=64$ 快路径在一个 Triton program 中完成：

1. 将 $64\times64$ 下三角区域拆成 $16\times16$ block。
2. 沿 head dimension 累加 $KK^\top$，中间 block 保持 fp32。
3. 乘 row-wise beta 和 pairwise gate，再施加严格下三角 mask。
4. 对四个对角块做短前代。
5. 用分块公式按 $16\to32\to64$ 合并三角逆。
6. 将 $A$ 按输入 dtype 写回 HBM。

融合 KKT 与 solve 避免了 KKT 矩阵的一次 HBM 往返；$C=16/32$ 以及部分 backend 使用 `chunk_scaled_dot_kkt_fwd -> solve_tril` 的通用两步路径。尾 chunk 的 token mask 必须在指数乘法前生效，避免越界 gate 导致 `0*inf=NaN`。

### 4.4 为什么显式生成 W/U

FLA 将 residual 写成

$$
U=A\operatorname{diag}(\beta)V,
\qquad
W=A\operatorname{diag}(\beta)\operatorname{diag}(e^\gamma)K,
$$

$$
V_{new}=U-WS_c.
$$

`recompute_w_u_fwd` 将 $W[T,H_v,K]$ 和 $U[T,H_v,V]$ 写入 HBM。它们不是算法必须保存的量，而是训练边界：forward 保存 $A$，backward 可以重建 $W/U$，再让多个梯度 kernel 复用。

### 4.5 State scan 与 output 为什么分开

state kernel 的 grid 按 `(sequence, value-dimension tile, state-head)` 展开。每个 program 将 state tile 保存在 fp32 register 中，沿 chunk 串行执行：保存当前 $H_c$、形成 $V_{new}=U-WS_c$、衰减旧状态、用 $K^\top V_{new}$ 更新状态，最后写 final state。

`chunk_fwd_o` 读取 $Q,K,V_{new},H_c,\gamma$ 计算 $O_c$。给定 $H_c$ 后，各 chunk 的输出彼此独立，所以 output 使用与 state scan 不同的并行 grid；代价是每个 chunk 的 boundary state $H_c$ 必须物化。

对于算法文档中的 Qwen shape、$C=64$，FLA 每个 chunk 约显式写出 1 MiB 的 $A$、2 MiB 的 $W$、2 MiB 的 $U$、2 MiB 的 $V_{new}$ 和 4 MiB 的 boundary state。精确流量还取决于 dtype 和读写次数，但这说明 FLA 的主要代价是中间量 HBM 流量，而不是算法不同。

### 4.6 FLA 的优化取舍

| 设计 | 获得 | 代价 |
| --- | --- | --- |
| $C=64$ KKT + solve 融合 | 少一次 $C^2$ 中间往返 | 快路径绑定 shape/backend |
| 显式 $W/U$ | backward 可重算与复用 | token-wise 中间张量 |
| state 与 output 分离 | 简单、通用的 grid | boundary state 落 HBM |
| state 常驻 register | chunk 间少读写 | chunk loop 串行，低并行度时欠占用 |
| 多个 constexpr 特化 | 覆盖多种配置 | 编译变体更多 |

FLA 也有默认关闭的 inference-only intra-card backend，它先计算 subsequence 局部变换和入口状态，再重用原 state kernel；这只替换 state 阶段，不改变 WY 与 output 边界。

## 5. FlashInfer：把同一组公式压入持久 kernel

公开入口是 `flashinfer/gdn_prefill.py::chunk_gated_delta_rule`。wrapper 负责 varlen metadata、输出与 state pool 分配、架构检查，以及 non-CP/CP 路由。

### 5.1 Non-CP 完整数据流

non-CP 路径将一条 `(sequence,state-head)` 状态链交给一个持久 CTA。CTA 载入 $S_0$ 后，按顺序遍历 64-token chunk：

```text
HBM: Q, K, V, alpha, beta, S0
                    |
          load one 64-token chunk
                    |
          alpha -> gamma -> pairwise G
                    |
        +-----------+--------------------+
        |                                |
 K K^T, beta, G -> triangular A     K S_c, Q S_c
        |                                |
        +--> residual + inverse apply -> V_new
                              |             |
                   gated Q K^T V_new        +-> K^T V_new
                              |                         |
                         store O_c             keep S_(c+1) on chip
                                                        |
                                                next chunk / final store
```

$KK^\top$、$QK^\top$、$A$、$V_{new}$ 和 output partial 只服务当前 chunk，尽量留在 SMEM/TMEM/register。跨 chunk 真正流动的只有 $S_c$；正常路径每个 chunk 写回 $O_c$，state 只在序列结束或 checkpoint 边界写回。

### 5.2 每个 chunk 的七组 GEMM

SM100 主实现 `flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py` 将公式组织为：

1. $KK^\top$：构造严格下三角 KKT 项。
2. $QK^\top$：构造 chunk 内 causal score。
3. $KS_c$：读取旧状态在 key 方向的内容。
4. $QS_c$：输出的跨 chunk 部分。
5. 三角逆作用到 residual，形成 $V_{new}$。
6. gated causal score 乘 $V_{new}$，形成局部输出。
7. $K^\top V_{new}$：更新 recurrent state。

与 FLA 不同，这七组 GEMM、gate cumsum、三角求逆和 epilogue 都在一个 kernel 内；`A/W/U/H/V_{new}` 不作为跨 kernel 接口。

### 5.3 Blackwell warp specialization

当前 SM100 kernel 使用 12 个 warp：

| warp | 职责                                     |
| ---- | ---------------------------------------- |
| 0–3  | pairwise gate、KK/QK epilogue、三角求逆  |
| 4–7  | state/value/output 的向量 epilogue       |
| 8    | KK/QK 一侧的 MMA issuer                  |
| 9    | Q/K/V 的 TMA producer                    |
| 10   | state/value/output 一侧的第二 MMA issuer |
| 11   | output store，并预取 gate/beta           |

核心不是 warp 编号，而是两条 MMA 依赖链：$KK^\top/QK^\top/A$ 不依赖入口状态，可以提前推进；$KS_c/QS_c/V_{new}/S_{c+1}$ 依赖当前状态，只能按 chunk 顺序消费。多级 SMEM buffer、TMEM accumulator 和 mbarrier 用来重叠两条链。

### 5.4 三角求逆与特化

$64\times64$ 单位下三角矩阵先对较小对角块做局部逆，再使用

$$
\begin{bmatrix}A&0\\B&D\end{bmatrix}^{-1}
=\begin{bmatrix}A^{-1}&0\\-D^{-1}BA^{-1}&D^{-1}\end{bmatrix}
$$

逐层合并。这样把 64 步长前代变成块内短递推和 Tensor Core 合并，并让逆直接流向后续 value GEMM。`initial_state=None` 还会选择零状态特化，首块可以省掉无意义的 $KS_c/QS_c$ 或 state load。

### 5.5 精确 CP：在并行度不足时拆分状态链

non-CP 的融合不能增加 `(sequence,state-head)` CTA 数。FlashInfer CP 将长 sequence 切成 segment，并利用任意一段都可以写成

$$
S_{out}=MS_{in}+N
$$

这一事实执行四个阶段：

```text
1. T precompute
   K + beta -> 每个 64-token block 的三角变换

2. M/N precompute
   每个 segment 从零状态运行 -> M_local, N_local

3. fixup
   按 segment 复合 (M, N) -> 每段真实 initial state

4. CP prefill
   所有 segment 用 fixed state 并行重放 -> O, final_state
```

这条路径没有丢弃历史，是逐 token GDN 的精确代数重排；浮点结果可能因 TF32/fp32 累加顺序不同产生正常误差。代价是物化 `T`、$(M,N)$ 和 segment initial states，并增加预处理 launch 与 GEMM，因此只在 non-CP 工作数不足时启用。

### 5.6 FlashInfer 的优化取舍

| 设计 | 获得 | 代价 |
| --- | --- | --- |
| non-CP 单个持久 kernel | 消除 $A/W/U/H/V_{new}$ 的 HBM 接口 | 片上资源占用高 |
| 两条 MMA issuer 链 | 重叠状态无关与状态相关计算 | warp specialization 和同步更复杂 |
| state 驻留 TMEM | 跨 chunk 少读写 | chunk 间仍串行 |
| 分块三角逆 | 缩短求逆依赖链 | 快路径依赖硬件与固定 tile |
| 精确 CP | 低 batch/head 时增加 segment 并行度 | 额外 workspace、GEMM 和 launch |

## 6. 如何理解两者的性能差异

FLA 和 FlashInfer 的性能差异不能解释为“一个算法更好”，而应沿同一公式的数据位置分析：

1. **三角求解：** FLA 把 $A$ 写回供后续与 backward 使用；FlashInfer non-CP 直接消费片上 $A$。
2. **Residual：** FLA 通过全局 $W/U$ 与 state kernel 形成 $V_{new}$；FlashInfer 在主流水中直接形成并立即消费。
3. **State/output：** FLA 为并行 output 保存所有 $H_c$；FlashInfer 让同一 CTA 一边更新 state、一边生成 output。
4. **低并行度：** FLA 默认保留长 state scan；FlashInfer 付出 $(M,N)$ workspace，把长链精确拆成可并行 segment。
5. **训练与推理目标：** FLA 的中间边界让 autograd 和多 backend 更容易复用；FlashInfer 用更强的硬件特化换取更少的 HBM 流量。

因此，FLA 是“显式分解并复用中间量”，FlashInfer 是“融合并让中间量片上流动”；两者在输入语义对齐后计算同一个精确 GDN。

## 源码地图

| 实现 | 环节 | 路径/符号 |
| --- | --- | --- |
| FLA | public forward | `fla/ops/gated_delta_rule/chunk.py` |
| FLA | KKT + solve | `fla/ops/gated_delta_rule/chunk_fwd.py` |
| FLA | W/U | `fla/ops/gated_delta_rule/wy_fast.py` |
| FLA | state scan | `fla/ops/common/chunk_delta_h.py` |
| FLA | output | `fla/ops/common/chunk_o.py` |
| FlashInfer | public prefill | `flashinfer/gdn_prefill.py::chunk_gated_delta_rule` |
| FlashInfer | SM100 non-CP | `gdn_kernels/blackwell/gated_delta_net_chunked.py` |
| FlashInfer | SM100 CP | `gdn_kernels/blackwell/gdn_cp_prefill.py::cp_delta_rule_dsl_sm100` |
| FlashInfer | CP heuristic | `gdn_kernels/delta_rule_dsl/varlen_helper.py` |

下一篇：[`FlashQLA_GDN_Blackwell.md`](FlashQLA_GDN_Blackwell.md)。
