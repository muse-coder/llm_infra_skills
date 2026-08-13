# FlashInfer GDN prefill：融合与精确分段扫描

前置阅读：[`GDN_Algorithm.md`](GDN_Algorithm.md)。本文基于 `flashinfer` commit `53a1c3bd7a3f`，重点解释公开 API 到 kernel 的真实调用链。

## 1. 入口与范围

本文只分析 `flashinfer/gdn_prefill.py::chunk_gated_delta_rule` 及其 GDN prefill kernel。

prefill API 使用扁平 varlen 布局：

```text
q: [total_tokens, Hq, D]
k: [total_tokens, Hk, D]
v: [total_tokens, Hv, D]
cu_seqlens: [num_sequences + 1]   # 必填
state: [N, Hs, D, D]
```

$H_s=\max(H_q,H_v)$。它同时支持 GQA 和 GVA；输出 head 数也是 $H_s$。

一个关键接口差异是：FlashInfer 的 `g` 是线性空间的遗忘因子 $\alpha\in(0,1]$，不是 FLA/FlashQLA 使用的 $\log\alpha$。`g=None` 和 `beta=None` 都按全 1 处理。

## 2. 从公式看 FlashInfer 优化了什么

对一个 chunk，算法必须完成四个核心量：

$$
A=\left[I+\operatorname{strictLower}
\left(G\odot\operatorname{diag}(\beta)KK^\top\right)\right]^{-1},
$$

$$
V_{\mathrm{new}}=A\operatorname{diag}(\beta)
\left(V-\operatorname{diag}(e^\gamma)KS_{\mathrm{prev}}\right),
$$

$$
S_{\mathrm{next}}=e^{\gamma_C}S_{\mathrm{prev}}
+\left[\operatorname{diag}(e^{\gamma_C-\gamma})K\right]^\top V_{\mathrm{new}},
$$

$$
O=s\left[\operatorname{diag}(e^\gamma)QS_{\mathrm{prev}}
+\left(G\odot\operatorname{Lower}(QK^\top)\right)V_{\mathrm{new}}\right].
$$

FlashInfer 的 non-CP kernel 直接围绕这四式融合，主要优化角度是：

1. **不物化算法中间量。** $KK^\top$、$QK^\top$、$A$、$V_{\mathrm{new}}$ 和 chunk boundary state 尽量留在 SMEM/TMEM/register 中。
2. **按依赖关系重叠计算。** $KK^\top$、$QK^\top$、$A$ 不依赖入口状态，可以和依赖 $S_{\mathrm{prev}}$ 的 $KS$、$QS$、state update 使用两条 MMA 发射链并行推进。
3. **让状态跨 chunk 片上驻留。** 同一 CTA 串行处理多个 chunk，避免每 64 token 把 $128\times128$ 状态写回再读入。
4. **并行度不足时改算法调度。** 把更长 segment 写成 $S_{out}=MS_{in}+N$，通过精确 scan/fixup 让不同 segment 并行。

公式到实现的对应关系如下：

| 公式部分 | non-CP kernel 内的工作 | CP 路径 |
| --- | --- | --- |
| $G$、$A$ | gate cumsum + $KK^\top$ + 分块三角逆 | 单独 T precompute |
| $V_{new}$ | $KS$、residual、inverse apply | CP prefill 中计算 |
| $S_{next}$ | 状态驻留 TMEM 并逐 chunk 更新 | M/N precompute + fixup 得到各段入口，再局部更新 |
| $O$ | $QS$ 与 gated causal $QK^\top V_{new}$ 融合 | 各段并行重放输出 |

## 3. Non-CP：一个 CTA 内完成整条 chunk 链

SM100 主实现位于 `flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`。固定基本 chunk $C=64$，一个工作 tile 持有某个 sequence/state-head 的完整状态并串行推进 chunk。

### 3.1 每个 chunk 的数据流

源码把主计算写成七组 GEMM：

1. $KK^\top$：构造严格下三角 KKT 项。
2. $QK^\top$：构造 chunk 内 causal score。
3. $KS_{\mathrm{prev}}$：读取旧状态在 key 上的内容。
4. $QS_{\mathrm{prev}}$：输出的跨 chunk 部分。
5. 三角逆作用到 residual/value，形成修正后的 value。
6. chunk 内 score 乘修正 value。
7. $K^\top V'$：更新 recurrent state。

gate cumsum、pairwise decay、beta、三角求逆、状态更新与输出 epilogue 都留在同一个 kernel 中。跨 chunk 只保留真正必须保留的状态，不把 FLA 的 $W/U$、chunk boundary state 和 output partial 全部写回 HBM。

这条路径的核心取舍是：

- 单个 `(sequence, head)` 工作的局部性和融合非常好。
- chunk 之间仍是严格串行；当 batch×head 很小时，单个巨核无法凭融合制造更多 CTA。

### 3.2 Blackwell warp specialization

当前 SM100 kernel 使用 12 个 warp（384 threads）：

| warp | 职责                                     |
| ---- | ---------------------------------------- |
| 0–3  | pairwise gate、KK/QK epilogue、三角求逆  |
| 4–7  | state/value/output 的向量 epilogue       |
| 8    | KK/QK 这一侧的 MMA issuer                |
| 9    | Q/K/V 的 TMA producer                    |
| 10   | state/value/output 一侧的第二 MMA issuer |
| 11   | output store，同时预取 gate/beta         |

这里值得关注的不是 warp 编号本身，而是两条独立 MMA 发射链：与状态无关的 KK/QK/solve 可以向前预取和计算；依赖状态的 KS/QS/value/state update 必须按 chunk 顺序消费。实现用多级 SMEM buffer、TMEM accumulator 和 mbarrier 把两条链重叠。

文件顶部的长 docstring 仍把 warp 10 描述成 gate loader；当前类初始化代码已经把它改成第二 MMA issuer，并让 warp 11 兼任 gate/beta load。分析时应以 `GatedDeltaNetChunkedKernel.__init__` 为准。

### 3.3 三角求逆

KKT 矩阵是 $64\times64$ 单位下三角矩阵。实现不是逐行做一个长达 64 步的全局前代，而是：

1. 对较小的对角块做局部逆。
2. 用分块下三角逆公式逐级合并。
3. 把逆直接送给后续 value GEMM。

分块公式为

$$
\begin{bmatrix}A&0\\B&D\end{bmatrix}^{-1}
=
\begin{bmatrix}
A^{-1}&0\\-D^{-1}BA^{-1}&D^{-1}
\end{bmatrix}.
$$

这样把长串行依赖改成“块内短递推 + Tensor Core 合并”，同时避免单独的 solve kernel 和全局中间张量。

### 3.4 零初始状态与尾块

`initial_state=None` 会选择零状态特化，首块可以消去无意义的 $KS$、$QS$ 或状态加载。varlen 尾块由同一 kernel 的边界 predicate 处理，不改变四个核心公式。

## 4. CP：用仿射复合精确增加并行度

这里的 CP 是 **intra-GPU context parallelism**。它把一条长序列切成更大的 CP segment，每段内部仍以 64-token block 计算，但不同 segment 可以并行。

SM100 路径 `gdn_cp_prefill.py::cp_delta_rule_dsl_sm100` 明确发射四个阶段：

```text
1. T precompute
   K + beta -> 每个 64-token block 的 signed/beta-folded triangular inverse

2. M/N precompute
   每个 CP segment 从零状态运行，得到
   S_out = M_local S_in + N_local

3. fixup
   按 segment 顺序复合 (M, N)，求出每段真实 initial state

4. CP prefill
   各段用自己的 fixed state 并行生成 token output
```

这就是算法文档中的仿射扫描。它没有丢弃历史，属于代数精确重排；浮点结果仍会因 TF32/fp32 累加顺序不同而与逐 chunk 路径存在正常数值差异。

### 4.1 为什么不总是开 CP

CP 增加了 $(M,N)$ workspace、三次预处理/fixup launch 和额外 GEMM。只有默认 non-CP 工作数不足时才值得付这个成本。

公开 wrapper 的 SM100 auto heuristic 是：

$$
4\times(\text{num\_sequences}\times H_s)<\text{num\_SM}.
$$

SM90/SM120 还根据 HBM/GDDR 设备类别使用不同阈值。segment 长度不是常数： `choose_cp_chunk_len_host` 在短任务上平衡 fixup 与重放成本，在长任务上选择能让预计算 CTA 接近一波 SM 的 512-token 对齐长度。

因此 CP 的收益取决于实际是否启用以及选择了多长的 segment。

## 5. 优化思路总结

| 优化角度 | 对应公式 | 做法 |
| --- | --- | --- |
| 融合 | $A,V_{new},S_{next},O$ | non-CP 单 kernel，避免算法中间量落 HBM |
| 并行流水 | 状态无关的 $A/QK^\top$ 与状态相关项 | 两条 MMA issuer 链重叠 |
| 状态局部性 | $S_c\to S_{c+1}$ | 同一 CTA 内让状态驻留 TMEM，串行推进 chunk |
| 降低求逆深度 | $A$ | 小块求逆后用 Schur 公式分层合并 |
| 增加序列并行度 | $S_{out}=MS_{in}+N$ | CP 四阶段精确 precompute/fixup/replay |

FlashInfer 的核心思想可以概括为：并行度足够时优化单条状态链的片上流水；并行度不足时不再只优化 kernel 常数，而是用仿射结合律把一条长链拆成多条可并行短链。

## 源码地图

| 主题 | 路径/符号 |
| --- | --- |
| public prefill | `flashinfer/gdn_prefill.py::chunk_gated_delta_rule` |
| SM100 non-CP | `gdn_kernels/blackwell/gated_delta_net_chunked.py` |
| SM100 CP orchestration | `gdn_kernels/blackwell/gdn_cp_prefill.py::cp_delta_rule_dsl_sm100` |
| CP heuristic | `gdn_kernels/delta_rule_dsl/varlen_helper.py` |
| 下一篇：[`FLA_Triton_Baseline.md`](FLA_Triton_Baseline.md)。 |
