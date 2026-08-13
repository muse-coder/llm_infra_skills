# flash-linear-attention 的 GDN kernel

前置阅读：[`GDN_Algorithm.md`](GDN_Algorithm.md)。本文基于
`flash-linear-attention` commit `ab181c671576`，只分析 FLA 的 Gated Delta
Rule 前向 kernel；反向只在解释中间量为何存在时涉及。

## 1. 入口与输入语义

公开入口是
`fla/ops/gated_delta_rule/chunk.py::chunk_gated_delta_rule`，输入通常为：

```text
q:    [B, T, Hqk, K]
k:    [B, T, Hqk, K]
v:    [B, T, Hv,  V]
g:    [B, T, Hv]
beta: [B, T, Hv]
state:[N, Hv, K, V]
```

FLA 使用 GVA：`Hv % Hqk == 0`，多个 value/state head 复用 q/k head。默认
`g` 是逐 token 的 $\log\alpha$，入口内部再做 chunk-local cumsum。可选项包括：

- `use_qk_l2norm_in_kernel`：在 wrapper 中执行 q/k L2 norm。
- `use_gate_in_kernel`：传 raw gate，同时给 `A_log`/`dt_bias`，融合 gate 激活和 cumsum。
- `use_beta_sigmoid_in_kernel`：把 beta logits 转成 sigmoid。
- `allow_neg_eigval`：使用 `2*sigmoid(beta)`，把 beta 范围扩到 $[0,2)$。
- `state_v_first`：把状态显存布局切换为 `[V,K]`。
- `chunk_size`：支持 16、32、64，默认 64。

## 2. 从公式看 FLA 怎样拆 kernel

对一个 chunk，先计算 gated triangular inverse：

$$
A=\left[I+\operatorname{strictLower}
\left(G\odot\operatorname{diag}(\beta)KK^\top\right)\right]^{-1}.
$$

FLA 不直接在一个 kernel 中形成 $V_{\mathrm{new}}$，而是使用 WY 表示：

$$
U=A\operatorname{diag}(\beta)V,
\qquad
W=A\operatorname{diag}(\beta)\operatorname{diag}(e^\gamma)K,
$$

$$
V_{\mathrm{new}}=U-WS_{\mathrm{prev}}.
$$

再分别计算

$$
S_{\mathrm{next}}=e^{\gamma_C}S_{\mathrm{prev}}
+\left[\operatorname{diag}(e^{\gamma_C-\gamma})K\right]^\top V_{\mathrm{new}},
$$

$$
O=s\left[\operatorname{diag}(e^\gamma)QS_{\mathrm{prev}}
+\left(G\odot\operatorname{Lower}(QK^\top)\right)V_{\mathrm{new}}\right].
$$

这直接决定了 kernel 边界：

| 公式部分 | FLA kernel | 为什么独立 |
|---|---|---|
| $\gamma$ | gate/cumsum kernel | 标量扫描，数据形状与矩阵阶段不同 |
| $A$ | KKT + `solve_tril` | 每个 chunk 独立，可完全并行 |
| $W,U$ | `recompute_w_u_fwd` | forward/backward 都能由保存的 $A$ 重建 |
| $V_{new},S_{next}$ | `chunk_gated_delta_rule_fwd_h` | 二者共同依赖跨 chunk 状态，必须一起串行 |
| $O$ | `chunk_fwd_o` | 给定 boundary state 后，各 chunk 可独立并行 |

所以 FLA 的首要优化角度不是把所有公式塞进一个 kernel，而是保留可复用的 WY
分解和训练反向边界，同时在最昂贵的局部步骤上做针对性融合。它获得通用性与完整
autograd，代价是 $A/W/U/H/V_{new}$ 等中间量的 HBM 流量。

## 3. 前向 kernel 链

FLA 原生 Triton 路径的主线是：

```text
gate/log-decay
  └─ chunk-local cumsum

K, beta, cumulative g
  └─ KKT + triangular solve -> A
  └─ recompute_w_u          -> W, U

K, W, U, g, initial_state
  └─ chunk state scan       -> boundary states H, V_new, final_state

Q, K, V_new, H, g
  └─ output kernel          -> O
```

若打开 q/k norm 或 beta sigmoid，前面还会有对应 kernel。核心不是固定“恰好几个
launch”，而是三个明确边界：intra-chunk WY 表示、chunk 间状态递推、输出组装。

## 4. Gate cumsum

`chunk_gated_delta_rule_fwd` 先产生 chunk 内累计 log decay $\gamma$：

$$
\gamma_i=\sum_{j\le i}\log\alpha_j.
$$

FLA 在写出 cumsum 时乘 $\log_2e$，后续统一使用 `exp2`：

$$
e^x=2^{x\log_2e}.
$$

因此 kernel 内部名为 `g` 的中间张量不是原始 log gate，而是 chunk-local cumsum
并已换到底数 2 的指数坐标。不能把它直接传给 FlashInfer 或当作原始模型 gate。

pairwise decay $G_{ij}=e^{\gamma_i-\gamma_j}$ 不落 HBM，需要时由两行 cumsum
之差现场生成。

## 5. KKT 与三角求逆

入口是
`fla/ops/gated_delta_rule/chunk_fwd.py::chunk_gated_delta_rule_fwd_intra`。

### 5.1 $C=64$ 快路径

非 Intel GPU 上，$C=64$ 使用
`chunk_gated_delta_rule_fwd_kkt_solve_kernel`，在一个 Triton program 中完成：

1. 把 $64\times64$ 下三角区域拆成 10 个 $16\times16$ block。
2. 沿 head dimension 累加 $KK^\top$，10 个 block 都留在 fp32 register。
3. 乘 row-wise beta 和 pairwise gate，并施加严格下三角 mask。
4. 对 4 个 $16\times16$ 对角块做前代。
5. 用分块公式合并为 $32\times32$，再合并为 $64\times64$。
6. 把最终三角逆 $A$ 按输入 dtype 写回 HBM。

融合的直接收益是避免把 KKT 中间矩阵写回 HBM 后再由 `solve_tril` 读回。合并阶段
在支持 TF32 的 NVIDIA GPU 上使用 TF32 dot，累加器保持 fp32。

尾 chunk 的 gate mask 必须在乘法之前生效。否则越界 gate 与有效 gate 做指数差
可能得到 `inf`，再出现 `0*inf=NaN`。源码把 token-valid mask 合进 `tl.where`
正是为了解决这个问题。

### 5.2 其他 chunk size

$C=16/32$，以及 Intel GPU 上的 $C=64$，走两步：

```text
chunk_scaled_dot_kkt_fwd -> solve_tril
```

它与融合路径数学等价，但 KKT 矩阵要多一次 HBM 往返。这个 fallback 也说明 FLA
首先追求 shape/backend 覆盖，$C=64$ NVIDIA 快路径是在通用实现上的专门优化。

## 6. WY 中间表示：为什么显式生成 W/U

求得 $A$ 后，`recompute_w_u_fwd` 生成

$$
U=A\operatorname{diag}(\beta)V,
$$

$$
W=A\operatorname{diag}(\beta)\operatorname{diag}(e^\gamma)K.
$$

随后

$$
V_{\mathrm{new}}=U-WS_{\mathrm{prev}}.
$$

这一步是单独的 Triton kernel，输出 $W[T,H_v,K]$ 和 $U[T,H_v,V]$ 到 HBM。
对纯推理而言，这比在一个 fused kernel 内直接形成 $V_{\mathrm{new}}$ 多了中间流量；
但对 FLA 的完整 forward/backward 很重要：forward 在 autograd context 中保存 $A$，
backward 用同一个 `recompute_w_u_fwd` 重建 $W/U$，再让多个梯度 kernel 复用。

因此 $W/U$ 不是算法必需输出，而是 FLA 为训练、可组合 backward 和通用 Triton
实现选择的边界。

## 7. Chunk 间 state kernel

状态递推由
`fla/ops/common/chunk_delta_h.py::chunk_gated_delta_rule_fwd_h` 发射
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`。

grid 按 `(sequence, value-dimension tile, state-head)` 展开。每个 program：

1. 把入口状态以 fp32 accumulator 保存在 register。
2. 按 chunk 串行循环。
3. 在每块开始时按需保存 boundary state $H_c$。
4. 计算 $V_{\mathrm{new}}=U-WS_c$。
5. 把 chunk 末 gate 乘到旧状态。
6. 计算加权 $K^\top V_{\mathrm{new}}$ 并更新状态。
7. 循环结束后写 final state。

kernel 把 $K$ 维按 64 展开，当前断言 $K\le256$。状态跨所有 chunk 常驻 register
能避免反复读写，但 chunk loop 是串行的。低 batch、低 $H_v$ 时，program 数可能
不足以填满 GPU；这正是 FlashInfer/FlashQLA 另外增加单卡分段路径的原因。

### 7.1 FLA 自带的单卡 CP backend

FLA 也有 `fla/ops/common/backends/intracard.py`，但默认关闭
（`FLA_INTRACARD_CP`），只在 `torch.inference_mode()` 且 varlen 输入下可用。它先
对 subsequence 计算局部变换、合并入口状态，再调用原始 state kernel 重放。

这条路径只替换 state 阶段，不改变前面的 WY 表示和后面的 output kernel；不要与
FLA 的 `FLACPContext` 跨卡训练 CP 混淆。

## 8. Output kernel

`fla/ops/common/chunk_o.py::chunk_fwd_o` 读取 $Q,K,V_{\mathrm{new}}$ 和每个
chunk 的入口状态 $H_c$，计算：

$$
O=s\left[
\operatorname{diag}(e^\gamma)QH_c
+\left(G\odot\operatorname{Lower}(QK^\top)\right)V_{\mathrm{new}}
\right].
$$

state kernel 与 output kernel 的 grid 不同：前者需要沿 chunk 串行以保持状态，后者
能按 chunk 并行。因此 boundary state $H_c$ 必须跨 kernel 物化。这是 FLA 原生
路径最大的中间张量之一，也是 fused inference kernel 能省掉的流量。

## 9. FLA 原生路径的优化思路

| 设计 | 获得 | 代价 |
|---|---|---|
| KKT + solve 融合（$C=64$） | 少一次 $C^2$ 中间往返 | 快路径绑定 shape/backend |
| 显式 $W/U$ | backward 可重算与复用 | 两个 token-wise 中间张量 |
| state 与 output 分离 | 简单、通用的 Triton grid | boundary state 落 HBM |
| state 常驻 register | chunk 间少读写 | chunk loop 串行，低并行度欠占用 |
| 多个 constexpr 特化 | 支持多配置且无运行时分支 | 编译变体多、源码分支多 |

## 源码地图

| 环节 | 路径/符号 |
|---|---|
| public forward | `fla/ops/gated_delta_rule/chunk.py` |
| KKT + solve | `fla/ops/gated_delta_rule/chunk_fwd.py` |
| W/U | `fla/ops/gated_delta_rule/wy_fast.py` |
| state scan | `fla/ops/common/chunk_delta_h.py` |
| output | `fla/ops/common/chunk_o.py` |
| intra-card CP | `fla/ops/common/intracard_cp.py` |

下一篇：[`FlashQLA_GDN_Blackwell.md`](FlashQLA_GDN_Blackwell.md)。
