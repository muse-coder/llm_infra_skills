# GDN（Gated DeltaNet）算法逻辑

> **本文只讲算法，不讲实现。** 三个算子库（FLA / FlashInfer / FlashQLA）的优化方案各自成文，
> 但它们全部建立在本文推导出的同一套 chunk 公式、同一张依赖图、同一组数学恒等式之上。
> 读那三篇之前应当先读完本文，尤其是 §5（依赖结构）、§6（性能问题）和 §7（四个数学杠杆）。

对象：Qwen3-Next / Qwen3.5 的线性注意力层
范围：推理 **prefill** 阶段（chunked 形式）；decode 作为 $C=1$ 的退化情形在 §9 单独交代

---

## 目录

1. [从线性注意力到 Gated DeltaNet](#1-从线性注意力到-gated-deltanet)
2. [Qwen3.5 中的参数化](#2-qwen35-中的参数化)
3. [Chunk 形式的推导](#3-chunk-形式的推导)
4. [Chunk size 为什么是 64](#4-chunk-size-为什么是-64)
5. [依赖结构：唯一的串行环节](#5-依赖结构唯一的串行环节)
6. [Prefill 的性能问题](#6-prefill-的性能问题)
7. [四个可用的数学杠杆](#7-四个可用的数学杠杆)
8. [数值性质](#8-数值性质)
9. [Decode：$C=1$ 的退化](#9-decodec1-的退化)
10. [符号表与库间 API 差异](#10-符号表与库间-api-差异)

---

## 1. 从线性注意力到 Gated DeltaNet

### 1.1 线性注意力：把注意力写成可累加的状态

去掉 softmax，注意力就变成一个固定大小的状态在序列上累加：

$$
S_t = S_{t-1} + k_t v_t^{\top}, \qquad S \in \mathbb{R}^{K \times V}
$$

$$
o_t = S_t^{\top} q_t
$$

状态大小 $K\times V$ 固定，不随序列长度增长——这是线性注意力全部吸引力的来源（KV cache 从 $O(T)$ 变成 $O(1)$）。

问题是这个状态**只会写、不会改**。同一个 $k$ 方向反复写入会互相叠加污染，状态容量很快饱和。

### 1.2 联想记忆视角：什么是"$k$ 方向"，为什么会污染

关键是把 $S$ 理解成一张**联想记忆表**，而不是一堆数字。把写入累加起来：

$$
S = \sum_j k_j v_j^{\top}
\qquad\Longrightarrow\qquad
o = S^{\top} q = \sum_j v_j \,(k_j \!\cdot\! q)
$$

看这个读取式：**每个 $v_j$ 被取出来的权重，就是 $q$ 与 $k_j$ 的内积**。所以

- $k_j$ 是**地址**，$v_j$ 是**内容**；
- $k$ 做过 L2 norm，$\lVert k\rVert_2 = 1$，长度恒为 1，**只有方向携带信息**；
- 所谓"$k$ 方向"，就是这个 token 的 key 向量在 $K$ 维单位球面上指向哪里，即**它占用了哪个地址槽**。

两个 token 的 $k$ 方向**正交** $\Rightarrow$ 用 $q = k_a$ 去读时 $k_b\!\cdot\! q = 0$，完全读不到 $v_b$，互不干扰。
两个 token 的 $k$ 方向**相同** $\Rightarrow$ 用了同一个地址，读出来的是两者的**混合**，再也分不开——这就是"污染"。

**最小例子**（$K=2$ 地址空间，$V=1$ 内容为标量，$k_a=(1,0)^{\top}$，$k_b=(0,1)^{\top}$）：

| 步骤 | 操作 | 状态 $S$ | 读 $S^{\top}k_a$ |
|---|---|---|---|
| ① | 写入 $(k_a,\,3)$ | $\binom{3}{0}$ | $3$ ✓ |
| ② | 写入 $(k_b,\,5)$ — 正交方向，安全 | $\binom{3}{5}$ | $3$ ✓ |
| ③ | 再写入 $(k_a,\,7)$ — **纯累加** | $\binom{10}{5}$ | $10$ ❌ |
| ④ | 再写入 $(k_a,\,7)$ — **delta rule** | $\binom{7}{5}$ | $7$ ✓ |

第 ③ 步读出 $10$，它既不是旧值 $3$ 也不是新值 $7$，是两次写入**叠加的和**且无法拆分。
第 ④ 步的做法是：先读出该地址的旧值 $S^{\top}k_a = 3$，只写入残差 $7-3=4$，于是变成**覆盖写**而非累加写。

### 1.3 Delta rule：先擦后写

**擦除算子做的正是这件事。** $\bigl(I - \beta k k^{\top}\bigr)$ 是 $K\times K$ 矩阵，从左边作用在 $S$ 的**地址维度**上。取 $\beta=1$，对存在 $k$ 地址上的内容 $S = k v^{\top}$：

$$
\bigl(I - k k^{\top}\bigr) k v^{\top} = \Bigl(k - k \underbrace{(k^{\top}k)}_{=1}\Bigr)v^{\top} = 0
\qquad\text{（恰好清零）}
$$

而对正交方向 $k_b \perp k$：

$$
\bigl(I - k k^{\top}\bigr) k_b v_b^{\top} = k_b v_b^{\top}
\qquad\text{（完全不受影响）}
$$

$\beta < 1$ 时擦掉 $\beta$ 的比例、保留 $1-\beta$，相当于软覆盖，$\beta$ 即覆盖强度（学习率）。

**这也解释了"容量"从何而来**：$K=128$ 维空间里近似正交的方向极多（高维随机向量天然接近正交），所以只要不同 token 的 key 落在互不相干的方向上，$S$ 就能同时存下大量键值对而互不干扰。但序列一长，新来的 $k$ 必然与已存方向有非零内积——纯累加型对此毫无办法，只能越叠越糊；delta rule 则先擦后写，主动清掉干扰。

写成递推：

$$
S_t \;=\; S_{t-1} + \beta_t\, k_t \bigl( v_t - S_{t-1}^{\top} k_t \bigr)^{\top}
\;=\; \bigl( I - \beta_t k_t k_t^{\top} \bigr) S_{t-1} + \beta_t k_t v_t^{\top}
$$

其中 $\bigl(v_t - S_{t-1}^{\top}k_t\bigr)$ 是"当前状态在 $k_t$ 方向读出的值"与"想写入的值"之差，即**误差**；$\beta_t$ 是学习率。本质是**把状态当作一个在线学习的线性映射，每来一个 token 做一步梯度下降**。

### 1.4 遗忘门：Gated DeltaNet

Delta rule 能覆盖旧内容，但不能**主动腾出容量**——没被新 key 命中的地址永远保留。GDN 再加一个标量遗忘门：

$$
\boxed{\;
S_t \;=\; \alpha_t \bigl( I - \beta_t k_t k_t^{\top} \bigr) S_{t-1} \;+\; \beta_t k_t v_t^{\top}
\;}
$$

$$
o_t = S_t^{\top} q_t
$$

$\alpha_t \in (0,1)$ 让整张记忆表**全局按比例淡出**。这一项看起来只是个标量缩放，但它是后面几乎所有优化的支点：

- §3.3 的换元把它从状态转移里完全消掉，是 chunk 推导能走通的前提；
- §7.1 的"门可以从待求逆矩阵里提出来"就是它带来的相似变换；
- §7.4 的近似上下文并行完全建立在它带来的指数衰减上；
- 状态做低精度 I/O 量化之所以安全，也是因为量化噪声会被它衰减掉（见 §8）。

### 1.5 与 Mamba2 的本质区别

| | 状态转移算子 | 性质 |
|---|---|---|
| Mamba2 / SSD | $\mathrm{diag}(a_t)$ 或标量 $a_t$ | **对角**，逐元素乘 |
| GDN | $\alpha_t\bigl(I - \beta_t k_t k_t^{\top}\bigr)$ | **满秩矩阵作用**，秩一修正 |

这一条差别决定了两件事：GDN 表达力更强（能做地址级覆盖，Mamba2 只能整体衰减）；GDN 的 chunk 化困难得多（矩阵连乘 vs 标量连乘）。**本文后面所有麻烦都源于此。**

---

## 2. Qwen3.5 中的参数化

### 2.1 符号与形状

| 符号 | 定义 | 形状 |
|---|---|---|
| $S$ | 递归状态 | $[K, V]$，$K = V = 128$，每 v-head 一个 |
| $\alpha_t$ | 遗忘门 $=\exp\bigl(-\mathrm{softplus}(A_{\log})\cdot\mathrm{softplus}(a_t + \text{dt\_bias})\bigr) \in (0,1)$ | **标量**，每 v-head 一个 |
| $\beta_t$ | 写入门 $=\sigma(b_t) \in (0,1)$ | **标量**，每 v-head 一个 |
| $q_t, k_t$ | 经过 **L2 norm**，故 $\lVert k_t\rVert_2 = 1$ | $[K]$ |
| $v_t$ | | $[V]$ |
| $g_t$ | $g_t=\sum_{i\le t}\log\alpha_i$，chunk 内累加，$g_0=0$ | 标量 |

注意 $\alpha_t,\beta_t$ 都是**标量**而不是向量——这比 Mamba2 的逐通道门更省，也让 §3 的推导里它们能以对角阵形式自由挪动。

### 2.2 GVA：k/q head 与 v head 的数量不同

Qwen3.5 系列的线性注意力层用的是 **grouped value attention**：v-head 比 k/q-head 多，多个 v-head 共享同一组 $q,k$。真实配置（$d = K = V = 128$ 恒定）：

| 模型 | $H_{qk}$ | $H_v$ | 比例 |
|---|---|---|---|
| Qwen3.5-397B / 122B | 16 | 64 | 4 |
| Qwen3.5-27B | 16 | 48 | 3 |
| Qwen3.5-35B / 9B / 4B | 16 | 32 | 2 |
| Qwen3.5-2B / 0.8B | 16 | 16 | 1 |

**状态的数量由 $H_v$ 决定**（每个 v-head 一个 $[K,V]$ 状态），而 $q,k$ 张量只有 $H_{qk}$ 份、被广播复用。这一点在实现上很重要：

- $KK^{\top}$、$A_{\mathrm{raw}}$、$\Lambda$ 是 $q,k$ 与门的函数。$KK^{\top}$ 只依赖 k-head，可以在同组的 $H_v/H_{qk}$ 个 v-head 之间共享；但 $\alpha,\beta$ 是 per-v-head 的，所以 $\Lambda$ 和 $\mathrm{diag}(\beta)KK^{\top}$ 仍必须逐 v-head 算。**共享的只有裸 $KK^{\top}$。**
- 反向传播时 $dq,dk$ 需要在组内 $H_v/H_{qk}$ 个 v-head 上求和归约。

后文除特别说明外，"head"一律指 **v-head**，$H := H_v$。

### 2.3 层内数据流

```
hidden → in_proj_qkvz → [q, k, v, z]      in_proj_ba → [b, a]
   ↓
causal_conv1d(kernel=4)        # 只作用在 q,k,v 上
   ↓
l2norm(q), l2norm(k);  α = exp(−softplus(A_log)·softplus(a+dt_bias));  β = sigmoid(b)
   ↓
Gated Delta Rule               # 本文的对象
   ↓
gated RMSNorm(o, z) → out_proj
```

三个库在"哪些前处理算在算子里"这件事上分歧很大（$l2norm$、$\sigma(b)$、$\alpha$ 的计算是否融进 kernel），这直接影响 kernel 数量和 API 兼容性，见各库文档。

### 2.4 三个决定性的性质

**性质 A：状态是矩阵，不是向量。** $S \in \mathbb{R}^{128\times128}$，每 head 64 KB（fp32）。这不是 RNN 那种几百维的 hidden state，而是需要矩阵乘单元参与、且要占据片上存储的对象。

**性质 B：状态转移是满秩矩阵作用，不是对角的**（§1.5）。既是表达力来源，也是难优化的根源。

**性质 C：天生串行。** 算 $S_t$ 必须先有 $S_{t-1}$。

---

## 3. Chunk 形式的推导

### 3.1 为什么必须改写

§1 的递推一次只处理一个 token。Prefill 有几千上万个 token，逐个执行意味着上万次很小的矩阵-向量运算——全是延迟，没有吞吐，硬件的矩阵乘单元完全用不上。

所以要把**连续 $C$ 个 token 的更新一次性算完**，让它变成大矩阵乘法。取 $C = 64$（理由见 §4）。

难点只有一个：直接展开会得到一串矩阵连乘

$$
\prod_{t=C}^{1}\bigl( I - \beta_t k_t k_t^{\top} \bigr)
$$

好在每个因子都是"单位阵减一个秩一项"，这类乘积可以收拢成**一个 $C\times C$ 单位下三角矩阵的逆**。整个推导就是把这件事做出来，分三步。

### 3.2 记号

固定一个 chunk，token 编号 $t = 1,\dots,C$，$S_0$ 是进入该 chunk 时的状态。

| 记号 | 含义 | 形状 |
|---|---|---|
| $K,\,Q,\,V$ | 该 chunk 的 key / query / value，第 $t$ 行是 $k_t^{\top}$ 等 | $[C,K]$, $[C,K]$, $[C,V]$ |
| $g_t$ | $g_t = \sum_{i\le t}\log\alpha_i$，**chunk 内**累加，$g_0=0$ | 标量 |
| $\mathrm{diag}(e^{g})$，$\mathrm{diag}(\beta)$ | 对角阵 | $[C,C]$ |
| $\Lambda$ | $\Lambda_{ij}=\exp(g_i-g_j)$ | $[C,C]$ |

### 3.3 第一步：换元，把遗忘门从状态转移里消掉

令 $S_t = e^{g_t} P_t$（故 $P_0 = S_0$）。代入递推、约去 $e^{g_t}$：

$$
\boxed{\;P_t = \bigl( I - \beta_t k_t k_t^{\top} \bigr) P_{t-1} + e^{-g_t}\beta_t\, k_t v_t^{\top}\;}
$$

> **在 $P$ 坐标系里，遗忘门从状态转移算子中完全消失**，只剩下对写入量的一个标量缩放。

这一步是后面一切的基础，两个后果：

1. 那串连乘里真正难处理的只有 $\prod(I-\beta kk^{\top})$，门可以最后再加回来；
2. 误差传播分析只需要研究擦除算子（见 §8.1），门贡献一个干净的指数因子。

### 3.4 第二步：秩一展开，得到单位下三角方程组（UT transform / WY）

上式是纯秩一更新：$P_t = P_{t-1} + k_t\tilde{u}_t^{\top}$，其中

$$
\tilde{u}_t = \beta_t\bigl( e^{-g_t}v_t - P_{t-1}^{\top}k_t \bigr)
$$

把 $P_{t-1} = P_0 + \sum_{j<t}k_j\tilde{u}_j^{\top}$ 代回去，$\tilde{u}_t$ 就只依赖它之前的 $\tilde{u}_j$：

$$
\tilde{u}_t + \beta_t\sum_{j<t}(k_t\!\cdot\! k_j)\,\tilde{u}_j = \beta_t\bigl( e^{-g_t}v_t - P_0^{\top}k_t \bigr)
$$

**这是一个关于 $\{\tilde{u}_t\}$ 的下三角线性方程组**，写成矩阵形式并求解：

$$
\boxed{\;
A_{\mathrm{raw}} = \Bigl( I + \mathrm{tril}\bigl(\mathrm{diag}(\beta)KK^{\top},\,-1\bigr) \Bigr)^{-1},
\qquad
\tilde{U} = A_{\mathrm{raw}}\,\mathrm{diag}(\beta)\bigl( \mathrm{diag}(e^{-g})V - K S_0 \bigr)
\;}
$$

这就是 **UT transform / WY 表示**。三个要点：

- $I + \mathrm{tril}(\cdot,-1)$ 是**单位下三角**（对角线严格为 1），行列式恒为 1，**逆一定存在**，且求逆过程**不需要除法**——这是实现上一个不小的便利；
- $A_{\mathrm{raw}}$ 里**没有遗忘门**——门只留在等式右边的 $\mathrm{diag}(e^{-g})$ 上。这个事实就是 §7.1 那个杠杆；
- 待求逆矩阵的元素有界：$\lvert (\mathrm{diag}(\beta)KK^{\top})_{ij}\rvert = \beta_i\lvert k_i\!\cdot\!k_j\rvert \le 1$，因为 $\lVert k\rVert_2=1$ 且 $\beta<1$。这是低精度求逆之所以敢做的依据。

### 3.5 第三步：换回原坐标，门以相似变换的形式回来

定义 $V_{\mathrm{new}} = \mathrm{diag}(e^{g})\tilde{U}$，把 $P$ 换回 $S$。整理后得到三个式子（推导只是把对角阵挪来挪去）：

$$
\textbf{①}\quad V_{\mathrm{new}} = \bigl( \Lambda \circ A_{\mathrm{raw}} \bigr)\,\mathrm{diag}(\beta)\bigl( V - \mathrm{diag}(e^{g})\,K\,S_{0} \bigr)
$$

$$
\textbf{②}\quad S_{C} = e^{g_C} S_{0} + K^{\top}\mathrm{diag}\!\left(e^{\,g_C-g}\right) V_{\mathrm{new}}
$$

$$
\textbf{③}\quad O = \underbrace{\mathrm{diag}(e^{g})\,Q\,S_{0}}_{\text{chunk 间}} + \underbrace{\bigl( \Lambda \circ (QK^{\top}) \bigr) V_{\mathrm{new}}}_{\text{chunk 内}}
$$

其中 $\Lambda\circ A_{\mathrm{raw}}$ 这一项的来历值得记住：

$$
\mathrm{diag}(e^{g})\,A_{\mathrm{raw}}\,\mathrm{diag}(e^{-g}) \;=\; \Lambda \circ A_{\mathrm{raw}}
$$

**它是一个相似变换**——门是"换坐标系时自然产生的"，而不是本来就长在矩阵里的。

### 3.6 结果：七个矩阵乘

预处理（不依赖 $S_{\mathrm{prev}}$）：$g = \mathrm{cumsum}(\log\alpha)$，$\Lambda$，$A_{\mathrm{raw}}$。

主体三式 ①②③ 展开成七个矩阵乘：

| # | 矩阵乘 | 规模 | 出现在 |
|---|---|---|---|
| 1 | $KK^{\top}$ | $[C,K]\times[K,C]$ | ② 的预处理 |
| 2 | $QK^{\top}$ | $[C,K]\times[K,C]$ | ③ intra |
| 3 | $K S_{\mathrm{prev}}$ | $[C,K]\times[K,V]$ | ① |
| 4 | $Q S_{\mathrm{prev}}$ | $[C,K]\times[K,V]$ | ③ inter |
| 5 | $(\Lambda\circ A_{\mathrm{raw}})\,\mathrm{diag}(\beta)(V-\ldots)$ | $[C,C]\times[C,V]$ | ① |
| 6 | $(\Lambda\circ QK^{\top})\,V_{\mathrm{new}}$ | $[C,C]\times[C,V]$ | ③ intra |
| 7 | $K^{\top}\mathrm{diag}(\cdot)V_{\mathrm{new}}$ | $[K,C]\times[C,V]$ | ② |

这七个乘法（外加一次 $C\times C$ 三角求逆和 $O(C^2)$ 次 $\exp$）就是三个库共同的工作量基准。请记住其中 **3、4、7 是 $C\!\times\!d\!\times\!d$ 的"大"乘法**，1、2、5、6 是 $C\!\times\!C\!\times\!d$ 的"小"乘法——$C=64,d=128$ 时前者各占两倍 MAC。

### 3.7 掩码与 scale 约定（易错点）

| 位置 | 掩码 |
|---|---|
| $\mathrm{tril}(\mathrm{diag}(\beta)KK^{\top},-1)$ | **严格**下三角（$j<t$），对角线由 $I$ 提供 |
| $\Lambda\circ(QK^{\top})$、$\Lambda\circ A_{\mathrm{raw}}$ | 下三角**含对角线**（$j\le t$） |

另外 $q$ 还要乘 $\text{scale}=1/\sqrt{K}$，实现中折进 $Q$ 或 $\Lambda\circ QK^{\top}$。

**还有一个纯实现层面的陷阱**：$\Lambda_{ij}=\exp(g_i-g_j)$ 在越界位置（tail chunk 的 padding 行）如果 $g$ 读成 0 而 $g_j$ 是有效负值，$\exp(0-g_j)$ 可能溢出成 $+\infty$，随后 $0\times\infty = \mathrm{NaN}$。正确做法是把边界掩码**并进 `where` 的条件里**，在乘法之前就把值置零，而不是乘完再掩。三个库都踩过这个坑。

### 3.8 $w$ / $u$ 形式与合并形式

FLA 最早的写法把 ① 拆成两步（记 $A_{\mathrm{gated}} = \Lambda\circ A_{\mathrm{raw}}$）：

$$
w = A_{\mathrm{gated}}\,\mathrm{diag}(\beta)\,\mathrm{diag}(e^{g})K,
\qquad
u = A_{\mathrm{gated}}\,\mathrm{diag}(\beta)V,
\qquad
V_{\mathrm{new}} = u - w S_{\mathrm{prev}}
$$

用分配律一合并就回到 ①。所以 **$w$ / $u$ 只是 ① 展开后的形式**。两种形式的代价对比：

| | $w$/$u$ 形式 | 合并形式（①） |
|---|---|---|
| 矩阵乘次数 | 4（$A\!\cdot\!\beta e^g K$、$A\!\cdot\!\beta V$、$wS$、减法） | 2（$KS_{\mathrm{prev}}$、$A(V-KS)$） |
| 中间张量 | $w\in\mathbb{R}^{C\times K}$、$u\in\mathbb{R}^{C\times V}$ 两个 | 无 |
| 依赖链 | $A \to w \to wS \to V_{\mathrm{new}}$ | $A$ 与 $KS$ 可并行准备，再一次乘 |
| 反向传播 | $w$ 会被反向多次复用，物化一次读多次更省 | 需重算 |

**合并形式适合推理，$w/u$ 形式适合训练。** FlashQLA 与 FlashInfer 走合并形式；FLA 保留 $w/u$ 并物化到 HBM，因为它的 backward 要读 $w$ 三次（详见 FLA 文档）。这是三个库第一处结构性分歧，且它的根源是**训练 vs 推理的定位差异**，不是谁更聪明。

### 3.9 Chunk 化的代价

原始递推里根本不存在、完全由 chunk 化引入的东西：

- $KK^{\top}$（$C\times C$）及其**三角求逆**
- $\Lambda$（$C\times C$ 次 $\exp$）
- $V_{\mathrm{new}}$、$K^{\top}V_{\mathrm{new}}$ 等中间量

换来的是能用 Tensor Core。**这是一笔"多算一些换吞吐"的交易**，只在 $C$ 足够大时划算。

### 3.10 正确性自检清单

实现新 kernel 时按这个顺序对齐，能省掉大量二分调试（这也是三个库测试文件里的实际做法）：

1. **$C=1$ 退化**：chunk 公式在 $C=1$ 时必须逐字退化成 §1.4 的递推（$A_{\mathrm{raw}}=I$，$\Lambda=1$）。
2. **$\alpha\equiv 1$（无遗忘）**：$g\equiv 0$，$\Lambda\equiv 1$，公式退化成纯 DeltaNet。
3. **$\beta\equiv 0$（不写入）**：$V_{\mathrm{new}}=0$，$S_C = e^{g_C}S_0$，$O$ 只剩 inter 项。
4. **$S_0=0$**：③ 的 inter 项与 ① 的 $KS_0$ 项整体消失（这个性质也是一处实打实的优化，见 FlashInfer 文档）。
5. **两个 chunk 拼接 = 一个双倍长 chunk**：验证 $S$ 的传递与 $\Lambda$ 的跨 chunk 约定。
6. **varlen 尾块**：序列长度非 $C$ 整数倍时，padding 行不得污染 $KK^{\top}$、$\Lambda$、$K^{\top}V_{\mathrm{new}}$。

---

## 4. Chunk size 为什么是 64

$C$ 在两股相反的力之间取平衡。

**压 $C$ 变大的力**：串行步数是 $T/C$，$C$ 越大串行链越短（§5 的核心矛盾）；而且 $C$ 是几个矩阵乘的 M 维和收缩维，太小的话矩阵乘单元根本吃不饱（极端情况 $C=1$ 就退化回纯递推，等于没做 chunk 化）。

**压 $C$ 变小的力**：chunk 化引入的额外计算随 $C$ 线性增长。把七个矩阵乘按维度分类，每 token 的计算量是

$$
\underbrace{4Cd}_{\text{chunk 内}(C\times C\text{ 项，即 }1,2,5,6)} + \underbrace{3d^2}_{\text{chunk 间}(d\times d\text{ 项，即 }3,4,7)}
\qquad\Longrightarrow\qquad
\text{冗余率} = 1 + \frac{4C}{3d}
$$

（分母 $3d^2$ 是纯递推每 token 的理论下界。）关键是**冗余率只取决于 $C/d$**：

| $C$（$d=128$） | 冗余率 | $\Lambda$ 的 $\exp$ 次数/chunk | 求逆规模 |
|---|---|---|---|
| 32 | 1.33× | 1024 | $32^2$ |
| **64** | **1.67×** | **4096** | $64^2$ |
| 128 | 2.33× | 16384 | $128^2$ |
| 256 | 3.67× | 65536 | $256^2$ |

再往上这笔交易就亏了。此外 $\Lambda$ 的 $\exp$ 数量、三角求逆的规模（$O(C^3)$ 工作量）、片上存储占用也都随 $C$ 超线性增长。

**平衡点**落在 $C = d/2 = 64$：冗余率还能接受，同时正好对齐硬件原生的矩阵乘 tile（Hopper 的 warpgroup MMA M 维固定为 64；Blackwell 的 `tcgen05` 同样以 64 为基本 M 粒度）。FLA、FlashQLA、FlashInfer 三家在 SM90/SM100 上都取 64。

**它确实是可调的**：片上资源更紧张的架构上可以改用 32——牺牲串行步数换更低的冗余率和更小的片上占用。所以 64 不是数学常数，是当前 $d=128$ + 当前硬件下的平衡点，$d$ 变了它就会变。

---

## 5. 依赖结构：唯一的串行环节

把 §3.5 的 ①②③ 拆成 5 个环节，标注跨 chunk 依赖：

| 环节 | 内容 | 需要 $S_{\mathrm{prev}}$ 吗 |
|---|---|---|
| ① 门前处理 | $g=\mathrm{cumsum}(\log\alpha)$、$\Lambda$ | ❌ **所有 chunk 可同时算** |
| ② UT transform | $KK^{\top}\to A_{\mathrm{raw}}=(I+\mathrm{tril}(\cdot))^{-1}$ | ❌ **所有 chunk 可同时算** |
| ③ value 修正 | $V_{\mathrm{new}}=(\Lambda\circ A_{\mathrm{raw}})\mathrm{diag}(\beta)(V-\mathrm{diag}(e^{g})KS_{\mathrm{prev}})$ | ✅ |
| ④ **状态递推** | $S_{\mathrm{next}}=e^{g_C}S_{\mathrm{prev}}+K^{\top}\mathrm{diag}(\cdot)V_{\mathrm{new}}$ | ✅ **严格串行** |
| ⑤ 输出组装 | $O=\mathrm{diag}(e^{g})QS_{\mathrm{prev}}+(\Lambda\circ QK^{\top})V_{\mathrm{new}}$ | ✅ |

**核心结论**：

> **④ 是唯一真正的串行环节。**
> ①② 天然是全 chunk 并行的；③⑤ 只要拿到 $S_{\mathrm{prev}}$ 就能并行。
> 所以 **GDN prefill 优化的第一命题 = 怎么让 ④ 不再是一条长度为 $T/C$ 的串行链。**

这张表同时也是**实现层面的分工蓝图**。三个库的 kernel 边界、warp 分组、缓冲级数几乎都能追回这张表：不依赖状态的工作可以任意超前发射，依赖状态的工作只能一步步走；把用同一批中间量的环节放进同一个执行单元。读三篇实现文档时请随时回看这里。

---

## 6. Prefill 的性能问题

### 6.1 算术强度：落在算力受限一侧

按 $C = 64$、$d = K = V = 128$、bf16 计算，单 chunk 单 head：

**计算量**（MAC 数）

| 矩阵乘 | 规模 | MAC |
|---|---|---|
| $KK^{\top}$ | $C\!\times\! C\!\times\! d$ | 524,288 |
| $QK^{\top}$ | $C\!\times\! C\!\times\! d$ | 524,288 |
| $K S_{\mathrm{prev}}$ | $C\!\times\! d\!\times\! d$ | 1,048,576 |
| $Q S_{\mathrm{prev}}$ | $C\!\times\! d\!\times\! d$ | 1,048,576 |
| $A\,(V-KS)$ | $C\!\times\! C\!\times\! d$ | 524,288 |
| $(\Lambda\circ QK^{\top})V_{\mathrm{new}}$ | $C\!\times\! C\!\times\! d$ | 524,288 |
| $K^{\top}V_{\mathrm{new}}$ | $d\!\times\! C\!\times\! d$ | 1,048,576 |
| **合计** | | **5,242,880 MAC $\approx$ 10.5 MFLOP** |

**访存量**：$Q,K,V$ 各 $64\times128\times2\,\text{B}$，合计约 $49\,\text{KB}$（状态跨 chunk 常驻片上，摊销后可忽略）。

$$
\text{算术强度} \;\approx\; \frac{10.5\times10^{6}}{49{,}152} \;\approx\; 213\ \text{FLOP/Byte}
$$

现代加速器的算力/带宽比在 $O(100\sim300)$ FLOP/Byte 量级，所以 **GDN prefill 恰好落在算力受限一侧**（前提是中间量不落显存，见下）。

> **推论**：prefill 的瓶颈既不是算力不足，也不是带宽不足，而是**能不能把机器填满**。

注意这个结论有个隐含前提：**七个矩阵乘之间的中间量必须留在片上**。如果像 FLA 那样把 $A$、$w$、$u$、$h$、$V_{\mathrm{new}}$ 全部往返 HBM，实测访存量会涨到理论值的十倍以上，算术强度掉到个位数——一个算力受限的公式被实现成了带宽受限的 kernel。**这是第二命题：融合。**

### 6.2 真正的问题：并行度只有 $B\times H_v$

环节 ④ 的串行性意味着：**一个序列的一个 head 只能由一个计算单元从头走到尾**。可用的并行维度只剩

$$
P \;=\; B \times H_v
$$

用 §2.2 的真实配置（单请求 $B=1$）：

| 场景 | 每卡 $H_v$ | 并行任务数 |
|---|---|---|
| Qwen3.5-397B，TP8 | 8 | **8** |
| Qwen3.5-397B，TP4 | 16 | 16 |
| Qwen3.5-397B，TP2 | 32 | 32 |
| Qwen3.5-397B，TP1 | 64 | 64 |
| Qwen3.5-35B，TP1 | 32 | 32 |
| Qwen3.5-2B，TP1 | 16 | 16 |

而一张现代数据中心 GPU 有一百多个 SM。**8 个任务对上百个执行单元，绝大部分在空转。**

雪上加霜的是三件事：

1. **张量并行切的正是 head 维度**，切得越多并行度越低。长上下文 + 高 TP 是最坏组合，也恰恰是最常见的组合。
2. **序列越长越糟**：$T$ 增大不增加并行度，只把每个任务的串行链拉长（$T=32\text{k}$，$C=64$ ⟹ 512 步依赖链）。
3. **单请求 prefill 无法靠 batch 救**。$B$ 大时问题自动缓解，但 TTFT 敏感的首包场景恰恰是 $B=1$。

如果只能记住本文一句话，记这个：**GDN prefill 的头号问题是并行度饥饿，不是算力也不是带宽。** §6.1 的融合问题是常数因子级的（几倍），并行度饥饿是数量级级的（十几倍）。

### 6.3 次要问题

| 问题 | 说明 | 量级 |
|---|---|---|
| 三角求逆是"异类" | $(I+L)^{-1}$ 教科书解法是 $C$ 步前代，依赖链长 $C$，跑标量/向量单元，用不上 Tensor Core，而且卡在关键路径上（$A \to V_{\mathrm{new}} \to S_{\mathrm{next}}$） | FLOP 占比 <1%，但延迟占比可以很高 |
| $\Lambda$ 的 $\exp$ 开销 | 每 chunk $C\times C = 4096$ 次 $\exp$，特殊函数单元吞吐远低于矩阵乘单元；而且 $\Lambda$ 在公式里出现两次 | 常数因子 |
| 中间量落显存 | §6.1 末尾 | 数倍 |
| varlen 尾块 | 非 $C$ 整数倍的尾块要正确处理边界，packed batch 下还会跨序列越界 | 正确性问题，不是性能问题 |
| 前处理碎 kernel | $l2norm$、$\sigma(b)$、$\alpha$、cumsum 如果各自一个 kernel，全是访存受限的小 kernel，launch 开销与带宽双输 | 常数因子 |

---

## 7. 四个可用的数学杠杆

以下四条恒等式/性质是三个库全部优化的数学素材。**每一条都在算法层面成立，与硬件无关**；三个库的差别在于用了哪几条、以及怎么落到硬件上。

### 7.1 杠杆一：门与求逆可交换（相似变换）

$$
\boxed{\;
\Bigl( I + \mathrm{tril}\bigl( \Lambda\circ(\mathrm{diag}(\beta)KK^{\top}),\,-1 \bigr) \Bigr)^{-1}
\;=\;
\Lambda \circ \Bigl( I + \mathrm{tril}\bigl( \mathrm{diag}(\beta)KK^{\top},\,-1 \bigr) \Bigr)^{-1}
\;}
$$

**证明**：记 $D=\mathrm{diag}(e^{g})$，则 $\Lambda\circ X = D X D^{-1}$ 对任意 $X$ 成立（因为 $(DXD^{-1})_{ij}=e^{g_i}X_{ij}e^{-g_j}$）。对角阵 $D$ 与 $\mathrm{diag}(\beta)$ 可交换，且 $\mathrm{tril}$ 与相似变换 $D(\cdot)D^{-1}$ 可交换（逐元素缩放不改变零模式），$I = DID^{-1}$。于是待求逆矩阵 $= D(I+\mathrm{tril}(\mathrm{diag}(\beta)KK^{\top},-1))D^{-1}$，而 $(DXD^{-1})^{-1} = DX^{-1}D^{-1} = \Lambda\circ X^{-1}$。∎

**这意味着有两种等价做法**：

| 做法 | 顺序 | 后果 |
|---|---|---|
| **门在外** | 先求 gate-free 的 $A_{\mathrm{raw}}$，再逐元素乘 $\Lambda$ | $C\times C$ 次 $\exp$ 从求逆的**关键路径上移走**；被求逆矩阵元素模长 $\le 1$，数值动态范围最小；$A_{\mathrm{raw}}$ 与门无关，可跨用途复用 |
| **门在内** | 先把 $\Lambda,\beta$ 乘进矩阵再求逆 | 那批 $\exp$ 留在关键路径上；矩阵元素含 $e^{g_i-g_j}$，动态范围更大 |

§3.4 的自然推导给出的**就是"门在外"**——$A_{\mathrm{raw}}$ 是在 $P$ 坐标系里得到的，本来就 gate-free；$\Lambda$ 是换回原坐标时才产生的。"门在内"是把它主动乘回去。

**谁用了**：

| 实现 | 门的位置 | 动机 |
|---|---|---|
| **FlashQLA** | **门在外** | 求逆里没有 $\exp$；且 $A_{\mathrm{raw}}$ 门无关 ⟹ **一份 `A` 同时服务前向、反向和 CP 的预热 pass** |
| **FLA** | 门在内（把 $\Lambda$ 乘进 $KK^{\top}$ 再求逆） | 一趟算完，无复用需求 |
| **FlashInfer 非 CP 路径** | 门在内 | 同上——一趟算完就丢 |
| **FlashInfer CP 路径** | **门在外** | 让 $T$ 门无关 ⟹ **每 64-token 块只算一次，被后续两个 kernel 复用** |

**注意最后两行**：同一个库的两条路走了相反的方向，而分界线恰好是"$T$ 有没有跨 pass 复用的需求"。**这说明这个杠杆的价值不在"门在内还是在外"本身，而在"门无关性是否换来了复用"。**

所以要修正一个常见判断：$A_{\mathrm{raw}}$ 门无关带来的"算一次多处复用"收益，**不只对训练有意义**（forward / backward 共用同一个 $A$）。FlashInfer 的 CP 路径是一个纯推理的反例——**跨 kernel 复用同样需要门无关**。只有在"单 kernel 一趟算完、无任何复用"的场景下，这处分歧才退化成"少一批 $\exp$ + 数值范围更小"两个常数因子。

### 7.2 杠杆二：三角求逆的分块 Schur 补（工作量换深度）

$(I+L)^{-1}$ 的教科书解法是 $C$ 步前代，**依赖链长度 $C$**。改成按 $2\times2$ 分块递归：

```
        ┌              ┐                    ┌                                    ┐
        │  L11     0   │                    │  L11⁻¹                0            │
  L  =  │              │        L⁻¹  =      │                                    │
        │  L21    L22  │                    │  −L22⁻¹·L21·L11⁻¹    L22⁻¹         │
        └              ┘                    └                                    ┘
```

即逆的三个非零块为

$$
(L^{-1})_{11} = L_{11}^{-1}, \qquad (L^{-1})_{22} = L_{22}^{-1}, \qquad (L^{-1})_{21} = -\,L_{22}^{-1}\,L_{21}\,L_{11}^{-1}
$$

**两个对角小块各自递归求逆（互相独立，可并行），再用两次矩阵乘把左下角块补上。** 一般地，对 $n\times n$ 分块的单位块下三角矩阵：

$$
(L^{-1})_{ij} \;=\; -\,L_{ii}^{-1}\!\!\sum_{j\le m<i}\! L_{im}\,(L^{-1})_{mj}
$$

按反对角波前（$i-j$ 从小到大）求解，波前内部全部独立。

- 依赖链长度从 $C$ 降到 $O(\log C)$ 级（若递归到底）或"块内前代 + $O(\log)$ 层合并"
- 合并步骤全是矩阵乘，能上 Tensor Core
- **总运算量反而增加**——典型的"用工作量换深度"

三家都做了这件事，只在分块粒度（$8\times8$ / $16\times16$）、合并层数和合并精度上有差别。

### 7.3 杠杆三：chunk 的作用是仿射变换，仿射变换可结合（→ 精确并行扫描）

由 §3.5 的 ①② 消去 $V_{\mathrm{new}}$：

$$
S_{\mathrm{next}} = e^{g_C} S_{\mathrm{prev}} + K^{\top}\mathrm{diag}\!\left(e^{g_C-g}\right)\bigl(\Lambda\circ A_{\mathrm{raw}}\bigr)\mathrm{diag}(\beta)\bigl( V - \mathrm{diag}(e^{g}) K S_{\mathrm{prev}} \bigr)
$$

$S_{\mathrm{prev}}$ 出现在两处，都是线性的。整理成

$$
\boxed{\; S_{\mathrm{out}} \;=\; M\, S_{\mathrm{in}} \;+\; N \;}
$$

$$
M = e^{g_C} I - K^{\top}\mathrm{diag}\!\left(e^{g_C-g}\right)\bigl(\Lambda\circ A_{\mathrm{raw}}\bigr)\mathrm{diag}(\beta)\,\mathrm{diag}(e^{g})\,K
\;\in \mathbb{R}^{K\times K}
$$

$$
N = K^{\top}\mathrm{diag}\!\left(e^{g_C-g}\right)\bigl(\Lambda\circ A_{\mathrm{raw}}\bigr)\mathrm{diag}(\beta)\,V
\;\in \mathbb{R}^{K\times V}
$$

（左乘 / 右乘的约定视状态布局而定，结构相同。）

**关键性质**：仿射变换的复合仍是仿射变换，

$$
M_2\bigl( M_1 S + N_1 \bigr) + N_2 \;=\; \bigl( M_2 M_1 \bigr) S + \bigl( M_2 N_1 + N_2 \bigr)
$$

**满足结合律 $\Longrightarrow$ 可以做并行前缀和（parallel scan）**，于是环节 ④ 从长度 $T/C$ 的串行链变成三段式：

1. **并行**：把序列切成 $N_{\mathrm{shard}}$ 段，所有段**同时**算出各自的 $(M_c, N_c)$——互不依赖；
2. **串行**：$S_{c+1} = M_c S_c + N_c$ 扫一遍，但这是在**粗粒度段**上扫，串行长度从 $T/C$ 降到 $N_{\mathrm{shard}}$；
3. **并行**：每段拿到**精确**的起始状态，独立算 ①③⑤。

这与 Mamba 的 selective scan 是同一套思路，区别只在 GDN 的转移是 $K\times K$ 矩阵而非标量，所以扫描每一步是一次 $[K,K]\times[K,V]$ 矩阵乘，而且 $M_c$ 本身要显式构造并存储（$128\times128$，不小）。

| | |
|---|---|
| 依据 | 仿射变换的**结合律**（纯代数结构） |
| 结果 | **代数上精确**（见下方警告） |
| 额外开销 | 显式构造并存储 $M_c$ + 多两趟遍历 |
| 失效条件 | 无 |

> ⚠️ **"精确"必须区分代数与浮点两层。** 仿射复合在数学上严格精确，fixup 恢复的就是真正的 chunk 边界状态。但浮点实现会引入非 CP 路径根本不存在的新误差：$M_c$ 的连乘是额外的矩阵乘，而 fixup 的扫描为了吞吐往往跑在 TF32 上。
> FlashInfer 的实测就是这样：它的 fixup 阶段测试容差是 $2\times10^{-3}$，CP vs 非 CP 的端到端容差是 $4\times10^{-2}$（见其文档 §10.6、§13.2）。
> **对照 §7.4 那条"近似"路线的 $4.5\times10^{-5}$——精确算法的浮点实现反而误差更大。** 所以选路时不能简单用"精确 vs 近似"决断，要看**实现的实际数值行为**。

**谁用了**：FlashInfer 的 CP 路径；FlashQLA 也用同一个 $M$ 做精确兜底；FLA 的 intracard CP 用同样的数学（它把 $h$ 与 $m$ 打包在一个 buffer 里）。**这一条是三家共有的"精确"武器，但只有 FlashInfer 把它当主路径。**

### 7.4 杠杆四：擦除算子非扩张 + 门指数衰减（→ 近似截断）

考察状态误差 $\Delta S$ 在 chunk 间的传播。从 §3.3 的 $P$ 空间递推

$$
P_t = \bigl( I - \beta_t k_t k_t^{\top} \bigr) P_{t-1} + \underbrace{e^{-g_t}\beta_t k_t v_t^{\top}}_{\text{与 } P_{t-1} \text{ 无关}}
$$

误差只经过擦除算子：

$$
\Delta P_t = \bigl( I - \beta_t k_t k_t^{\top} \bigr)\, \Delta P_{t-1}
$$

由于 $\lVert k_t\rVert_2 = 1$（L2 norm）且 $\beta_t = \sigma(\cdot) \in (0,1)$，矩阵 $I - \beta_t k_t k_t^{\top}$ 的特征值为

$$
\{\, \underbrace{1-\beta_t}_{\text{沿 } k_t \text{ 方向}},\ \underbrace{1,\ 1,\ \dots,\ 1}_{K-1 \text{ 个正交方向}} \,\},
\qquad 0 < 1-\beta_t < 1
$$

它是对称的，故其**谱范数恰好等于 1**，是**非扩张**（non-expansive）算子：$\lVert \Delta P_t \rVert \le \lVert \Delta P_{t-1} \rVert$。换回原坐标 $S_t = e^{g_t} P_t$：

$$
\boxed{\;
\lVert \Delta S_t \rVert \;\le\; \exp\!\Bigl( \sum_{i \le t} \log \alpha_i \Bigr)\cdot \lVert \Delta S_0 \rVert
\;=\; e^{\,g_t}\,\lVert \Delta S_0 \rVert
\;}
$$

> **状态误差只会被遗忘门单调衰减，永远不会被 delta rule 放大。**

**这条性质有两个完全不同的用途**：

1. **近似上下文并行**（FlashQLA）：既然久远历史的影响会指数衰减，那就允许每段"假装从零状态开始"，只往前多算几个 chunk 预热。见 §8.2 的误差分析和 FlashQLA 文档。
2. **状态低精度 I/O 的安全性**（FlashInfer）：状态在显存里存 bf16/fp8 引入的量化噪声，进入状态后同样按 $e^{\sum g}$ 衰减，**有界且自愈**，不会像纯累加型线性注意力那样无限累积。

| | |
|---|---|
| 依据 | 状态的**指数遗忘性质**（模型训练出来的性质，非纯代数） |
| 结果 | 近似（用途 1）/ 安全性论证（用途 2） |
| 额外开销 | 只多预热几个 chunk，近似**几乎免费** |
| 失效条件 | 模型的门普遍接近 1（几乎不遗忘）⟹ 退回精确路径，加速归零 |

**注意 §7.3 与 §7.4 是两条独立的路**：前者是代数恒等式，任意输入都精确；后者是模型性质，便宜但有近似。一个实现可以同时用两条（近似为主、精确兜底），这正是 FlashQLA 的结构。

---

## 8. 数值性质

### 8.1 误差不放大定理

见 §7.4 的方框式。要强调的是**这是绝对误差界**：它约束的是误差的绝对大小 $\lVert\Delta S_t\rVert$，右边没有除以任何东西。

### 8.2 从绝对界到相对误差（近似 CP 的真正依据）

设某段（shard）从零状态起步，但往前多算若干 chunk 做"预热"，预热区间覆盖的累计对数衰减记为 $\sum g$。shard 的"初始误差"恰好就是被丢弃的那段真实状态：

$$
\Delta S_0 = 0 - S_{\text{真}}(\text{预热起点}) = -\,S_{\text{真}}(\text{预热起点})
$$

代入绝对界，传到 shard 起点：

$$
\lVert \Delta S(\text{shard 起点}) \rVert \;\le\; e^{\sum g}\cdot \lVert S_{\text{真}}(\text{预热起点}) \rVert
$$

而 shard 起点的真实状态可拆成两块：

$$
S_{\text{真}}(\text{shard 起点}) = \underbrace{(\text{预热区间的转移})\,S_{\text{真}}(\text{预热起点})}_{\text{被丢掉的部分}} + \underbrace{\text{预热期间新写入的内容}}_{\text{保留下来了}}
$$

于是相对误差为

$$
\boxed{\;
\frac{\lVert \Delta S \rVert}{\lVert S_{\text{真}} \rVert}
\;\le\;
e^{\sum g}\cdot \frac{\lVert S_{\text{真}}(\text{预热起点}) \rVert}{\lVert S_{\text{真}}(\text{shard 起点}) \rVert}
\;}
$$

**只有当后面那个范数比是 $O(1)$ 时，相对误差才 $\approx e^{\sum g}$。** 这在训练好的模型里成立——GDN 的状态处于稳态（新写入与遗忘衰减平衡），范数大致平稳；但这是一个**额外的经验假设**，不是从算子范数直接得出的。若某 head 在预热窗口内状态范数发生数量级塌缩，相对误差会被放大。

取阈值 $\sum g < -10$：

$$
\frac{\lVert \Delta S \rVert}{\lVert S_{\text{真}} \rVert} \;\lesssim\; e^{-10} \approx 4.5\times10^{-5}
$$

注意阈值本身是个**相对判据**——$\sum g$ 度量的正是"旧状态还剩多少比例活着"。所以这套机制的**意图是控相对误差，但严格证明只覆盖到绝对界**。

### 8.3 与浮点分辨率的对照

| 参照 | 相对分辨率 | 与 $4.5\times10^{-5}$ 的关系 |
|---|---|---|
| bf16（8 bit 尾数） | $3.9\times10^{-3}$ | 近似误差**低两个数量级**，被舍入吞掉 |
| fp16（11 bit 尾数） | $4.9\times10^{-4}$ | 低一个数量级 |
| fp32（24 bit 尾数） | $6\times10^{-8}$ | **高三个数量级** |

准确表述：**bf16 / fp16 输出下该近似不可观测；但相对 fp32 精确递推它是有偏的。**

### 8.4 范数界的两点局限

1. **相对误差的成立依赖 §8.2 那个范数比假设**，严格证明只给到绝对界。
2. **范数界是 Frobenius / 谱意义上的全局界，不是逐元素保证。** 误差传到输出是 $\Delta o = \Delta S^{\top} q$，若 $q$ 恰好指向被丢弃分量占主导的方向，或 $o$ 本身因抵消而很小，该分量的相对误差可显著大于 $4.5\times10^{-5}$。

### 8.5 精度分配的一般原则

三个库的精度策略高度一致，可以概括成三条：

| 量 | 精度 | 理由 |
|---|---|---|
| $g$（$\log\alpha$ 的 cumsum）、$\beta$ | **fp32** | $g$ 是要取 $\exp$ 的指数，且是累加量；这里省精度会直接放大成乘性误差 |
| 状态 $S$ 的**片上累加** | **fp32** 恒定 | 它是跨 $T/C$ 步的累加器，唯一不能降的地方 |
| 所有矩阵乘的累加器 | fp32 | 硬件本来就是 bf16×bf16→fp32 |
| $Q,K,V$、$A^{-1}$、$V_{\mathrm{new}}$ 等操作数 | bf16 / fp16 | 走 Tensor Core 的必要条件 |
| 状态在**显存里**的表示 | fp32 / bf16 / fp16 / fp8 均可 | 有 §7.4 用途 2 的自愈性质保底；这是 **I/O 量化，不是计算量化** |
| 三角求逆的中间累加 | fp32 | 依赖链长，且结果要被后续所有环节乘 |

一个实用的经验：**分不清该用什么精度时，看这个量会不会被累加很多次、或者会不会被送进 $\exp$。** 会的话上 fp32。

---

## 9. Decode：$C=1$ 的退化

Decode 阶段每步只有一个 token（MTP / speculative decoding 时是少数几个），$C=1$：

$$
A_{\mathrm{raw}} = I,\quad \Lambda = 1,\quad
S_{\mathrm{next}} = \alpha\bigl(I-\beta kk^{\top}\bigr)S_{\mathrm{prev}} + \beta k v^{\top},\quad
o = S_{\mathrm{next}}^{\top}q
$$

七个矩阵乘退化成两个矩阵-向量乘（$S^{\top}k$、$S^{\top}q$）加一个秩一更新。**性质完全反转**：

| | Prefill | Decode |
|---|---|---|
| 每 token 计算量 | $\sim 4Cd+3d^2$ | $\sim 3d^2$ |
| 每 token 访存量 | $Q,K,V$ 三行 | **整个状态 $K\!\times\!V$** |
| 算术强度 | $O(200)$ FLOP/Byte，**算力受限** | $O(1)$ FLOP/Byte，**彻底带宽受限** |
| 并行度 | $B\times H_v$（饥饿） | $B\times H_v$，但 $B$ 通常很大（不饥饿） |
| 优化目标 | 填满机器、缩短串行链 | **压状态读写带宽** |

所以 decode kernel 的设计重心跟本文讲的完全不同：状态布局（是否预转置以便 $S^{\top}q$ 连续访问）、状态量化（bf16/fp8 直接砍带宽）、分页状态池、MTP 多 token 摊销。三个库的 decode 实现都独立于 prefill 实现，本知识库只在需要对照时提及。

---

## 10. 符号表与库间 API 差异

### 10.1 符号表

| 本文 | 含义 | FLA 常用名 | FlashQLA / FlashInfer 常用名 |
|---|---|---|---|
| $C$ | chunk 长度 | `BT`, `chunk_size` | `BT`, `CHUNK_SIZE` |
| $K=V=d$ | head dim | `K`, `V` | `head_dim`, `D` |
| $H_v$ | v-head 数（= 状态个数） | `HV` | `h_v`, `num_v_heads` |
| $H_{qk}$ | k/q-head 数 | `H` | `h_qk` |
| $\alpha_t$ | 遗忘门（线性空间，$\in(0,1)$） | 由 `g` + `A_log` + `dt_bias` 算出 | `g`（FlashInfer 的 `g` 直接就是 $\alpha$） |
| $g_t$ | $\log\alpha$ 的 chunk 内 cumsum | `g`（已 cumsum，fp32） | `g_cumsum` |
| $\beta_t$ | 写入门 | `beta` | `beta` |
| $A_{\mathrm{raw}}$ | gate-free 的三角逆 | — | `A_inv`, `Tinv` |
| $\Lambda\circ A_{\mathrm{raw}}$ | 带门的三角逆 | `A`（FLA 直接存这个） | `A_gated` |
| $V_{\mathrm{new}}$ | 修正后的 value | `v_new` | `v_new` |
| $S$ | 状态 | `h` / `initial_state` / `final_state` | `state`, `h0` |
| $M, N$ | 段级仿射变换 | `m`（打包在 `hm` 里） | `M`, `transition` |

### 10.2 三处必须注意的 API 语义差异

踩过一次就会记住的三个坑：

1. **门是线性空间还是对数空间。** FlashInfer 的 `g` 参数是**线性空间的 $\alpha\in(0,1)$**（全 1 表示不衰减）；传入 log-space 的值（比如 `logsigmoid` 的输出）会直接产出 NaN。FLA 的 `g` 则是**已经做过 chunk 内 cumsum 的对数量**（fp32），且内部还乘了 $\log_2 e$ 以便用 `exp2`。
2. **前处理边界。** FLA 可以把 $l2norm(q,k)$、$\sigma(b)$、$\alpha$ 的计算都融进算子（`use_qk_l2norm_in_kernel` / `use_gate_in_kernel` / `use_beta_sigmoid_in_kernel`）；FlashQLA / FlashInfer 通常要求调用方先算好。混用时极易出现"门被算了两次"或"k 没归一化"。
3. **状态布局。** $S$ 是 $[K,V]$ 还是 $[V,K]$（FLA 的 `state_v_first`）、状态池是否连续、第 0 维 stride 是否被 padding，三家约定都不同。而 §7.4 的误差分析、§8.5 的精度分配全部假设"状态以 fp32 在片上累加"，只有显存表示可以自由选。

---

## 相关文档

- [`FLA_Triton_Baseline.md`](FLA_Triton_Baseline.md)：flash-linear-attention 的 Triton 多 kernel 实现——参考基线与通用回退路径
- [`FlashInfer_GDN_Blackwell.md`](FlashInfer_GDN_Blackwell.md)：FlashInfer 在 SM100/SM103 上的 CuTe DSL 全融合实现 + 精确并行扫描
- [`FlashQLA_GDN_Blackwell.md`](FlashQLA_GDN_Blackwell.md)：FlashQLA（Qwen 官方，TileLang）的三段式调度 + 门驱动近似上下文并行

一句话对照（详见各文档）：

> GDN chunk 形式里唯一的串行瓶颈是状态递推（环节 ④），它把并行度锁死在 $B\times H_v$。
>
> - **FLA**：**不解决它。** 8 个 kernel、中间量全物化，用可移植性与训练支持换性能。它是正确性基准与万能回退。
> - **FlashInfer**：**两条路。** 非 CP 主 kernel 同样不解决 ④（每 tile 一个 CTA 串行走 chunk），但把融合做到极致（384 线程、12 warp、双 MMA 发射流、TMEM 512 列恰好用满）；并行度不足时（$B H_{\mathrm{sab}}\cdot4 < num\_sm$）切到 **CP 路径，用杠杆三代数精确地**把串行递推变成四阶段并行扫描。
> - **FlashQLA**：**用杠杆四近似地**把长串行链剪成若干条独立短链（逐 head 判断 $\sum g < -10$），并用杠杆三做精确兜底。三个库里唯一利用了"GDN 状态会指数遗忘"这个模型性质的实现。

选型倾向：

| 场景 | 倾向 |
|---|---|
| 训练；非 128 head dim；非 NVIDIA；需要算子内融合门激活 | **FLA**（其余两家不支持） |
| 长序列 + 低 head 数（高 TP 的单请求 prefill） | **FlashQLA**（近似 CP 在这个区间收益最大，且不必付 $M$ 的显存） |
| 高 head 数 + 多序列（GPU 本已饱和）；需要分页状态池或 fp8 状态 I/O | **FlashInfer**（CP 关闭时纯 kernel 质量更优，且是唯一支持分页池与状态量化的） |

**两条反直觉的提醒**：

1. **"精确"不等于"更准"**（§7.3 的警告）。FlashInfer CP 的实际浮点误差量级（$\sim2\times10^{-3}$，因为 fixup 走 TF32）比 FlashQLA 近似路径的理论界（$4.5\times10^{-5}$）**更大**。
2. **两家的 CP 都有启用门槛，落在门槛外时性能会突变。** FlashQLA 在 SM100 上前向要求 ≥256 chunk（16k token），导致 8k–12k 区间出现比 Hopper 还慢 2 倍的坑；FlashInfer 要求 $B H_{\mathrm{sab}}\cdot4 < num\_sm$。**排查 GDN prefill 性能问题的第一步，永远是确认 CP 到底有没有被启用。**
