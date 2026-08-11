# GDN (Gated DeltaNet) Prefill 优化笔记

> 对象：Qwen3-Next / Qwen3.5 的线性注意力层
> 范围：**推理的 prefill 阶段**（chunked 形式）
> 对比：FlashQLA（Qwen 官方）vs FlashInfer

---

## 目录

1. [GDN 算法本身](#1-gdn-算法本身)
2. [Chunk GDN 的推导](#2-chunk-gdn-的推导)
3. [Chunk 形式的依赖结构](#3-chunk-形式的依赖结构)
4. [Prefill 的性能问题](#4-prefill-的性能问题)
5. [FlashQLA 的解法：利用状态遗忘](#5-flashqla-的解法利用状态遗忘)
6. [FlashInfer 的解法：利用仿射变换的结合律](#6-flashinfer-的解法利用仿射变换的结合律)
7. [FlashInfer SM100 实现：算法结构如何映射到硬件](#7-flashinfer-sm100-实现算法结构如何映射到硬件)
8. [其余环节上的共同改写与分歧](#8-其余环节上的共同改写与分歧)
9. [总结对照](#9-总结对照)
10. [参考实现位置](#10-参考实现位置)

---

## 1. GDN 算法本身

### 1.1 演化脉络

**线性注意力**去掉 softmax，把注意力写成一个可累加的状态：

$$
S_t = S_{t-1} + k_t v_t^{\top}, \qquad S \in \mathbb{R}^{K \times V}
$$

$$
o_t = S_t^{\top} q_t
$$

状态大小固定，不随序列增长。问题是**只会写、不会改**——同一个 $k$ 方向反复写入会互相叠加污染，状态容量很快饱和。

---

#### 插叙：什么是"$k$ 方向"，为什么会污染

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

---

**Delta rule**：写之前先把旧的擦掉。

$$
S_t \;=\; S_{t-1} + \beta_t\, k_t \bigl( v_t - S_{t-1}^{\top} k_t \bigr)^{\top}
\;=\; \bigl( I - \beta_t k_t k_t^{\top} \bigr) S_{t-1} + \beta_t k_t v_t^{\top}
$$

其中 $\bigl(v_t - S_{t-1}^{\top} k_t\bigr)$ 是"当前状态在 $k_t$ 方向读出的值"与"想写入的值"之差，即**误差**；$\beta_t$ 是学习率。本质是**把状态当作一个在线学习的线性映射，每来一个 token 做一步梯度下降**。

**Gated Delta Net (GDN)**：再加一个遗忘门，让状态能主动衰减腾出容量。

$$
\boxed{\;
S_t \;=\; \alpha_t \bigl( I - \beta_t k_t k_t^{\top} \bigr) S_{t-1} \;+\; \beta_t k_t v_t^{\top}
\;}
$$

$$
o_t = S_t^{\top} q_t
$$

### 1.2 Qwen 的参数化

| 符号 | 定义 | 形状 |
|---|---|---|
| $S$ | 递归状态 | $[K, V]$，$K = V = 128$，每 head 一个 |
| $\alpha_t$ | 遗忘门 $=\exp\bigl(-\mathrm{softplus}(A_{\log})\cdot\mathrm{softplus}(a_t + \text{dt\_bias})\bigr) \in (0,1)$ | **标量**，每 head 一个 |
| $\beta_t$ | 写入门 $=\sigma(b_t) \in (0,1)$ | **标量**，每 head 一个 |
| $q_t, k_t$ | 经过 **L2 norm**，故 $\lVert k_t\rVert_2 = 1$ | $[K]$ |
| $v_t$ | | $[V]$ |

层内完整流程：

```
hidden → in_proj_qkvz → [q, k, v, z]      in_proj_ba → [b, a]
   ↓
causal_conv1d(kernel=4)        # 只作用在 q,k,v 上
   ↓
l2norm(q), l2norm(k);  α = exp(−softplus(A_log)·softplus(a+dt_bias));  β = sigmoid(b)
   ↓
Gated Delta Rule               # 本笔记的对象
   ↓
gated RMSNorm(o, z) → out_proj
```

### 1.3 三个决定性的性质

**性质 A：状态是矩阵，不是向量。** $S \in \mathbb{R}^{128\times128}$。这不是 RNN 那种几百维的 hidden state，而是需要矩阵运算参与的对象。

**性质 B：状态转移是满秩矩阵作用，不是对角的。**
这是 GDN 与 Mamba2 的本质区别（Mamba2 的转移是逐元素乘）。它既是 GDN 表达力强的来源，也是难优化的根源。

**性质 C：天生串行。** 算 $S_t$ 必须先有 $S_{t-1}$。

---
## 2. Chunk GDN 的推导

### 2.0 为什么必须改写

§1 的递推一次只处理一个 token。Prefill 有几千上万个 token，逐个执行意味着上万次很小的矩阵-向量运算——全是延迟，没有吞吐，硬件的矩阵乘单元完全用不上。

所以要把**连续 $C$ 个 token 的更新一次性算完**，让它变成大矩阵乘法。取 $C = 64$。

难点只有一个：直接展开会得到一串矩阵连乘

$$
\prod_{t=C}^{1}\bigl( I - \beta_t k_t k_t^{\top} \bigr)
$$

好在每个因子都是"单位阵减一个秩一项"，这类乘积可以收拢成**一个 $C\times C$ 三角矩阵的逆**。整个推导就是把这件事做出来，分三步。

---

### 2.1 记号

固定一个 chunk，token 编号 $t = 1,\dots,C$，$S_0$ 是进入该 chunk 时的状态。

| 记号 | 含义 | 形状 |
|---|---|---|
| $K,\,Q,\,V$ | 该 chunk 的 key / query / value，第 $t$ 行是 $k_t^{\top}$ 等 | $[C,K]$, $[C,K]$, $[C,V]$ |
| $g_t$ | $g_t = \sum_{i\le t}\log\alpha_i$，**chunk 内**累加，$g_0=0$ | 标量 |
| $\mathrm{diag}(e^{g})$，$\mathrm{diag}(\beta)$ | 对角阵 | $[C,C]$ |
| $\Lambda$ | $\Lambda_{ij}=\exp(g_i-g_j)$ | $[C,C]$ |

---

### 2.2 第一步：换元，把遗忘门从状态转移里消掉

令 $S_t = e^{g_t} P_t$（故 $P_0 = S_0$）。代入递推、约去 $e^{g_t}$：

$$
\boxed{\;P_t = \bigl( I - \beta_t k_t k_t^{\top} \bigr) P_{t-1} + e^{-g_t}\beta_t\, k_t v_t^{\top}\;}
$$

> **在 $P$ 坐标系里，遗忘门从状态转移算子中完全消失**，只剩下对写入量的一个标量缩放。
> 这一步是后面一切的基础：它意味着那串连乘里真正难处理的只有 $\prod(I-\beta kk^{\top})$，门可以最后再加回来。

---

### 2.3 第二步：秩一展开，得到下三角方程组

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

这就是 **UT transform / WY 表示**。两个要点：

- $I + \mathrm{tril}(\cdot,-1)$ 是**单位下三角**，行列式恒为 1，**逆一定存在**；
- $A_{\mathrm{raw}}$ 里**没有遗忘门**——门只留在等式右边的 $\mathrm{diag}(e^{-g})$ 上。

---

### 2.4 第三步：换回原坐标，门以相似变换的形式回来

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

**它是一个相似变换**——门是"换坐标系时自然产生的"，而不是本来就长在矩阵里的。§8.2 的"门剥离"优化就是直接利用这一点。

**掩码约定（易错点）**：

| 位置 | 掩码 |
|---|---|
| $\mathrm{tril}(\mathrm{diag}(\beta)KK^{\top},-1)$ | **严格**下三角（$j<t$），对角线由 $I$ 提供 |
| $\Lambda\circ(QK^{\top})$、$\Lambda\circ A_{\mathrm{raw}}$ | 下三角**含对角线**（$j\le t$） |

另外 $q$ 还要乘 $\text{scale}=1/\sqrt{K}$，实现中折进 $Q$ 或 $\Lambda\circ QK^{\top}$。

---

### 2.5 结果：七个矩阵乘

预处理（不依赖 $S_{\mathrm{prev}}$）：$g = \mathrm{cumsum}(\log\alpha)$，$\Lambda$，$A_{\mathrm{raw}}$。

主体三式 ①②③ 展开成七个矩阵乘：

$$
KK^{\top},\quad QK^{\top},\quad K S_{\mathrm{prev}},\quad Q S_{\mathrm{prev}},\quad A(V - KS),\quad (\Lambda\circ QK^{\top})V_{\mathrm{new}},\quad K^{\top}V_{\mathrm{new}}
$$

---

### 2.6 与 FLA 原始 $w$ / $u$ 形式的关系

FLA 把 ① 拆成两步（记 $A_{\mathrm{gated}} = \Lambda\circ A_{\mathrm{raw}}$）：

$$
w = A_{\mathrm{gated}}\,\mathrm{diag}(\beta)\,\mathrm{diag}(e^{g})K,
\qquad
u = A_{\mathrm{gated}}\,\mathrm{diag}(\beta)V,
\qquad
V_{\mathrm{new}} = u - w S_{\mathrm{prev}}
$$

用分配律一合并就回到 ①。所以 **$w$ / $u$ 只是 ① 展开后的形式**，合并回去可以：

- 矩阵乘从 4 次降到 2 次（只需 $KS_{\mathrm{prev}}$ 和 $A(V-KS)$）
- 消掉两个中间张量 $w\in\mathbb{R}^{C\times K}$、$u\in\mathbb{R}^{C\times V}$
- 缩短依赖链：$A$ 与 $KS$ 可并行准备

FlashQLA 与 FlashInfer 都用合并形式。

---

### 2.7 Chunk 化的代价

原始递推里根本不存在、完全由 chunk 化引入的东西：

- $KK^{\top}$（$C\times C$）及其**三角求逆**
- $\Lambda$（$C\times C$ 次 $\exp$）
- $V_{\mathrm{new}}$、$K^{\top}V_{\mathrm{new}}$ 等中间量

换来的是能用 Tensor Core。**这是一笔"多算一些换吞吐"的交易**，只在 $C$ 足够大时划算。

---

### 2.8 为什么 chunk size 取 64

$C$ 在两股相反的力之间取平衡。

**压 $C$ 变大的力**：串行步数是 $T/C$，$C$ 越大串行链越短（§3 的核心矛盾）；而且 $C$ 是几个矩阵乘的 M 维和收缩维，太小的话矩阵乘单元根本吃不饱（极端情况 $C=1$ 就退化回纯递推，等于没做 chunk 化）。

**压 $C$ 变小的力**：chunk 化引入的额外计算随 $C$ 线性增长。把七个矩阵乘按维度分类，每 token 的计算量是

$$
\underbrace{4Cd}_{\text{chunk 内}(C\times C\text{ 项})} + \underbrace{3d^2}_{\text{chunk 间}(d\times d\text{ 项})}
\qquad\Longrightarrow\qquad
\text{冗余率} = 1 + \frac{4C}{3d}
$$

（分母 $3d^2$ 是纯递推每 token 的理论下界。）关键是**冗余率只取决于 $C/d$**：$d=128$ 时，$C=64$ 多算 67%，$C=128$ 多算 133%，$C=256$ 多算 267%——再往上这笔交易就亏了。此外 $\Lambda$ 的 $\exp$ 数量、三角求逆的规模、片上存储占用也都随 $C$（或 $C^2$）增长。

**平衡点**落在 $C = d/2 = 64$：冗余率还能接受，同时正好对齐硬件原生的矩阵乘 tile（Hopper 的 warpgroup 矩阵乘 M 维固定为 64）。FLA、FlashQLA、FlashInfer 三家都取 64。

**它确实是调出来的**：FlashQLA 在 SM90 / SM100 上用 64，但在片上资源更紧张的 SM120 上改用 **32**——牺牲串行步数换更低的冗余率和更小的片上占用。平衡点随硬件条件移动。

---

## 3. Chunk 形式的依赖结构

把 §2.4 的 ①②③ 拆成 5 个环节，标注跨 chunk 依赖：

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
> 所以 **GDN prefill 优化的全部命题 = 怎么让 ④ 不再是一条长度为 $T/C$ 的串行链。**

---

## 4. Prefill 的性能问题

### 4.1 算术强度：落在算力受限一侧

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

现代加速器的算力/带宽比在 $O(100\sim300)$ FLOP/Byte 量级，所以 **GDN prefill 恰好落在算力受限一侧**。

> **推论**：prefill 的瓶颈既不是算力不足，也不是带宽不足，而是**能不能把机器填满**。

### 4.2 真正的问题：并行度只有 batch × head

环节 ④ 的串行性意味着：**一个序列只能由一个计算单元从头走到尾**。可用的并行维度只剩

$$
P \;=\; B \times H
$$

| 场景 | 并行任务数 |
|---|---|
| TP8，模型 64 个 v-head → 每卡 8 个，单请求 | **8** |
| TP4 → 每卡 16 个，单请求 | 16 |
| TP1 → 每卡 64 个，单请求 | 64 |

而一张现代加速器有一百多个流式多处理器。**8 个任务对上百个执行单元，绝大部分在空转。**

雪上加霜的是：**张量并行切的正是 head 维度**，切得越多并行度越低。长上下文 + 高 TP 是最坏组合，也恰恰是最常见的组合。

### 4.3 次要问题

| 问题 | 说明 |
|---|---|
| 三角求逆是"异类" | $(I+L)^{-1}$ 教科书解法是 $C$ 步前代，依赖链长 $C$，跑标量单元，用不上 Tensor Core，而且卡在关键路径上（$A \to V_{\mathrm{new}} \to S_{\mathrm{next}}$） |
| $\Lambda$ 的 $\exp$ 开销 | 每 chunk $C\times C = 4096$ 次 $\exp$，特殊函数单元吞吐远低于矩阵乘单元 |
| 中间量落显存 | 若按 FLA 那样拆成 6 个独立 kernel，所有中间量往返显存，把一个算力密集的公式跑成访存瓶颈 |

但这些都是**常数因子级**的问题。§4.2 的并行度饥饿是**数量级级**的。

---

## 5. FlashQLA 的解法：利用状态遗忘

**出发点**：串行是因为需要精确的 $S_{\mathrm{prev}}$。但 GDN 的状态一直在衰减，很久以前的历史真的还有影响吗？

### 5.1 误差界的证明

考察状态误差 $\Delta S$ 在 chunk 间的传播。从 §2.2 的 $P$ 空间递推

$$
P_t = \bigl( I - \beta_t k_t k_t^{\top} \bigr) P_{t-1} + \underbrace{e^{-g_t}\beta_t k_t v_t^{\top}}_{\text{与 } P_{t-1} \text{ 无关}}
$$

误差只经过擦除算子：

$$
\Delta P_t = \bigl( I - \beta_t k_t k_t^{\top} \bigr)\, \Delta P_{t-1}
$$

由于 Qwen 对 $k$ 做了 **L2 norm**（$\lVert k_t\rVert_2 = 1$）且 $\beta_t = \sigma(\cdot) \in (0,1)$，矩阵 $I - \beta_t k_t k_t^{\top}$ 的特征值为

$$
\{\, \underbrace{1-\beta_t}_{\text{沿 } k_t \text{ 方向}},\ \underbrace{1,\ 1,\ \dots,\ 1}_{K-1 \text{ 个正交方向}} \,\},
\qquad 0 < 1-\beta_t < 1
$$

故其**谱范数恰好等于 1**，是**非扩张**（non-expansive）算子：

$$
\lVert \Delta P_t \rVert \;\le\; \lVert \Delta P_{t-1} \rVert
$$

换回原坐标 $S_t = e^{g_t} P_t$：

$$
\boxed{\;
\lVert \Delta S_t \rVert \;\le\; \exp\!\Bigl( \sum_{i \le t} \log \alpha_i \Bigr)\cdot \lVert \Delta S_0 \rVert
\;=\; e^{\,g_t}\,\lVert \Delta S_0 \rVert
\;}
$$

> **状态误差只会被遗忘门单调衰减，永远不会被 delta rule 放大。**

**注意这是绝对误差界**：它约束的是误差的绝对大小 $\lVert\Delta S_t\rVert$，右边没有除以任何东西。要用来论证"近似可忽略"，还需要下一节的一步转换。

### 5.2 算法：门驱动的近似上下文并行

把长序列切成若干段，**每段假装自己从零状态开始**，但往前多算几个 chunk 做"预热"，预热区间覆盖的累计对数衰减记为 $\sum g$。

#### 从绝对界到相对误差

shard 从零起步，所以"初始误差"恰好就是被丢弃的那段真实状态：

$$
\Delta S_0 = 0 - S_{\text{真}}(\text{预热起点}) = -\,S_{\text{真}}(\text{预热起点})
$$

代入 §5.1 的绝对界，传到 shard 起点：

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

#### 阈值

取

$$
\sum g \;<\; -10
\qquad\Longrightarrow\qquad
\frac{\lVert \Delta S \rVert}{\lVert S_{\text{真}} \rVert} \;\lesssim\; e^{-10} \approx 4.5\times10^{-5}
$$

注意阈值本身是个**相对判据**——$\sum g$ 度量的正是"旧状态还剩多少比例活着"。所以这套机制的**意图是控相对误差，但严格证明只覆盖到绝对界**。

```
原来：chunk₀ → chunk₁ → chunk₂ → … → chunk_{N−1}      一条长串行链
现在：[预热|段₀]   [预热|段₁]   [预热|段₂]  …           若干条互不相干的短链
```

**序列方向变成了新的并行维度**：

$$
P \;=\; B \times H \;\longrightarrow\; P \;=\; B \times H \times N_{\mathrm{shard}}
$$

**两个关键细节**：

1. **逐 head 判断**。不同 head 的遗忘速度差异极大，各自独立决定要预热几个 chunk。
2. **精确兜底**。若某 head 遗忘太慢、预热到头也没跌破阈值，则退回精确路径：把整段的作用写成 $S_{\mathrm{out}} = M\,S_{\mathrm{in}} + N$（$M \in \mathbb{R}^{K\times K}$ 为转移矩阵），串行修正。

所以算法是"**近似为主、精确兜底**"的混合结构，且是自我保护的——门衰减慢就自动走精确路径。

### 5.3 误差量级

| 参照 | 相对分辨率 | 与 $4.5\times10^{-5}$ 的关系 |
|---|---|---|
| bf16（8 bit 尾数） | $3.9\times10^{-3}$ | 近似误差**低两个数量级**，被舍入吞掉 |
| fp16（11 bit 尾数） | $4.9\times10^{-4}$ | 低一个数量级 |
| fp32（24 bit 尾数） | $6\times10^{-8}$ | **高三个数量级** |

准确表述：**bf16 / fp16 输出下该近似不可观测；但相对 fp32 精确递推它是有偏的。**

**两点保留**：

1. **相对误差的成立依赖 §5.2 那个范数比假设**，严格证明只给到绝对界。
2. **范数界是 Frobenius 意义上的全局界，不是逐元素保证。** 误差传到输出是 $\Delta o = \Delta S^{\top} q$，若 $q$ 恰好指向被丢弃分量占主导的方向，或 $o$ 本身因抵消而很小，该分量的相对误差可显著大于 $4.5\times10^{-5}$。

### 5.4 性质总结

| | |
|---|---|
| 依据 | 状态的**指数遗忘性质**（模型特性） |
| 结果 | 近似，误差 $\sim 10^{-5}$ |
| 额外开销 | 只多预热几个 chunk，**近似免费** |
| 失效条件 | 模型的门普遍接近 1（几乎不遗忘）时，所有 head 退回精确路径，加速归零 |

---

## 6. FlashInfer 的解法：利用仿射变换的结合律

**出发点**是纯代数的，与模型性质无关。

### 6.1 观察：chunk 的作用是仿射变换

由 §2.4 的 ①② 消去 $V_{\mathrm{new}}$：

$$
S_{\mathrm{next}} = e^{g_C} S_{\mathrm{prev}} + K^{\top}\mathrm{diag}\!\left(e^{g_C-g}\right)\bigl(\Lambda\circ A_{\mathrm{raw}}\bigr)\mathrm{diag}(\beta)\bigl( V - \mathrm{diag}(e^{g}) K S_{\mathrm{prev}} \bigr)
$$

整理成关于 $S_{\mathrm{prev}}$ 的形式：

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

（实现中左乘 / 右乘的约定视状态布局而定，结构相同。）

### 6.2 关键性质：结合律

仿射变换的复合仍是仿射变换：

$$
M_2\bigl( M_1 S + N_1 \bigr) + N_2 \;=\; \bigl( M_2 M_1 \bigr) S + \bigl( M_2 N_1 + N_2 \bigr)
$$

**满足结合律 $\Longrightarrow$ 可以做并行前缀和（parallel scan）。**

### 6.3 算法：三段式 blocked scan

1. **并行阶段**：把序列切成若干段，所有段**同时**算出各自的 $(M_c, N_c)$。这一步互不依赖，完全并行。
2. **串行阶段**：$S_{c+1} = M_c S_c + N_c$ 扫一遍。但这是在**粗粒度的段**上扫，串行长度从 $T/C$ 降到 $T/L_{\mathrm{shard}}$。
3. **并行阶段**：每段拿到**精确**的起始状态，独立算 ①③。

这与 Mamba 的 selective scan 是同一套思路，区别只在 GDN 的转移是 $K\times K$ 矩阵而非标量，所以扫描每一步是矩阵乘。

### 6.4 性质总结

| | |
|---|---|
| 依据 | 仿射变换的**结合律**（代数结构） |
| 结果 | **精确** |
| 额外开销 | 显式构造并存储 $M_c$（$128\times128$，不小）+ 多两趟遍历 |
| 失效条件 | 无 |

---

## 7. FlashInfer SM100 实现：算法结构如何映射到硬件

SM100（数据中心 Blackwell）不是 SM90 版本的移植，是按 Blackwell 的资源模型重写的一版。它值得单独看，因为**它几乎每一处硬件决策都能追溯到 §3 的依赖结构**。

规模：384 线程（12 warp）、SMEM 约 225 KB、TMEM 全部 512 列、寄存器 CTA 预算 64512。

---

### 7.1 起点：依赖图直接决定 warp 分工

§3 的结论是：环节 ①② 不依赖状态（全 chunk 可并行），③④⑤ 被串行链锁住。SM100 把这条界线**原封不动地画成了 warp 分组的边界**：

| 分组 | 负责 | 依赖状态？ |
|---|---|---|
| **CG0**（4 warp） | $\Lambda$ 的构造、$KK^{\top}$ / $QK^{\top}$ 的后处理、**三角求逆** | ❌ 不依赖 |
| **CG1**（4 warp） | $K S$、$V - KS$、$V_{\mathrm{new}}$、状态更新、输出组装 | ✅ 依赖 |
| **MMA issuer A**（1 warp） | 只发 $KK^{\top}$、$QK^{\top}$ 两类 GEMM | ❌ |
| **MMA issuer B**（1 warp） | 发其余五个状态/输出 GEMM | ✅ |
| **TMA warp**（1） | 搬 Q / K / V | — |
| **Epilogue warp**（1） | 写出 $O$，顺带预取 gate / beta | — |

**两条独立的 MMA 发射流**是这里最关键的设计。Blackwell 的 `tcgen05` MMA 是单线程发射、异步执行的，所以"谁来发指令"是可以自由安排的。把不依赖状态的 GEMM 交给一个独立的发射 warp，意味着**它可以任意超前**——只要 TMA 把 K、Q 搬进来了，$KK^{\top}$ 就能发，完全不用等状态链。而状态链那条流只能一步一步走。

配套的一个决定是：**两个计算组各有独立的 TMEM 累加器 ring**，而不是共用一个。多占 TMEM，换来把"这块累加器现在归谁"的所有权交接从关键路径上彻底拿掉。

轻活也做了合并：写出 $O$ 是轻负载，于是 gate / beta 的预取挂在同一个 warp 上，不再单独占一个 warp。

---

### 7.2 链式中间量常驻 TMEM，全程不经过 SMEM

**算法观察**：chunk 公式里 ①③ 构成一条链

$$
S_{\mathrm{prev}} \;\to\; K S_{\mathrm{prev}} \;\to\; (V - \mathrm{diag}(e^{g})KS_{\mathrm{prev}}) \;\to\; V_{\mathrm{new}} \;\to\; \{\,O,\; S_{\mathrm{next}}\,\}
$$

链上每个中间量**只被下一步用一次**，用完就死。既然是一次性中间量，就没有任何理由让它落到 shared memory 再读回来。

**硬件对应**：Blackwell 的 `tcgen05` MMA 允许 A 操作数**直接从 TMEM 读取**（而不是必须从 SMEM）。于是：

- 状态 $S$ 以 fp32 累加器**常驻 TMEM**，从第一个 chunk 到最后一个 chunk 不搬家
- $S$ 转成低精度后**仍然放在 TMEM 里**，直接作为 $KS$、$QS$ 的操作数
- $V - KS$ 和 $V_{\mathrm{new}}$ 也写进 TMEM 的操作数缓冲，直接喂给下一个 GEMM

结果：七个 GEMM 里只有 $KK^{\top}$、$QK^{\top}$ 的操作数来自 SMEM（因为 Q、K 是 TMA 刚搬进来的），**其余链式量全程在 TMEM 内流转**。

> **与 SM90 版的对照**：SM90 用的是完全相同的思想，但载体是**寄存器**——Hopper 的 `wgmma` 支持 A 操作数来自寄存器，于是把上一个 GEMM 的 fp32 累加器就地转成低精度、重新解释成 A 操作数布局，直接发下一条 MMA。
> SM100 换成 TMEM，是因为 `tcgen05` 的累加器本来就在 TMEM，就近取用最省。
> **同一个算法性质（链式一次性中间量），在两代硬件上找到了两个不同的落点。**

---

### 7.3 TMEM 预算：恰好用满 512 列

TMEM 一共 512 列，SM100 版的分配是：

| 用途 | 列数 | 为什么是这个数 |
|---|---|---|
| 状态 $S$（fp32 累加器） | 128 | $128\times128$ fp32 |
| $QS$ 累加器 | 64 | **单独占一块，不进共享 ring** |
| $S$ 的低精度操作数副本 | 64 | $128\times128$ 低精度 = 半个 fp32 宽度 |
| CG0 的累加器 ring | 2×64 | $KK^{\top}$、$QK^{\top}$ 各一级 |
| CG1 的累加器 ring | 1×64 | $KS$、$V_{\mathrm{new}}$ 轮用 |
| 链式操作数缓冲（低精度） | 2×32 | $V-KS$ 与 $V_{\mathrm{new}}$ |
| **合计** | **512** | 一列不剩 |

值得单独说的是 **$QS$ 为什么要单独一块累加器**。看输出式 ③：

$$
O = \underbrace{\mathrm{diag}(e^{g})\,Q\,S_{\mathrm{prev}}}_{\text{chunk 间}} + \underbrace{(\Lambda \circ QK^{\top})\,V_{\mathrm{new}}}_{\text{chunk 内}}
$$

$QS$ 要在 GEMM 4 算出来，然后**一直等到 GEMM 6 把 intra 部分算完才能相加**。它的生命期横跨中间好几个 GEMM，不能被别人复用——所以只能独占一块，不能塞进轮转的共享 ring 里。这是公式的结构直接决定的。

---

### 7.4 两 chunk 成对流水：为了藏住三角求逆的延迟

**问题**：三角求逆（环节 ②）虽然 FLOP 占比很小（每 token 摊下来不到 1%），但它是**长依赖链 + 标量单元**，延迟很长，而且卡在关键路径上：$A \to V_{\mathrm{new}} \to S_{\mathrm{next}}$。单 chunk 流水的话，这段延迟没有东西可以拿来掩盖——后面的 GEMM 都在等它。

**解法**：一次处理**两个 chunk**（pair）。发射顺序被刻意安排成

```
KK⁰  →  KK¹  →  QK⁰  →  QK¹
```

**先把两个 chunk 的 $KK^{\top}$ 都发出去**，让两个三角求逆尽早开始；然后用 chunk 0 的状态链计算时间去掩盖 chunk 1 求逆的延迟。$QK^{\top}$ 排在后面，因为它只服务输出、不在状态链上，可以晚一点。

这个流水结构反过来定死了各缓冲的深度。SM100 版每个 stage 数都是从活跃集算出来的，不是拍的：

| 缓冲 | 级数 | 理由 |
|---|---|---|
| K | **4** | K 是四个 GEMM 的操作数，生命期最长；要让 TMA 把第 4 份搬好的同时，第 1 份还活着给状态更新用，这样下一对的两个 $KK^{\top}$ 才能背靠背发出去 |
| V | **3** | 打断"双 $KK^{\top}$ 前瞻"造成的依赖环，同时让 TMA 保持超前 |
| $A^{-1}$ | **3** | 跨 pair 前瞻时的活跃集是「本对的第二个逆 + 下一对的两个逆」= 3 |
| gate / beta | **5** | 标量极便宜，开深让标量生产者彻底跑飞，与主流水解耦 |
| Q、$QK$、O | 2 | 生命期短 |

---

### 7.5 $\Lambda$ 只算一次，两处共用

**算法观察**：$\Lambda_{ij} = e^{g_i - g_j}$ 在公式里出现**两次**——一次在待求逆的矩阵 $\Lambda\circ(\mathrm{diag}(\beta)KK^{\top})$ 里，一次在 chunk 内输出 $\Lambda\circ(QK^{\top})$ 里。

**优化**：CG0 把 $\Lambda$ 算在寄存器里，先给 $KK^{\top}$ 的后处理用，**同一批寄存器接着给 $QK^{\top}$ 的后处理用**。$\exp$ 的次数直接减半。

这是"同一个数学量在公式两处出现 ⟹ 只算一次"最直接的应用。它之所以能成立，是因为 CG0 同时拥有这两个后处理任务——又回到 §7.1 那个分组决策：**把用同一批中间量的工作放进同一个 warp 组**。

---

### 7.6 零初始状态时剥掉第一个 chunk 的两个 GEMM

**算法观察**：如果 $S_{\mathrm{prev}} = 0$（序列的第一个 chunk，且没有传入初始状态），那么

$$
K S_{\mathrm{prev}} = 0,\qquad Q S_{\mathrm{prev}} = 0
$$

这两个 $[C,d]\times[d,d]$ 的 GEMM **完全不用算**——而它们恰好是七个 GEMM 里最大的两个（各占单 chunk 总 MAC 的 20%）。而且 $V - KS = V$、$O$ 的 chunk 间部分为零，累加器第一次写入也不需要 accumulate。

**实现**：编译期特化出一条剥离路径，首 chunk 跳过这两个 GEMM。

**代价**：剥离路径要多带一套流水游标，寄存器需求变了，所以寄存器预算要重新分配（把 CG1 的一部分让给轻量 warp），保持 CTA 总量不变。这是"算法上省掉两个 GEMM"必须付的实现税。

---

### 7.7 七个 GEMM 的操作数布局协同

`tcgen05` 的 MMA 对每个操作数都要指定 major mode（K-major 还是 MN-major）和来源（SMEM 还是 TMEM）。七个 GEMM 的要求各不相同，被归成四组：

| 组 | GEMM | A 的来源 / major |
|---|---|---|
| 1 | $KK^{\top}$、$QK^{\top}$ | SMEM，K-major |
| 2 | $KS$、$QS$ | **TMEM** |
| 3 | $A^{-1}V$、$V_{\mathrm{new}}$ 与 $QK$ 的乘积 | **TMEM**（首个 tile 的 $V$ 来自 SMEM，故额外配一个 SMEM 版本） |
| 4 | $K^{\top}V_{\mathrm{new}}$ | B 是 MN-major |

问题在于 **SMEM 布局必须同时满足所有消费者**。$K$ 被四个 GEMM 用到，其中 $K^{\top}V_{\mathrm{new}}$ 需要的是转置视图——而这个转置需求**是 chunk 公式 ② 本身带来的**（状态更新式里 $K$ 就是转置的），不是实现随意选的。所以 $K$ 必须在 SMEM 里同时以两种视图可读。

这是个典型的**布局协同设计**问题：算法决定了哪些张量需要哪些视图，实现要找到一组 swizzle 布局同时满足全部约束，还不能为此多存一份数据。

---

### 7.8 SMEM 的时间复用与一处精度取舍

SMEM 只有约 225 KB，而 chunk 公式要同时容纳 Q、K（4 级）、V（3 级）、$A^{-1}$（3 级）、$QK$、$O$、状态 staging、标量。于是做了**时间维度上的复用**：

- 存 $QK$ 分数的缓冲，等 $O$ 算完之后被 $O$ 的写出暂存**覆写**——两者生命期不重叠
- 状态在 GMEM ↔ TMEM 之间中转用的缓冲，和 $V$ 的衰减缩放的暂存**共用**

还有一处值得注意的**精度决策**：求逆用的 scratch 缓冲，在「求逆精度 == I/O 精度」时直接别名到 $A^{-1}$ 缓冲本身，**不额外占空间**。SM100 强制这两者相等（SM90 允许 I/O 是 bf16 而求逆用 fp16）。

代价是求逆精度从 fp16 的 11 bit 尾数降到 bf16 的 8 bit；换来的是省掉一次 dtype 转换 pass 和一块 SMEM。之所以敢这么换，是因为 $\lVert k\rVert=1$、$\beta<1$ 保证了待求逆矩阵的元素有界（$\lvert k_i\!\cdot\! k_j\rvert \le 1$），加上 §8.3 的分治求逆全程 fp32 累加，误差可控。

---

### 7.9 varlen：TMA descriptor 运行时打补丁

**问题**：TMA 的 tensor descriptor 在 host 上创建，`global_dim` 是固定的。packed varlen batch 里，**非最后一条序列的尾块（不满 $C$ 个 token）会跨界读写到下一条序列的地盘上**。

**解法**：每个 CTA 在显存里有一组 descriptor 槽位（Q / K / V / O 各一个），运行时把 `global_dim` 改成本序列的真实边界，只对尾块使用这个临时 descriptor。判断逻辑上还做了优化——以下情形可以直接用原始 descriptor、省掉打补丁：

- 中间的满 tile
- 序列长度恰好是 $C$ 的整数倍
- **最后一条序列**（越界正好落在整个 allocation 之外，TMA 自带钳位）

注意 SM90 版只需要给 $O$ 打补丁，SM100 版要给 **Q / K / V / O 四个都打**——因为 SM100 的 Q、K、V 也全部走 TMA tile load，读越界会把下一条序列的数据混进来。

另一个 varlen 相关的约束是：**head 数、序列长度、batch 大小全部保持运行时值**，只有 head_dim、chunk size 和几个布尔特化是编译期常量。一份编译产物服务所有 shape——serving 场景不能每换一个 batch 就 JIT 一次。

---

### 7.10 消灭 prologue peeling

首尾 chunk 的特殊情况用**运行时谓词**处理，而不是展开一份专门的首轮代码。

理由很实在：这个 kernel 的 pair 主体极其庞大（七个 GEMM + 三角求逆 + 多个后处理），复制一份会让指令 cache 压力翻倍。用运行时谓词换来 SASS 里**只有一份 pair body**。

---

### 7.11 推理侧的两个设计

这两点和 chunk 算法本身无关，但决定了这个 kernel 能不能真的用在服务里。

**① 分页状态池就地读写。** 状态存在一个 `[N_pool, H, V, K]` 的池子里，且第 0 维的 stride 可能被 padding（因为和 conv state 打包在一起，布局非紧凑）。kernel 直接按 slot id 读写，**免掉"先 gather 到连续 buffer、算完再 scatter 回去"这两次完整的状态拷贝**。

代价是非紧凑布局无法给最内维加整除性提示，状态拷贝的向量化宽度可能变窄——但状态拷贝在整个 kernel 里占比极小，这个交换是划算的。

**② 状态的 I/O 量化。** 状态在显存里可以是 fp32 / bf16 / fp16 / **fp8**（e4m3 或 e5m2），但**片上累加恒为 fp32**。这是 I/O 量化，不是计算量化：读进来立刻升到 fp32，写回时再降。

算法上的依据和 §5.1 同源：GDN 有遗忘门，量化误差进入状态后随 $e^{\sum g}$ 指数衰减，**有界且自愈**，不会像纯累加型线性注意力那样无限累积。

---

### 7.12 汇总：每处优化在权衡什么

| 优化 | 算法依据 | 换来什么 | 付出什么 |
|---|---|---|---|
| 依赖图 → warp 分组 + 双 MMA 发射流 | ①② 无状态依赖，③④⑤ 有 | 无依赖的 GEMM 可任意超前 | 组间同步逻辑复杂 |
| CG0 / CG1 独立累加器 ring | 同上 | 关键路径上没有所有权交接 | 多占 TMEM |
| 链式中间量常驻 TMEM | 链上中间量只用一次 | 五个 GEMM 免掉 SMEM 往返 | 数学式子要按操作数位置改写 |
| $QS$ 独占累加器 | 输出式里 inter 与 intra 相隔多个 GEMM | 不必把它挪来挪去 | 64 列 TMEM |
| 两 chunk 成对流水 | 求逆延迟长且在关键路径 | 求逆延迟被邻居 chunk 掩盖 | 活跃集变大 → 缓冲级数增加 |
| $\Lambda$ 两处共用 | $\Lambda$ 在公式里出现两次 | $\exp$ 次数减半 | 需要把两处后处理放同一 warp 组 |
| 零状态首 chunk 剥离 | $S=0 \Rightarrow KS=QS=0$ | 省掉最大的两个 GEMM | 多一套流水游标，寄存器要重排 |
| 四组 TiledMMA + $K$ 双视图 | 公式 ② 需要 $K^{\top}$ | 不必存第二份 $K$ | 布局约束求解复杂 |
| SMEM 时间复用 | 生命期不重叠 | 省 SMEM | 顺序被锁死 |
| 求逆精度降到 I/O 精度 | $\lVert k\rVert=1,\beta<1$ ⟹ 元素有界 | 省一次转换 pass + 一块 SMEM | 尾数从 11 bit 降到 8 bit |
| descriptor 运行时打补丁 | — | varlen 尾块正确 | 每个 CTA 一组 descriptor 槽位 |
| 运行时谓词代替 peeling | — | SASS 只有一份主体 | 每次迭代多几条谓词判断 |
| 分页状态池就地读写 | — | 免两次全状态拷贝 | 状态拷贝向量化变窄 |
| 状态 fp8 I/O | 遗忘门让量化误差自愈 | 显存与带宽减到 1/4 | 状态读写有量化噪声 |

**贯穿始终的一条线**：这一版几乎所有硬件层面的决策，都能追到 chunk 公式的某个结构性质上——哪些量不依赖状态、哪些中间量只用一次、哪个量的生命期横跨多个 GEMM、哪个量在公式里出现两次、哪个矩阵需要转置视图。**先把算法的依赖图和生命期画清楚，硬件资源的分配方式基本就唯一了。**

---

## 8. 其余环节上的共同改写与分歧

### 8.1 环节 ③：消去 $w$ / $u$（两家都做）

见 §2.6。合并后矩阵乘从 4 次降到 2 次，消掉两个中间张量，依赖链变短。

### 8.2 环节 ②：门要不要留在被求逆的矩阵里（分歧）

由 §2.3–2.4 的推导可知，**门天然不在待求逆矩阵里**：$A_{\mathrm{raw}}$ 是在 $P$ 坐标系中得到的，本来就是 gate-free；$\Lambda$ 是换回原坐标时产生的**相似变换**。

对应的恒等式：

$$
\Bigl( I + \mathrm{tril}\bigl( \Lambda\circ(\mathrm{diag}(\beta)KK^{\top}),\,-1 \bigr) \Bigr)^{-1}
\;=\;
\Lambda \circ \Bigl( I + \mathrm{tril}\bigl( \mathrm{diag}(\beta)KK^{\top},\,-1 \bigr) \Bigr)^{-1}
$$

证明依据：$\Lambda\circ(KK^{\top}) = D\,KK^{\top}D^{-1}$，其中 $D=\mathrm{diag}(e^{g})$；而对角阵 $D$ 与 $\mathrm{diag}(\beta)$ 可交换，且相似变换与求逆可交换：$(D X D^{-1})^{-1} = D X^{-1} D^{-1}$。

| | 做法 | 后果 |
|---|---|---|
| **FlashQLA** | 按自然形式：先求 gate-free 的 $A_{\mathrm{raw}}$，再逐元素乘 $\Lambda$ | $C\times C$ 次 $\exp$ 从求逆的关键路径上移走；$\Lambda$ 不参与消元，数值动态范围更小；且 $A_{\mathrm{raw}}$ 与门无关 |
| **FlashInfer** | 先把 $\Lambda$ 和 $\beta$ 乘进矩阵再求逆 | 那 $C\times C$ 次 $\exp$ 留在关键路径上 |

补充：$A_{\mathrm{raw}}$ 门无关意味着它可以算一次多处复用——但这个收益只对训练（forward / backward 共用同一个 $A$）有意义。纯推理场景下，这处分歧只剩"少一批 $\exp$"这个常数因子。

### 8.3 环节 ②：三角求逆的分治化（两家都做）

$(I+L)^{-1}$ 的教科书解法是 $C$ 步前代，依赖链长 $C$。两家都改成**分块 Schur 补递归**：把下三角矩阵按 $2\times2$ 分块，

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

**两个对角小块各自递归求逆，再用一次矩阵乘把左下角块补上。** 递归到底时小对角块直接消元，然后两两合并、再两两合并，一路拼回整个 $C\times C$。

- 依赖链长度从 $C$ 降到 $O(\log C)$ 级
- 合并步骤全是矩阵乘，能上 Tensor Core
- 总运算量反而增加——典型的**"用工作量换深度"**

两家只在分块粒度上有差别（起始块 $8\times8$ 还是 $16\times16$）。

---

## 9. 总结对照

### 9.1 逐环节

| 环节 | 性质 | FlashQLA | FlashInfer |
|---|---|---|---|
| ① 门前处理 | 全并行 | 独立预计算 | 融进主流程 |
| ② UT transform | 全并行，内部串行 | 分治求逆 **+ 门保持在相似变换之外** | 分治求逆（分块更细），门乘进矩阵内 |
| ③ value 修正 | 依赖 $S_{\mathrm{prev}}$ | 合并形式，消去 $w$ / $u$ | 同左 |
| ④ **状态递推** | **唯一串行瓶颈** | **遗忘截断 $\Rightarrow$ 近似并行** | **结合律 $\Rightarrow$ 精确 scan 并行** |
| ⑤ 输出组装 | 依赖 $S_{\mathrm{prev}}$ | — | — |

### 9.2 两条路的本质

|  | FlashQLA | FlashInfer |
|---|---|---|
| 依据什么 | 状态会**指数遗忘**（模型的性质） | 仿射变换满足**结合律**（代数结构） |
| 数学结果 | 近似，误差 $\sim 10^{-5}$（bf16 下不可观测） | 精确 |
| 额外开销 | 预热几个 chunk，近似免费 | 显式算并存 $M_c$，多两趟遍历 |
| 失效条件 | 门接近 1（不遗忘）时全部退回串行 | 无 |
| 适配范围 | 依赖 GDN 训练出的门分布 | 任意输入 |

### 9.3 一句话

> GDN chunk 形式里**唯一的串行瓶颈是状态递推（环节 ④）**，它把并行度锁死在 $B\times H$，在长序列 + 高 TP 下造成数量级的硬件闲置。
> **FlashQLA** 用"很久以前的历史已被遗忘门衰减掉"这个**模型性质**，近似地把长串行链剪成若干条独立短链；
> **FlashInfer** 用"chunk 的作用是可结合的仿射变换"这个**代数事实**，精确地把串行递推变成并行扫描。
> 一个便宜但有近似，一个精确但要多跑两趟。
> 其余环节（②的分治求逆、③的合并化简）两家做法趋同，属于把同一个数学式子写得更适合硬件，量级上是常数因子。

---

## 10. 参考实现位置

### FlashQLA（Qwen 官方，TileLang）

| 内容 | 路径 |
|---|---|
| 顶层调度（`cumsum → kkt_solve → fused_fwd` 三段） | `flash_qla/ops/gated_delta_rule/chunk/__init__.py` |
| 三角求逆（gate-free + 分治） | `.../chunk/hopper/kkt_solve.py` |
| 主循环（①②③⑤ 融合） | `.../chunk/hopper/fused_fwd.py` |
| 门驱动 CP：预热 chunk 数计算、精确修正 | `.../chunk/hopper/cp_fwd.py` |
| CP 调度决策（分片长度、开关阈值） | `.../chunk/cp_context.py` |
| 转移矩阵 $M$ 的计算 | `.../chunk/hopper/prepare_h.py` |
| 数学参考实现（可读性最好） | `tests/ref_gdr.py` |

### FlashInfer（CuTe DSL）

| 内容 | 路径 |
|---|---|
| Prefill 入口与后端派发 | `flashinfer/gdn_prefill.py` |
| SM90 全融合主 kernel | `flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm90.py` |
| SM100 版本（含完整算法说明的文件头注释） | `flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py` |
| 精确 CP：$(M,N)$ 预计算 + fixup scan | `flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm90.py` |
| CP 分片长度与开关的解析模型 | `flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py` |
| 分治三角求逆 | `flashinfer/gdn_kernels/delta_rule_dsl/collective_inverse_hmma.py` |

### FLA 基线（Triton，6-kernel 拆分）

$$
\text{cumsum} \to KK^{\top} \to \text{solve\_tril} \to (w,u) \to (h, V_{\mathrm{new}}) \to O
$$
