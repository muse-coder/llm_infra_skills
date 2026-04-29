# TurboQuant KV Cache 量化知识库

> 本文档覆盖 TurboQuant（ICLR 2026, arXiv 2504.19874）的核心算法原理、代码实现逻辑、工程决策与实测结论。  
> 仓库地址：`turboquant_plus`，llama.cpp fork：`llama-cpp-turboquant`。

---

## 目录

1. [先看结论](#1-先看结论)
2. [TurboQuant 是什么](#2-turboquant-是什么)
3. [整体算法架构](#3-整体算法架构)
4. [Stage 1：PolarQuant（MSE 最优量化）](#4-stage-1polarquantmse-最优量化)
5. [Stage 2：QJL 残差修正](#5-stage-2qjl-残差修正)
6. [KV Cache 差异化策略](#6-kv-cache-差异化策略)
7. [非整数 bit 率：Outlier 通道策略](#7-非整数-bit-率outlier-通道策略)
8. [关键工程决策](#8-关键工程决策)
9. [核心代码逐模块解析](#9-核心代码逐模块解析)
10. [实测效果与验证数据](#10-实测效果与验证数据)
11. [使用方式](#11-使用方式)
12. [常见误区](#12-常见误区)
13. [TurboQuant 在 LLM 推理流程中的位置](#13-turboquant-在-llm-推理流程中的位置)
14. [完整计算流程详解](#14-完整计算流程详解)
15. [非对称 KV 量化的发现（Asymmetric KV Discovery）](#15-非对称-kv-量化的发现asymmetric-kv-discovery)
16. [上下文退化问题与修复（Context Scaling）](#16-上下文退化问题与修复context-scaling)
17. [注意力门控优化（Attention-Gated Optimizations）](#17-注意力门控优化attention-gated-optimizations)

---

## 1. 先看结论

### 1.1 最重要的三句话

1. **TurboQuant = 随机旋转 Gaussianization + Lloyd-Max 最优量化 + 范数提取**，三步组合在极低 bit 率下仍保持高质量。
2. **K Cache 和 V Cache 用不同量化器**：K 需要内积保持（注意力分数），V 需要 MSE 最小（加权求和）。
3. **QJL 在生产中被关闭**：QJL 增加方差，softmax 放大后反而损害质量，纯 PolarQuant 更好。

### 1.2 压缩效果速查

| 模式 | bits/val | 压缩比 vs fp16 | PPL 劣化（32K ctx） | 速度 vs q8_0 |
|------|----------|--------------|-------------------|-------------|
| `turbo3` | 3.5 | **4.9x** | +1.64% | ≈ 持平 |
| `turbo4` | 4.25 | **3.8x** | +0.93% | ≈ 持平 |
| `q8_0`（基线） | 8 | 2.0x | — | 基线 |
| `q4_0`（参考） | 4 | 4.0x | +0.31% | — |

---

## 2. TurboQuant 是什么

TurboQuant 是专为 **LLM KV Cache** 设计的极致压缩量化算法，核心思路：

- **问题**：KV Cache 向量分布是重尾、高峰度的（真实 Qwen3-1.7B kurtosis 高达 900），直接量化效果极差。
- **解法**：先用随机旋转把分布变成近似 Gaussian，再用对 Gaussian 最优的 Lloyd-Max 量化器。
- **实现**：作为 llama.cpp 的 KV Cache 类型扩展，新增 `turbo3` / `turbo4` 两种 cache type，无需修改模型权重。

**论文**：
- TurboQuant: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)（ICLR 2026）
- PolarQuant: [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)（AISTATS 2026）

---

## 3. 整体算法架构

```
输入：KV Cache 向量 x ∈ R^d（一个 attention head 的向量）
         │
         ├─ 1. 提取 L2 范数：γ = ||x||₂，x̂ = x / γ
         │      （论文 page 5：范数用 float32 单独存储，解量化时还原尺度）
         │
         ├─ 2. 随机旋转（Gaussianization）
         │      WHT（Walsh-Hadamard Transform）+ 随机符号翻转
         │      旋转后：每个坐标 ~ N(0, 1/d)
         │
         ├─ 3. Lloyd-Max 最优标量量化（PolarQuant）
         │      turbo4: 16 个质心（4-bit）
         │      turbo3: 8 个质心（3-bit）
         │      turbo2: 4 个质心（2-bit）
         │
         ├─ 4. [可选] QJL 1-bit 残差修正（生产中关闭）
         │      消除内积偏差，但增加方差，softmax 放大后反而更差
         │
         └─ 输出：量化索引 + 范数（每 block 一个 float32）
```

**文件结构**：

```
turboquant/
├── rotation.py       # Walsh-Hadamard Transform + 随机符号翻转
├── codebook.py       # Lloyd-Max 最优质心计算
├── polar_quant.py    # PolarQuant：范数提取 + 旋转 + 标量量化
├── qjl.py            # QJL 1-bit 量化（保留参考，生产不用）
├── turboquant.py     # 完整 TurboQuant 流水线（PolarQuant + QJL）
├── kv_cache.py       # KV Cache 集成层（K/V 差异化）
└── outlier.py        # Outlier 通道策略（2.5-bit、3.5-bit）
```

---

## 4. Stage 1：PolarQuant（MSE 最优量化）

### 4.1 随机旋转 Gaussianization（`rotation.py`）

**核心原理**：随机正交旋转不改变向量的 L2 范数，但会把各坐标的能量均匀分散，使分布趋向 Gaussian。

```python
# 方法一：Dense Haar 旋转（精确，O(d²)，Python 原型用）
def random_rotation_dense(d, rng):
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)           # QR 分解得到正交矩阵
    signs = np.sign(np.diag(R))
    Q = Q * signs                     # 修正符号 → Haar 均匀分布
    sign, _ = np.linalg.slogdet(Q)
    if sign < 0:
        Q[:, 0] = -Q[:, 0]           # 确保 det(Q) = +1（真正的旋转）
    return Q

# 方法二：快速结构化旋转（近似，O(d log d)，GPU 实现用）
# 形式：D₂ · H · D₁
#   D₁, D₂：随机 ±1 对角矩阵
#   H：Walsh-Hadamard 矩阵（蝴蝶运算，GPU 高效）
def random_rotation_fast(d, rng):
    padded_d = next_power_of_2(d)
    signs1 = rng.choice([-1.0, 1.0], size=padded_d)
    signs2 = rng.choice([-1.0, 1.0], size=padded_d)
    return signs1, signs2, padded_d
```

**Walsh-Hadamard Transform（WHT）蝴蝶运算**：

```python
def fast_walsh_hadamard_transform(x):
    # O(n log n) 蝴蝶运算，类似 FFT
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = x[j], x[j + h]
                x[j] = a + b          # 蝴蝶加
                x[j + h] = a - b      # 蝴蝶减
        h *= 2
    return x / sqrt(n)               # 归一化
```

**验证结果**（真实 Qwen3-1.7B KV 张量）：
```
原始峰度 kurtosis: 900.4 → 旋转后: 2.9（Gaussian 理论值 = 3.0）✅
旋转后标准差: 0.088388 = 1/√d（理论值完全吻合）✅
```

### 4.2 Lloyd-Max 最优质心（`codebook.py`）

旋转后每个坐标服从 `N(0, 1/d)`，对此分布用 **Lloyd-Max 算法**迭代求最优标量量化质心：

```python
def optimal_centroids(bit_width, d):
    n_centroids = 1 << bit_width

    # 1-bit 和 2-bit 有解析解
    if bit_width == 1:
        c = sqrt(2.0 / (pi * d))
        return [-c, c]
    if bit_width == 2:
        return [-1.51, -0.453, 0.453, 1.51] / sqrt(d)

    # 3-bit 及以上：Lloyd-Max 数值迭代
    return _lloyds_gaussian(n_centroids, sigma=1.0 / sqrt(d))

def _lloyds_gaussian(n_centroids, sigma, n_iter=100):
    # 初始化：从 Gaussian 分位数均匀采样边界
    boundaries = norm.ppf(linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma)

    for _ in range(n_iter):
        # 更新边界：相邻质心的中点（Voronoi 划分）
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # 更新质心：每个区间内的条件期望
        # E[X | a < X < b] = σ · (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
        for i in range(n_centroids):
            centroids[i] = gaussian_conditional_expectation(sigma, boundaries[i-1], boundaries[i])

    return sort(centroids)
```

**最近质心查找**（O(n log k) 二分搜索）：

```python
def nearest_centroid_indices(values, centroids):
    # 利用质心已排序，用 searchsorted 代替暴力 O(n·k) 比较
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return searchsorted(boundaries, values.ravel()).reshape(values.shape)
```

### 4.3 完整 PolarQuant 量化/反量化（`polar_quant.py`）

```python
class PolarQuant:
    def __init__(self, d, bit_width, seed=42, norm_correction=True):
        self.rotation = random_rotation_dense(d, rng)       # 固定旋转矩阵
        self.centroids = optimal_centroids(bit_width, d)    # 预计算质心

    def quantize(self, x):
        # 1. 提取范数，归一化（论文 page 5 明确要求）
        norms = linalg.norm(x, axis=1)
        x_normalized = x / norms[:, newaxis]

        # 2. 随机旋转（Gaussianization）
        y = (self.rotation @ x_normalized.T).T

        # 3. 最近质心查找
        indices = nearest_centroid_indices(y, self.centroids)
        return indices, norms

    def dequantize(self, indices, norms):
        # 1. 查表还原旋转域向量
        y_hat = self.centroids[indices]

        # 2. 范数修正（关键！防止量化误差导致范数漂移）
        if self.norm_correction:
            y_hat = y_hat / ||y_hat||   # 重归一化到单位球面

        # 3. 逆旋转（rotation 是正交矩阵，转置即逆）
        x_hat_unit = (self.rotation.T @ y_hat.T).T

        # 4. 用原始范数还原尺度
        return x_hat_unit * norms[:, newaxis]
```

**存储开销**：
- 每个坐标：`b` bits（量化索引）
- 每个向量额外：`32` bits（float32 范数）
- 压缩比 vs fp16：`16d / (b·d + 32)`

---

## 5. Stage 2：QJL 残差修正

### 5.1 为什么需要 QJL

PolarQuant 是 **MSE 最优**的，但对**内积保持**有偏差。  
K Cache 的核心操作是 `Q @ K^T`（注意力分数），需要内积无偏。  
QJL（Quantized Johnson-Lindenstrauss）提供 1-bit 无偏内积估计。

### 5.2 QJL 算法（`qjl.py`）

```python
# 数学原理：
# 量化：z = sign(S · r)，S ~ N(0,1)^(d×d)，r 是 PolarQuant 残差
# 还原：r̃ = √(π/2) / d · γ · Sᵀ · z
# 性质：E[<r̃, q>] = <r, q>（无偏内积估计）

class QJL:
    def __init__(self, d, seed=123):
        self.S = rng.standard_normal((d, d))   # 随机投影矩阵

    def quantize(self, r):
        norms = linalg.norm(r, axis=1)
        projected = (self.S @ r.T).T           # 随机投影
        signs = sign(projected).astype(int8)   # 1-bit 量化（仅存符号）
        return signs, norms

    def dequantize(self, signs, norms):
        reconstructed = (self.S.T @ signs.T).T
        scale = sqrt(π/2) / d * norms          # 无偏缩放因子
        return reconstructed * scale
```

### 5.3 完整两阶段流水线（`turboquant.py`）

```python
class TurboQuant:
    def quantize(self, x):
        # Stage 1: PolarQuant (b-1) bits → MSE 最优重建 + 残差
        mse_indices, vector_norms, residual = self.polar_quant.quantize_and_residual(x)

        # Stage 2: QJL 1-bit 量化残差 → 消除内积偏差
        qjl_signs, residual_norms = self.qjl.quantize(residual)

        return CompressedVector(mse_indices, vector_norms, qjl_signs, residual_norms, bit_width)

    def dequantize(self, compressed):
        x_mse = self.polar_quant.dequantize(compressed.mse_indices, compressed.vector_norms)
        x_qjl = self.qjl.dequantize(compressed.qjl_signs, compressed.residual_norms)
        return x_mse + x_qjl    # 两阶段叠加

# 生产用：MSE-only（关闭 QJL）
class TurboQuantMSE:
    def quantize(self, x):
        return self.polar_quant.quantize(x)   # 直接 PolarQuant，无 QJL

    def dequantize(self, indices, norms):
        return self.polar_quant.dequantize(indices, norms)
```

---

## 6. KV Cache 差异化策略

K Cache 和 V Cache 的数学操作不同，量化目标也不同：

| Cache | 操作 | 量化目标 | 使用量化器 |
|-------|------|---------|-----------|
| **K Cache** | `Q @ Kᵀ`（注意力分数） | **内积保持** | `TurboQuant`（PolarQuant + QJL） |
| **V Cache** | `attn_weights @ V`（加权求和） | **MSE 最小** | `TurboQuantMSE`（纯 PolarQuant） |

```python
class KVCacheCompressor:
    def __init__(self, head_dim, k_bits=3, v_bits=3):
        # K: 完整两阶段（内积无偏）
        self.k_quantizer = TurboQuant(head_dim, bit_width=k_bits)
        # V: 仅 PolarQuant（MSE 最优，更简单）
        self.v_quantizer = TurboQuantMSE(head_dim, bit_width=v_bits)

    def compress(self, k_cache, v_cache):
        # k_cache shape: (num_layers, num_heads, seq_len, head_dim)
        # 量化粒度：每个 (head_dim,) 向量独立量化
        for layer in range(num_layers):
            for head in range(num_heads):
                k_vecs = k_cache[layer, head]   # (seq_len, head_dim)
                v_vecs = v_cache[layer, head]
                k_compressed = self.k_quantizer.quantize(k_vecs)
                v_indices, v_norms = self.v_quantizer.quantize(v_vecs)
```

**推荐配置**（来自实测）：

| 场景 | K Cache | V Cache | 说明 |
|------|---------|---------|------|
| 最佳质量 | `turbo4` | `turbo4` | PPL +0.93% |
| 最佳压缩 | `turbo3` | `turbo3` | PPL +1.64%，4.9x 压缩 |
| **推荐（Q4_K_M 模型）** | `q8_0` | `turbo4` | 避免 Q4_K_M 对称量化退化 |
| AMD GPU | `q8_0` | `turbo4` | AMD 上 symmetric turbo 有问题 |

---

## 7. 非整数 bit 率：Outlier 通道策略

实现 2.5-bit、3.5-bit 等**非整数精度**（`outlier.py`）：

```
2.5-bit = 25% 通道用 3-bit + 75% 通道用 2-bit
          → (0.25×3 + 0.75×2) = 2.5 avg bits/channel

3.5-bit = 50% 通道用 4-bit + 50% 通道用 3-bit
          → (0.5×4 + 0.5×3) = 3.5 avg bits/channel
```

```python
def _compute_channel_split(d, target_bits):
    low_bits = floor(target_bits)
    high_bits = low_bits + 1
    frac = target_bits - low_bits       # 高精度通道比例
    n_outlier = round(d * frac)         # 高精度通道数
    n_normal = d - n_outlier
    return n_outlier, high_bits, n_normal, low_bits

class OutlierTurboQuant:
    def quantize(self, x):
        x_outlier = x[:, self.outlier_idx]   # 高精度通道
        x_normal  = x[:, self.normal_idx]    # 低精度通道

        # 分别用不同 bit 宽的 PolarQuant 量化
        out_idx, out_norms, out_residual = self.pq_outlier.quantize_and_residual(x_outlier)
        norm_idx, norm_norms, norm_residual = self.pq_normal.quantize_and_residual(x_normal)

        # QJL 作用于完整残差（跨通道）
        full_residual = concat(out_residual, norm_residual)
        qjl_signs, residual_norms = self.qjl.quantize(full_residual)
```

**实测压缩质量**（Python 原型）：

| 配置 | 压缩比 | Cosine Sim | MSE |
|------|--------|-----------|-----|
| 2-bit | 7.1× | 0.79 | 0.0047 |
| 2.5-bit（outlier） | 4.9× | 0.86 | 0.0029 |
| 3-bit | 4.9× | 0.91 | 0.0018 |
| 3.5-bit（outlier） | 3.8× | 0.95 | 0.0009 |
| 4-bit | 3.8× | 0.96 | 0.0007 |

---

## 8. 关键工程决策

### 8.1 为什么关闭 QJL

原论文用 QJL 做 1-bit 误差修正，但实测发现：
- QJL 增加重建方差
- Softmax 对方差非常敏感，会放大 QJL 引入的噪声
- 更多质心（纯 PolarQuant）在 MSE 和内积质量上都优于 MSE+QJL 分裂
- 已被 5 个独立团队验证

**结论**：生产中用 `TurboQuantMSE`（纯 PolarQuant），不用 `TurboQuant`（含 QJL）。

### 8.2 范数修正（norm_correction）

反量化时对 `y_hat` 重归一化到单位球面，原因：
- 量化误差会导致 `||y_hat|| ≠ 1`
- 不修正会导致范数漂移，累积误差随序列长度增大
- 修正后范数误差消除，只剩方向误差

### 8.3 非对称 K/V（Asymmetric KV）

Q4_K_M 等 4-bit 权重量化模型上，对称 turbo 会导致 decode 速度退化 37.9%。  
解决方案：K 用 `q8_0`，V 用 `turbo4`，兼顾速度和压缩。

### 8.4 Block Size 32 → 128

将量化 block 从 32 增大到 128：
- 压缩比提升 12%（范数开销从 32/32=1bit/val 降到 32/128=0.25bit/val）
- 质量零损失（更大 block 内分布更稳定）

### 8.5 Sparse V（稀疏 V 反量化跳过）

注意力权重极低的 token 对输出贡献可忽略，跳过其 V 向量的反量化：
- MoE 模型 decode 速度 +22.8%
- PPL 无影响（跳过的 token 注意力权重 < 1e-6）
- 已提交 llama.cpp upstream PR #21119

---

## 9. 核心代码逐模块解析

### 9.1 `rotation.py` — 旋转矩阵生成

| 函数 | 作用 | 复杂度 |
|------|------|--------|
| `random_rotation_dense` | Haar 均匀分布旋转矩阵（QR 分解） | O(d²) |
| `random_rotation_fast` | 结构化旋转参数（D·H·D） | O(1) 生成 |
| `fast_walsh_hadamard_transform` | WHT 蝴蝶运算 | O(n log n) |
| `apply_fast_rotation` | 应用结构化旋转到单向量 | O(d log d) |
| `apply_fast_rotation_batch` | 批量向量旋转（向量化） | O(batch·d log d) |

### 9.2 `codebook.py` — 质心计算

| 函数 | 作用 |
|------|------|
| `optimal_centroids` | 入口：1/2-bit 解析解，3+bit Lloyd-Max |
| `_lloyds_gaussian` | Lloyd-Max 迭代（100 次收敛） |
| `_gaussian_conditional_expectation` | E[X\|a<X<b]，数值稳定实现 |
| `nearest_centroid_indices` | 二分搜索最近质心，O(n log k) |

### 9.3 `polar_quant.py` — PolarQuant 核心

| 方法 | 作用 |
|------|------|
| `__init__` | 预计算旋转矩阵和质心（固定，推理时复用） |
| `quantize` | 范数提取 → 旋转 → 最近质心 |
| `dequantize` | 查表 → 范数修正 → 逆旋转 → 范数还原 |
| `quantize_and_residual` | 量化 + 返回残差（供 QJL 第二阶段用） |

### 9.4 `turboquant.py` — 完整流水线

| 类 | 用途 |
|----|------|
| `TurboQuant` | 完整两阶段：PolarQuant(b-1 bit) + QJL(1 bit) |
| `TurboQuantMSE` | 仅 PolarQuant（生产推荐，无 QJL） |
| `CompressedVector` | 压缩向量容器（indices + norms + qjl_signs） |

### 9.5 `kv_cache.py` — KV Cache 集成

| 类/方法 | 作用 |
|---------|------|
| `KVCacheCompressor` | K 用 TurboQuant，V 用 TurboQuantMSE |
| `compress` | 全量压缩 (num_layers, num_heads, seq_len, head_dim) |
| `decompress` | 全量解压 |
| `compress_token` | 流式压缩（逐 token，推理时用） |

---

## 10. 实测效果与验证数据

### 10.1 Gaussianization 验证（Qwen3-1.7B 真实 KV 张量）

```
原始峰度 kurtosis:  900.4
旋转后峰度:           2.9  （Gaussian 理论值 = 3.0）
旋转后标准差:    0.088388  = 1/√d（理论值完全吻合）
比值:               1.000  （完美）
```

### 10.2 长文本 PPL（wikitext-103，32K ctx，50 chunks，CI ±0.021）

| 配置 | PPL | vs q8_0 |
|------|-----|---------|
| q8_0（8-bit KV） | 7.0638 | — |
| q4_0（4-bit KV） | 7.0857 | +0.31% |
| turbo3（3.5-bit） | 7.1796 | +1.64% |
| turbo3 + sparse V | 7.1796 | +1.64%（sparse V 零影响） |

### 10.3 NIAH 检索（Qwen3.5-35B-A3B，M5 Max 128GB）

| 配置 | 单针检索（9 位置） | 多键检索（32K） |
|------|-----------------|--------------|
| q8_0 | 7/9 | 100% |
| turbo3 | 7/9 | 100% |
| **turbo3 + sparse V** | **9/9 (100%)** | 100% |

### 10.4 速度（M5 Max，prefill tok/s）

| 优化步骤 | Prefill tok/s | vs q8_0 |
|---------|--------------|---------|
| turbo3 fp32 WHT（初始） | 739 | 0.27x |
| + fp16 WHT | 1074 | 0.40x |
| + half4 向量化蝴蝶 | 1411 | 0.52x |
| + graph-side WHT 旋转 | 2095 | 0.78x |
| + block-32 存储 | 2747 | 1.02x |
| **+ 优化 dequant** | **2524** | **0.98x** |

### 10.5 KV Cache 内存（262K context）

| KV 类型 | Cache MiB | 节省 | 压缩比 |
|---------|-----------|------|--------|
| q8_0 | 2782 | — | baseline |
| turbo4 | 1422 | 1360 MiB | 1.96x |
| q8_0-K + turbo4-V | 2102 | 680 MiB | 1.32x |

---

## 11. 使用方式

### 11.1 构建 llama.cpp（Apple Silicon）

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache

cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 验证 turbo 类型可用
./build/bin/llama-server --help | grep turbo
```

### 11.2 推理命令

```bash
# Server 模式（推荐）
./build/bin/llama-server \
  -m models/your-model.gguf \
  --alias "model-turbo" \
  --jinja -ngl 99 -c 262144 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -np 1 --metrics --host 0.0.0.0 --port 8080

# CLI 快速测试
./build/bin/llama-cli \
  -m models/your-model.gguf \
  -ngl 99 -c 2048 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -n 100 -p "Hello world"
```

### 11.3 Python 原型验证

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -v          # 141 个测试
python3 benchmarks/demo.py           # 快速压缩 demo
python3 benchmarks/validate_real_model.py  # 真实模型验证（需下载 Qwen3-1.7B）
```

### 11.4 Cache Type 速查

| Flag | bits/val | 压缩比 vs fp16 | 说明 |
|------|----------|--------------|------|
| `turbo3` | 3.5 | **4.6x** | 3-bit PolarQuant + WHT，最高压缩，q8_0 速度 |
| `turbo4` | 4.25 | **3.8x** | 4-bit PolarQuant，最佳质量 |
| `q8_0` | 8 | 2.0x | llama.cpp 默认量化 cache |
| `q4_0` | 4 | 4.0x | llama.cpp 4-bit cache |

---

## 12. 常见误区

### ❌ 误区 1：TurboQuant 需要对特定模型做集成

**正确**：TurboQuant 作用于 KV Cache 层，与模型权重无关。任何能跑在 llama.cpp 上的模型（Qwen、Llama、Mistral 等）直接加 `--cache-type-k turbo3` 即可，无需修改模型。

### ❌ 误区 2：QJL 是必须的，关掉会损失内积精度

**正确**：QJL 在理论上消除内积偏差，但实测中 QJL 引入的方差经 softmax 放大后反而损害质量。生产中关闭 QJL，纯 PolarQuant 效果更好。

### ❌ 误区 3：K 和 V 应该用相同量化器

**正确**：K 需要内积保持（用于注意力分数计算），V 需要 MSE 最小（用于加权求和）。两者目标不同，应差异化处理。

### ❌ 误区 4：对称 turbo 在所有模型上都能用

**正确**：Q4_K_M 等 4-bit 权重量化模型上，对称 turbo（K 和 V 都用 turbo）会导致 decode 速度退化 37.9%，AMD GPU 上 symmetric turbo 也有问题。推荐 `q8_0-K + turbo4-V` 的非对称配置。

### ❌ 误区 5：旋转矩阵每次推理都要重新生成

**正确**：旋转矩阵由固定 seed 生成，在 `__init__` 时预计算一次，推理时复用。量化和反量化用同一个旋转矩阵，保证可逆性。

---

## 13. TurboQuant 在 LLM 推理流程中的位置

### 13.1 整体推理流程定位

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM 推理流程                                │
│                                                                 │
│  Input Tokens → Embedding                                       │
│       ↓                                                         │
│  ┌─── Transformer Layer × N ─────────────────────────────────┐  │
│  │                                                            │  │
│  │   Q = x @ W_q                                             │  │
│  │   K = x @ W_k  ──→ 【TurboQuant 压缩】──→ 写入 K Cache    │  │
│  │   V = x @ W_v  ──→ 【PolarQuant 压缩】──→ 写入 V Cache    │  │
│  │                                                            │  │
│  │   Decode 时：                                              │  │
│  │   K Cache ──→ 【反量化】──→ Attention Score (Q·Kᵀ)        │  │
│  │   V Cache ──→ 【反量化】──→ 加权求和 (attn_weights·V)     │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│       ↓                                                         │
│  FFN → Output Token                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 13.2 两个推理阶段的具体作用时机

| 阶段 | 操作 | TurboQuant 的角色 |
|------|------|-----------------|
| **Prefill（输入处理）** | 处理所有输入 tokens，计算每层 K/V | 压缩写入 Cache：K → TurboQuant，V → PolarQuant |
| **Decode（逐 token 生成）** | 每次生成一个新 token | 读取全部历史 K/V，反量化后参与 Attention 计算 |

**关键**：TurboQuant **完全不涉及训练**，不改变模型权重，只压缩推理时动态生成的 KV Cache。

### 13.3 为什么 KV Cache 是瓶颈

```
Llama-3 70B，128K context，fp16 KV Cache ≈ 64GB
→ 直接撑爆消费级显存

TurboQuant turbo3：压缩 4.9x → 约 13GB
→ 在 Mac M2 Pro 64GB 上可跑 128K context
```

Decode 阶段每生成一个 token，都要读取**全部历史 KV Cache**，KV Cache 越大，内存带宽压力越大，速度越慢。压缩 KV Cache 同时解决了**内存容量**和**带宽**两个瓶颈。

---

## 14. 完整计算流程详解

### 14.1 量化流程（Prefill 写入时）

```
输入向量 x，shape: (head_dim=128,)
│
▼ ── PolarQuant Stage 1 (b-1 bits) ──────────────────────────
│
│  Step 1: 提取 L2 范数并归一化（论文 page 5 要求）
│    norm = ||x||₂
│    x_unit = x / norm          ← 归一化到单位球面
│
│  Step 2: 随机旋转（WHT Gaussianization）
│    y = R @ x_unit             ← 旋转后各坐标趋向高斯分布
│                                  原始峰度 900 → 旋转后 2.9（≈ 高斯的 3.0）
│
│  Step 3: 最优质心量化（Lloyd-Max 质心）
│    idx[i] = argmin_c ||y[i] - centroid[c]||   ← 每坐标独立查最近质心
│
│  Step 4: 计算残差（供 QJL 用，生产中跳过）
│    residual = x - dequantize(idx, norm)
│
▼ ── QJL Stage 2 (1 bit，生产中关闭) ────────────────────────
│  ⚠️ 生产中关闭原因：QJL 增加重建方差，softmax 对方差敏感会放大噪声，
│     实测纯 PolarQuant 效果优于 PolarQuant+QJL（见第 8.1 节）
│
│  sign[i] = sign(P @ residual)   ← 随机投影 + 符号量化
│  residual_norm = ||residual||₂
│
▼ ── 存储（生产实际存储，无 QJL 部分）────────────────────────

CompressedVector {
    mse_indices:    (128,) int8    # 质心索引，b bits/coord（生产中 b=3 或 b=4）
    vector_norms:   float32        # 原始 L2 范数（32 bits overhead）
    # qjl_signs 和 residual_norms 在生产中不存储
}
```

### 14.2 反量化流程（Decode 读取时）

```
CompressedVector
│
▼ ── PolarQuant 反量化 ───────────────────────────────────────
│
│  Step 1: 查表还原旋转域向量
│    y_hat = centroids[mse_indices]
│
│  Step 2: 范数修正（norm_correction，关键！）
│    y_hat = y_hat / ||y_hat||    ← 重归一化到单位球面
│                                    消除量化误差导致的范数漂移
│
│  Step 3: 逆旋转（正交矩阵，转置即逆）
│    x_hat_unit = Rᵀ @ y_hat
│
│  Step 4: 乘回原始范数
│    x_mse = x_hat_unit * vector_norms
│
▼ ── QJL 反量化（生产中跳过）────────────────────────────────
│
│  x_qjl = Pᵀ @ qjl_signs * residual_norms / √d
│
▼ ── 叠加输出 ────────────────────────────────────────────────

x_hat = x_mse (+ x_qjl)   → 参与 Attention 计算
```

### 14.3 压缩比计算示例

以 `head_dim=128, turbo3（3-bit）` 为例：

```
原始 fp16：128 × 16 = 2048 bits

压缩后（生产，无 QJL）：
  K: 128 × 3 + 32(norm) = 416 bits
  V: 128 × 3 + 32(norm) = 416 bits

单向量压缩比：2048 / 416 ≈ 4.9x
```

---

## 15. 非对称 KV 量化的发现（Asymmetric KV Discovery）

> 来源：`docs/asymmetric-kv-discovery.md`，2026-03-28/29 调试记录

### 15.1 问题起因

在 Mac Mini M2 Pro 上测试时，发现 Q4_K_M 权重模型 + turbo KV 出现灾难性 PPL：

| 配置 | PPL |
|------|-----|
| q8_0 KV（基线） | 6.58 |
| turbo3 KV | 3556 ← 灾难 |
| turbo4 KV | 218 ← 灾难 |

初始怀疑是 M2 Metal 硬件 Bug，经过 8 小时排查后发现真相。

### 15.2 排查过程（五个阶段）

| 阶段 | 假设 | 结论 |
|------|------|------|
| Phase 1 | M2 Metal 硬件 Bug | ❌ 排除：M2/M5 每个张量字节完全一致 |
| Phase 2 | 模型差异 | ✅ 真相：Q4_K_M 权重 + turbo = 灾难，Q8_0 权重 + turbo = 正常 |
| Phase 3 | 非对称救援测试 | ✅ q8_0-K + turbo3-V = PPL 6.68（仅 +1.6%） |
| Phase 4 | turbo4-V NaN 根因 | ✅ 缺少 `kq8_0_vturbo4` Metal 内核实例化 |
| Phase 5 | 完整修复验证 | ✅ 新增 150 个内核，所有混合对正常工作 |

### 15.3 核心发现：量化叠加效应

```
Q4_K_M 权重量化
    → K/V 激活值已含噪声
    → 再加 turbo 量化（WHT 把噪声扩散到所有 128 个坐标）
    → PolarQuant 质心针对干净分布优化，对噪声分布效果差
    → K 的误差经 softmax 指数级放大
    → PPL 灾难性劣化

Q8_0 权重量化
    → 激活值足够干净
    → turbo 量化在容忍范围内
    → PPL 正常
```

### 15.4 非对称救援结果（Qwen2.5-7B Q4_K_M）

| K | V | PPL | vs 基线 | V 压缩比 |
|---|---|-----|---------|---------|
| q8_0 | q8_0 | 6.58 | — | 1.0x |
| q8_0 | turbo4 | **6.64** | +1.0% | 2.0x |
| q8_0 | turbo3 | **6.71** | +2.0% | 2.3x |
| q8_0 | turbo2 | **6.91** | +5.1% | 3.2x |
| turbo3 | turbo3 | 3556 | 灾难 | — |

**结论**：K 精度是主导质量因素。K 决定注意力路由，误差经 softmax 指数级放大；V 的误差只是线性叠加，即使 2-bit 也只有 +5.1% PPL 损失。

### 15.5 使用建议

| 场景 | 推荐配置 |
|------|---------|
| Q4_K_M 模型（敏感） | `-ctk q8_0 -ctv turbo4` |
| Q8_0 或更高权重量化 | `-ctk turbo3 -ctv turbo3` |
| 最大 V 压缩（实验性） | `-ctk q8_0 -ctv turbo2`（3.2x，+5.1% PPL） |
| Q4_K_M 大模型（如 Mistral-24B） | 先试 `-ctk turbo3 -ctv turbo3`，不行再换非对称 |

---

## 16. 上下文退化问题与修复（Context Scaling）

> 来源：`docs/context-scaling-deep-dive.md`，Issue #32

### 16.1 问题现象

turbo3 在短上下文时速度与 q8_0 持平，但随上下文增长差距扩大：

| Context | turbo3/q8_0 |
|---------|-------------|
| 1024 | 0.976x |
| 2048 | 0.960x |
| 4096 | **0.921x** ← 越来越慢 |

极端情况：M1 Max 64GB，42K context，decode 从 11 t/s 跌到 4 t/s（**0.36x**）。

### 16.2 排查：三个红鲱鱼

| 假设 | 测试方法 | 结论 |
|------|---------|------|
| WHT 旋转 matmul 随 context 线性增长 | 实现 O(d log d) 自定义 WHT 算子 | ❌ 性能几乎不变，旋转不是瓶颈 |
| `ggml_cont` 开销 | 跳过不必要的连续化操作 | ❌ +1%，可忽略 |
| 旋转组从 128 缩到 32 | 减少计算量 | ❌ PPL 从 6.19 劣化到 7.06，质量不达标 |

### 16.3 真正根因：Flash Attention 内核里的逐位置反量化

每处理一个缓存 token，都要做 turbo3 反量化：

```
turbo3 反量化（每 4 个元素）：
  - 读 qs 字节（1 次设备内存读）
  - 位移 + 掩码提取 low2 bits（2 次 ALU × 4 元素）
  - 读 signs 字节（1 次设备内存读）
  - 位移 + 掩码提取 hi1 bit（2 次 ALU × 4 元素）
  - 组合索引（2 次 ALU × 4 元素）
  - 查质心表（1 次常量内存读 × 4 元素）
  - 乘范数（1 次乘法 × 4 元素）
  总计：~2 次设备读 + 7 次 ALU + 4 次常量读 + 4 次乘法

q8_0 反量化（每 4 个元素）：
  - 读 4 个 int8（1 次设备内存读）
  - 乘 scale（4 次乘法）
  总计：~1 次设备读 + 4 次乘法
```

**turbo3 反量化计算量约是 q8_0 的 3-4 倍**，乘以所有缓存位置数，随 context 线性放大。

短上下文时，turbo3 的内存带宽优势（KV Cache 小 2.3x）能覆盖额外计算开销；长上下文时，反量化计算量主导，带宽优势相对缩小。

### 16.4 修复：优化反量化实现

**根本原因**：原实现对同一个 qs/signs 字节读取 4 次（每元素读一次），改为**批量读取一次 + 循环展开**。

**修复效果：**

| Context | 修复前 turbo3/q8_0 | 修复后 turbo3/q8_0 |
|---------|-----------------|-----------------|
| 1024 | 0.976x | **0.981x** |
| 2048 | 0.960x | **0.989x** |
| 4096 | 0.921x | **0.981x** |
| 8192 | — | **0.995x** |
| 32768 | — | **0.995x** |

**2K → 32K 全范围验证，比值稳定在 0.987x ~ 0.995x，退化趋势完全消除，质量（PPL）不受影响。**

---

## 17. 注意力门控优化（Attention-Gated Optimizations）

> 来源：`docs/attention-gated-optimizations.md`

**核心思路**：Flash Attention 计算过程中，注意力权重（`ss[]`）已经算出来了，可以用它来**门控后续计算**——权重极小的位置跳过对应计算，节省开销。

### 17.1 四个候选优化点

| 优化点 | 状态 | 结论 |
|--------|------|------|
| **Tile 级 V skip** | ❌ 已验证，不值得做 | MoE +0.6-0.9%，稠密模型 -0.8-1.1%。扫描 tile max（读 8 次 ss[]）的开销比节省的还多 |
| **f16 路径 Sparse V** | ❌ 已验证，不值得做 | 短上下文 -0.3%，长上下文 +1.3%。f16 路径太便宜，分支开销抵消收益 |
| **O 重缩放跳过** | ⏳ 仓库待实现 | `O = diag(ms)*O` 在 max 未变时（`ms ≈ 1.0`）是无效乘法，加 `if (|ms-1.0|>ε)` 门控。代码位置：`ggml-metal.metal` ~line 7298-7303。预期收益小，风险低但需注意浮点边界 |
| **exp() 跳过** | ⏳ 仓库待实现 | `s-M < -20` 时 `exp(s-M) ≈ 2e-9 ≈ 0`，直接写 0 跳过 exp()。代码位置：`ggml-metal.metal` ~line 7291。exp() 在 GPU 上开销大，长上下文时大多数 score 远低于 max，预期收益中等，数学上完全等价 |

### 17.2 已实现的 Sparse V（最重要）

Sparse V 是已经落地的注意力门控优化，逻辑：

```
for each cached token position i:
    if ss[i] < threshold τ:
        skip V dequant and accumulate   ← 注意力权重极小，跳过
    else:
        dequant V[i] and accumulate     ← 正常处理
```

**实测效果**：
- MoE 模型（Qwen3.5-35B-A3B）decode 速度 +22.8%
- PPL 无影响（跳过的 token 注意力权重 < 1e-6）
- 已提交 llama.cpp upstream PR #21119
