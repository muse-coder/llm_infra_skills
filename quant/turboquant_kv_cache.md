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
