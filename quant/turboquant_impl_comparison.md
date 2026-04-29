# TurboQuant 实现对比分析：turboquant_plus vs vLLM PR #38479

> 本文档对比两个 TurboQuant KV Cache 量化实现与论文原始算法的差异，供内部评估参考。  
> 对比基准：TurboQuant 原论文（ICLR 2026, arXiv 2504.19874）。  
> 相关文档：`turboquant_kv_cache.md`（turboquant_plus 知识库）、`turboquant_arxiv/paper.md`（论文原文）。

---

## 目录

1. [概述](#1-概述)
2. [论文算法回顾](#2-论文算法回顾)
3. [逐维度对比总表](#3-逐维度对比总表)
4. [关键差异深度分析](#4-关键差异深度分析)
5. [一致点](#5-一致点)
6. [结论与建议](#6-结论与建议)

---

## 1. 概述

### 1.1 两个实现的定位

| | turboquant_plus | vLLM PR #38479 |
|---|---|---|
| **仓库** | `turboquant_plus` + `llama-cpp-turboquant` | `vllm-project/vllm` (PR by @vibhavagarwal5) |
| **目标平台** | llama.cpp（CPU/Metal/CUDA） | vLLM（GPU/Triton） |
| **实现语言** | Python 原型 + C/Metal 内核 | Python + Triton 内核 |
| **论文忠实度** | 高（K/V 都用 PolarQuant） | 中（仅 K 用 TurboQuant，V 用 uniform） |
| **状态** | 已在 llama.cpp fork 中可用 | PR 已获 APPROVE，pre-commit 失败待修 |
| **社区验证** | 由 TheTom 主导，多方验证 PPL/NIAH | 5+ 独立团队验证，mgoin(vLLM maintainer) APPROVED |

### 1.2 预设配置对比

**turboquant_plus**:

| Cache Type | bits/val | 压缩比 vs fp16 |
|-----------|----------|--------------|
| `turbo3` | 3.5 | 4.9x |
| `turbo4` | 4.25 | 3.8x |

**vLLM PR**:

| Preset | Keys | Values | 压缩比 |
|--------|------|--------|--------|
| `turboquant_k8v4` | FP8 (8-bit) | 4-bit uniform | ~2.6x |
| `turboquant_4bit_nc` | 4-bit MSE | 4-bit uniform | ~3.8x |
| `turboquant_k3v4_nc` | 3-bit MSE | 4-bit uniform | ~3.5x |
| `turboquant_3bit_nc` | 3-bit MSE | 3-bit uniform | ~4.9x |

---

## 2. 论文算法回顾

### 2.1 Algorithm 1: TurboQuant_mse（MSE 最优量化）

```
输入：x ∈ S^(d-1)（单位球面上的 d 维向量），bit-width b

1. 生成随机旋转矩阵 Π ∈ R^(d×d)（通过 QR 分解 Haar 分布）
2. 旋转：y = Π · x
   → 每个坐标 y_j ~ Beta((d-1)/2, (d-1)/2)，高维近似 N(0, 1/d)
3. 标量量化：idx_j = argmin_k |y_j - c_k|，c_k 为 Lloyd-Max 最优质心
4. 存储：idx（b-bit 索引）

反量化：
1. 查表：ỹ_j = c_{idx_j}
2. 逆旋转：x̃ = Πᵀ · ỹ
```

**保证**：`D_mse ≤ (√3·π/2) · 1/4^b`，距信息论下界仅差 ~2.7 倍常数因子。

### 2.2 Algorithm 2: TurboQuant_prod（内积最优量化）

```
输入：x ∈ S^(d-1)，bit-width b

1. 用 (b-1) bits 执行 TurboQuant_mse → 得到 idx 和残差 r = x - dequant(idx)
2. QJL 1-bit 量化残差：qjl = sign(S · r)，S ~ N(0,1)^(d×d)
3. 存储：(idx, qjl, ||r||₂)

反量化：
1. x̃_mse = dequant_mse(idx)
2. x̃_qjl = √(π/2)/d · ||r|| · Sᵀ · qjl
3. x̃ = x̃_mse + x̃_qjl
```

**保证**：内积估计**无偏**，`E[<y, x̃>] = <y, x>`。

### 2.3 论文关键要素清单

| 要素 | 论文要求 |
|------|---------|
| 旋转矩阵 | Haar 分布随机正交矩阵（QR 分解） |
| 标量量化 | Lloyd-Max 最优质心，对 Beta/Gaussian 分布 |
| QJL | 残差 1-bit 量化，保证内积无偏 |
| K/V 处理 | **对称**：K 和 V 都用 TurboQuant |
| 范数存储 | float 精度单独存储 |
| 维度要求 | 任意 d（QR 分解支持任意维度） |

---

## 3. 逐维度对比总表

| 维度 | 论文 | turboquant_plus | vLLM PR #38479 | 差异等级 |
|------|------|----------------|----------------|---------|
| **V Cache 量化** | TurboQuant（旋转+Lloyd-Max） | PolarQuant（旋转+Lloyd-Max） | **Uniform min-max**（无旋转） | **严重偏离** |
| **旋转结构** | Haar 随机正交 (QR) | D₂·H·D₁（双层随机+WHT） | D·H（单层随机+WHT） | 中等偏离 |
| **非 pow2 dim** | 任意 d (QR) | pad 到 next_power_of_2 | **不处理（bug）** | **严重** |
| **QJL** | 有（内积无偏） | 有（生产关闭） | 移除 | 一致（合理） |
| **范数精度** | float 精度 | float32 | **fp16** | 中等偏离 |
| **Lloyd-Max** | 数值求解 | 闭式条件期望 | 梯形积分 n=200 | 轻微偏离 |
| **Norm correction** | 无 | 有 | 有 | 一致（工程增强） |
| **Sparse V** | 无 | 已实现 (+22.8%) | 有 env var，不确定 | — |
| **量化叠加效应** | 未讨论 | 发现并提出非对称救援 | 未讨论 | — |
| **Outlier 通道** | 有（2.5/3.5-bit） | 有 | 无 | — |

---

## 4. 关键差异深度分析

### 4.1 Value 量化算法 — 最核心的差异

**这是两个实现之间最本质的区别。**

#### 论文要求

论文中 K 和 V 都是 "高维向量"，都需要量化后保持欧几里得结构。Algorithm 1 的证明对 K 和 V 同样成立——两者都应经过随机旋转 + Lloyd-Max 最优标量量化。

#### turboquant_plus 实现

K 和 V 都使用 PolarQuant（旋转 + Lloyd-Max）：

```python
class KVCacheCompressor:
    def __init__(self, head_dim, k_bits=3, v_bits=3):
        # K: 完整两阶段（内积保持）
        self.k_quantizer = TurboQuant(head_dim, bit_width=k_bits)
        # V: PolarQuant（MSE 最优）— 同样是旋转+Lloyd-Max
        self.v_quantizer = TurboQuantMSE(head_dim, bit_width=v_bits)
```

V 经过旋转后每个坐标近似 N(0, 1/d)，Lloyd-Max 质心对此分布是 MSE 最优的。

#### vLLM PR 实现

V 使用简单的 uniform min-max 量化，**没有旋转，没有 Lloyd-Max**：

```python
# triton_turboquant_store.py — _store_quantized_value 函数
val_min = tl.min(val_vec)
val_max = tl.max(val_vec)
v_scale = (val_max - val_min) / 15.0   # 4-bit: 16 均匀 levels
q_all = round((val_vec - val_min) / v_scale)  # 线性量化
```

#### 数学影响

对于 N(0, σ²) 分布，uniform 量化 vs Lloyd-Max 最优量化的 MSE 比较：

| bit-width | Lloyd-Max MSE | Uniform MSE | Uniform 劣化 |
|-----------|-------------|-------------|-------------|
| 2-bit | 0.1175/d | 0.167/d | +42% |
| 3-bit | 0.0340/d | 0.042/d | +24% |
| 4-bit | 0.0094/d | 0.011/d | +17% |

Lloyd-Max 质心集中在概率密度高的区域（靠近均值），而 uniform 均匀分布在 [min, max] 范围，在尾部浪费编码空间。

此外，V 未经旋转直接量化意味着：
- 原始 V 向量的坐标分布可能是**重尾、高峰度**的（turboquant_plus 验证了 Qwen3-1.7B 上 kurtosis 高达 900）
- Uniform 量化对重尾分布效果更差——尾部 outlier 把 [min, max] 范围撑大，中间大量值的量化精度降低

#### 影响评估

**严重**。社区测试中反复出现的 V 精度问题（MidasMining: 2-bit V 导致推理质量崩塌；vipin-sa-16319: 吞吐量大幅下降）可能部分源于 V 的 uniform 量化质量不够。V 的误差在 attention 输出中是**线性叠加**的（`output = Σ attn_weight_i · V_i`），虽然不像 K 那样经 softmax 指数放大，但在长序列中误差会累积。

---

### 4.2 旋转矩阵结构 — 单层 vs 双层随机化

#### 论文要求

Haar 分布随机正交矩阵 Π，通过对随机高斯矩阵 G ~ N(0,1)^(d×d) 做 QR 分解获得。这保证旋转后向量均匀分布在单位球面上，每个坐标精确服从 Beta 分布，且不同坐标近独立。

#### turboquant_plus：D₂ · H · D₁

```python
def random_rotation_fast(d, rng):
    padded_d = next_power_of_2(d)
    signs1 = rng.choice([-1.0, 1.0], size=padded_d)  # D₁
    signs2 = rng.choice([-1.0, 1.0], size=padded_d)  # D₂
    return signs1, signs2, padded_d

# 应用：y = D₂ · H · D₁ · x
```

使用**两个**独立的随机 ±1 对角矩阵夹 Hadamard 变换。这是随机线性代数中的标准结构（Subsampled Randomized Hadamard Transform, SRHT），两层随机化提供更好的伪随机混合。

#### vLLM PR：D · H

```python
# turboquant_attn.py — _ensure_on_device
H = _build_hadamard(D, str(device))
layer._tq_PiT = (signs.unsqueeze(1) * H).contiguous()  # D · H
```

只使用**一个**随机符号矩阵。Hadamard 矩阵 H 本身有非常规则的结构（所有元素为 ±1/√d），单层符号翻转可能不足以打破这种规则性，导致：
- 某些坐标间的相关性未被充分消除
- 旋转后的分布与理论 N(0, 1/d) 的偏差更大

#### 数学分析

设输入向量 x 的某两个坐标高度相关。

- **D₂·H·D₁**：D₁ 先打乱符号 → H 混合所有坐标 → D₂ 再次打乱符号。两次独立的随机符号翻转使得输出坐标的 pairwise 相关性以 O(1/d) 衰减。
- **D·H**：H 混合所有坐标 → D 打乱符号。但如果输入坐标 i,j 在 H 矩阵中对应相同的蝴蝶模式，单次符号翻转无法打破相关性。

#### 影响评估

**中等**。对于 d ≥ 128 的高维情况，差异较小（Central Limit Theorem 的收敛效应主导）。对于 d=64 或 d=96 等较小维度，双层随机化的质量优势更明显。

---

### 4.3 非 2 的幂 head_dim — 硬性 Bug

#### 论文

QR 分解对任意 d 都有效，无维度限制。

#### turboquant_plus

显式处理 padding：

```python
def random_rotation_fast(d, rng):
    padded_d = next_power_of_2(d)  # d=96 → padded_d=128
    signs1 = rng.choice([-1.0, 1.0], size=padded_d)
    signs2 = rng.choice([-1.0, 1.0], size=padded_d)
    return signs1, signs2, padded_d
```

量化前 pad 到 128 维，反量化后截断回 96 维。

#### vLLM PR

```python
@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:                    # d=96 → 生成 128×128 矩阵
        H = torch.cat([torch.cat([H, H], 1),
                       torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(...)         # ← 用 d=96 归一化 128×128 矩阵
```

问题链条：
1. `_build_hadamard(96)` 生成 128×128 矩阵
2. `signs` 的形状是 `(96,)`（来自 `generate_wht_signs(d=96, ...)`)
3. `PiT = signs.unsqueeze(1) * H` → 形状不匹配：`(96,1) * (128,128)` → 广播错误或静默错误
4. 即使用 `H[:d, :d]` 截取 96×96 子矩阵，截断后的矩阵**不再正交**，旋转质量劣化

#### 受影响的模型

| 模型 | head_dim | 是否 pow2 | 受影响 |
|------|----------|----------|--------|
| Llama-3 | 128 | 是 | 否 |
| Qwen-2.5 | 128 | 是 | 否 |
| **Phi-3/Phi-4** | **96** | **否** | **是** |
| **Gemma-2** | **256** | 是 | 否 |
| **Gemma-4** (global) | **512** | 是 | 否 |

#### 影响评估

**严重（对特定模型）**。Phi-3-mini (head_dim=96) 等模型会直接崩溃或产生错误结果。

---

### 4.4 范数存储精度

#### 论文 & turboquant_plus

范数用 float32（32 bits）存储：

```python
# turboquant_plus — polar_quant.py
norms = linalg.norm(x, axis=1)   # float32
# 存储：CompressedVector.vector_norms: float32
```

#### vLLM PR

范数用 fp16（16 bits）存储：

```python
# triton_turboquant_store.py — _tq_fused_store_mse
vn_f16 = tl.load(Norms_ptr + pid).to(tl.float16)    # 截断为 fp16
vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
tl.store(KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8))
```

#### 影响分析

反量化时范数是**乘性因子**：`x̂ = x̂_unit × norm`。fp16 的精度特性：

| 范围 | fp16 精度 | 相对误差 |
|------|----------|---------|
| [1, 2] | 2⁻¹⁰ ≈ 0.001 | 0.1% |
| [1024, 2048] | 1.0 | 0.05-0.1% |
| [0.001, 0.002] | 2⁻²⁰ ≈ 10⁻⁶ | 0.1% |
| **[0, 6e-5]** | **flush to zero** | **100%** |

对于 attention head 中范数非常小的向量（接近零向量），fp16 会直接 flush to zero，完全丢失信息。

turboquant_plus 的存储开销：每向量 32 bits 额外开销（float32 norm）。  
vLLM PR 的存储开销：每向量 16 bits（fp16 norm），节省 16 bits/vector，但牺牲精度。

#### 影响评估

**中等**。在大多数情况下 fp16 精度足够，但对于范数极小的向量或需要高精度的长序列推理场景，可能引入可观测的误差。每向量节省的 16 bits（对 head_dim=128 的 3-bit 量化，总存储 ~416 bits）只占约 3.8% 的存储开销，收益很小。

---

### 4.5 Lloyd-Max 求解器实现

#### turboquant_plus

对 1-bit 和 2-bit 使用**解析解**，3+ bit 使用 Gaussian 条件期望的**闭式公式**：

```python
# 解析解
if bit_width == 1:
    c = sqrt(2.0 / (pi * d))
    return [-c, c]
if bit_width == 2:
    return [-1.51, -0.453, 0.453, 1.51] / sqrt(d)

# 3+ bit: 精确的 Gaussian 条件期望（闭式）
# E[X | a < X < b] = σ · (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
# φ = Gaussian PDF, Φ = Gaussian CDF
```

条件期望 `E[X | a < X < b]` 对 Gaussian 分布有精确的闭式解，无需数值积分。

#### vLLM PR

所有 bit-width 都使用**梯形法则数值积分**（n=200 步）：

```python
def _trapz(f, a: float, b: float, n: int = 200) -> float:
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h

# Lloyd-Max 迭代中的质心更新：
num = _trapz(lambda x: x * pdf(x), a, b)   # ∫ x·f(x) dx
den = _trapz(pdf, a, b)                      # ∫ f(x) dx
new_centroid = num / den
```

#### 精度差异

对于 N(0, 1/d) 在 d=128 时，σ = 1/√128 ≈ 0.0884。Gaussian 在 ±3σ ≈ ±0.265 范围内集中了 99.7% 的概率。

梯形法则的误差：

- 积分区间 [-0.31, 0.31]，n=200 步，步长 h ≈ 0.003
- 梯形法则误差 = O(h²) ≈ O(10⁻⁵)
- 对于尾部区间（概率密度很低），被积函数变化剧烈，200 步可能不够精确

对于实际的 4-bit (16 质心) 量化，最外侧质心的区间概率极低（~0.3%），这些区间的条件期望计算受梯形法则精度限制最大。

#### 影响评估

**轻微**。Lloyd-Max 迭代本身会收敛到正确的质心（即使单步积分有小误差，多次迭代会修正），且质心被 `@lru_cache` 缓存，只计算一次。实际质心差异可能在第 4-5 位有效数字，对量化质量影响微乎其微。

---

### 4.6 量化叠加效应

#### turboquant_plus 的发现

在 Q4_K_M 权重量化模型上，symmetric turbo（K 和 V 都用 TurboQuant）导致灾难性 PPL 劣化：

```
Q4_K_M 权重量化 → 激活值含噪声
→ WHT 把噪声扩散到所有 128 个坐标
→ Lloyd-Max 质心针对干净 N(0,1/d) 优化，对噪声分布效果差
→ K 的误差经 softmax 指数级放大
→ PPL: 6.58 → 3556（灾难）
```

**非对称救援**：K 用 q8_0（高精度），V 用 turbo（压缩） → PPL 恢复到 6.64（仅 +1.0%）。

#### vLLM PR

未讨论此问题。vLLM 主要面向 GPU 上的 fp16/bf16 模型，不太会用 Q4_K_M。但如果用户在以下场景启用 TurboQuant，可能遇到类似问题：
- FP8 权重量化模型 + TurboQuant KV
- AWQ/GPTQ 4-bit 权重模型 + TurboQuant KV（社区已有人测试 Gemma-4 AWQ + TQ）

vLLM PR 的 `turboquant_k8v4`（FP8 keys + 4-bit values）预设在一定程度上避免了此问题——FP8 keys 不经过旋转/量化，保持了 K 的精度。但其他预设（如 `turboquant_3bit_nc`）在低精度权重模型上可能出问题。

---

## 5. 一致点

以下方面两个实现与论文的偏离方向一致：

### 5.1 QJL 关闭

两个实现都选择在生产中关闭 QJL：

- **论文**：QJL 保证内积无偏，理论上是必要的
- **实践共识**：QJL 增加方差，softmax 对方差敏感会放大噪声。5+ 独立团队验证纯 PolarQuant 效果优于 PolarQuant + QJL
- **数学解释**：在 b ≥ 3 时 MSE 量化器的内积偏差已经很小（<3%），QJL 引入的方差代价大于偏差纠正的收益

### 5.2 Norm Correction

两个实现都加入了论文没有的**范数修正**（反量化时将 ỹ 重归一化到单位球面）：

```python
# 两个实现相同的逻辑
y_hat = centroids[indices]
y_hat = y_hat / ||y_hat||   # 重归一化，消除量化导致的范数漂移
x_hat = Rᵀ @ y_hat
x_hat = x_hat * original_norm
```

这是一个合理的工程增强——量化误差会导致 `||ỹ|| ≠ 1`，不修正会在逆旋转后引入方向误差。

### 5.3 WHT 替代 QR

两个实现都用 Walsh-Hadamard Transform 替代论文的 QR 分解随机旋转：
- QR 分解：O(d³) 构造，精确 Haar 分布
- WHT：O(d log d) 或 O(d²) matmul，近似但更快

对 d ≥ 64 的 attention head dimension，WHT 提供足够好的近似。

### 5.4 Gaussian 近似 Beta 分布

两个实现都直接用 N(0, 1/d) 计算 Lloyd-Max 质心，而非论文中的精确 Beta 分布：

```
精确：f_X(x) = Γ(d/2) / (√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)
近似：f_X(x) ≈ N(0, 1/d)
```

对 d ≥ 64，Beta 与 Gaussian 的 KL 散度 < 10⁻⁴，近似完全可接受。

---

## 6. 结论与建议

### 6.1 总体评估

| 实现 | 论文忠实度 | 工程完善度 | K 量化质量 | V 量化质量 |
|------|----------|----------|----------|----------|
| turboquant_plus | 高 | 高 | 优 | 优 |
| vLLM PR | 中 | 中 | 良 | **差** |

vLLM PR 的准确描述应该是 **"TurboQuant Keys + Uniform Values"**，而非完整的 TurboQuant。

### 6.2 如果要改进 vLLM PR

按优先级排序：

1. **[P0] 修复非 pow2 head_dim**：对 Hadamard 矩阵构造增加 padding/truncation 处理，或在 `from_cache_dtype` 中检查 head_dim 是否为 2 的幂并给出明确错误
2. **[P1] V 量化改用 PolarQuant**：将 V 的 uniform 量化替换为旋转 + Lloyd-Max，与 K 共享旋转矩阵和质心生成逻辑
3. **[P2] 旋转改为双层 D₂·H·D₁**：增加第二个随机符号矩阵，提升伪随机质量
4. **[P3] 范数改用 float32 或 bfloat16**：fp16 的精度收益太小（节省 2 bytes/vector），不值得精度损失

### 6.3 如果接受现状

vLLM PR 在以下场景仍然实用：
- `turboquant_k8v4`（FP8 keys）：K 精度不受 TQ 量化影响，V 虽然是 uniform 但 4-bit 精度尚可，提供 ~2.6x 压缩
- 大模型长上下文：即使 V 量化不够好，内存节省带来的批处理能力提升（4x 容量）可能抵消质量损失
- 头 d=128 的主流模型：head_dim 是 2 的幂，不触发 Hadamard bug

---

*文档版本：2026-04-14*  
*对比基准：TurboQuant 论文 (arXiv 2504.19874)、turboquant_plus 仓库、vLLM PR #38479 (commit a8d08c6b)*
