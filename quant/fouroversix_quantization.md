# Four Over Six (4/6) 量化算法详解

> 论文：[arXiv:2512.02010](https://arxiv.org/abs/2512.02010)、[arXiv:2603.28765](https://arxiv.org/abs/2603.28765)
> 核心代码：`src/fouroversix/quantize/pytorch/reference.py`

---

## 一、核心思想

标准 NVFP4 对每个 16 元素的块用一个固定的 E4M3 缩放因子把数据归一化到 E2M1 范围，然后量化。

**4/6 的创新**：为每个块并行计算**两种缩放方案**，再逐块选择误差最小的那个。

> **命名含义**："6"和"4"指的是 E2M1 量化值的有效范围上限，不是位宽。
> - **6 方案**：块内最大值映射到 E2M1 的 ±6（用满全部动态范围）
> - **4 方案**：块内最大值映射到 E2M1 的 ±4（只用精细的低值区间）
>
> 两种方案的量化值都是 **E2M1（4-bit）**，缩放因子都是 **E4M3（8-bit）**，总开销不变。

**两级量化结构**：

```
原始值  →  [E4M3 缩放因子]  →  归一化值  →  [E2M1 量化]  →  量化码
```

---

## 二、关键常量与数值

### 2.1 ScaleRule —— 控制走哪条路径

```python
# src/fouroversix/utils.py
class ScaleRule(str, Enum):
    mse     = "mse"      # 4/6 模式（默认）：均方误差选择
    mae     = "mae"      # 4/6 模式：平均绝对误差选择
    abs_max = "abs_max"  # 4/6 模式：最大绝对误差选择
    static_6 = "static_6" # 标准 NVFP4：不选择，固定 6 方案
    static_4 = "static_4" # 固定 4 方案（不常用）

    @property
    def is_static(self) -> bool:
        return self in {ScaleRule.static_4, ScaleRule.static_6}
```

- `mse / mae / abs_max` → **4/6 自适应模式**，`is_static = False`
- `static_6 / static_4` → **固定模式**，`is_static = True`

### 2.2 max_scale_factor：448 vs 256 的本质

```python
# ScaleType.nv（E4M3 缩放因子）
def get_maximum_value(self, scale_rule) -> int:
    return 448 if scale_rule.is_static else 256
```

| scale_rule | is_static | max_scale_factor | 说明 |
|---|---|---|---|
| `static_6` | True | **448** | 用满 E4M3 全部范围 |
| `static_4` | True | **448** | 用满 E4M3 全部范围 |
| `mse/mae/abs_max` | False | **256** | 为 ×1.5 留空间 |

**为什么 4/6 模式用 256 而非 448？唯一原因：**

> 4/6 模式需要在 6 方案基础上再生成一个 `scale × 1.5` 的 4 方案。
> 如果上限是 448，则 448 × 1.5 = 672，超出 E4M3 物理上限（448），无法存储。
> 上限设为 256，则 256 × 1.5 = **384 ≤ 448**，4 方案的缩放因子可以正常表示。

**注意：256 vs 448 与内层量化映射到 4 还是 6 完全无关**（详见第三节推导）。

### 2.3 max_quantized_value

```python
# QuantizedValueType.fp4
def get_maximum_value(self, scale_rule) -> int:
    if scale_rule == ScaleRule.static_4:
        return 4   # 固定 4 方案
    return 6       # 所有其他情况（包括 4/6 模式）都返回 6
```

注意：4/6 模式下 `max_quantized_value` 始终为 **6**，不管最终选了 4 还是 6 方案。"4"是由 `scale_expansion_factor=1.5` 在运行时产生的效果，不是通过修改这个常量实现的。

### 2.4 三种 case 完整对照

| | `static_6`（标准 NVFP4） | 4/6 模式 — 6 方案 | 4/6 模式 — 4 方案 |
|---|---|---|---|
| `max_quantized_value` | 6 | 6 | **6**（不变） |
| `max_scale_factor` | **448** | **256** | **256**（不变） |
| `encode_scale` 分母 | 6×448=**2688** | 6×256=**1536** | 6×256=**1536** |
| `scale_hp` for max block | **448** | **256** | 256×1.5=**384** |
| E4M3 存储值 | 448 | 256 | 384 |
| `decode_scale` 分母 | 6×448=**2688** | 6×256=**1536** | 6×256=**1536**（同 6 方案） |
| **内层最大映射值** | **6.0** | **6.0** | **4.0** |

---

## 三、为什么 4 方案内层最大值是 4.0（数学推导）

这是理解整个算法的核心，展开推导消除疑惑：

```
x_block_scaled = x_scale_blocks / (decode_scale × scale)
```

其中：

```
decode_scale = x_amax / (max_q × max_s)    # max_q=6, max_s=256 in 4/6 mode

scale_6 = block_amax / max_q × (max_q × max_s / x_amax)
        = block_amax / x_amax × max_s
        = block_amax / x_amax × 256

scale_4 = scale_6 × 1.5
        = block_amax / x_amax × 384
```

代入求 block_amax 归一化后的值：

```
6 方案：x_block_scaled_max = block_amax / (decode_scale × scale_6)
      = block_amax / (x_amax/1536 × 256)
      = block_amax × 1536 / (x_amax × 256)
      = block_amax / x_amax × 6.0   →  最大值 = 6.0 ✓

4 方案：x_block_scaled_max = block_amax / (decode_scale × scale_4)
      = block_amax / (x_amax/1536 × 384)
      = block_amax × 1536 / (x_amax × 384)
      = block_amax / x_amax × 4.0   →  最大值 = 4.0 ✓
```

**关键洞察：max_s（256）在分子分母完全抵消**。内层映射值只取决于：
- `max_quantized_value`（6）
- `scale_expansion_factor`（1.5，使分母多一个 1.5）

**256 vs 448 只影响 E4M3 中实际存储的数值大小，不影响内层量化范围。**

---

## 四、E2M1 精度结构 —— 两种方案互补的根本原因

E2M1 可表示的正数值：`{0, 0.5, 1, 1.5, 2, 3, 4, 6}`

量化步长按指数区间递增（**高值区间精度粗糙**）：

| 区间 | 步长 | 可表示值 |
|---|---|---|
| [0, 2) | **0.5** | 0, 0.5, 1, 1.5 |
| [2, 4) | **1** | 2, 3 |
| [4, 6] | **2** | 4, 6 |

- **6 方案**：块内最大值映射到 6，部分值会落入 [4, 6] 的粗糙区间（步长 2）
- **4 方案**：块内最大值映射到 4，所有值都在步长 ≤ 1 的精细区间内

当块内数据分布"扁平"（多数值接近最大值）时，4 方案能减少大量舍入误差。当数据分布"集中在小值"时，6 方案有更大的动态范围，误差更小。**逐块选择总能得到两者中更优的那个。**

---

## 五、E4M3 相对精度分析

> 这里澄清一个常见误解：**"E4M3 在 [0, 256] 范围精度更高"是错误的。**

E4M3 是浮点格式，每个指数区间内相对精度恒为 6.67%~12.5%：

| 区间 | 平均相对误差 |
|---|---|
| [128, 256) | ~9.07% |
| [256, 448) | ~9.79% |

两者几乎没有差别。选 256 作为上限与 E4M3 精度无关，唯一原因是为 ×1.5 留出空间（见第二节）。

---

## 六、核心算法代码详解

### 6.1 compute_nv_scale_factors —— 计算缩放因子（第 184 行）

```python
def compute_nv_scale_factors(
    x_scale_blocks: torch.Tensor,    # 已分块数据 (num_blocks, 16)
    x_amax: torch.Tensor,            # 全局最大绝对值
    *,
    fp4_format: DataType,
    scale_rule: ScaleRule,
    round_style: RoundStyle,
    scale_expansion_factor: float | None = None,  # 4/6 关键：传 1.5 生成 4 方案
) -> tuple[torch.Tensor, torch.Tensor]:

    # 获取格式常量
    max_quantized_value = fp4_format.quantized_value_type.get_maximum_value(scale_rule)
    # nvfp4 在 4/6 模式下 → 6
    max_scale_factor = fp4_format.scale_type.get_maximum_value(scale_rule)
    # nvfp4 在 4/6 模式下 → 256

    # encode_scale = (6 × 256) / x_amax  [4/6 模式下]
    encode_scale = (max_quantized_value * max_scale_factor * adj) / x_amax

    # 每块缩放因子：本质是 block_amax / x_amax × max_scale_factor
    x_scales_hp = block_amax / max_quantized_value * encode_scale

    # ★ 4 方案在此乘 1.5，6 方案不走这里
    if scale_expansion_factor is not None:
        x_scales_hp = x_scales_hp * scale_expansion_factor

    # 截断到 E4M3 精度（6 方案存 ~256，4 方案存 ~384）
    x_scales = x_scales_hp.to(torch.float8_e4m3fn)

    # decode_scale = x_amax / (6 × 256)  两种方案使用同一个 decode_scale！
    decode_scale = x_amax / (max_quantized_value * max_scale_factor * adj)

    # 归一化：6 方案最大值→6.0，4 方案最大值→4.0
    x_block_scaled = x_scale_blocks / (decode_scale * x_scales)

    return x_block_scaled, x_scales
```

**重点**：两种方案共用同一个 `decode_scale`（分母均为 `6 × 256`），这是反量化误差可以公平比较的前提。

### 6.2 nvfp4_fouroversix_block_scaled_quantization —— 4/6 主流程（第 519 行）

```python
def nvfp4_fouroversix_block_scaled_quantization(
    x_scale_blocks, x_amax, *, scale_rule, round_style,
):
    # 6 方案：scale_expansion_factor=None，最大值映射到 6.0
    x_block_scaled_6, scales_6 = quantize_to_nvfp4(
        x_scale_blocks, x_amax,
        scale_rule=scale_rule, round_style=round_style,
    )

    # 4 方案：scale_expansion_factor=1.5，最大值映射到 4.0
    x_block_scaled_4, scales_4 = quantize_to_nvfp4(
        x_scale_blocks, x_amax,
        scale_rule=scale_rule, round_style=round_style,
        scale_expansion_factor=1.5,
    )

    # 逐块比较误差，选最优
    return select_fouroversix(
        x_scale_blocks,
        x_block_scaled_6, scales_6,
        x_block_scaled_4, scales_4,
        x_amax,
        scale_rule=scale_rule, round_style=round_style,
    )
```

### 6.3 select_fouroversix —— 逐块选择（第 262 行）

```python
def select_fouroversix(
    x_scale_blocks,          # 原始数据块 (num_blocks, 16)
    x_block_scaled_6,        # 6 方案归一化值（范围 [-6, 6]）
    scales_6,                # 6 方案 E4M3 缩放因子（≈ 256 for max block）
    x_block_scaled_4,        # 4 方案归一化值（范围 [-4, 4]）
    scales_4,                # 4 方案 E4M3 缩放因子（= scales_6 × 1.5，≈ 384）
    x_amax,
    *, scale_rule, round_style,
):
    # 步骤 1：模拟量化（两种方案都用 fake_quantize_to_e2m1 舍入）
    x_fake_6 = fake_quantize_to_e2m1(x_block_scaled_6)
    x_fake_4 = fake_quantize_to_e2m1(x_block_scaled_4)

    # 步骤 2：反量化回原始尺度（公式相同，保证公平比较）
    # dequant = fake_q × scale × x_amax / (6 × 256)
    # 两种方案用同一个分母 E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX = 6 × 256 = 1536
    x_deq_6 = x_fake_6 * scales_6 * x_amax / 1536
    x_deq_4 = x_fake_4 * scales_4 * x_amax / 1536

    # 步骤 3：按 scale_rule 计算每块误差
    # mse:     error = sum((deq - original)²)
    # mae:     error = sum(|deq - original|)
    # abs_max: error = max(|deq - original|)

    # 步骤 4：逐块选择（不同块可以选不同方案）
    select_4 = (error_4 < error_6)
    x_fake_quantized = where(select_4, x_fake_4, x_fake_6)
    scales = where(select_4, scales_4, scales_6)

    return x_fake_quantized, scales
```

### 6.4 fake_quantize_to_e2m1 —— E2M1 模拟量化（第 30 行）

```python
def fake_quantize_to_e2m1(x, *, round_style=RoundStyle.nearest):
    """
    三段式处理，对应 E2M1 不同指数区间的步长：
      |x| < 2:  步长 0.5  → step1 = round(2x) / 2
      2 ≤ |x| < 4: 步长 1  → step2 = round(x)
      |x| ≥ 4:  步长 2  → step3 = 2 × round(x/2)
    """
    step1 = torch.round(2 * x.abs()) / 2
    step2 = torch.round(x.abs())
    step3 = 2 * torch.round(x.abs() / 2)

    mask1 = x.abs() < 2
    mask2 = x.abs() < 4

    return x.sign() * (
        step1 * mask1
        + step2 * (~mask1) * mask2
        + step3 * (~mask1) * (~mask2)
    )
```

---

## 七、完整数据流（4/6 模式，最差块 block_amax = x_amax）

```
输入: x_scale_blocks (num_blocks × 16), x_amax

        ┌──────────────────────────────────────────────┐
        │              compute_nv_scale_factors         │
        │                                               │
        │  encode_scale = 6×256 / x_amax = 1536/x_amax │
        │                                               │
        │  scale_hp_6 = block_amax/6 × encode_scale     │
        │             = block_amax/x_amax × 256          │
        │             → E4M3: 存 256                     │
        │                                               │
        │  scale_hp_4 = scale_hp_6 × 1.5                │
        │             = block_amax/x_amax × 384          │
        │             → E4M3: 存 384                     │
        │                                               │
        │  decode_scale = x_amax / 1536  (两方案相同)   │
        │                                               │
        │  x_scaled_6 = x / (decode_scale × 256)        │
        │             → max value = 6.0                  │
        │                                               │
        │  x_scaled_4 = x / (decode_scale × 384)        │
        │             → max value = 4.0                  │
        └──────────────────────────────────────────────┘
                       ↓                    ↓
           fake_quantize_to_e2m1   fake_quantize_to_e2m1
           (舍入到 E2M1 格点)       (舍入到 E2M1 格点)
                       ↓                    ↓
             反量化: ×scale_6×x_amax/1536   反量化: ×scale_4×x_amax/1536
                       ↓                    ↓
                   误差_6                误差_4
                       └──────── 比较 ────────┘
                                   ↓
                          逐块选 min(误差_4, 误差_6)
                          输出: fake_quantized, scales
```

---

## 八、标准 NVFP4 vs 4/6 对比

| 维度 | `static_6`（标准 NVFP4） | `mse/mae/abs_max`（4/6） |
|---|---|---|
| `max_scale_factor` | **448** | **256** |
| E4M3 存储值（max block） | 448 | 256（6方案）/ 384（4方案） |
| 内层归一化范围 | [-6, 6] | [-6, 6] 或 [-4, 4]，逐块选 |
| 缩放因子计算次数 | 1 次 | **2 次** |
| 是否有选择过程 | 无 | **逐块误差比较** |
| 计算开销 | 低 | ~2 倍 |
| 精度 | 标准 | **更高** |

---

## 九、Hadamard 变换（可选预处理）

```python
# src/fouroversix/quantize/pytorch/reference.py 第 570 行
if had is not None:
    x = (x.reshape(-1, had.shape[0]) @ had.to(x.dtype)).reshape_as(x)
```

在量化前应用随机 Hadamard 变换（RHT），使权重分布更均匀（减少异常值），从而让两种量化方案都能得到更低的误差。与 4/6 选择机制正交，可单独使用也可组合使用。

---

## 十、Triton 高性能实现

Triton 版本在 `src/fouroversix/kernels/triton/quantize.py` 第 196 行，与 PyTorch 参考实现算法完全一致：

```python
@triton.jit
def compute_error_and_select_kernel(
    original_values,
    dequantized_1,
    dequantized_2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SCALE_RULE: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr,
    BLOCK_SCALE_2D: tl.constexpr,
) -> None:
    diff_1 = dequantized_1 - original_values
    diff_2 = dequantized_2 - original_values

    if SCALE_RULE == SCALE_RULE_ABS_MAX:
        error_1 = tl.max(tl.abs(diff_1), axis=-1)
        error_2 = tl.max(tl.abs(diff_2), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MAE:
        error_1 = tl.sum(tl.abs(diff_1), axis=-1)
        error_2 = tl.sum(tl.abs(diff_2), axis=-1)
    elif SCALE_RULE == SCALE_RULE_MSE:
        error_1 = tl.sum(diff_1 * diff_1, axis=-1)
        error_2 = tl.sum(diff_2 * diff_2, axis=-1)
    # ... 后续处理 BLOCK_SCALE_2D 情况及逐块选择
```
