# NVFP4 量化原理

## 概述

NVFP4 是 NVIDIA Blackwell GPU 支持的 4-bit 浮点量化格式。本文档详细介绍 NVFP4 的数据格式、量化数学过程和两级缩放机制。

---

## 1. FP4 数据格式（E2M1）

### 1.1 位宽分配

| 组成部分 | 位数 | 说明 |
|---------|------|------|
| 符号位 | 1 bit | 隐含在值中 |
| 指数位 | 2 bits | E2 |
| 尾数位 | 1 bit | M1 |
| **总计** | **4 bits** | |

### 1.2 可表示的值

FP4 E2M1 格式可以表示以下 16 个离散值：

```python
# 正值（8 个）
[0, 0.5, 1, 1.5, 2, 3, 4, 6]

# 负值（8 个）
[0, -0.5, -1, -1.5, -2, -3, -4, -6]

# 完整的 E2M1 值表（索引 0-15）
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
```

**代码位置**：`modelopt/torch/quantization/qtensor/nvfp4_tensor.py` 第 20-21 行

### 1.3 量化阈值

将连续值量化到 FP4 时使用的阈值：

| 输入范围 | 量化值 |
|---------|--------|
| abs(x) ≤ 0.25 | 0 |
| 0.25 < abs(x) < 0.75 | 0.5 |
| 0.75 ≤ abs(x) ≤ 1.25 | 1.0 |
| 1.25 < abs(x) < 1.75 | 1.5 |
| 1.75 ≤ abs(x) ≤ 2.5 | 2.0 |
| 2.5 < abs(x) < 3.5 | 3.0 |
| 3.5 ≤ abs(x) ≤ 5.0 | 4.0 |
| abs(x) > 5.0 | 6.0 |

---

## 2. 两级缩放机制（Per-block + Global）

NVFP4 使用两级缩放来最大化量化精度：

### 2.1 缩放因子结构

```
┌─────────────────────────────────────────────────────────────┐
│                    Global Scale (FP32)                       │
│                 global_scale = global_amax / (6.0 × 448.0)   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Block Scale 0 │     │ Block Scale 1 │     │ Block Scale N │
│    (FP8)      │     │    (FP8)      │     │    (FP8)      │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  16 个 FP4 值  │     │  16 个 FP4 值  │     │  16 个 FP4 值  │
└───────────────┘     └───────────────┘     └───────────────┘
```

### 2.2 缩放因子计算

**Global Scale（全局缩放因子）**：
```python
global_amax = reduce_amax(input, axis=None)  # 整个张量的最大绝对值
global_scale = global_amax / (6.0 * 448.0)   # 6.0 是 FP4 最大值，448.0 是 FP8 最大值
```

**Per-block Scale（分块缩放因子）**：
```python
per_block_amax = reduce_amax(input_block, axis=-1)  # 每个块的最大绝对值
per_block_scale = per_block_amax / (6.0 * global_scale)
```

### 2.3 为什么使用两级缩放？

1. **精度优化**：Per-block scale 捕获局部动态范围，Global scale 捕获全局动态范围
2. **存储效率**：Per-block scale 使用 FP8 存储（4 位指数 + 3 位尾数），比 FP32 节省空间
3. **硬件支持**：Blackwell GPU 原生支持这种两级缩放结构

---

## 3. 量化数学过程

### 3.1 完整量化流程

```
输入张量 X (FP16/BF16/FP32)
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 1: 计算全局最大绝对值           │
│ global_amax = max(|X|)              │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 2: 计算全局缩放因子             │
│ global_scale = global_amax / 2688.0 │
│ (2688.0 = 6.0 × 448.0)              │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 3: 重塑张量为块                 │
│ X_blocks = X.view(..., -1, 16)      │
│ (block_size = 16)                   │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 4: 计算每块的最大绝对值         │
│ per_block_amax = max(|X_block|)     │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 5: 计算分块缩放因子             │
│ per_block_scale = per_block_amax /  │
│                   (6.0 × global_scale)│
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 6: 缩放输入                     │
│ scaled_X = X / (per_block_scale ×   │
│                 global_scale)        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 7: 量化到 E2M1                  │
│ q_val = cast_to_fp4(scaled_X)       │
│ 映射到 [0, 0.5, 1, 1.5, 2, 3, 4, 6] │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 8: 打包存储                     │
│ packed = (q[1::2] << 4) | q[0::2]   │
│ 2 个 FP4 值 → 1 个 uint8            │
└─────────────────────────────────────┘
         │
         ▼
输出: (packed_weight, per_block_scale, global_scale)
```

### 3.2 核心代码实现

**量化函数**（`modelopt/torch/quantization/qtensor/nvfp4_tensor.py` 第 130-220 行）：

```python
@classmethod
def quantize(
    cls,
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor: torch.Tensor | None = None,
    weights_scaling_factor_2: torch.Tensor | None = None,
    keep_high_precision: bool = False,
    try_tensorrt: bool = False,
):
    # 步骤 1-2: 计算全局缩放因子
    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = cls.get_weights_scaling_factor_2(input)
        # 计算: reduce_amax(input) / (6.0 * 448.0)
    
    # 步骤 3-5: 计算分块缩放因子
    if weights_scaling_factor is None:
        weights_scaling_factor, _ = cls.get_weights_scaling_factor(
            input, block_size, weights_scaling_factor_2
        )
        # 计算: per_block_amax / (6.0 * weights_scaling_factor_2)
    
    # 步骤 3: 重塑张量
    input = input.view((*tuple(input.shape[:-1]), -1, block_size))
    
    # 步骤 6: 应用缩放
    scaled_weight = input / (
        (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
    )
    
    # 步骤 7: 转换为 FP4
    q_weight = cls._cast_fp4(scaled_weight)
    # 使用 E2M1 格式: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    
    # 步骤 8: 打包权重
    packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
    
    return (cls(input_shape, input_dtype, packed_weight),
            weights_scaling_factor,
            weights_scaling_factor_2)
```

**Triton Fake Quantization 内核**（`modelopt/torch/quantization/triton/fp4_kernel.py` 第 140-180 行）：

```python
@triton.jit
def static_blockwise_fp4_fake_quant_kernel(
    x_ptr, y_ptr, scale_ptr, NUM_FP4_BLOCKS, BLOCK_SIZE: tl.constexpr, OUT_DTYPE: tl.constexpr
):
    # 加载输入和缩放因子
    x = tl.load(x_ptr + idx).to(tl.float32)
    scale = tl.load(scale_ptr + pid).to(tl.float32)
    
    # 计算缩放后的绝对值
    abs_scaled = x_abs / scale_safe
    
    # 量化到 FP4 值（使用阈值判断）
    q_val = tl.where(
        abs_scaled <= 0.25, 0.0,
        tl.where(abs_scaled < 0.75, 0.5,
        tl.where(abs_scaled <= 1.25, 1.0,
        tl.where(abs_scaled < 1.75, 1.5,
        tl.where(abs_scaled <= 2.5, 2.0,
        tl.where(abs_scaled < 3.5, 3.0,
        tl.where(abs_scaled <= 5.0, 4.0, 6.0)))))))
    
    # 恢复符号并存储
    x_quant = tl.where(x >= 0, q_val * scale_safe, -q_val * scale_safe)
    tl.store(y_ptr + idx, x_quant.to(OUT_DTYPE))
```

---

## 4. 反量化过程

### 4.1 反量化流程

```
packed_weight (uint8) + per_block_scale (FP8) + global_scale (FP32)
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 1: 解包                         │
│ unpacked[1::2] = packed >> 4        │
│ unpacked[0::2] = packed & 0x0F      │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 2: 查表获取 E2M1 值             │
│ fp_values = e2m1_table[unpacked]    │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 3: 应用缩放因子恢复原始值       │
│ output = fp_values × per_block_scale│
│          × global_scale             │
└─────────────────────────────────────┘
         │
         ▼
输出张量 (FP16/BF16/FP32)
```

### 4.2 反量化代码

```python
def dequantize(self, dtype: torch.dtype = None, fast=False, **kwarg):
    # 步骤 1: 解包张量
    unpacked[..., 1::2] = input >> 4
    unpacked[..., 0::2] = input & 0x0F
    
    # 步骤 2: 查表获取 E2M1 值
    unpacked = self.get_e2m1_values(input.device)[unpacked.long()]
    
    # 步骤 3: 应用缩放因子恢复原始值
    output = unpacked * scale * global_scale
    
    return output.to(dtype)
```

---

## 5. 与其他量化格式的对比

| 特性 | NVFP4 (E2M1) | FP8 (E4M3) | INT8 | INT4 |
|------|-------------|------------|------|------|
| 位宽 | 4 bits | 8 bits | 8 bits | 4 bits |
| 动态范围 | 0 ~ 6 | 0 ~ 448 | -128 ~ 127 | -8 ~ 7 |
| 可表示值数量 | 16 | 256 | 256 | 16 |
| 缩放方式 | 两级（FP8 + FP32） | 单级 | 单级 | 单级 |
| 块大小 | 16 | 通常无 | 通常无 | 32/128 |
| 硬件支持 | Blackwell | Hopper+ | 广泛 | 广泛 |
| 精度损失 | 中等 | 低 | 中等 | 高 |
| 压缩比 | 4x | 2x | 2x | 4x |

### 5.1 NVFP4 的优势

1. **浮点表示**：相比 INT4，FP4 能更好地表示接近零的小值
2. **两级缩放**：Per-block + Global 缩放提供更好的动态范围适应
3. **硬件加速**：Blackwell GPU 原生支持，无需软件模拟
4. **权重+激活量化**：同时支持 W4A4（权重和激活都是 4-bit）

### 5.2 NVFP4 的限制

1. **硬件依赖**：仅 Blackwell GPU 支持推理
2. **精度损失**：相比 FP8，精度损失更大
3. **校准需求**：需要校准数据来确定最优的缩放因子

---

## 相关文档

- [NVFP4 核心代码位置](./02_nvfp4_code_structure.md)
- [NVFP4 校准流程详解](./03_nvfp4_calibration.md)
- [NVFP4 权重导出机制](./04_nvfp4_weight_export.md)
- [NVFP4 选择性量化使用指南](../NVFP4_Selective_Quantization_Guide.md)
