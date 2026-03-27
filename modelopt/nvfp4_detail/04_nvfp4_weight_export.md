# NVFP4 权重导出机制

## 概述

本文档详细介绍 NVFP4 量化后的权重是如何导出和保存的，包括导出流程、保存的张量、Fake Quantization 到 Real Quantization 的转换，以及 `modelopt_state` 元数据结构。

---

## 1. export_hf_checkpoint 流程

### 1.1 文件位置

**文件**：`modelopt/torch/export/unified_export_hf.py`

### 1.2 函数签名

```python
def export_hf_checkpoint(
    model: Any,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
    save_modelopt_state: bool = False,
    components: list[str] | None = None,
    extra_state_dict: dict[str, torch.Tensor] | None = None,
    **kwargs,
) -> None:
    """Export quantized model to HuggingFace checkpoint format.
    
    Args:
        model: 量化后的模型
        dtype: 导出的数据类型（如 torch.bfloat16）
        export_dir: 导出目录
        save_modelopt_state: 是否保存 modelopt_state
        components: 要导出的组件列表（用于 diffusers）
        extra_state_dict: 额外要保存的张量
    """
```

### 1.3 导出流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    export_hf_checkpoint(model, export_dir)               │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 1: 检测模型类型                                                     │
│ - transformers 模型 → _export_transformers_checkpoint()                  │
│ - diffusers 模型 → _export_diffusers_checkpoint()                        │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 2: 融合共享输入的层                                                 │
│ - QKV 投影层融合                                                         │
│ - gate/up 投影层融合                                                     │
│ - 重新平滑和重量化融合层                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 3: 处理量化模块 (_process_quantized_modules)                        │
│ - 遍历所有量化模块                                                       │
│ - 调用 _export_quantized_weight() 导出量化权重                           │
│ - 注册 weight_scale, weight_scale_2, input_scale 等 buffer               │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 4: 后处理 state_dict (postprocess_state_dict)                       │
│ - 过滤不必要的 quantizer 状态                                            │
│ - 转换 KV cache 缩放因子格式                                             │
│ - 重命名张量键名                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 5: 生成量化配置 (get_quant_config)                                  │
│ - 创建 quant_config 字典                                                 │
│ - 包含算法类型、块大小、排除的模块等                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 6: 保存文件                                                         │
│ - model.safetensors: 量化后的权重                                        │
│ - config.json: 模型配置 + 量化配置                                       │
│ - hf_quant_config.json: 向后兼容的量化配置                               │
│ - modelopt_state.pt: modelopt 状态（可选）                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 导出的张量列表

### 2.1 NVFP4 量化层导出的张量

对于每个被量化的层（如 `model.layers.0.mlp.gate_proj`），导出以下张量：

| 张量名 | 数据类型 | 形状 | 说明 |
|--------|---------|------|------|
| `weight` | uint8 | `[out_features, in_features // 2]` | 打包的 FP4 权重（2 个 FP4 值打包到 1 个 uint8） |
| `weight_scale` | float8_e4m3fn | `[out_features, in_features // 16]` | Per-block 缩放因子（FP8 格式） |
| `weight_scale_2` | float32 | `[1]` 或 `[out_features]` | Global 缩放因子 |
| `input_scale` | float32 | `[1]` 或 `[in_features // 16]` | 输入激活缩放因子（如果启用激活量化） |

### 2.2 KV Cache 量化张量

如果启用了 KV cache 量化，还会导出：

| 张量名 | 数据类型 | 说明 |
|--------|---------|------|
| `k_proj.k_scale` | float32 | K 投影的 KV cache 缩放因子 |
| `v_proj.v_scale` | float32 | V 投影的 KV cache 缩放因子 |
| `k_proj.k_bias` | float32 | K 投影的 KV cache 偏置（仿射量化时） |
| `v_proj.v_bias` | float32 | V 投影的 KV cache 偏置（仿射量化时） |

### 2.3 导出代码

**文件**：`modelopt/torch/export/quant_utils.py`（第 508-608 行）

```python
def _export_quantized_weight(sub_module, weight_name, quantizer_attrs):
    """Export quantized weight and scaling factors."""
    
    # 注册 weight_scale (per-block scaling factor)
    sub_module.register_buffer(
        quantizer_attrs.weight_scale,
        get_weight_scaling_factor(sub_module, weight_name)
    )
    
    # 注册 weight_scale_2 (global scaling factor) - NVFP4 特有
    sub_module.register_buffer(
        quantizer_attrs.weight_scale_2,
        get_weight_scaling_factor_2(sub_module, weight_name).squeeze(),
    )
    
    # 注册 input_scale (activation scaling factor)
    sub_module.register_buffer(
        quantizer_attrs.input_scale,
        get_activation_scaling_factor(sub_module, ...).squeeze(),
    )
    
    # 转换权重为量化格式
    quantized_weight = to_quantized_weight(
        weight,
        weights_scaling_factor,
        quantization="nvfp4",
        weights_scaling_factor2=weights_scaling_factor_2,
        block_size=16,
    )
    
    # 替换原始权重
    sub_module.weight = nn.Parameter(quantized_weight, requires_grad=False)
```

---

## 3. Fake Quantization → Real Quantization

### 3.1 概念区别

| 类型 | 说明 | 用途 |
|------|------|------|
| **Fake Quantization** | 模拟量化效果，但权重仍以 FP16/BF16 存储 | 训练、校准、精度评估 |
| **Real Quantization** | 权重真正转换为低精度格式（如 FP4 打包到 uint8） | 推理、部署 |

### 3.2 转换函数

**文件**：`modelopt/torch/export/quant_utils.py`（第 902-970 行）

```python
def to_quantized_weight(
    weight: torch.Tensor,
    weights_scaling_factor: torch.Tensor,
    quantization: str,
    weights_scaling_factor2: torch.Tensor | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    """Convert fake-quantized weight to real quantized format.
    
    Args:
        weight: 原始权重（FP16/BF16/FP32）
        weights_scaling_factor: Per-block 缩放因子
        quantization: 量化类型（"nvfp4"）
        weights_scaling_factor2: Global 缩放因子（NVFP4 特有）
        block_size: 块大小（默认 16）
    
    Returns:
        打包的量化权重（uint8）
    """
    if quantization == "nvfp4":
        # 调用 NVFP4QTensor.quantize() 进行真实量化
        quantized_tensor, _, _ = NVFP4QTensor.quantize(
            weight,
            block_size,
            weights_scaling_factor=weights_scaling_factor,
            weights_scaling_factor_2=weights_scaling_factor2,
            try_tensorrt=True,
        )
        return quantized_tensor.packed_tensor  # 返回打包的 uint8 张量
```

### 3.3 转换过程详解

```
原始权重 (FP16/BF16)
[out_features, in_features]
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 1: 应用缩放                     │
│ scaled = weight / (scale × scale_2) │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 2: 量化到 FP4                   │
│ 映射到 [0, 0.5, 1, 1.5, 2, 3, 4, 6] │
│ 得到 4-bit 索引 (0-15)              │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 步骤 3: 打包                         │
│ packed[i] = (fp4[2i+1] << 4) |      │
│             fp4[2i]                  │
│ 2 个 FP4 值 → 1 个 uint8            │
└─────────────────────────────────────┘
         │
         ▼
打包的权重 (uint8)
[out_features, in_features // 2]
```

---

## 4. fold_weight 的作用

### 4.1 文件位置

**文件**：`modelopt/torch/quantization/nn/modules/quant_module.py`（第 130-160 行）

### 4.2 功能说明

`fold_weight()` 将 fake quantization 操作"折叠"到权重中，用于加速推理：

```python
def fold_weight(self, keep_attrs: bool = False):
    """Fold the weight for faster eval.
    
    将 fake quantization 预计算到权重中，推理时不再需要动态量化。
    
    Args:
        keep_attrs: 是否保留量化相关属性（如 _amax）
    """
    for name in dir(self):
        attr = getattr(self, name)
        if (
            name.endswith("weight_quantizer")
            and isinstance(attr, TensorQuantizer)
            and attr.fake_quant
        ):
            weight_name = name[:-10]  # 移除 _weight_quantizer 后缀
            weight = getattr(self, weight_name)
            
            # 将 fake quantization 应用到权重上
            # weight = quantize(dequantize(weight))
            weight.data.copy_(attr(weight.float()).to(weight.dtype))
            
            # 禁用量化器（推理时不再需要）
            attr.disable()
            
            # 删除量化相关属性（如果 keep_attrs=False）
            if not keep_attrs:
                for attr_name in ["_pre_quant_scale", "_amax"]:
                    if hasattr(attr, attr_name):
                        delattr(attr, attr_name)
```

### 4.3 使用场景

1. **导出前**：在导出 checkpoint 前调用，确保权重已经是量化后的值
2. **推理优化**：减少推理时的计算开销
3. **精度评估**：评估量化后模型的精度

---

## 5. modelopt_state 元数据

### 5.1 定义

`modelopt_state` 是 Model Optimizer 用于保存和恢复优化状态的数据结构。

**文件**：`modelopt/torch/opt/mode.py`（第 40-60 行）

```python
ModeloptStateList = list[tuple[str, ModeState]]
ModeState = dict[str, ConfigDict | MetadataDict]
```

### 5.2 结构

```python
modelopt_state = [
    (
        "quantization",  # 模式名称
        {
            "config": {
                "quant_cfg": {...},  # 量化配置
                "algorithm": "max",  # 校准算法
            },
            "metadata": {
                "quantizer_state": {
                    "model.layers.0.mlp.gate_proj.weight_quantizer": {
                        "_amax": tensor(...),
                        "_global_amax": tensor(...),
                        "_num_bits": (2, 1),
                        "_block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                        ...
                    },
                    "model.layers.0.mlp.gate_proj.input_quantizer": {
                        "_amax": tensor(...),
                        ...
                    },
                    ...
                }
            }
        }
    ),
    # 可能有其他优化模式（如 sparsity）
]
```

### 5.3 存储位置

- **模型属性**：`model._modelopt_state`
- **文件**：`modelopt_state.pt`（当 `save_modelopt_state=True` 时）

### 5.4 用途

1. **恢复量化状态**：从 checkpoint 恢复量化器的 amax 等参数
2. **继续训练**：支持量化感知训练（QAT）的断点续训
3. **调试**：检查量化配置和校准结果

### 5.5 保存和加载

**保存**：
```python
# 在 export_hf_checkpoint 中
if save_modelopt_state:
    torch.save(model._modelopt_state, export_dir / "modelopt_state.pt")
```

**加载**：
```python
# 恢复 modelopt_state
modelopt_state = torch.load("modelopt_state.pt")
model._modelopt_state = modelopt_state

# 或使用 restore_from_modelopt_state
from modelopt.torch.opt import restore_from_modelopt_state
restore_from_modelopt_state(model, modelopt_state)
```

---

## 6. 导出的文件结构

### 6.1 目录结构

```
export_dir/
├── model.safetensors           # 量化后的权重（或分片的 model-00001-of-00002.safetensors）
├── model.safetensors.index.json # 分片索引（大模型）
├── config.json                 # 模型配置 + 量化配置
├── hf_quant_config.json        # 向后兼容的量化配置
├── tokenizer.json              # Tokenizer 配置
├── tokenizer_config.json       # Tokenizer 配置
├── special_tokens_map.json     # 特殊 token 映射
└── modelopt_state.pt           # ModelOpt 状态（可选）
```

### 6.2 config.json 中的量化配置

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  ...
  "quantization_config": {
    "producer": {
      "name": "modelopt",
      "version": "0.17.0"
    },
    "quantization": {
      "quant_algo": "nvfp4",
      "kv_cache_quant_algo": "fp8_cast",
      "exclude_modules": ["lm_head"]
    }
  }
}
```

### 6.3 hf_quant_config.json

```json
{
  "producer": {
    "name": "modelopt",
    "version": "0.17.0"
  },
  "quantization": {
    "quant_algo": "nvfp4",
    "kv_cache_quant_algo": "fp8_cast"
  },
  "layers": {
    "model.layers.0.quantization": "nvfp4",
    "model.layers.0.awq_block_size": 16,
    ...
  }
}
```

---

## 7. 导出最佳实践

### 7.1 基本导出

```python
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,
        dtype=torch.bfloat16,
        export_dir="./quantized_model",
    )
```

### 7.2 保存 modelopt_state

```python
export_hf_checkpoint(
    model,
    export_dir="./quantized_model",
    save_modelopt_state=True,  # 保存 modelopt_state.pt
)
```

### 7.3 大模型分片导出

对于大模型，safetensors 会自动分片：

```python
export_hf_checkpoint(
    model,
    export_dir="./quantized_model",
    max_shard_size="5GB",  # 每个分片最大 5GB
)
```

---

## 相关文档

- [NVFP4 量化原理](./01_nvfp4_quantization_principle.md)
- [NVFP4 核心代码位置](./02_nvfp4_code_structure.md)
- [NVFP4 校准流程详解](./03_nvfp4_calibration.md)
- [NVFP4 MoE 模型处理](./05_nvfp4_moe_handling.md)
- [NVFP4 选择性量化使用指南](../NVFP4_Selective_Quantization_Guide.md)
