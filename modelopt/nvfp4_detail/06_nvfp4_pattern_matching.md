# NVFP4 量化器模式匹配规则

## 概述

Model Optimizer 使用通配符模式匹配来确定哪些模块的量化器应该被启用。理解这些匹配规则对于正确配置选择性量化（如 `nvfp4_mlp_only`）至关重要。

---

## 1. 通配符匹配机制

### 1.1 基本规则

| 符号 | 含义 | 示例 |
|------|------|------|
| `*` | 匹配任意字符序列（包括 `.`） | `*mlp*` 匹配 `model.layers.0.mlp.gate_proj` |
| 无通配符 | 精确匹配 | `nn.Linear` 仅匹配 `nn.Linear` |

### 1.2 匹配对象

模式匹配的对象是**量化器的完整路径名**，格式为：

```
{模块路径}.{量化器名称}
```

**示例**：
- `model.layers.0.mlp.gate_proj.weight_quantizer`
- `model.layers.0.self_attn.q_proj.input_quantizer`
- `model.layers.0.block_sparse_moe.experts.0.w1.weight_quantizer`

### 1.3 匹配示例

```python
# 配置
quant_cfg = {
    "*mlp*weight_quantizer": {"enable": True, ...},
    "*mlp*input_quantizer": {"enable": True, ...},
    "default": {"enable": False},
}

# 匹配结果
"model.layers.0.mlp.gate_proj.weight_quantizer"  # → *mlp*weight_quantizer ✅
"model.layers.0.mlp.up_proj.input_quantizer"     # → *mlp*input_quantizer ✅
"model.layers.0.self_attn.q_proj.weight_quantizer"  # → default ✅ (不量化)
"model.layers.0.self_attn.o_proj.weight_quantizer"  # → default ✅ (不量化)
```

---

## 2. 匹配优先级

### 2.1 优先级规则

当一个量化器同时匹配多个模式时，**更具体的模式优先**。

**具体性判断**：
1. 精确匹配 > 通配符匹配
2. 通配符越少越具体
3. 通配符位置越靠后越具体

### 2.2 优先级示例

```python
quant_cfg = {
    "*o_proj*weight_quantizer": {"enable": True},   # 模式 A
    "*mlp*weight_quantizer": {"enable": True},      # 模式 B
    "default": {"enable": False},                    # 模式 C
}

# 对于 "model.layers.0.self_attn.o_proj.weight_quantizer"
# 匹配 A: *o_proj*weight_quantizer ✅
# 匹配 B: *mlp*weight_quantizer ❌ (路径中没有 mlp)
# 结果: 使用模式 A，启用量化

# 对于 "model.layers.0.mlp.gate_proj.weight_quantizer"
# 匹配 A: *o_proj*weight_quantizer ❌ (路径中没有 o_proj)
# 匹配 B: *mlp*weight_quantizer ✅
# 结果: 使用模式 B，启用量化

# 对于 "model.layers.0.self_attn.q_proj.weight_quantizer"
# 匹配 A: *o_proj*weight_quantizer ❌
# 匹配 B: *mlp*weight_quantizer ❌
# 匹配 C: default ✅
# 结果: 使用模式 C，禁用量化
```

### 2.3 顺序无关性

Python dict 的插入顺序**不影响**匹配优先级。优先级完全由模式的具体性决定。

```python
# 以下两种配置等价
config1 = {
    "*mlp*": {...},
    "*o_proj*": {...},
}

config2 = {
    "*o_proj*": {...},
    "*mlp*": {...},
}
```

---

## 3. 默认禁用的量化器列表

### 3.1 完整列表

**文件**：`modelopt/torch/quantization/config.py`

```python
_default_disabled_quantizer_cfg = {
    # BatchNorm 层（量化会破坏其统计特性）
    "nn.BatchNorm1d": {"*": {"enable": False}},
    "nn.BatchNorm2d": {"*": {"enable": False}},
    "nn.BatchNorm3d": {"*": {"enable": False}},
    
    # 激活函数（通常不需要量化）
    "nn.LeakyReLU": {"*": {"enable": False}},
    
    # 语言模型头（对精度敏感）
    "*lm_head*": {"enable": False},
    "*proj_out.*": {"enable": False},           # Whisper 模型的 lm_head
    "*output_layer*": {"enable": False},
    "output.*": {"enable": False},
    
    # MoE 路由器（必须保持高精度）
    "*block_sparse_moe.gate*": {"enable": False},
    "*router*": {"enable": False},
    "*mlp.gate.*": {"enable": False},            # 注意：不是 gate_proj
    "*mlp.shared_expert_gate.*": {"enable": False},
    
    # 特殊层
    "*linear_attn.conv1d*": {"enable": False},
    "*mixer.conv1d*": {"enable": False},         # Mamba conv1d
    
    # 默认规则（所有未匹配的量化器）
    "default": {"enable": False},
}
```

### 3.2 各规则说明

| 模式 | 说明 | 原因 |
|------|------|------|
| `nn.BatchNorm*` | BatchNorm 层 | 量化会破坏其运行时统计特性 |
| `*lm_head*` | 语言模型输出头 | 对精度极其敏感，量化会严重影响输出质量 |
| `*block_sparse_moe.gate*` | MoE 路由器 | 路由决策需要高精度，否则会导致专家选择错误 |
| `*router*` | MoE 路由器（通用模式） | 同上 |
| `*mlp.gate.*` | MoE 门控（非 gate_proj） | 区分 MoE 门控和 MLP 的 gate_proj |
| `*mixer.conv1d*` | Mamba 模型的 conv1d | 特殊架构，量化可能导致问题 |
| `default` | 所有未匹配的量化器 | 保守策略，默认不量化 |

### 3.3 重要区分

**`*mlp.gate.*` vs `*mlp*gate_proj*`**：

```python
# MoE 路由器门控（不量化）
"model.layers.0.mlp.gate.weight"  # → *mlp.gate.* ✅ 禁用

# MLP 的 gate_proj 层（量化）
"model.layers.0.mlp.gate_proj.weight"  # → *mlp*weight_quantizer ✅ 启用
```

---

## 4. NVFP4_MLP_ONLY_CFG 的匹配逻辑

### 4.1 配置定义

```python
_nvfp4_quantizer = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
}

_nvfp4_mlp_only_quant_cfg = {
    # 启用 MLP 层量化
    "*mlp*weight_quantizer": _nvfp4_quantizer,
    "*mlp*input_quantizer": _nvfp4_quantizer,
    
    # 启用 MoE 专家层量化
    "*block_sparse_moe*weight_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*input_quantizer": _nvfp4_quantizer,
    
    # 继承默认禁用规则
    **_default_disabled_quantizer_cfg,
}

NVFP4_MLP_ONLY_CFG = {
    "quant_cfg": _nvfp4_mlp_only_quant_cfg,
    "algorithm": "max",
}
```

### 4.2 匹配结果表

| 模块 | 量化器路径示例 | 匹配模式 | 是否量化 |
|------|---------------|---------|---------|
| MLP gate_proj | `*.mlp.gate_proj.weight_quantizer` | `*mlp*weight_quantizer` | ✅ |
| MLP up_proj | `*.mlp.up_proj.weight_quantizer` | `*mlp*weight_quantizer` | ✅ |
| MLP down_proj | `*.mlp.down_proj.weight_quantizer` | `*mlp*weight_quantizer` | ✅ |
| MoE 专家 | `*.block_sparse_moe.experts.0.w1.weight_quantizer` | `*block_sparse_moe*weight_quantizer` | ✅ |
| Attention q_proj | `*.self_attn.q_proj.weight_quantizer` | `default` | ❌ |
| Attention k_proj | `*.self_attn.k_proj.weight_quantizer` | `default` | ❌ |
| Attention v_proj | `*.self_attn.v_proj.weight_quantizer` | `default` | ❌ |
| Attention o_proj | `*.self_attn.o_proj.weight_quantizer` | `default` | ❌ |
| lm_head | `lm_head.weight_quantizer` | `*lm_head*` | ❌ |
| MoE 路由器 | `*.block_sparse_moe.gate.weight_quantizer` | `*block_sparse_moe.gate*` | ❌ |

### 4.3 为什么 attention 层不被量化？

关键在于 `default: {"enable": False}` 规则：

1. `*mlp*weight_quantizer` 只匹配路径中包含 `mlp` 的量化器
2. `model.layers.0.self_attn.q_proj.weight_quantizer` 路径中没有 `mlp`
3. 因此落入 `default` 规则，被禁用

---

## 5. NVFP4_OMLP_ONLY_CFG 的匹配逻辑

### 5.1 配置定义

```python
NVFP4_OMLP_ONLY_CFG = {
    "quant_cfg": {
        # 额外启用 o_proj 层量化
        "*o_proj*weight_quantizer": _nvfp4_quantizer,
        "*o_proj*input_quantizer": _nvfp4_quantizer,
        
        # 继承 MLP-only 配置
        **_nvfp4_mlp_only_quant_cfg,
    },
    "algorithm": "max",
}
```

### 5.2 与 MLP-only 的区别

| 模块 | MLP-only | OMLP-only |
|------|----------|-----------|
| MLP 层 | ✅ 量化 | ✅ 量化 |
| MoE 专家 | ✅ 量化 | ✅ 量化 |
| Attention o_proj | ❌ 不量化 | ✅ 量化 |
| Attention q_proj | ❌ 不量化 | ❌ 不量化 |
| Attention k_proj | ❌ 不量化 | ❌ 不量化 |
| Attention v_proj | ❌ 不量化 | ❌ 不量化 |

---

## 6. 自定义模式匹配

### 6.1 添加更多层

```python
import copy
import modelopt.torch.quantization as mtq

# 在 MLP-only 基础上也量化 q_proj 和 k_proj
custom_cfg = copy.deepcopy(mtq.NVFP4_MLP_ONLY_CFG)
custom_cfg["quant_cfg"]["*q_proj*weight_quantizer"] = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
}
custom_cfg["quant_cfg"]["*q_proj*input_quantizer"] = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
}
custom_cfg["quant_cfg"]["*k_proj*weight_quantizer"] = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
}
custom_cfg["quant_cfg"]["*k_proj*input_quantizer"] = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
}
```

### 6.2 跳过特定层

```python
custom_cfg = copy.deepcopy(mtq.NVFP4_OMLP_ONLY_CFG)

# 跳过第 0 层的 MLP
custom_cfg["quant_cfg"]["*layers.0.mlp*"] = {"enable": False}

# 跳过所有 down_proj
custom_cfg["quant_cfg"]["*down_proj*"] = {"enable": False}
```

### 6.3 按层索引选择性量化

```python
custom_cfg = copy.deepcopy(mtq.NVFP4_MLP_ONLY_CFG)

# 只量化前 16 层
for i in range(16, 32):
    custom_cfg["quant_cfg"][f"*layers.{i}.*"] = {"enable": False}
```

### 6.4 混合精度配置

```python
# 不同层使用不同的量化配置
custom_cfg = {
    "quant_cfg": {
        # 前 8 层使用 FP8
        "*layers.[0-7].mlp*weight_quantizer": {
            "num_bits": (4, 3),  # FP8 E4M3
            "enable": True,
        },
        # 后面的层使用 NVFP4
        "*layers.[8-9]*.mlp*weight_quantizer": mtq._nvfp4_quantizer,
        "*layers.1[0-9].mlp*weight_quantizer": mtq._nvfp4_quantizer,
        "*layers.2[0-9].mlp*weight_quantizer": mtq._nvfp4_quantizer,
        "*layers.3[0-1].mlp*weight_quantizer": mtq._nvfp4_quantizer,
        **mtq._default_disabled_quantizer_cfg,
    },
    "algorithm": "max",
}
```

---

## 7. 调试模式匹配

### 7.1 打印量化器状态

```python
import modelopt.torch.quantization as mtq

# 量化后检查哪些量化器被启用
model = mtq.quantize(model, mtq.NVFP4_MLP_ONLY_CFG, forward_loop)

for name, module in model.named_modules():
    if hasattr(module, "weight_quantizer"):
        wq = module.weight_quantizer
        print(f"{name}.weight_quantizer: enabled={wq.is_enabled}, amax={wq.amax}")
    if hasattr(module, "input_quantizer"):
        iq = module.input_quantizer
        print(f"{name}.input_quantizer: enabled={iq.is_enabled}, amax={iq.amax}")
```

### 7.2 使用 verbose 模式

```bash
python hf_ptq.py \
    --pyt_ckpt_path meta-llama/Llama-3.1-8B \
    --qformat nvfp4_mlp_only \
    --verbose  # 打印量化摘要
```

### 7.3 检查配置合并结果

```python
from modelopt.torch.quantization.config import get_quant_cfg

# 获取最终的量化配置
final_cfg = get_quant_cfg(model, mtq.NVFP4_MLP_ONLY_CFG)
for pattern, cfg in final_cfg.items():
    print(f"{pattern}: {cfg}")
```

---

## 8. 常见问题

### 8.1 某些层没有被量化

**检查步骤**：
1. 确认模式是否正确匹配模块路径
2. 检查是否被 `_default_disabled_quantizer_cfg` 中的规则覆盖
3. 使用 verbose 模式查看量化摘要

### 8.2 不想量化的层被量化了

**解决方案**：
```python
# 添加更具体的禁用规则
custom_cfg["quant_cfg"]["*specific_layer*"] = {"enable": False}
```

### 8.3 MoE 专家没有被量化

**可能原因**：
1. MoE 架构不是 Mixtral 风格（`block_sparse_moe`）
2. 需要使用 `register_sparse_moe_on_the_fly()` 动态注册

**解决方案**：
```python
from modelopt.torch.quantization.plugins.huggingface import register_sparse_moe_on_the_fly

# 在量化前调用
register_sparse_moe_on_the_fly(model)
```

---

## 相关文档

- [NVFP4 量化原理](./01_nvfp4_quantization_principle.md)
- [NVFP4 核心代码位置](./02_nvfp4_code_structure.md)
- [NVFP4 校准流程详解](./03_nvfp4_calibration.md)
- [NVFP4 权重导出机制](./04_nvfp4_weight_export.md)
- [NVFP4 MoE 模型处理](./05_nvfp4_moe_handling.md)
- [NVFP4 选择性量化使用指南](../NVFP4_Selective_Quantization_Guide.md)
