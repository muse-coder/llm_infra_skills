# NVFP4 核心代码位置

## 概述

本文档详细列出 NVIDIA Model Optimizer 中 NVFP4 量化相关的核心代码文件和关键函数，帮助开发者快速定位和理解实现细节。

---

## 1. 配置定义

### 1.1 文件位置

**文件**：`modelopt/torch/quantization/config.py`

### 1.2 核心配置

#### `_nvfp4_quantizer`（第 639-643 行）

NVFP4 量化器的基础配置：

```python
_nvfp4_quantizer = {
    "num_bits": (2, 1),  # E2M1: 2 位指数 + 1 位尾数
    "block_sizes": {
        -1: 16,           # 沿最后一个维度的块大小
        "type": "dynamic", # 动态缩放
        "scale_bits": (4, 3)  # FP8 缩放因子: 4 位指数 + 3 位尾数
    },
    "enable": True,
}
```

#### `NVFP4_MLP_ONLY_CFG`（第 645-655 行）

仅量化 MLP 层的配置：

```python
_nvfp4_mlp_only_quant_cfg = {
    "*mlp*weight_quantizer": _nvfp4_quantizer,
    "*mlp*input_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*weight_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*input_quantizer": _nvfp4_quantizer,
    **_default_disabled_quantizer_cfg,
}

NVFP4_MLP_ONLY_CFG = {
    "quant_cfg": _nvfp4_mlp_only_quant_cfg,
    "algorithm": "max",
}
```

#### `NVFP4_OMLP_ONLY_CFG`（第 657-665 行）

量化 MLP + o_proj 层的配置：

```python
NVFP4_OMLP_ONLY_CFG = {
    "quant_cfg": {
        "*o_proj*weight_quantizer": _nvfp4_quantizer,
        "*o_proj*input_quantizer": _nvfp4_quantizer,
        **_nvfp4_mlp_only_quant_cfg,
    },
    "algorithm": "max",
}
```

#### `_default_disabled_quantizer_cfg`

默认禁用的量化器配置：

```python
_default_disabled_quantizer_cfg = {
    "nn.BatchNorm1d": {"*": {"enable": False}},
    "nn.BatchNorm2d": {"*": {"enable": False}},
    "nn.BatchNorm3d": {"*": {"enable": False}},
    "nn.LeakyReLU": {"*": {"enable": False}},
    "*lm_head*": {"enable": False},
    "*proj_out.*": {"enable": False},           # Whisper 模型的 lm_head
    "*block_sparse_moe.gate*": {"enable": False}, # MoE 路由器
    "*router*": {"enable": False},               # MoE 路由器
    "*mlp.gate.*": {"enable": False},            # MoE 路由器（注意：不是 gate_proj）
    "*mlp.shared_expert_gate.*": {"enable": False}, # 共享专家 gate
    "*linear_attn.conv1d*": {"enable": False},
    "*mixer.conv1d*": {"enable": False},         # Mamba conv1d
    "*output_layer*": {"enable": False},
    "output.*": {"enable": False},
    "default": {"enable": False},                # 所有未匹配的量化器
}
```

---

## 2. 量化入口

### 2.1 文件位置

**文件**：`modelopt/torch/quantization/model_quant.py`

### 2.2 核心函数

#### `mtq.quantize()`（第 88-130 行）

量化和校准模型的主入口：

```python
def quantize(
    model: nn.Module,
    config: dict[str, Any | QuantizeConfig],
    forward_loop: ForwardLoop | None = None,
) -> nn.Module:
    """Quantizes and calibrates the model in-place.
    
    Args:
        model: 要量化的 PyTorch 模型
        config: 量化配置（如 NVFP4_MLP_ONLY_CFG）
        forward_loop: 校准数据迭代函数
    
    Returns:
        量化后的模型（原地修改）
    """
    # 1. 应用量化模式配置
    apply_mode(
        model,
        mode=get_modelike_from_algo_cfg(algorithm),
        mode_kwargs={"forward_loop": forward_loop},
    )
    
    # 2. 验证量化器属性
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            for attr_name in ["_amax", "_pre_quant_scale"]:
                module.validate_attr(attr_name=attr_name, warn_error=True, name=name)
    
    return model
```

#### `calibrate()`（第 47-86 行）

校准函数：

```python
def calibrate(
    model: nn.Module,
    algorithm: QuantizeAlgoCfgType = "max",
    forward_loop: ForwardLoop | None = None,
) -> nn.Module:
    """Adjusts weights and scaling factors based on selected algorithms."""
    with forward_with_reshard(model):
        apply_mode(
            model,
            mode=get_modelike_from_algo_cfg(algorithm),
            mode_kwargs={"forward_loop": forward_loop},
        )
    return model
```

---

## 3. TensorQuantizer 实现

### 3.1 文件位置

**文件**：`modelopt/torch/quantization/nn/modules/tensor_quantizer.py`

### 3.2 核心类

#### `TensorQuantizer`

基础量化器类，负责 fake quantization：

```python
class TensorQuantizer(nn.Module):
    """Tensor quantizer module."""
    
    def __init__(
        self,
        num_bits: int | tuple[int, int] = 8,
        axis: int | None = None,
        fake_quant: bool = True,
        unsigned: bool = False,
        narrow_range: bool = False,
        learn_amax: bool = False,
        **kwargs,
    ):
        ...
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply quantization to inputs."""
        if self._disabled:
            return inputs
        
        if self._if_calib:
            self._calibrator.collect(inputs)
        
        if self._if_quant:
            outputs = self._fake_quantize(inputs)
        else:
            outputs = inputs
        
        return outputs
```

#### `NVFP4StaticQuantizer`（第 740-760 行）

NVFP4 静态量化器：

```python
class NVFP4StaticQuantizer(TensorQuantizer):
    """Static NVFP4 quantizer with pre-computed amax."""
    
    def _fake_quantize(self, inputs):
        """Fake quantization using two-level scaling with _amax and _global_amax."""
        if self.amax is not None:
            return static_blockwise_fp4_fake_quant(
                inputs,
                self.amax,
                self.global_amax,  # Can be None, will be computed internally
                True,  # quantize_block_scales
                inputs.dtype,
                self._pass_through_bwd,
            )
        return super()._fake_quantize(inputs)
```

### 3.3 关键方法

#### `_fake_quantize()`（第 840-860 行）

Fake quantization 实现：

```python
def _fake_quantize(self, inputs):
    if self.block_sizes is not None and self.block_sizes.get("type", "static") == "dynamic":
        # 动态块量化
        block_size = self.block_sizes.get(-1, None) or self.block_sizes.get(
            inputs.dim() - 1, None
        )
        outputs = dynamic_block_quant(
            inputs,
            block_size,
            amax,
            self._get_bias(inputs),
            self._num_bits,
            self.block_sizes.get("scale_bits", None),
            ...
        )
    return outputs
```

#### `_real_quantize()`（第 700-735 行）

真实量化（用于导出）：

```python
def _real_quantize(self, inputs):
    """Real quantization for NVFP4."""
    if self._block_sizes.get("scale_bits") == (4, 3):
        # NVFP4 default quantization
        outputs, _weights_scaling_factor, _weights_scaling_factor_2 = NVFP4QTensor.quantize(
            inputs,
            self._block_sizes[-1],
            weights_scaling_factor_2=self.amax.float() / (448.0 * 6.0)
            if self.amax is not None
            else None,
            try_tensorrt=True,
        )
        buffer_to_register["_scale"] = _weights_scaling_factor
        buffer_to_register["_double_scale"] = _weights_scaling_factor_2
```

---

## 4. NVFP4QTensor 实现

### 4.1 文件位置

**文件**：`modelopt/torch/quantization/qtensor/nvfp4_tensor.py`

### 4.2 核心类

#### `NVFP4QTensor`

NVFP4 量化张量类：

```python
class NVFP4QTensor:
    """NVFP4 quantized tensor representation."""
    
    # E2M1 值表
    e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    
    @classmethod
    def quantize(cls, input, block_size, weights_scaling_factor=None, 
                 weights_scaling_factor_2=None, keep_high_precision=False, try_tensorrt=False):
        """Quantize input tensor to NVFP4 format."""
        ...
    
    def dequantize(self, dtype=None, fast=False, **kwargs):
        """Dequantize NVFP4 tensor back to floating point."""
        ...
    
    @classmethod
    def get_weights_scaling_factor_2(cls, input):
        """Calculate global scaling factor."""
        return reduce_amax(input) / (6.0 * 448.0)
    
    @classmethod
    def get_weights_scaling_factor(cls, input, block_size, weights_scaling_factor_2):
        """Calculate per-block scaling factor."""
        ...
```

### 4.3 关键方法

#### `quantize()`（第 130-220 行）

量化方法：

```python
@classmethod
def quantize(cls, input: torch.Tensor, block_size: int, ...):
    # 1. 计算全局缩放因子
    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = cls.get_weights_scaling_factor_2(input)
    
    # 2. 计算分块缩放因子
    if weights_scaling_factor is None:
        weights_scaling_factor, _ = cls.get_weights_scaling_factor(
            input, block_size, weights_scaling_factor_2
        )
    
    # 3. 重塑张量
    input = input.view((*tuple(input.shape[:-1]), -1, block_size))
    
    # 4. 应用缩放
    scaled_weight = input / (
        (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
    )
    
    # 5. 转换为 FP4
    q_weight = cls._cast_fp4(scaled_weight)
    
    # 6. 打包权重
    packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
    
    return (cls(input_shape, input_dtype, packed_weight),
            weights_scaling_factor,
            weights_scaling_factor_2)
```

#### `dequantize()`（第 230-280 行）

反量化方法：

```python
def dequantize(self, dtype: torch.dtype = None, fast=False, **kwarg):
    # 1. 解包张量
    unpacked[..., 1::2] = input >> 4
    unpacked[..., 0::2] = input & 0x0F
    
    # 2. 查表获取 E2M1 值
    unpacked = self.get_e2m1_values(input.device)[unpacked.long()]
    
    # 3. 应用缩放因子恢复原始值
    output = unpacked * scale * global_scale
    
    return output.to(dtype)
```

---

## 5. Triton 内核

### 5.1 文件位置

**文件**：`modelopt/torch/quantization/triton/fp4_kernel.py`

### 5.2 核心函数

#### `static_blockwise_fp4_fake_quant()`

静态分块 FP4 fake quantization：

```python
def static_blockwise_fp4_fake_quant(
    x: torch.Tensor,
    amax: torch.Tensor,
    global_amax: torch.Tensor | None = None,
    quantize_block_scales: bool = True,
    output_dtype: torch.dtype = torch.bfloat16,
    pass_through_bwd: bool = False,
) -> torch.Tensor:
    """Apply static blockwise FP4 fake quantization."""
    ...
```

#### `static_blockwise_fp4_fake_quant_kernel()`（第 140-180 行）

Triton JIT 编译的量化内核：

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
    
    # 量化到 FP4 值
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

## 6. 校准算法

### 6.1 文件位置

**文件**：`modelopt/torch/quantization/calib/max.py`

### 6.2 核心类

#### `MaxCalibrator`（第 24-85 行）

Max 校准器：

```python
class MaxCalibrator(_Calibrator):
    """Max calibrator, tracks the maximum value globally."""
    
    def __init__(self, num_bits=8, axis=None, unsigned=False, track_amax=False):
        super().__init__(num_bits, axis, unsigned)
        self._track_amax = track_amax
        if self._track_amax:
            self._amaxs = []
        self._calib_amax = None
    
    @torch.no_grad()
    def collect(self, x):
        """Tracks the absolute max of all tensors."""
        reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
        
        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax = torch.max(self._calib_amax, local_amax)
    
    def compute_amax(self):
        """Return the absolute max of all tensors collected."""
        return self._calib_amax
```

---

## 7. 导出功能

### 7.1 文件位置

**文件**：`modelopt/torch/export/unified_export_hf.py`

### 7.2 核心函数

#### `export_hf_checkpoint()`（第 427-547 行）

导出 HuggingFace 格式 checkpoint：

```python
def export_hf_checkpoint(
    model: Any,
    dtype: torch.dtype | None = None,
    export_dir: Path | str = tempfile.gettempdir(),
    save_modelopt_state: bool = False,
    components: list[str] | None = None,
    extra_state_dict: dict[str, torch.Tensor] | None = None,
    **kwargs,
):
    """Export quantized model to HuggingFace checkpoint format."""
    ...
```

---

## 8. 代码结构总览

```
modelopt/torch/
├── quantization/
│   ├── config.py                    # 量化配置定义
│   ├── model_quant.py               # mtq.quantize() 入口
│   ├── model_calib.py               # 校准入口
│   ├── nn/
│   │   └── modules/
│   │       ├── tensor_quantizer.py  # TensorQuantizer 实现
│   │       └── quant_module.py      # QuantModule 基类
│   ├── qtensor/
│   │   └── nvfp4_tensor.py          # NVFP4QTensor 实现
│   ├── calib/
│   │   └── max.py                   # MaxCalibrator 实现
│   ├── triton/
│   │   └── fp4_kernel.py            # Triton FP4 内核
│   ├── plugins/
│   │   └── huggingface.py           # HuggingFace 模型支持（含 MoE）
│   └── utils/
│       └── core_utils.py            # 工具函数（含 sync_moe_expert_amax）
└── export/
    ├── unified_export_hf.py         # export_hf_checkpoint()
    └── quant_utils.py               # 导出工具函数
```

---

## 相关文档

- [NVFP4 量化原理](./01_nvfp4_quantization_principle.md)
- [NVFP4 校准流程详解](./03_nvfp4_calibration.md)
- [NVFP4 权重导出机制](./04_nvfp4_weight_export.md)
- [NVFP4 MoE 模型处理](./05_nvfp4_moe_handling.md)
- [NVFP4 选择性量化使用指南](../NVFP4_Selective_Quantization_Guide.md)
