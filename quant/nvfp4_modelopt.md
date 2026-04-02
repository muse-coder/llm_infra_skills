# ModelOpt NVFP4 PTQ 知识库

> 覆盖范围：NVFP4 的数据格式定义、校准逻辑、Fake Quant 量化逻辑、Real Quant 权重压缩、以及导出逻辑（scaling factor 提取）。
> 代码库：`nvidia/modelopt` — `modelopt/torch/quantization/` & `modelopt/torch/export/`

---

## 一、NVFP4 数据格式基础

### 1.1 格式定义

NVFP4 是 **E2M1 格式的 4-bit 浮点数**，两级缩放设计：

| 级别 | 名称 | 精度 | 粒度 | 来源 |
|------|------|------|------|------|
| Level-1 | `weights_scaling_factor_2`（global scale） | FP32 scalar | per-tensor | 校准得到（静态） |
| Level-2 | `weights_scaling_factor`（per-block scale） | FP8 E4M3 | per-16-elements | 权重导出时计算 |

E2M1 可表示的值集合（8 个非零绝对值）：
```
{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}    maxbound = 6.0
```

**文件**：`modelopt/torch/quantization/qtensor/nvfp4_tensor.py`

```python
# E2M1 边界和值的查找表（CPU 路径用）
e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
```

---

## 二、配置定义（Config）

**文件**：`modelopt/torch/quantization/config.py`

### 2.1 核心量化器配置

```python
_nvfp4_quantizer = {
    "num_bits": (2, 1),                                        # E2M1 格式
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},  # block=16, FP8 scale
    "enable": True,
}
```

- `block_sizes[-1]: 16`：沿最后维每 16 个元素为一个量化块
- `type: "dynamic"`：per-block scale 在量化时动态计算（不需要提前校准）
- `scale_bits: (4, 3)`：per-block scale 用 FP8 E4M3 存储

### 2.2 预定义 Config

```python
# 全网络 NVFP4
NVFP4_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": _nvfp4_quantizer,
        "*input_quantizer": _nvfp4_quantizer,
        **_default_disabled_quantizer_cfg,   # lm_head/router 等默认 disable
    },
    "algorithm": "max",
}

# 只量化 MLP 层（不量化 attention QKV/O）
_nvfp4_mlp_only_quant_cfg = {
    "*mlp*weight_quantizer": _nvfp4_quantizer,
    "*mlp*input_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*weight_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*input_quantizer": _nvfp4_quantizer,
    **_default_disabled_quantizer_cfg,
}
NVFP4_MLP_ONLY_CFG = {"quant_cfg": _nvfp4_mlp_only_quant_cfg, "algorithm": "max"}

# 量化 MLP + o_proj（不量化 q/k/v_proj）
NVFP4_OMLP_ONLY_CFG = {
    "quant_cfg": {
        "*o_proj*weight_quantizer": _nvfp4_quantizer,
        "*o_proj*input_quantizer": _nvfp4_quantizer,
        **_nvfp4_mlp_only_quant_cfg,
    },
    "algorithm": "max",
}
```

### 2.3 `_default_disabled_quantizer_cfg`（通用禁用规则）

```python
_default_disabled_quantizer_cfg = {
    "nn.BatchNorm1d": {"*": {"enable": False}},
    # ...
    "*lm_head*": {"enable": False},
    "*block_sparse_moe.gate*": {"enable": False},   # MOE router
    "*router*": {"enable": False},
    "*mlp.gate.*": {"enable": False},
    # ...
    "default": {"enable": False},   # ← 最关键：未命中规则的层默认 disable
}
```

**核心规则**：通配符匹配采用 `fnmatch`，**后面的规则覆盖前面的规则**，`"default"` 兜底禁用所有未显式启用的层。

---

## 三、校准逻辑（Calibration）

### 3.1 入口

**文件**：`examples/llm_ptq/hf_ptq.py`

```python
# PTQ 主流程
language_model = mtq.quantize(language_model, quant_cfg, forward_loop=calibrate_loop)
```

`mtq.quantize()` 内部两步：
1. 将 `nn.Linear` 替换为 `QuantLinear`（附加 `weight_quantizer` / `input_quantizer`）
2. 按 `quant_cfg` 配置每个 `TensorQuantizer` 参数，然后调用 `calibrate()`

### 3.2 校准主函数

**文件**：`modelopt/torch/quantization/model_calib.py`

```python
@torch.no_grad()
def max_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    distributed_sync=True,
):
    """Max calibration: 三步走"""
    # Step 1: 切换所有 TQ 到统计收集模式
    enable_stats_collection(model)

    # Step 2: 执行数据前向，收集激活 amax；或纯权重量化（不需要 forward_loop）
    if forward_loop is None:
        weight_only_quantize(model)   # 只统计权重
    else:
        forward_loop(model)           # 数据驱动，激活+权重

    # Step 3: 写入 amax buffer
    finish_stats_collection(model)

    # Step 4: 分布式同步（DP/TP/EP）
    # ... amax sync across data/tensor/expert parallel groups
```

### 3.3 Step 1：`enable_stats_collection()`

```python
def enable_stats_collection(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module._disabled:
            if module._use_constant_amax:
                module.disable_quant()    # 固定 amax 的量化器：关闭量化，不收集
                continue
            elif module._calibrator is not None:
                module.disable_quant()    # _if_quant = False
                module.enable_calib()     # _if_calib = True
            else:
                module.disable()
```

NVFP4 config 中 `enable: True` 的量化器：`_if_quant=False, _if_calib=True`
未命中（`default: {enable: False}`）的量化器：`_disabled=True`，直接跳过

### 3.4 Step 2：数据前向 → `MaxCalibrator.collect()`

前向传播中每个启用的 `TensorQuantizer.forward()` 执行：

**文件**：`modelopt/torch/quantization/nn/modules/tensor_quantizer.py`

```python
# TensorQuantizer.forward() 核心路径（calib 阶段）
if self._if_calib and not self._dynamic:
    # 注意：虽然 block_sizes.type == "dynamic"，但 _dynamic 属性对应的是
    # quantizer 本身是否是纯动态模式（无需 amax），NVFP4 需要 global amax，所以 _dynamic=False
    self.collect(inputs)

def collect(self, inputs) -> None:
    if not self._if_calib or self._dynamic:
        return
    self._calibrator.collect(inputs)    # 调用 MaxCalibrator
```

**文件**：`modelopt/torch/quantization/calib/max.py`

```python
class MaxCalibrator(_Calibrator):

    @torch.no_grad()
    def collect(self, x):
        """跨所有 batch 追踪全局 absolute max"""
        # axis=None → per-tensor scalar amax
        reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()

        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            # Running max，取历史最大值
            self._calib_amax = torch.max(self._calib_amax, local_amax)

    def compute_amax(self):
        return self._calib_amax
```

- **激活**（`input_quantizer`）：每个 batch 都会 collect，running max 跨所有样本
- **权重**（`weight_quantizer`）：权重是常数，第一次 forward 时 collect 一次即可

### 3.5 Step 3：`finish_stats_collection()`

```python
def finish_stats_collection(model: nn.Module, method=None, **kwargs):
    for _, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or module._disabled:
            continue
        cal = getattr(module, "_calibrator", None)
        if cal and not getattr(module, "_dynamic", False):
            if cal.compute_amax(**kwargs) is not None:
                module.load_calib_amax(**kwargs)   # ← 核心：写入 _amax buffer

        module.enable_quant()     # 恢复量化
        module.disable_calib()    # 关闭统计收集
```

`load_calib_amax()` 实现：

```python
def load_calib_amax(self, *args, **kwargs):
    calib_amax = self._calibrator.compute_amax(*args, **kwargs)
    if not hasattr(self, "_amax"):
        self.register_buffer("_amax", calib_amax.clone().detach())
    else:
        self._amax.data.copy_(calib_amax.clone().detach())
```

校准完成后，每个 NVFP4 量化器持有：
- `_amax`：scalar，global per-tensor amax（激活/权重各自独立）

### 3.6 Step 4：分布式 amax 同步

**文件**：`modelopt/torch/quantization/model_calib.py`（`max_calibrate()` 尾部）

```python
# DP/EP 同步：确保 data parallel 所有 rank 的 amax 一致
def sync_quantizer_amax_across_dp_ep(quantizer, parallel_state):
    if getattr(quantizer, "_amax", None) is not None:
        quantizer.sync_amax_across_distributed_group(parallel_state.data_parallel_group)
        quantizer.sync_amax_across_distributed_group(parallel_state.expert_model_parallel_group)

# TP 同步
def sync_quantizer_amax_across_tp(quantizer, ...):
    # NVFP4 (dynamic block type) 的 amax 不需要 TP 同步，直接 return
    if quantizer.block_sizes is not None:
        if getattr(quantizer.block_sizes, "type", None) == "dynamic":
            return    # ← NVFP4 跳过 TP sync
    if quantizer.axis in axes_for_sync and quantizer.amax is not None:
        quantizer.sync_amax_across_distributed_group(parallel_state.tensor_parallel_group)
```

---

## 四、量化逻辑（Fake Quantization）

PTQ 阶段评估时，`_fake_quantize=True`，`TensorQuantizer.forward()` → `_fake_quantize()`。

### 4.1 路由逻辑

**文件**：`modelopt/torch/quantization/nn/modules/tensor_quantizer.py`

```python
def _fake_quantize(self, inputs):
    amax = self._get_amax(inputs)   # 返回校准得到的 _amax（scalar）

    if self.block_sizes is not None and self.block_sizes.get("type", "static") == "dynamic":
        # NVFP4 走这里（block_sizes.type == "dynamic"）
        block_size = self.block_sizes.get(-1, None)   # = 16

        outputs = dynamic_block_quant(
            inputs,
            block_size,                               # = 16
            amax,                                     # global amax（scalar）
            self._get_bias(inputs),                   # None for standard NVFP4
            self._num_bits,                           # = (2, 1)
            self.block_sizes.get("scale_bits", None), # = (4, 3) → FP8 E4M3
            ...
        )
```

`dynamic_block_quant` = `DynamicBlockQuantizationFunction.apply`

**文件**：`modelopt/torch/quantization/tensor_quant.py`

```python
def _dynamic_block_quantize_forward(ctx, inputs, block_size, amax, num_bits, scale_bits, ...):
    # num_bits=(2,1): exponent_bits=2, total_bits=4
    # scale_bits=(4,3): FP8 E4M3 的 per-block scale
    exponent_bits = num_bits[0]   # = 2
    num_bits_total = num_bits[0] + num_bits[1] + 1  # = 4
    scale_exponent_bits = scale_bits[0]  # = 4
    scale_num_bits = scale_bits[0] + scale_bits[1] + 1  # = 8

    outputs = dynamic_block_quantize_op(
        inputs, block_size, amax,
        num_bits_total, exponent_bits, scale_num_bits, scale_exponent_bits,
    )
    return outputs
```

`dynamic_block_quantize_op` = `torch.ops.tensorrt.dynamic_block_quantize_op`，实现在：

```python
def _dynamic_block_quantize_impl(inputs, block_size, amax, num_bits, ...):
    num_bits_tuple = (exponent_bits, num_bits - exponent_bits - 1)  # = (2, 1) = E2M1
    scale_bits_tuple = (scale_exponent_bits, scale_num_bits - scale_exponent_bits - 1)  # = (4, 3) = E4M3

    if (
        num_bits_tuple == (2, 1)
        and scale_bits_tuple == (4, 3)
        and triton_kernel.IS_AVAILABLE
        and hasattr(triton_kernel, "fp4_fake_quant_block")  # 需要 compute >= 8.9
        and amax is not None
    ):
        # 路径 A：Triton kernel（Hopper+）
        return triton_kernel.fp4_fake_quant_block(inputs, amax)

    # 路径 B：CUDA C++ extension（fallback）
    cuda_ext_mx = get_cuda_ext_mx(raise_if_failed=True)
    return cuda_ext_mx.fused_amax_convert(inputs, block_size, E2M1, E4M3, amax)
```

### 4.2 路径 A：Triton Kernel（Hopper+，compute >= 8.9）

**文件**：`modelopt/torch/quantization/triton/fp4_kernel_hopper.py`

此文件包含**两个** Triton kernel，通过环境变量 `MODELOPT_FOUROVERSIX` 选择：

#### 4.2.1 标准 NVFP4 kernel：`fp4_fake_quant_block`

```python
def fp4_fake_quant_block(x, global_amax, block_size=16, ...):
    # global_scale = global_amax / (6.0 * 448.0)   ← 标准 NVFP4 用 448
    global_scale = (global_amax.float() / (6.0 * 448.0)).to(x.device)
    fp4_fake_quant_kernel[grid](x, y, M, N, global_scale, ...)
```

Triton kernel 核心逻辑（每个 thread block 处理 `[TILE_M, TILE_N]`）：

```python
@triton.jit
def fp4_fake_quant_kernel(x_ptr, y_ptr, M, N, global_scale_ptr, ...):
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    global_scale_safe = tl.where(global_scale > 0.0, global_scale, 1e-12)

    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)
    block_max = tl.max(x_abs, axis=2, keep_dims=True)

    # per-block FP8 scale，clip 到 448
    block_max_scaled = block_max / (6.0 * global_scale_safe)
    block_max_scaled = tl.minimum(block_max_scaled, 448.0)
    block_max_quant = block_max_scaled.to(tl.float8e4nv).to(tl.float32) * global_scale
    block_max_quant = tl.where(block_max_quant >= 1e-5, block_max_quant, 1.0)

    abs_scaled = x_abs / block_max_quant
    q_val = tl.where(abs_scaled <= 0.25, 0.0,
              tl.where(abs_scaled < 0.75, 0.5, ...))   # E2M1 离散化

    x_rescaled = q_val * block_max_quant * sign(x)
    tl.store(y_block_ptr, x_rescaled.to(OUT_DTYPE), ...)
```

**标准 NVFP4 Fake Quant 公式：**
```
global_scale         = global_amax / (6.0 × 448.0)
per_block_scale_fp8  = clip(block_max / (6.0 × global_scale), 448.0) → FP8 E4M3
dequant_scale        = per_block_scale_fp8(fp32) × global_scale
x_fake               = E2M1_round(|x| / dequant_scale) × dequant_scale × sign(x)
```

#### 4.2.2 FourOverSix kernel：`fp4_fouroversix_fake_quant_block`

当 `MODELOPT_FOUROVERSIX=1` 时，`_dynamic_block_quantize_impl` 优先调用此 kernel。

```python
def fp4_fouroversix_fake_quant_block(x, global_amax, block_size=16, ...):
    # global_scale = global_amax / (6.0 * 256.0)   ← fouroversix 用 256，为 ×1.5 留空间
    global_scale = (global_amax.float() / (6.0 * 256.0)).to(x.device)
    fp4_fouroversix_fake_quant_kernel[grid](x, y, M, N, global_scale, ...)
```

```python
@triton.jit
def fp4_fouroversix_fake_quant_kernel(...):
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)
    block_max = tl.max(x_abs, axis=2, keep_dims=True)

    # ── 候选 6 方案（标准，最大值映射到 6.0）──
    scale_6_hp = block_max / (6.0 * global_scale_safe)
    scale_6_hp = tl.minimum(scale_6_hp, 256.0)          # clip 到 256
    scale_6 = scale_6_hp.to(tl.float8e4nv).to(tl.float32)

    # ── 候选 4 方案（1.5× 扩张，最大值映射到 4.0）──
    scale_4_hp = scale_6_hp * 1.5
    scale_4_hp = tl.minimum(scale_4_hp, 256.0)          # clip 到 256（≈384 after ×1.5）
    scale_4 = scale_4_hp.to(tl.float8e4nv).to(tl.float32)

    # 两套 dequant scale
    deq_6 = scale_6 * global_scale      # 小于 1e-5 时替换为 1.0
    deq_4 = scale_4 * global_scale

    # 两路 E2M1 fake-quant
    q6 = _e2m1_round(x_abs / deq_6)    # 结果 ∈ {0,0.5,1,1.5,2,3,4,6}
    q4 = _e2m1_round(x_abs / deq_4)

    recon_6 = q6 * deq_6 * sign        # 反量化重建
    recon_4 = q4 * deq_4 * sign

    # 逐块 MSE 比较
    err_6 = tl.sum((recon_6 - tile_reshaped)², axis=2)
    err_4 = tl.sum((recon_4 - tile_reshaped)², axis=2)

    # 逐块选择误差更小的方案
    pick_4 = (err_4 < err_6)
    result = tl.where(pick_4, recon_4, recon_6)

    tl.store(y_block_ptr, result.to(OUT_DTYPE), ...)
```

**FourOverSix Fake Quant 公式：**
```
global_scale = global_amax / (6.0 × 256.0)      ← 256 不是 448

scale_6 = clip(block_max / (6.0 × global_scale), 256) → FP8 E4M3
scale_4 = clip(scale_6 × 1.5, 256)              → FP8 E4M3（≈block_max/(4.0×global_scale)）

deq_6   = scale_6(fp32) × global_scale
deq_4   = scale_4(fp32) × global_scale

recon_6 = E2M1_round(|x|/deq_6) × deq_6 × sign(x)
recon_4 = E2M1_round(|x|/deq_4) × deq_4 × sign(x)

MSE_6 = sum((recon_6 - x)²)   per block
MSE_4 = sum((recon_4 - x)²)   per block

output  = recon_4 if MSE_4 < MSE_6 else recon_6   (逐块独立选择)
```

### 4.3 路径 B：CUDA C++ Extension（fallback）

**文件**：`modelopt/torch/quantization/src/tensor_quant_mx.cu`

```cpp
at::Tensor fused_amax_convert(at::Tensor x, const int blocksize,
                               Types format, Types scale_format,
                               std::optional<at::Tensor> global_amax) {
    // CUDA kernel 完成相同的两级缩放逻辑
    // format=E2M1, scale_format=E4M3
    // 通过 pybind11 暴露为 cuda_ext_mx.fused_amax_convert
}
```

### 4.4 NVFP4StaticQuantizer（静态 per-block amax 变种）

`NVFP4StaticQuantizer` 是 `TensorQuantizer` 的子类，用于权重使用静态 per-block amax 的场景（如 `mse` 校准后）。

**文件**：`modelopt/torch/quantization/nn/modules/tensor_quantizer.py`

```python
class NVFP4StaticQuantizer(TensorQuantizer):
    """持有 _amax（per-block amax）和 _global_amax（全局 amax）两级静态 scale"""

    def _fake_quantize(self, inputs):
        if self.amax is not None:
            # 调用 static_blockwise_fp4_fake_quant（Triton kernel）
            return static_blockwise_fp4_fake_quant(
                inputs,
                self.amax,        # per-block amax（shape: [N//16]）
                self.global_amax, # scalar global amax
                True,             # quantize_block_scales=True，scale 量化为 FP8
                inputs.dtype,
                self._pass_through_bwd,
            )
        return super()._fake_quantize(inputs)
```

**文件**：`modelopt/torch/quantization/triton/fp4_kernel.py`

```python
def static_blockwise_fp4_fake_quant(x, amax, global_amax=None, quantize_block_scales=True, ...):
    """静态 per-block amax 的 Fake Quant"""
    amax = amax.float()
    scale = amax / 6.0    # per-block FP4 max = 6.0

    if quantize_block_scales:
        # 把 per-block scale 量化为 FP8（模拟实际存储精度）
        scale_fp8_quant_amax = global_amax / 6.0
        scale = scaled_e4m3_impl(scale, scale_fp8_quant_amax)   # → FP8 E4M3

    # 启动 Triton kernel：每个 block 一个线程块
    static_blockwise_fp4_fake_quant_kernel[grid](x_flat, y_flat, scale_flat, ...)

@triton.jit
def static_blockwise_fp4_fake_quant_kernel(x_ptr, y_ptr, scale_ptr, ...):
    scale = tl.load(scale_ptr + pid).to(tl.float32)
    x = tl.load(x_ptr + idx).to(tl.float32)
    abs_scaled = tl.abs(x) / scale_safe

    # E2M1 离散化（同上）
    q_val = tl.where(abs_scaled <= 0.25, 0.0, ...)
    x_quant = tl.where(x >= 0, q_val * scale_safe, -(q_val * scale_safe))
    tl.store(y_ptr + idx, x_quant.to(OUT_DTYPE))
```

---

## 五、Real Quantization（权重压缩）

当需要把模型权重真正压缩（int4 packed）时，调用 `_real_quantize()`。通常在模型导出（checkpoint save）之前触发。

### 5.1 `_real_quantize()` 路由

**文件**：`modelopt/torch/quantization/nn/modules/tensor_quantizer.py`

```python
from ...qtensor.nvfp4_tensor import _FP8_MAX_SCALE, _FOUROVERSIX
# _FP8_MAX_SCALE = 256.0 if _FOUROVERSIX else 448.0

def _real_quantize(self, inputs):
    # ... MX / FP8 / INT8 路由省略 ...

    elif self._block_sizes.get("scale_bits") == (4, 3):
        # NVFP4 real quantization
        outputs, _weights_scaling_factor, _weights_scaling_factor_2 = NVFP4QTensor.quantize(
            inputs,
            self._block_sizes[-1],          # block_size = 16
            weights_scaling_factor_2=(
                self.amax.float() / (_FP8_MAX_SCALE * 6.0)
                # 标准: amax / (448 × 6)；fouroversix: amax / (256 × 6)
                if self.amax is not None else None
            ),
            try_tensorrt=not _FOUROVERSIX,  # fouroversix 时跳过 TRT-LLM fp4_quantize
        )
        buffer_to_register["_scale"] = _weights_scaling_factor
        buffer_to_register["_double_scale"] = _weights_scaling_factor_2

    self._dequantize = True
    return outputs
```

> **为什么 fouroversix 必须跳过 `try_tensorrt`？**
>
> TRT-LLM 的 `torch.ops.trtllm.fp4_quantize` 是标准 NVFP4（单方案，`max_s=448`），内部实现与 fouroversix
> 的双方案逻辑不兼容。若不跳过，TRT-LLM 会用 `1/wsf2 = 6×448/amax` 作为输入 scale 直接量化，
> 完全绕过 fouroversix 的逐块选择逻辑，导出结果不正确。
> 因此，设 `try_tensorrt=False`，强制走 PyTorch fallback 路径（`get_weights_scaling_factor`），
> 在那里执行 fouroversix 的双方案选择。

### 5.2 `NVFP4QTensor.quantize()` 完整流程

**文件**：`modelopt/torch/quantization/qtensor/nvfp4_tensor.py`

```python
# 顶部全局常量（受 MODELOPT_FOUROVERSIX 控制）
_FOUROVERSIX   = os.environ.get("MODELOPT_FOUROVERSIX", "0") == "1"
_FP8_MAX_SCALE = 256.0 if _FOUROVERSIX else 448.0

@classmethod
def quantize(cls, input, block_size, weights_scaling_factor=None,
             weights_scaling_factor_2=None, keep_high_precision=False, try_tensorrt=False):

    input = reduce_block_padding(input, block_sizes={-1: block_size})

    # Step 1: global scale（scalar）
    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = cls.get_weights_scaling_factor_2(input)
        # 标准:  amax / (6.0 × 448.0)
        # 4/6:   amax / (6.0 × 256.0)   ← 由调用方传入，这里是兜底

    # Step 2: 尝试 TRT-LLM（仅标准 NVFP4，fouroversix 时 try_tensorrt=False）
    if fp4_compatible() and try_tensorrt and ...:
        packed_weight, weights_scaling_factor = torch.ops.trtllm.fp4_quantize(
            input, 1.0 / weights_scaling_factor_2, block_size, False
        )
        return (cls(...), weights_scaling_factor, weights_scaling_factor_2)

    # Step 3: PyTorch fallback —— 计算 per-block FP8 scale
    if weights_scaling_factor is None:
        weights_scaling_factor, _ = cls.get_weights_scaling_factor(
            input, block_size, weights_scaling_factor_2
        )
        # ← fouroversix 时此函数内部执行双方案选择（见 5.4）

    # Step 4–6: 归一化、E2M1 离散化、打包（同原流程）
    ...
```

### 5.3 `_cast_fp4()`：E2M1 编码

```python
@classmethod
def _cast_fp4(cls, weight: torch.Tensor):
    """将归一化后的浮点权重编码为 E2M1 uint4"""
    # e2m1_bounds = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]

    # 1. 提取符号位
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs_()

    # 2. 二分查找：找到 |w| 在 e2m1_bounds 中的 ordinal（对应 E2M1 编码）
    e2m1_bounds = cls.get_e2m1_bounds(weight.device)
    ord = torch.searchsorted(e2m1_bounds, weight_abs, out_int32=True).to(torch.uint8)

    # 3. 处理边界舍入（odd-indexed bounds: 0.75, 1.75, 2.5 处向上舍入）
    odd_bounds = e2m1_bounds[[1, 3, 5]]
    equals_odd_bounds = torch.any(weight_abs.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.uint8)

    # 4. 最终 uint4 编码：高位 sign，低 3 位 ordinal
    return (sign_bit << 3) + ord + equals_odd_bounds
    # 例：+1.5 → sign=0, ord=3, 编码=3 (二进制 0011)
    # 例：-3.0 → sign=1, ord=5, 编码=13 (二进制 1101)
```

### 5.4 `get_weights_scaling_factor()`：per-block FP8 scale 计算

此函数根据 `_FOUROVERSIX` 走不同路径：

#### 标准 NVFP4 路径（`_FOUROVERSIX=False`）

```python
@classmethod
def get_weights_scaling_factor(cls, input, block_size, weights_scaling_factor_2=None, ...):
    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = cls.get_weights_scaling_factor_2(input)

    per_block_amax = reduce_block_amax(input, block_sizes={-1: block_size}).float()

    # per-block scale = block_amax / (6.0 × wsf2) ≈ block_amax/x_amax × 448
    per_block_scale = per_block_amax / (6.0 * weights_scaling_factor_2.to(per_block_amax.device))
    per_block_scale[per_block_scale == 0] = 1.0

    if not keep_high_precision:
        per_block_scale = per_block_scale.to(torch.float8_e4m3fn)

    return per_block_scale, weights_scaling_factor_2
```

#### FourOverSix 路径（`_FOUROVERSIX=True`）

```python
@classmethod
def get_weights_scaling_factor(cls, input, block_size, weights_scaling_factor_2=None, ...):
    per_block_amax = reduce_block_amax(input, block_sizes={-1: block_size}).float()
    wsf2_device = weights_scaling_factor_2.to(per_block_amax.device)

    # 候选 6 方案（clamp 到 256，不是 448）
    scale_6_hp = (per_block_amax / (6.0 * wsf2_device)).clamp(max=_FP8_MAX_SCALE)  # max=256
    scale_6 = scale_6_hp.to(torch.float8_e4m3fn)

    # 候选 4 方案（6 方案 × 1.5，最大值映射到 4.0）
    scale_4_hp = (scale_6_hp * 1.5).clamp(max=_FP8_MAX_SCALE)    # 384 clamp → FP8
    scale_4 = scale_4_hp.to(torch.float8_e4m3fn)

    # 模拟 E2M1 fake-quant，计算重建误差
    # （用 searchsorted + e2m1 查表对两套 scale 各做一次量化-反量化）
    input_view = input.view(..., -1, block_size)

    deq_6 = scale_6.float() * wsf2_device                     # per-block dequant scale
    deq_4 = scale_4.float() * wsf2_device
    recon_6 = fake_quant_e2m1(input_view, deq_6)
    recon_4 = fake_quant_e2m1(input_view, deq_4)

    err_6 = ((recon_6 - input_view) ** 2).sum(dim=-1)         # MSE per block
    err_4 = ((recon_4 - input_view) ** 2).sum(dim=-1)

    # 逐块选择：误差更小的方案胜出
    per_block_scale = torch.where(err_4 < err_6, scale_4, scale_6)
    return per_block_scale, weights_scaling_factor_2
```

**关键差异总结：**

| 维度 | 标准 NVFP4 | FourOverSix |
|------|-----------|-------------|
| `_FP8_MAX_SCALE` | 448 | **256** |
| `scale_hp` for max block | `≈ 448` | `256`（6方案）/ `≈ 384`（4方案） |
| 候选方案数量 | 1 | **2**（6方案 + 4方案） |
| 选择逻辑 | 无 | **逐块 MSE 比较** |
| `try_tensorrt` | True | **False** |

---

## 六、导出逻辑（Export）

导出时从量化后的模型中提取 scaling factors，写入 `LinearConfig`，供 TRT-LLM 消费。

### 6.1 格式识别：`get_quantization_format()`

**文件**：`modelopt/torch/export/quant_utils.py`

```python
def get_quantization_format(module) -> str | None:
    weight_quantizer = getattr(layer, "weight_quantizer", None)
    input_quantizer = getattr(layer, "input_quantizer", None)

    # ...
    if weight_quantizer.num_bits == (2, 1):
        block_sizes = weight_quantizer.block_sizes
        scale_bits = block_sizes.get("scale_bits")

        if input_quantizer and hasattr(input_quantizer, "_pre_quant_scale"):
            return QUANTIZATION_NVFP4_AWQ
        if input_quantizer and hasattr(weight_quantizer, "svdquant_lora_a"):
            return QUANTIZATION_NVFP4_SVDQUANT
        if (block_sizes.get("type") == "dynamic"
                and scale_bits == (4, 3)
                and input_quantizer.num_bits == (4, 3)):
            return QUANTIZATION_W4A8_NVFP4_FP8
        if scale_bits == (4, 3):
            return QUANTIZATION_NVFP4       # ← 标准 NVFP4
        elif scale_bits == (8, 0):
            return QUANTIZATION_MXFP4
```

NVFP4 相关的格式常量：
- `QUANTIZATION_NVFP4`：标准 NVFP4（W4A4）
- `QUANTIZATION_NVFP4_AWQ`：NVFP4 + AWQ smooth scaling
- `QUANTIZATION_NVFP4_SVDQUANT`：NVFP4 + SVD 低秩
- `QUANTIZATION_W4A8_NVFP4_FP8`：W4（NVFP4）+ A8（FP8）混合

### 6.2 导出 global scale（`weights_scaling_factor_2`）

```python
def get_weight_scaling_factor_2(module, weight_name="weight") -> torch.Tensor:
    ...
    return NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(weight_quantizer)
```

`NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer()` 受 fouroversix 影响：

```python
@classmethod
def get_weights_scaling_factor_2_from_quantizer(cls, weight_quantizer):
    # _FP8_MAX_SCALE = 256.0 if _FOUROVERSIX else 448.0
    if cls._is_static_quantizer(weight_quantizer):
        return weight_quantizer.global_amax.float() / (6.0 * _FP8_MAX_SCALE)
    else:
        return weight_quantizer._amax.float() / (6.0 * _FP8_MAX_SCALE)
    # 标准:  amax / (6 × 448) = amax / 2688
    # 4/6:   amax / (6 × 256) = amax / 1536
```

> 导出的 `weights_scaling_factor_2` 在 fouroversix 模式下**数值更大**（约 1.75×）。
> 这与 FP8 per-block scale 的数值更小（max 256 而非 448）完全对应，
> 两者乘积恢复权重原始尺度不变：`wsf × wsf2` 仍然等于 `block_amax / 6.0` 对于最大块。

### 6.3 导出 per-block scale（`weights_scaling_factor`）

```python
def get_weight_scaling_factor(module, weight_name="weight") -> torch.Tensor:
    weight = getattr(module, weight_name)
    weight_quantizer = getattr(module, "weight_quantizer")
    quantization_format = get_quantization_format(module)

    if quantization_format in [QUANTIZATION_NVFP4, QUANTIZATION_NVFP4_AWQ, ...]:
        _ensure_weight_quantizer_calibrated(weight_quantizer, weight, module_name)
        weight_scaling_factor_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(
            weight_quantizer
        )
        # 统一处理静态/动态量化器
        return NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
            weight_quantizer, weight,
            weight_scaling_factor_2.to(weight.device),
        )[0]
        # 返回 FP8 E4M3 格式的 per-block scale，shape: [out_features, in_features // 16]
```

`get_weights_scaling_factor_from_quantizer()` 分支：

```python
@classmethod
def get_weights_scaling_factor_from_quantizer(cls, weight_quantizer, weight,
                                               weights_scaling_factor_2=None, ...):
    if cls._is_static_quantizer(weight_quantizer):
        # 静态路径：使用已存的 per-block amax
        per_block_amax = weight_quantizer._amax.float()
        per_block_scale = per_block_amax / 6.0
        per_block_scale[per_block_scale == 0] = 1.0
        # 量化为 FP8
        per_block_scale = (per_block_scale * 448.0 / (global_amax / 6.0)).to(torch.float8_e4m3fn)
        return per_block_scale, weights_scaling_factor_2
    else:
        # 动态路径：从权重张量现场计算
        return cls.get_weights_scaling_factor(weight, block_size, weights_scaling_factor_2)
```

### 6.4 导出激活 scale（`activation_scaling_factor`）

```python
def get_activation_scaling_factor(module, input_quantizer_name="input_quantizer"):
    input_quantizer = getattr(module, input_quantizer_name, None)

    if get_quantization_format(module) in [QUANTIZATION_NVFP4, QUANTIZATION_NVFP4_AWQ, ...]:
        return NVFP4QTensor.get_activation_scaling_factor(input_quantizer)

@classmethod  # NVFP4QTensor 方法
def get_activation_scaling_factor(cls, quantizer):
    amax = quantizer.export_amax()
    # _FP8_MAX_SCALE = 256 if _FOUROVERSIX else 448
    activation_scaling_factor = amax.float() / (quantizer.maxbound * _FP8_MAX_SCALE)
    # 标准:  amax / (6.0 × 448.0) = amax / 2688
    # 4/6:   amax / (6.0 × 256.0) = amax / 1536
    return activation_scaling_factor
```

> fouroversix 导出的激活 scale 也使用 256 作为分母，与权重 global scale 保持一致，
> 确保推理引擎解量化激活时使用正确的动态范围。

### 6.5 写入 `LinearConfig`

**文件**：`modelopt/torch/export/layer_utils.py`

```python
def build_linear_config(module, linear_type):
    config = LinearConfig(linear_type=linear_type)
    config.weight = module.weight

    config.activation_scaling_factor = get_activation_scaling_factor(module)
    config.weights_scaling_factor   = get_weight_scaling_factor(module)    # per-block FP8 scale
    config.weights_scaling_factor_2 = get_weight_scaling_factor_2(module)  # global scalar scale
    config.prequant_scaling_factor  = get_prequant_scaling_factor(module)  # AWQ 平滑 scale
    config.awq_block_size           = get_weight_block_size(module)        # = 16
    config.quantization             = get_quantization_format(module)      # = "nvfp4"
    return config
```

导出后 `LinearConfig` 中各字段对应含义：

| 字段 | NVFP4 含义 | shape |
|------|-----------|-------|
| `weight` | 原始 BF16/FP16 权重（fake quant 后）或 packed uint8 | `[out, in]` |
| `weights_scaling_factor` | per-block FP8 E4M3 scale | `[out, in//16]` |
| `weights_scaling_factor_2` | global FP32 scalar scale | `[1]` |
| `activation_scaling_factor` | 激活 global scale = amax/(6×448) | `[1]` |
| `quantization` | `"nvfp4"` | - |

---

## 七、Dequantize（解量化）

**文件**：`modelopt/torch/quantization/qtensor/nvfp4_tensor.py`

```python
def dequantize(self, dtype=None, fast=False, **kwarg):
    """解量化 NVFP4 packed tensor"""
    if fast:
        # 快速路径：Triton kernel
        from ..triton.fp4_kernel import fp4_dequantize
        return fp4_dequantize(
            self._quantized_data,     # packed uint8
            kwarg["scale"],           # per-block FP8 scale
            kwarg["double_scale"],    # global FP32 scale
            block_size=kwarg["block_sizes"][-1],
            dtype=dtype,
        )
    else:
        # PyTorch fallback
        q_per_block_scale = kwarg["scale"].to(torch.float32)   # FP8 → FP32
        per_block_scale = q_per_block_scale * kwarg["double_scale"]  # 恢复原始 scale
        deq_data = _unpack_tensor(self._quantized_data)             # uint8 → E2M1 值
        deq_data = deq_data.view(..., -1, block_size) * per_block_scale.unsqueeze(-1)
        return deq_data.reshape(self.metadata["shape"]).to(dtype)
```

Triton dequantize kernel 使用 PTX 指令 `cvt.rn.f16x2.e2m1x2` 直接解码 E2M1：

```python
@triton.jit
def fp4_dequantize_kernel(packed_ptr, scale_ptr, global_scale_ptr, output_ptr, ...):
    # 使用 PTX inline asm 解码 E2M1（每次解码 4 个 uint4 → 4 个 FP16）
    x_f16x2_packed = tl.inline_asm_elementwise(
        asm="""
        {
            cvt.rn.f16x2.e2m1x2 $0, byte0;  // 2个 E2M1 → 1个 FP16x2
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            ...
        }
        """,
        ...
    )
    result = val * scale.to(tl.float32) * global_scale
    tl.store(output_ptr + ..., result)
```

---

## 八、完整调用栈

```
【校准阶段】
mtq.quantize(model, NVFP4_DEFAULT_CFG, forward_loop)
  └─ model_quant.quantize()
       ├─ apply_mode("quantize")  → QuantLinear + TensorQuantizer 初始化
       └─ calibrate(model, "max", forward_loop)
            └─ max_calibrate()                          [model_calib.py]
                 ├─ enable_stats_collection()
                 │    └─ TQ.enable_calib() → _if_calib=True, _if_quant=False
                 ├─ forward_loop(model)
                 │    └─ TQ.forward() → TQ.collect()
                 │         └─ MaxCalibrator.collect()   [calib/max.py]
                 │              └─ reduce_amax(x) → running max → _calib_amax
                 └─ finish_stats_collection()
                      └─ TQ.load_calib_amax()           → _amax buffer（scalar）

【Fake Quant 阶段（评估/QAT）】
TQ.forward()
  └─ _fake_quantize()                                   [tensor_quantizer.py]
       └─ dynamic_block_quant(inputs, 16, amax, (2,1), (4,3))
                                                         [tensor_quant.py]
            └─ _dynamic_block_quantize_impl()
                 ├─ [Hopper+] fp4_fake_quant_block()    [triton/fp4_kernel_hopper.py]
                 │    └─ fp4_fake_quant_kernel (Triton kernel)
                 │         global_scale → per_block_amax → FP8 scale → E2M1 round → 反缩放
                 └─ [fallback] cuda_ext_mx.fused_amax_convert()
                                                         [src/tensor_quant_mx.cu]

【Real Quant（权重压缩）】
TQ._real_quantize()                                     [tensor_quantizer.py]
  └─ NVFP4QTensor.quantize()                            [qtensor/nvfp4_tensor.py]
       ├─ [TRT-LLM] torch.ops.trtllm.fp4_quantize()
       └─ [PyTorch fallback]
            ├─ get_weights_scaling_factor_2() = amax / (6×448)    → FP32 scalar
            ├─ get_weights_scaling_factor()   = block_amax / 6    → FP8 E4M3
            ├─ scaled = weight / (fp8_scale × global_scale)
            ├─ _cast_fp4() → searchsorted + sign_bit              → uint4
            └─ pack: [q[1::2]<<4 | q[0::2]]                       → uint8

【导出阶段】
get_weight_scaling_factor(module)                       [export/quant_utils.py]
  └─ NVFP4QTensor.get_weights_scaling_factor_from_quantizer()
       → FP8 per-block scale [out, in//16]

get_weight_scaling_factor_2(module)
  └─ NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer()
       → FP32 scalar global scale [1]

get_activation_scaling_factor(module)
  └─ NVFP4QTensor.get_activation_scaling_factor()
       → amax / (6.0 × 448.0)  [标准] 或  amax / (6.0 × 256.0)  [fouroversix]

build_linear_config(module)                             [export/layer_utils.py]
  → LinearConfig { weights_scaling_factor, weights_scaling_factor_2,
                   activation_scaling_factor, quantization="nvfp4" }
```

---

## 九、关键数值公式汇总

### 标准 NVFP4

```
【校准】
  global_amax = max(|x|)    across all calibration batches（per-tensor scalar）

【Fake Quant（Triton kernel）】
  global_scale         = global_amax / (6.0 × 448.0)
  per_block_fp8_scale  = clip(block_max / (6.0 × global_scale), 448) → FP8 E4M3
  dequant_scale        = per_block_fp8_scale(fp32) × global_scale
  x_norm               = |x| / dequant_scale
  q                    = E2M1_round(x_norm)    ∈ {0, 0.5, 1, 1.5, 2, 3, 4, 6}
  x_fake               = q × dequant_scale × sign(x)

【Real Quant（权重压缩）】
  weights_scaling_factor_2 = amax / (6.0 × 448.0)          # global scalar（FP32）
  per_block_fp8_scale      = block_amax / 6.0 → FP8 E4M3   # [out, in//16]
  scaled_weight            = weight / (per_block_fp8_scale(fp32) × wsf2)
  q_uint4                  = E2M1_encode(scaled_weight)
  packed_uint8             = q_uint4[1::2] << 4 | q_uint4[0::2]

【导出（给 TRT-LLM）】
  weights_scaling_factor   = per_block_fp8_scale    FP8 E4M3, [out, in//16]
  weights_scaling_factor_2 = global_amax / (6×448)  FP32 scalar
  activation_scaling_factor= act_amax / (6×448)     FP32 scalar
```

### FourOverSix（`MODELOPT_FOUROVERSIX=1`）差异

```
【Fake Quant（Triton kernel fp4_fouroversix_fake_quant_block）】
  global_scale  = global_amax / (6.0 × 256.0)    ← 256 不是 448
  scale_6       = clip(block_max / (6.0 × global_scale), 256) → FP8 E4M3
  scale_4       = clip(scale_6 × 1.5, 256) → FP8 E4M3
  deq_6         = scale_6(fp32) × global_scale
  deq_4         = scale_4(fp32) × global_scale
  recon_6/4     = E2M1_round(|x|/deq) × deq × sign(x)
  MSE_6/4       = sum((recon - x)²) per block
  x_fake        = recon_4 if MSE_4 < MSE_6 else recon_6   ← 逐块选择

【Real Quant（PyTorch fallback，跳过 TRT-LLM）】
  weights_scaling_factor_2 = amax / (6.0 × 256.0)
  scale_6_hp    = block_amax / (6.0 × wsf2) clamp(max=256)  → FP8 E4M3
  scale_4_hp    = scale_6_hp × 1.5 clamp(max=256)           → FP8 E4M3
  选 MSE 更小的 per-block scale 作为最终 weights_scaling_factor

【导出（给 TRT-LLM/vLLM）】
  weights_scaling_factor   = 逐块选优的 FP8 scale，FP8 E4M3, [out, in//16]
  weights_scaling_factor_2 = global_amax / (6×256)  FP32 scalar（≈1.75× 标准值）
  activation_scaling_factor= act_amax / (6×256)     FP32 scalar
```

---

## 十、不同 Config 对比

| Config | 量化层 | 跳过层 | 备注 |
|--------|--------|--------|------|
| `NVFP4_DEFAULT_CFG` | 全部 Linear（W+A） | lm_head / router | 最激进 |
| `NVFP4_AWQ_LITE_CFG` | 全部 Linear（W+A） | lm_head / router | + AWQ smooth |
| `NVFP4_MLP_ONLY_CFG` | MLP W+A，MoE W+A | attn QKV/O，lm_head | 保守 |
| `NVFP4_OMLP_ONLY_CFG` | MLP W+A + o\_proj W+A | attn QKV，lm_head | 中等 |
| `NVFP4_W4A4_WEIGHT_MSE_FP8_SWEEP_CFG` | W static，A dynamic | lm_head | MSE 搜索 weight scale |

所有 Config 共同特征：
- `algorithm: "max"` → MaxCalibrator 收集 per-tensor global amax
- `block_sizes: {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}` → per-block scale 在量化时动态计算，不是校准时确定
- `"default": {"enable": False}` → 未显式命中的层均禁用量化

---

## 十一、FourOverSix 集成到 ModelOpt 的完整改动说明

> 此节记录在 `main` 分支基础上（commit `839fa3d658`）将 4/6 算法集成进 ModelOpt 的所有改动。
> 开关：环境变量 `MODELOPT_FOUROVERSIX=1`，默认 `0`（标准 NVFP4）。

### 11.1 改动文件清单

| 文件 | 改动性质 | 说明 |
|------|---------|------|
| `modelopt/torch/quantization/triton/fp4_kernel_hopper.py` | 新增函数 | 添加 `fp4_fouroversix_fake_quant_block` 及其 Triton kernel |
| `modelopt/torch/quantization/tensor_quant.py` | 修改 | `_dynamic_block_quantize_impl` 中优先走 fouroversix kernel |
| `modelopt/torch/quantization/qtensor/nvfp4_tensor.py` | 修改 | `_FP8_MAX_SCALE`、`get_weights_scaling_factor` 双方案选择 |
| `modelopt/torch/quantization/nn/modules/tensor_quantizer.py` | 修改 | `_real_quantize` 使用 `_FP8_MAX_SCALE`，`try_tensorrt=not _FOUROVERSIX` |
| `modelopt/torch/quantization/utils/core_utils.py` | 修改 | `sync_moe_expert_amax` 支持 `nn.Module` 类型的 experts 容器 |
| `modelopt/torch/quantization/model_quant.py` | 修改 | `print_quant_summary` 确保输出目录存在 |
| `examples/llm_ptq/quant.sh` | 修改 | 增加 fouroversix/standard 双模式循环运行 |
| `tests/gpu/torch/quantization/test_fouroversix_kernel.py` | 新增 | Triton kernel 功能验证测试 |

### 11.2 `tensor_quant.py` 路由逻辑

**文件**：`modelopt/torch/quantization/tensor_quant.py`

```python
import os
ENABLE_FOUROVERSIX = os.environ.get("MODELOPT_FOUROVERSIX", "0") == "1"

def _dynamic_block_quantize_impl(inputs, block_size, amax, num_bits, ...):
    num_bits_tuple = ...    # = (2, 1) for E2M1
    scale_bits_tuple = ...  # = (4, 3) for E4M3

    # ① FourOverSix 优先路径（仅 NVFP4 格式 + 环境变量开启 + Triton 可用）
    if (
        num_bits_tuple == (2, 1)
        and scale_bits_tuple == (4, 3)
        and ENABLE_FOUROVERSIX
        and triton_kernel.IS_AVAILABLE
        and hasattr(triton_kernel, "fp4_fouroversix_fake_quant_block")
        and not DISABLE_TRITON_KERNEL
        and amax is not None
    ):
        return triton_kernel.fp4_fouroversix_fake_quant_block(inputs, amax)

    # ② 标准 NVFP4 Triton 路径
    if (
        num_bits_tuple == (2, 1)
        and scale_bits_tuple == (4, 3)
        and triton_kernel.IS_AVAILABLE
        and hasattr(triton_kernel, "fp4_fake_quant_block")
        and amax is not None
    ):
        return triton_kernel.fp4_fake_quant_block(inputs, amax)

    # ③ CUDA C++ fallback
    return cuda_ext_mx.fused_amax_convert(inputs, block_size, E2M1, E4M3, amax)
```

### 11.3 `core_utils.py` MoE experts 修复

原始 `sync_moe_expert_amax` 假设 `experts` 是直接可迭代列表。
Qwen3MoeExperts（3D 参数张量容器，非列表）会触发 `TypeError: object is not iterable`。

修改后逻辑：

```python
def sync_moe_expert_amax(experts):
    if isinstance(experts, nn.Module):
        # 扫描所有 TensorQuantizer 子模块
        quantizer_items = [
            (name, module)
            for name, module in experts.named_modules()
            if isinstance(module, TensorQuantizer)
        ]

        def _normalize_quantizer_name(name: str) -> str:
            # "0.gate_proj.input_quantizer" → "gate_proj.input_quantizer"
            return ".".join(part for part in name.split(".") if not part.isdigit())

        # 按归一化名称分组，取 max amax
        groups = {}
        for name, tq in quantizer_items:
            group_name = _normalize_quantizer_name(name)
            if tq.amax is not None:
                groups.setdefault(group_name, []).append(tq.amax)

        for group_name, amaxes in groups.items():
            max_amax = torch.stack(amaxes).max(dim=0).values
            for name, tq in quantizer_items:
                if _normalize_quantizer_name(name) == group_name:
                    tq._amax.copy_(max_amax)
        return

    # 原始逻辑（iterable experts 列表）
    for expert in experts:
        ...
```

### 11.4 Triton E2M1 边界条件（`_e2m1_round`）

标准 PyTorch 参考（`torch.searchsorted`）与 Triton kernel 在边界值行为略有差异：

| 边界值 | `torch.searchsorted` (纯 `<`) | Triton `_e2m1_round` (混合 `<=`/`<`) |
|--------|-------------------------------|---------------------------------------|
| 0.25 | → 0（`<0.25` 不满足，searchsorted → 0） | → 0（`<= 0.25`） |
| 0.75 | → 1（`<0.75` 不满足，→ 1） | → 0.5（`< 0.75`） |
| 1.25 | → 2（→ 1.0） | → 1.0（`<= 1.25`） |
| 1.75 | → 3（→ 1.5） | → 1.5（`< 1.75`） |
| 2.5  | → 4（→ 2.0） | → 2.0（`<= 2.5`） |
| 3.5  | → 5（→ 3.0） | → 3.0（`< 3.5`） |
| 5.0  | → 6（→ 4.0） | → 4.0（`<= 5.0`） |

在验证测试中，测试函数 `_fake_quantize_e2m1_triton_compat` 精确复现 Triton 的混合 `<=`/`<` 逻辑，
保证 `block winner agreement = 100%`（两路 MSE 比较结果完全一致）。

### 11.5 使用方式

```bash
# 启用 fouroversix 运行 PTQ
export MODELOPT_FOUROVERSIX=1
python examples/llm_ptq/hf_ptq.py \
    --model_name /path/to/model \
    --qformat nvfp4 \
    --calib_size 512 \
    --export_path /path/to/output

# 或使用 quant.sh（内部循环 fouroversix=1 和 fouroversix=0）
bash examples/llm_ptq/quant.sh
```

`quant.sh` 自动以 `four_over_six-{qformat}` 和 `standard-{qformat}` 命名输出目录。

### 11.6 兼容性说明

- **校准阶段**：环境变量**不影响**校准过程，两种模式共用相同的 `amax` 收集逻辑
- **Fake Quant**：fouroversix 使用专用 Triton kernel，标准使用 `fp4_fake_quant_block`
- **Real Quant**：fouroversix 强制 PyTorch fallback（`try_tensorrt=False`），在 `get_weights_scaling_factor` 内做双方案选择
- **导出**：`weights_scaling_factor_2` 使用 `_FP8_MAX_SCALE`（256 或 448），格式标识仍为 `"nvfp4"`，对下游引擎透明
- **MoE 模型**：`sync_moe_expert_amax` 已修复对 `Qwen3_5MoeExperts` 的支持
