# NVFP4 校准流程详解

## 概述

校准（Calibration）是 NVFP4 量化的关键步骤，通过在代表性数据上运行模型来收集激活值的统计信息（主要是 amax），从而确定最优的量化缩放因子。本文档详细介绍 `nvfp4_mlp_only` 和 `nvfp4_omlp_only` 配置中校准集的使用方式。

---

## 1. forward_loop 的作用

### 1.1 定义

`forward_loop` 是用户提供的校准数据迭代函数，负责将校准数据前向传播通过模型。

```python
def forward_loop(model: nn.Module) -> None:
    """用户实现的校准数据迭代函数。
    
    Args:
        model: 要校准的模型
    
    该函数应该：
    1. 迭代校准数据集
    2. 将每个 batch 前向传播通过模型
    3. 不需要计算梯度
    """
    for batch in calibration_dataloader:
        model(**batch)
```

### 1.2 典型实现

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载校准数据
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:512]")

def get_calib_dataloader(batch_size=1, max_length=512):
    """创建校准数据加载器。"""
    def tokenize(examples):
        return tokenizer(
            examples["article"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized = dataset.map(tokenize, batched=True)
    return torch.utils.data.DataLoader(tokenized, batch_size=batch_size)

calib_dataloader = get_calib_dataloader()

def forward_loop(model):
    """校准数据迭代函数。"""
    for batch in calib_dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        model(input_ids=input_ids, attention_mask=attention_mask)
```

### 1.3 在 hf_ptq.py 中的使用

`hf_ptq.py` 自动创建 `forward_loop`：

```python
# examples/llm_ptq/hf_ptq.py
def get_forward_loop(model, dataloader, device):
    """Create forward loop for calibration."""
    def forward_loop(model):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
    return forward_loop
```

---

## 2. 校准流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        mtq.quantize(model, config, forward_loop)         │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 1: enable_stats_collection(model)                                   │
│ - 遍历所有 TensorQuantizer                                               │
│ - 设置 _if_calib = True                                                  │
│ - 初始化 MaxCalibrator                                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 2: forward_loop(model)                                              │
│ - 用户提供的校准数据迭代                                                  │
│ - 每个 batch 前向传播                                                    │
│ - TensorQuantizer.forward() 调用 calibrator.collect()                    │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 3: MaxCalibrator.collect(x)                                         │
│ - 计算当前 batch 的 local_amax = max(|x|)                                │
│ - 更新全局 amax: calib_amax = max(calib_amax, local_amax)                │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 4: finish_stats_collection(model)                                   │
│ - 调用 calibrator.compute_amax() 获取最终 amax                           │
│ - 调用 load_calib_amax() 将 amax 加载到量化器                            │
│ - 设置 _if_calib = False                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 步骤 5: 验证量化器属性                                                    │
│ - 检查 _amax 是否已设置                                                  │
│ - 检查 _pre_quant_scale 是否有效                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              量化后的模型
```

---

## 3. MaxCalibrator 实现

### 3.1 文件位置

**文件**：`modelopt/torch/quantization/calib/max.py`

### 3.2 核心实现

```python
class MaxCalibrator(_Calibrator):
    """Max calibrator, tracks the maximum value globally.
    
    对于 NVFP4 量化，使用 "max" 算法是最简单和最快的校准方法。
    它跟踪所有校准数据中每个量化器位置的最大绝对值。
    """
    
    def __init__(self, num_bits=8, axis=None, unsigned=False, track_amax=False):
        super().__init__(num_bits, axis, unsigned)
        self._track_amax = track_amax
        if self._track_amax:
            self._amaxs = []  # 可选：跟踪每个 batch 的 amax
        self._calib_amax = None
    
    @torch.no_grad()
    def collect(self, x):
        """Tracks the absolute max of all tensors.
        
        Args:
            x: 输入张量（激活值或权重）
        
        该方法在每次前向传播时被调用，收集 amax 统计信息。
        """
        # 确定要 reduce 的轴
        reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)
        
        # 计算当前 batch 的 amax
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
        
        # 更新全局 amax
        if self._calib_amax is None:
            self._calib_amax = local_amax
        else:
            if local_amax.shape != self._calib_amax.shape:
                raise RuntimeError("amax shape changed!")
            self._calib_amax = torch.max(self._calib_amax, local_amax)
        
        # 可选：记录每个 batch 的 amax
        if self._track_amax:
            self._amaxs.append(local_amax.cpu().numpy())
    
    def compute_amax(self):
        """Return the absolute max of all tensors collected.
        
        Returns:
            torch.Tensor: 校准期间收集的全局最大绝对值
        """
        return self._calib_amax
```

### 3.3 amax 的含义

- **amax**（Absolute Maximum）：张量中所有元素的最大绝对值
- **用途**：用于计算量化缩放因子 `scale = amax / max_representable_value`
- **对于 NVFP4**：`scale = amax / 6.0`（6.0 是 FP4 E2M1 的最大可表示值）

---

## 4. input_quantizer vs weight_quantizer

### 4.1 区别

| 特性 | input_quantizer | weight_quantizer |
|------|-----------------|------------------|
| **量化对象** | 激活值（activations） | 权重参数（weights） |
| **数据来源** | 前向传播时的中间结果 | 模型的静态参数 |
| **是否需要 forward_loop** | ✅ 必须 | ❌ 可选（可直接从权重计算） |
| **amax 变化** | 随输入数据变化 | 固定（权重不变） |
| **校准时机** | 每个 batch 都收集 | 可以一次性计算 |

### 4.2 在 NVFP4_MLP_ONLY_CFG 中的配置

```python
_nvfp4_mlp_only_quant_cfg = {
    # 权重量化器
    "*mlp*weight_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*weight_quantizer": _nvfp4_quantizer,
    
    # 激活量化器
    "*mlp*input_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*input_quantizer": _nvfp4_quantizer,
    
    **_default_disabled_quantizer_cfg,
}
```

### 4.3 校准行为

**input_quantizer 校准**：
```python
# 在 TensorQuantizer.forward() 中
def forward(self, inputs):
    if self._if_calib:
        # 收集激活值的 amax
        self._calibrator.collect(inputs)
    ...
```

**weight_quantizer 校准**：
```python
# 如果没有 forward_loop，可以直接从权重计算
def weight_only_quantize(model):
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module, "weight_quantizer"):
            # 直接从权重计算 amax
            module.weight_quantizer._calibrator.collect(module.weight)
```

---

## 5. 动态缩放 vs 静态缩放

### 5.1 动态缩放（Dynamic Scaling）

**配置**：
```python
"block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
```

**特点**：
- 缩放因子在**推理时**动态计算
- 每个块的 scale 根据当前输入实时计算
- 不需要在校准阶段固定 scale
- 更灵活，但推理时计算开销稍大

**适用场景**：
- 激活量化（激活值随输入变化）
- 需要高精度的场景

**实现**：
```python
def dynamic_block_quant(inputs, block_size, amax, ...):
    # 动态计算每个块的 scale
    per_block_amax = reduce_amax(inputs.view(..., -1, block_size), axis=-1)
    scale = per_block_amax / 6.0
    
    # 应用量化
    scaled = inputs / scale.unsqueeze(-1)
    quantized = cast_to_fp4(scaled)
    
    # 反量化（fake quantization）
    return quantized * scale.unsqueeze(-1)
```

### 5.2 静态缩放（Static Scaling）

**配置**：
```python
"block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)}
```

**特点**：
- 缩放因子在**校准阶段**预计算并固定
- 使用两级缩放：per-block scale + global scale
- 推理时直接使用预计算的 scale，效率更高
- 需要校准数据来确定 scale

**适用场景**：
- 权重量化（权重是静态的）
- 追求推理速度的场景

**实现**：
```python
def static_blockwise_fp4_fake_quant(inputs, amax, global_amax, ...):
    # 使用预计算的 amax
    scale = amax / 6.0
    
    # 如果需要量化 block scales
    if quantize_block_scales:
        scale_fp8_quant_amax = global_amax / 6.0
        scale = scaled_e4m3_impl(scale, scale_fp8_quant_amax)
    
    # 应用量化
    scaled = inputs / scale.unsqueeze(-1)
    quantized = cast_to_fp4(scaled)
    
    return quantized * scale.unsqueeze(-1)
```

### 5.3 对比总结

| 特性 | 动态缩放 | 静态缩放 |
|------|---------|---------|
| Scale 计算时机 | 推理时动态计算 | 校准时预计算并固定 |
| 灵活性 | 高，适应不同输入 | 中，基于校准集 |
| 推理效率 | 稍低（需要计算） | 高（直接查表） |
| 内存占用 | 低（不存储 scale） | 高（存储 per-block 和 global scale） |
| 适用场景 | 激活量化、变化大的数据 | 权重量化、追求推理速度 |

---

## 6. 校准后固定的值

### 6.1 固定到模型中的值

校准完成后，以下值被固定并成为模型权重的一部分：

| 值 | 说明 | 存储位置 | 用途 |
|----|------|---------|------|
| `_amax` | Per-block 最大绝对值 | TensorQuantizer 的 buffer | 计算 per-block scale |
| `_global_amax` | 全局最大绝对值 | TensorQuantizer 的 buffer | 计算 global scale |
| `_pre_quant_scale` | 预量化缩放因子（可选） | TensorQuantizer 的 buffer | SmoothQuant 等算法使用 |

### 6.2 导出时的转换

导出 checkpoint 时，这些值被转换为：

| 校准值 | 导出后的张量名 | 说明 |
|--------|---------------|------|
| `_amax` | `weight_scale` | Per-block 缩放因子 (FP8) |
| `_global_amax` | `weight_scale_2` | Global 缩放因子 |
| input `_amax` | `input_scale` | 输入激活缩放因子 |

### 6.3 代码位置

**加载校准结果**（`modelopt/torch/quantization/nn/modules/tensor_quantizer.py` 第 583-607 行）：

```python
def load_calib_amax(self):
    """Load calibrated amax into the quantizer."""
    if self._calibrator is not None:
        amax = self._calibrator.compute_amax()
        if amax is not None:
            self._amax = amax
            # 对于 NVFP4 静态量化，还需要计算 global_amax
            if self._is_nvfp4_static:
                self._global_amax = reduce_amax(amax, axis=None)
```

**保存到 modelopt_state**：

```python
def get_modelopt_state(self, properties_only: bool = False) -> dict[str, Any]:
    """Get the state for saving."""
    modelopt_state = {}
    for k in self._get_properties_for_modelopt_state():
        modelopt_state[k] = getattr(self, k)
    # 包括 _amax, _global_amax, _pre_quant_scale 等
    return modelopt_state
```

---

## 7. 校准参数

### 7.1 hf_ptq.py 校准相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--calib_size` | 512 | 校准样本数量 |
| `--calib_seq` | 512 | 校准序列最大长度 |
| `--batch_size` | 0（自动） | 校准批大小，0 表示自动检测 |
| `--dataset` | None | 校准数据集名称（如 "cnn_dailymail"） |

### 7.2 校准数据量建议

| 模型大小 | 建议 calib_size | 建议 calib_seq |
|---------|----------------|----------------|
| < 7B | 256-512 | 512 |
| 7B-13B | 512 | 512 |
| 13B-70B | 512-1024 | 512-1024 |
| > 70B | 1024+ | 1024 |

### 7.3 校准数据集选择

- **通用 LLM**：`cnn_dailymail`、`wikitext`、`c4`
- **代码模型**：`codeparrot/github-code`
- **多语言模型**：`mc4`
- **自定义数据**：使用与目标任务相似的数据

---

## 8. 校准最佳实践

### 8.1 数据选择

1. **代表性**：校准数据应该代表实际推理时的输入分布
2. **多样性**：包含不同长度、不同主题的样本
3. **数量**：通常 512-1024 个样本足够

### 8.2 内存优化

```bash
# 使用低内存模式
python hf_ptq.py \
    --pyt_ckpt_path meta-llama/Llama-3.1-8B \
    --qformat nvfp4_mlp_only \
    --low_memory_mode \
    --batch_size 1
```

### 8.3 分布式校准

```bash
# 多 GPU 校准
accelerate launch --num_processes 4 \
    hf_ptq.py \
    --pyt_ckpt_path meta-llama/Llama-3.1-70B \
    --qformat nvfp4_mlp_only \
    --calib_size 512
```

---

## 相关文档

- [NVFP4 量化原理](./01_nvfp4_quantization_principle.md)
- [NVFP4 核心代码位置](./02_nvfp4_code_structure.md)
- [NVFP4 权重导出机制](./04_nvfp4_weight_export.md)
- [NVFP4 MoE 模型处理](./05_nvfp4_moe_handling.md)
- [NVFP4 选择性量化使用指南](../NVFP4_Selective_Quantization_Guide.md)
