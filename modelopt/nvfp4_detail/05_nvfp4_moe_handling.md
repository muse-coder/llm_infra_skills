# NVFP4 MoE 模型处理

## 概述

Mixture of Experts (MoE) 模型在 NVFP4 量化中需要特殊处理，因为：
1. 每个专家可能在校准期间收到不同数量的 token
2. 需要同步所有专家的量化参数以保持一致性
3. 不同的 MoE 架构有不同的实现方式

本文档详细介绍 Model Optimizer 中 MoE 模型的 NVFP4 量化处理机制。

---

## 1. _QuantSparseMoe 类

### 1.1 文件位置

**文件**：`modelopt/torch/quantization/plugins/huggingface.py`（第 444-575 行）

### 1.2 功能说明

`_QuantSparseMoe` 是通用的量化包装器，支持所有 HuggingFace 稀疏 MoE 模块：

- Mixtral
- Qwen3Moe / Qwen2Moe
- Qwen3Next
- Llama4
- MiniMax
- NemotronH
- 等等

### 1.3 核心实现

```python
class _QuantSparseMoe(QuantModule):
    """Quantization wrapper for HuggingFace sparse MoE blocks.

    Supports ``layer_sync_moe_local_experts_amax`` to sync input quantizer amax across experts.

    Optionally supports two config-driven features (disabled by default):
    - ``_moe_calib_experts_ratio``: force-forward tokens to more experts during calibration.
    - ``_moe_count_expert_calib_tokens``: count tokens routed to each expert during calibration.

    When both are disabled, forward is a direct pass-through with zero overhead.
    """

    def _setup(self):
        """Initialize MoE-specific attributes."""
        self._moe_calib_experts_ratio = None
        self._moe_count_expert_calib_tokens = False
        self._token_counting_initialized = False

    def layer_sync_moe_local_experts_amax(self):
        """Sync input_quantizer amax across experts so all share the same amax per quantizer.
        
        对于融合专家模块（如 Qwen3MoeExperts），由于所有专家权重存储在单个 3D 张量中，
        无需同步 per-expert quantizers。
        """
        if not hasattr(self.experts, "__iter__"):
            # Fused expert module (e.g. Qwen3MoeExperts in transformers>=5.0)
            # stores all expert weights in a single 3D tensor — no per-expert
            # quantizers to sync.
            return
        sync_moe_expert_amax(self.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional calibration enhancements."""
        is_calib = any(
            isinstance(m, TensorQuantizer) and m._if_calib
            for m in self.modules()
        )
        
        if is_calib and self._moe_calib_experts_ratio:
            # 校准时强制激活更多专家
            self._count_expert_tokens = True
            assert 0 < self._moe_calib_experts_ratio <= 1
            
            # 临时增加 top_k 以激活更多专家
            original_top_k = self.gate.top_k
            self.gate.top_k = max(
                original_top_k,
                round(self.gate.num_experts * self._moe_calib_experts_ratio)
            )
            result = super().forward(hidden_states)
            self.gate.top_k = original_top_k
            return result
        
        return super().forward(hidden_states)
```

---

## 2. 自动检测与注册机制

### 2.1 文件位置

**文件**：`modelopt/torch/quantization/plugins/huggingface.py`（第 1303-1376 行）

### 2.2 检测逻辑

`register_sparse_moe_on_the_fly()` 基于结构模式而非类名检测 MoE 模块，实现向前兼容：

```python
def _has_num_experts(obj):
    """Check if object has num_experts attribute."""
    # n_routed_experts: NemotronH-style MoE
    return hasattr(obj, "num_experts") or hasattr(obj, "n_routed_experts")

def _is_sparse_moe_block(module):
    """Check if a module is structurally a sparse MoE block compatible with _QuantSparseMoe.

    All HuggingFace MoE blocks (Mixtral, Qwen3Moe, Qwen2Moe, Qwen3Next, Llama4, MiniMax,
    NemotronH, etc.) share a common structural pattern: a ``gate`` (TopKRouter) sub-module with
    routing attributes (``top_k`` and ``num_experts`` or ``n_routed_experts``), and an ``experts``
    sub-module.

    This function detects that pattern instead of relying on class names, making it forward-compatible
    with new MoE architectures.
    """
    if not hasattr(module, "experts"):
        return False

    # Primary: gate sub-module has topk/top_k + num_experts (standard TopKRouter pattern)
    if hasattr(module, "gate"):
        gate = module.gate
        if hasattr(gate, "top_k") and _has_num_experts(gate):
            return True

    # Fallback: top_k + num_experts on the block itself (older transformers, e.g. v4.x Qwen3Next)
    if hasattr(module, "top_k"):
        if not _has_num_experts(module) and hasattr(module.experts, "__len__"):
            module.num_experts = len(module.experts)
        return _has_num_experts(module)

    return False
```

### 2.3 注册机制

```python
def register_sparse_moe_on_the_fly(model):
    """Auto-detect and register MOE modules as _QuantSparseMoe.

    Walks the model tree, identifies MoE blocks by their structural attributes
    (``gate`` + ``experts``), and registers unregistered ones with ``_QuantSparseMoe``.
    """
    visited_types = set()
    for name, module in model.named_modules():
        mod_type = type(module)

        # Avoid duplicate registration: skip if we already processed this type
        # in this walk, or if it was previously registered in the QuantModuleRegistry.
        if mod_type in visited_types or QuantModuleRegistry.get(mod_type) is not None:
            continue

        visited_types.add(mod_type)

        if _is_sparse_moe_block(module):
            print(
                f"\033[1mDetected MOE module '{name}' of type {mod_type.__name__}, "
                f"registering with _QuantSparseMoe.\033[0m"
            )
            QuantModuleRegistry.register({mod_type: f"hf.{mod_type.__name__}"})(_QuantSparseMoe)
```

### 2.4 支持的检测模式

| 模式 | 说明 | 示例 |
|------|------|------|
| **主要模式** | `gate` 子模块有 `top_k` + `num_experts` | Mixtral, Qwen3Moe |
| **回退模式** | 模块本身有 `top_k` + `num_experts` | 旧版 transformers (v4.x) |

---

## 3. sync_moe_expert_amax 同步机制

### 3.1 文件位置

**文件**：`modelopt/torch/quantization/utils/core_utils.py`（第 519-565 行）

### 3.2 功能说明

`sync_moe_expert_amax()` 有两个核心功能：

1. **同步 input_quantizer amax**：取所有专家每个 `input_quantizer` amax 的元素级最大值，写回所有专家
2. **修复缺失的 weight amax**：对于校准期间未收到 token 的专家，运行 weight-only `max_calibrate`

### 3.3 实现代码

```python
def sync_moe_expert_amax(experts):
    """Sync input_quantizer amax across MoE experts and fix missing weight amax.

    1. Takes the element-wise max of each ``input_quantizer`` amax across all experts
       and writes it back, so every expert shares the same input amax.
    2. For any ``weight_quantizer`` that is enabled but has ``amax is None`` (expert
       received no tokens during calibration), runs a weight-only ``max_calibrate``
       to populate the missing amax.
    """
    from ..nn import TensorQuantizer

    # 步骤 1: 收集所有专家的 input_quantizer amax
    amax_dict: dict[str, torch.Tensor] = {}
    for expert in experts:
        for name, module in expert.named_modules():
            if (
                isinstance(module, TensorQuantizer)
                and module.amax is not None
                and "input_quantizer" in name
            ):
                stored_amax = amax_dict.get(name)
                amax_tensor = module.amax.detach().clone()
                amax_dict[name] = (
                    amax_tensor if stored_amax is None else torch.maximum(stored_amax, amax_tensor)
                )

    # 步骤 2: 将同步后的 amax 写回所有专家
    for expert in experts:
        for name, module in expert.named_modules():
            if isinstance(module, TensorQuantizer) and name in amax_dict:
                module.amax = amax_dict[name].detach().clone()

    # 步骤 3: 修复未激活专家的 weight amax
    from ..model_calib import max_calibrate

    for expert in experts:
        for name, module in expert.named_modules():
            if name.endswith("weight_quantizer") and module.is_enabled and module.amax is None:
                weight = expert.state_dict().get(name.replace("weight_quantizer", "weight"))
                if weight is not None:
                    max_calibrate(module, lambda m, w=weight: m(w), distributed_sync=False)
```

### 3.4 调用位置

- `modelopt/torch/quantization/plugins/huggingface.py` 第 570 行：`sync_moe_expert_amax(self.experts)`
- `modelopt/torch/quantization/plugins/megatron.py` 第 594 行：`sync_moe_expert_amax(self.local_experts)`

---

## 4. _moe_calib_experts_ratio 参数

### 4.1 作用

在校准期间强制将 token 路由到更多专家，以提高校准覆盖率。

### 4.2 问题背景

MoE 模型的路由机制通常只激活 top-k 个专家（如 top-2）。这意味着：
- 某些专家可能在校准期间收到很少或没有 token
- 这些专家的 amax 可能不准确或缺失
- 导致量化后精度下降

### 4.3 解决方案

通过 `_moe_calib_experts_ratio` 临时增加 `top_k`：

```python
# 在 _QuantSparseMoe.forward() 中
if is_calib and self._moe_calib_experts_ratio:
    assert 0 < self._moe_calib_experts_ratio <= 1
    
    # 临时增加 top_k
    original_top_k = self.gate.top_k
    self.gate.top_k = max(
        original_top_k,
        round(self.gate.num_experts * self._moe_calib_experts_ratio)
    )
    
    # 执行前向传播
    result = super().forward(hidden_states)
    
    # 恢复原始 top_k
    self.gate.top_k = original_top_k
    return result
```

### 4.4 使用方式

**命令行**：
```bash
python hf_ptq.py \
    --pyt_ckpt_path mistralai/Mixtral-8x7B-v0.1 \
    --qformat nvfp4_mlp_only \
    --moe_calib_experts_ratio 0.5  # 校准时激活 50% 的专家
```

**Python API**：
```python
from examples.llm_ptq.example_utils import build_quant_cfg

quant_cfg = build_quant_cfg(
    qformat="nvfp4_mlp_only",
    quant_cfg=mtq.NVFP4_MLP_ONLY_CFG,
    moe_calib_experts_ratio=0.5,
)
```

### 4.5 配置示例

```python
# examples/llm_ptq/example_utils.py
if moe_calib_experts_ratio:
    assert 0 < moe_calib_experts_ratio <= 1
    if isinstance(quant_cfg["algorithm"], str):
        quant_cfg["algorithm"] = {
            "method": quant_cfg["algorithm"],
            "moe_calib_experts_ratio": moe_calib_experts_ratio,
        }
    elif isinstance(quant_cfg["algorithm"], dict):
        quant_cfg["algorithm"]["moe_calib_experts_ratio"] = moe_calib_experts_ratio
```

---

## 5. 不同 MoE 架构的处理差异

### 5.1 架构对比

| 架构 | 量化类 | 特点 |
|------|--------|------|
| Mixtral | `_QuantSparseMoe` | 标准实现，per-expert Linear 层 |
| Llama4 | `_QuantLlama4TextExperts` | 转置量化，3D 权重张量 |
| Qwen3.5 | `_QuantQwen35MoeExperts` | Per-expert 容器模块 |
| Qwen3 VL | `_QuantQwen3VLMoeTextExperts` | 融合权重拆分 |
| DBRX | `_QuantDbrxExperts` | 同时处理 DbrxExperts 和 DbrxExpertGLU |
| GPT-OSS | `_QuantGptOssExperts` | 函数式量化 |

### 5.2 Mixtral（标准实现）

**特点**：
- 每个专家是独立的 `nn.Linear` 层
- 使用标准的 `_QuantSparseMoe` 包装器
- 通过 `*block_sparse_moe*` 模式匹配

**配置匹配**：
```python
"*block_sparse_moe*weight_quantizer": _nvfp4_quantizer,
"*block_sparse_moe*input_quantizer": _nvfp4_quantizer,
```

### 5.3 Llama4（转置量化）

**文件**：`modelopt/torch/quantization/plugins/huggingface.py`（第 573-592 行）

**特点**：
- 专家权重形状为 `(num_experts, in_dim, out_dim)`
- 使用 `torch.bmm` 进行批量矩阵乘法
- 需要转置量化 (`_transposed_quantize`)

```python
class _QuantLlama4TextExperts(QuantModule):
    def _setup(self):
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizer = TensorQuantizer()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        
        # 使用转置量化
        gate_up = torch.bmm(
            self.gate_up_proj_input_quantizer(hidden_states),
            _transposed_quantize(self.gate_up_proj, self.gate_up_proj_weight_quantizer),
        )
        gate, up = gate_up.chunk(2, dim=-1)
        
        next_states = torch.bmm(
            self.down_proj_input_quantizer(up * self.act_fn(gate)),
            _transposed_quantize(self.down_proj, self.down_proj_weight_quantizer),
        )
        next_states = next_states.view(-1, self.hidden_size)
        return next_states
```

### 5.4 Qwen3.5（Per-expert 容器）

**文件**：`modelopt/torch/quantization/plugins/huggingface.py`（第 786-870 行）

**特点**：
- 创建 per-expert `_Qwen35MoeExpertModule` 容器
- 产生命名模式：`experts.{id}.gate_proj.weight`
- 实现 `__len__`、`__iter__`、`__getitem__` 使模块可迭代

```python
class _Qwen35MoeExpertModule(nn.Module):
    """Container for a single Qwen3.5 MoE expert's linear layers."""
    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.down_proj = nn.Linear(expert_dim, hidden_dim, bias=False)

class _QuantQwen35MoeExperts(QuantModule):
    def _setup(self):
        """Modify the Qwen3_5MoeExperts by using per-expert nn.Module containers."""
        # 将融合的 3D 权重拆分为 per-expert Linear 层
        ...
```

### 5.5 Qwen3 VL（融合权重拆分）

**文件**：`modelopt/torch/quantization/plugins/huggingface.py`（第 692-762 行）

**特点**：
- 将融合的 3D 权重张量拆分为 per-expert `nn.Linear` 层
- 处理 `intermediate_size` 和 `intermediate_dim` 的版本差异

```python
class _QuantQwen3VLMoeTextExperts(QuantModule):
    def _setup(self):
        """Modify the Qwen3VLMoeTextExperts by using nn.Linear layers."""
        from accelerate import init_empty_weights

        # 将融合的 gate_up_proj 和 down_proj 拆分为 per-expert Linear 层
        ...
```

### 5.6 GPT-OSS（函数式量化）

**文件**：`modelopt/torch/quantization/plugins/huggingface.py`（第 1185-1280 行）

**特点**：
- 使用 `_QuantFunctionalMixin`
- 通过动态属性进行权重量化
- 拦截 `torch.Tensor.__matmul__` / `torch.bmm` 进行激活量化
- 使用上下文管理器 `quantize_weight()` 控制权重量化范围

```python
class _QuantGptOssExperts(_QuantFunctionalMixin):
    @staticmethod
    def _get_quantized_weight(quantizer, module, weight):
        """Get quantized weight with caching."""
        # MoE weight is accessed for each expert in one forward pass, so cache it
        if module._enable_weight_quantization:
            if hasattr(quantizer, "_cached_quant_val"):
                return getattr(quantizer, "_cached_quant_val")
            quantizer._cached_quant_val = _transposed_quantize(weight, quantizer)
            return quantizer._cached_quant_val
        return weight
```

---

## 6. MoE 量化最佳实践

### 6.1 校准建议

1. **增加校准数据量**：MoE 模型需要更多校准数据以覆盖所有专家
2. **使用 moe_calib_experts_ratio**：设置为 0.5-1.0 以确保所有专家都被校准
3. **检查专家覆盖率**：使用 `_moe_count_expert_calib_tokens` 统计每个专家收到的 token 数

### 6.2 命令行示例

```bash
# Mixtral 8x7B
python hf_ptq.py \
    --pyt_ckpt_path mistralai/Mixtral-8x7B-v0.1 \
    --qformat nvfp4_mlp_only \
    --calib_size 1024 \
    --moe_calib_experts_ratio 0.5

# Qwen3 MoE
python hf_ptq.py \
    --pyt_ckpt_path Qwen/Qwen3-MoE-15B \
    --qformat nvfp4_mlp_only \
    --calib_size 1024 \
    --trust_remote_code
```

### 6.3 配置匹配

对于 MoE 模型，`nvfp4_mlp_only` 配置通过以下模式匹配专家层：

```python
"*block_sparse_moe*weight_quantizer": _nvfp4_quantizer,
"*block_sparse_moe*input_quantizer": _nvfp4_quantizer,
```

**注意**：此模式主要匹配 Mixtral 风格的 MoE。对于其他 MoE 架构，ModelOpt 通过 `register_sparse_moe_on_the_fly()` 动态注册量化包装器。

---

## 7. 常见问题

### 7.1 某些专家 amax 为 None

**原因**：校准期间该专家未收到任何 token

**解决方案**：
1. 增加 `--moe_calib_experts_ratio`
2. 增加 `--calib_size`
3. `sync_moe_expert_amax()` 会自动修复（运行 weight-only calibrate）

### 7.2 MoE 模型未被正确量化

**原因**：MoE 模块未被自动检测

**解决方案**：
1. 确保模型有 `gate` + `experts` 结构
2. 检查是否需要 `--trust_remote_code`
3. 手动注册量化包装器

### 7.3 量化后精度下降严重

**原因**：专家间 amax 差异过大

**解决方案**：
1. 使用 `sync_moe_expert_amax()` 同步 amax
2. 考虑使用 `nvfp4_mlp_only` 而非全量 `nvfp4`
3. 增加校准数据多样性

---

## 相关文档

- [NVFP4 量化原理](./01_nvfp4_quantization_principle.md)
- [NVFP4 核心代码位置](./02_nvfp4_code_structure.md)
- [NVFP4 校准流程详解](./03_nvfp4_calibration.md)
- [NVFP4 权重导出机制](./04_nvfp4_weight_export.md)
- [NVFP4 选择性量化使用指南](../NVFP4_Selective_Quantization_Guide.md)
