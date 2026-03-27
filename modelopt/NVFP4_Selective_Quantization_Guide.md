# NVFP4 选择性量化使用指南 (nvfp4_mlp_only / nvfp4_omlp_only)

## 概述

NVIDIA Model Optimizer (ModelOpt) 提供了两种选择性 NVFP4 量化配置，专门用于在保持较高模型精度的同时实现显著的模型压缩。这两种配置通过仅量化 MLP 层（以及可选的 attention 输出投影层）来避免对精度敏感的 attention QKV 投影层进行量化。

### 适用范围

- **目标硬件**: NVIDIA Blackwell GPU（需要 TensorRT-LLM v0.17+）
- **适用模型**: 所有 HuggingFace Transformers 支持的 LLM 模型（包括 MoE 模型）
- **部署框架**: TensorRT-LLM、vLLM、SGLang

---

## 两种配置的区别

| 配置 | 量化范围 | 适用场景 |
|------|---------|---------|
| `nvfp4_mlp_only` | MLP 层 + MoE 专家层 | 追求最高精度 |
| `nvfp4_omlp_only` | MLP 层 + MoE 专家层 + Attention o_proj | 精度与压缩的平衡 |

### 量化层对比

| 模块 | nvfp4_mlp_only | nvfp4_omlp_only |
|------|---------------|-----------------|
| MLP (gate_proj / up_proj / down_proj) | ✅ 量化 | ✅ 量化 |
| MoE 专家层 | ✅ 量化 | ✅ 量化 |
| Attention o_proj | ❌ 不量化 | ✅ 量化 |
| Attention q_proj / k_proj / v_proj | ❌ 不量化 | ❌ 不量化 |
| lm_head | ❌ 不量化 | ❌ 不量化 |
| MoE 路由器 | ❌ 不量化 | ❌ 不量化 |

---

## 使用方式

### 方式一：hf_ptq.py 命令行

```bash
# 基本用法 - 仅量化 MLP
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path meta-llama/Llama-3.1-8B \
    --qformat nvfp4_mlp_only \
    --export_path ./quantized_model \
    --kv_cache_qformat fp8_cast

# 基本用法 - 量化 MLP + o_proj
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path meta-llama/Llama-3.1-8B \
    --qformat nvfp4_omlp_only \
    --export_path ./quantized_model \
    --kv_cache_qformat fp8_cast

# 禁用 KV cache 量化以获得最高精度
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path meta-llama/Llama-3.1-8B \
    --qformat nvfp4_mlp_only \
    --kv_cache_qformat none \
    --export_path ./quantized_model

# 自定义校准参数
python examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path meta-llama/Llama-3.1-8B \
    --qformat nvfp4_mlp_only \
    --calib_size 512 \
    --calib_seq 512 \
    --batch_size 0 \
    --export_path ./quantized_model
```

### 方式二：Shell 脚本

```bash
export HF_PATH=meta-llama/Llama-3.1-8B

# 单 GPU
scripts/huggingface_example.sh --model $HF_PATH --quant nvfp4_mlp_only --tp 1

# 多 GPU 张量并行
scripts/huggingface_example.sh --model $HF_PATH --quant nvfp4_omlp_only --tp 4
```

### 方式三：Python API

```python
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16)
model = model.to("cuda")

# 2. 准备校准数据
calib_dataloader = ...  # 包含 tokenize 后样本的 DataLoader

def forward_loop(model):
    for batch in calib_dataloader:
        model(**batch)

# 3. 量化
model = mtq.quantize(model, mtq.NVFP4_MLP_ONLY_CFG, forward_loop)

# 4. 可选：添加 KV cache 量化
kv_cache_cfg = mtq.FP8_KV_CFG
quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(
    mtq.NVFP4_MLP_ONLY_CFG, kv_cache_cfg["quant_cfg"]
)

# 5. 导出
with torch.inference_mode():
    export_hf_checkpoint(model, export_dir="./quantized_model")
```

### 方式四：AutoQuantize 混合精度搜索

```bash
# 在 nvfp4_mlp_only 和 fp8 之间自动搜索最优混合精度
scripts/huggingface_example.sh \
    --model $HF_PATH \
    --quant nvfp4_mlp_only,fp8 \
    --auto_quantize_bits 4.75 \
    --calib_batch_size 4
```

### 方式五：多节点 FSDP2

```bash
accelerate launch --config_file fsdp2.yaml \
    --num_machines=2 \
    --machine_rank=0 \
    multinode_ptq.py \
    --pyt_ckpt_path <模型路径> \
    --qformat nvfp4_mlp_only \
    --kv_cache_qformat fp8 \
    --batch_size 4 \
    --calib_size 512 \
    --export_path <输出路径>
```

---

## KV Cache 量化选项

KV cache 量化通过 `--kv_cache_qformat` 参数独立控制，与模型权重/激活量化分开：

| 格式 | 说明 | 是否需要校准 |
|------|------|------------|
| `none` | 不量化 KV cache | - |
| `fp8_cast`（默认） | FP8 KV cache，使用常量 amax | ❌ |
| `fp8` | FP8 KV cache，数据驱动校准 | ✅ |
| `fp8_affine` | FP8 仿射 KV cache | ✅ |
| `nvfp4_cast` | NVFP4 KV cache，使用常量 amax | ❌ |
| `nvfp4` | NVFP4 KV cache，数据驱动校准 | ✅ |
| `nvfp4_affine` | NVFP4 仿射 KV cache | ✅ |
| `nvfp4_rotate` | NVFP4 旋转 KV cache | ✅ |

**注意**：`_cast` 后缀的格式使用 `use_constant_amax=True`，不需要额外的校准数据。

---

## 自定义配置

### 在 MLP-only 基础上添加更多层

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
```

### 跳过特定层

```python
custom_cfg = copy.deepcopy(mtq.NVFP4_OMLP_ONLY_CFG)
# 跳过第 0 层的 MLP
custom_cfg["quant_cfg"]["*layers.0.mlp*"] = {"enable": False}
```

### 修改校准算法

```python
custom_cfg = copy.deepcopy(mtq.NVFP4_MLP_ONLY_CFG)
# 使用 AWQ Lite 校准代替 max
custom_cfg["algorithm"] = "awq_lite"
```

### 修改 MoE 校准专家比例

```python
from examples.llm_ptq.example_utils import build_quant_cfg

quant_cfg = build_quant_cfg(
    qformat="nvfp4_mlp_only",
    quant_cfg=mtq.NVFP4_MLP_ONLY_CFG,
    awq_block_size=0,
    model_type=None,
    moe_calib_experts_ratio=0.5,  # 校准时只激活 50% 的专家
)
```

---

## 参数参考

### hf_ptq.py 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--qformat` | `fp8` | 量化格式，可选 `nvfp4_mlp_only`、`nvfp4_omlp_only` 等 |
| `--kv_cache_qformat` | `fp8_cast` | KV cache 量化格式 |
| `--calib_size` | `512` | 校准样本数 |
| `--calib_seq` | `512` | 校准最大序列长度 |
| `--batch_size` | `0`（自动） | 校准批大小，0 表示自动检测 |
| `--export_path` | `exported_model` | 导出路径 |
| `--dataset` | `None` | 校准数据集名称 |
| `--low_memory_mode` | `False` | 低内存模式（先压缩权重再校准） |
| `--moe_calib_experts_ratio` | `1.0` | MoE 校准时激活的专家比例 |
| `--trust_remote_code` | `False` | 是否信任远程代码 |
| `--verbose` | `True` | 是否打印量化摘要 |

### NVFP4 量化器属性

| 属性 | 值 | 说明 |
|------|-----|------|
| `num_bits` | `(2, 1)` | FP4: 2 位指数 + 1 位尾数 |
| `block_sizes[-1]` | `16` | 沿最后一个维度的块大小 |
| `block_sizes["type"]` | `"dynamic"` | 动态缩放 |
| `block_sizes["scale_bits"]` | `(4, 3)` | FP8 缩放因子: 4 位指数 + 3 位尾数 |
| `enable` | `True` | 启用量化 |

---

## 已知问题

1. **小模型精度损失更大**：模型越小，PTQ 后的精度损失越明显。对于小模型，建议使用 `nvfp4_mlp_only` 而非 `nvfp4`（全量）
2. **KV cache 量化可能影响精度**：某些模型（如 QWen 2/2.5）的 KV cache 量化可能导致较大的精度损失。可通过 `--kv_cache_qformat none` 禁用
3. **需要 Blackwell GPU**：NVFP4 推理需要 Blackwell GPU，校准可以在 Hopper/Ada GPU 上进行
4. **AutoQuantize 不支持 Llama-4**：Llama-4 不支持反向传播，因此不能使用 AutoQuantize
5. **MoE 模型的 `*block_sparse_moe*` 模式**：此模式主要匹配 Mixtral 风格的 MoE。对于其他 MoE 架构（如 DeepSeek、Qwen3 MoE），ModelOpt 通过 `register_sparse_moe_on_the_fly()` 动态注册 `_QuantSparseMoe` 来处理

---

## 深度技术文档

如需了解 NVFP4 量化的深度技术细节，请参阅以下文档：

| 文档 | 内容 |
|------|------|
| [NVFP4 量化原理](./nvfp4_detail/01_nvfp4_quantization_principle.md) | FP4 数据格式、两级缩放机制、量化数学过程 |
| [NVFP4 核心代码位置](./nvfp4_detail/02_nvfp4_code_structure.md) | 配置定义、量化入口、TensorQuantizer、NVFP4QTensor |
| [NVFP4 校准流程详解](./nvfp4_detail/03_nvfp4_calibration.md) | forward_loop、MaxCalibrator、动态/静态缩放 |
| [NVFP4 权重导出机制](./nvfp4_detail/04_nvfp4_weight_export.md) | export_hf_checkpoint、导出张量、modelopt_state |
| [NVFP4 MoE 模型处理](./nvfp4_detail/05_nvfp4_moe_handling.md) | _QuantSparseMoe、amax 同步、不同架构处理 |
| [NVFP4 量化器模式匹配](./nvfp4_detail/06_nvfp4_pattern_matching.md) | 通配符匹配、优先级规则、自定义配置 |

---

## 变更记录

- **v2.0**：拆分为 Guide + Detail 文档结构，Guide 聚焦使用方式，Detail 聚焦技术细节
- **v1.0**：初始版本，基于 Model-Optimizer 源码分析
