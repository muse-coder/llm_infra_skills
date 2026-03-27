---
name: modelopt-nvfp4-selective
description: "处理 NVIDIA Model Optimizer NVFP4 选择性量化任务时必须使用。涵盖 nvfp4_mlp_only、nvfp4_omlp_only、NVFP4_MLP_ONLY_CFG、NVFP4_OMLP_ONLY_CFG 的配置定义、hf_ptq.py 中的使用方式、量化器模式匹配、KV cache 选项、精度与压缩的权衡，适用于 Blackwell GPU 部署。"
---

# ModelOpt NVFP4 选择性量化完整知识 (nvfp4_mlp_only / nvfp4_omlp_only)

> 你是 Model Optimizer PTQ 专家。当用户需要使用 NVFP4 选择性量化格式（仅量化 MLP 层，以及可选的 attention 输出投影层）来获得更高精度时，应用此知识。

---

## 核心概念

NVIDIA Model Optimizer 提供两种选择性 NVFP4 量化配置，以少量压缩换取显著更好的精度（相比全量 NVFP4 `NVFP4_DEFAULT_CFG`）：

| 配置 | 命令行名称 | 量化的层 | 不量化的层 |
|------|----------|---------|-----------|
| `NVFP4_MLP_ONLY_CFG` | `nvfp4_mlp_only` | MLP (gate/up/down_proj) + MoE 专家 | Attention (q/k/v/o_proj)、lm_head、路由器 |
| `NVFP4_OMLP_ONLY_CFG` | `nvfp4_omlp_only` | MLP + MoE 专家 + `o_proj` | Attention (q/k/v_proj)、lm_head、路由器 |

**推荐**：进行 NVFP4 PTQ 时，优先使用 `nvfp4_mlp_only` 或 `nvfp4_omlp_only`，而非 `nvfp4`（全量），以保留精度敏感的 attention QKV 投影层。

---

## 配置定义（源自 `modelopt/torch/quantization/config.py`）

### 基础 NVFP4 量化器

```python
_nvfp4_quantizer = {
    "num_bits": (2, 1),           # FP4: 2 位指数 + 1 位尾数
    "block_sizes": {
        -1: 16,                   # 沿最后一个维度的块大小为 16
        "type": "dynamic",        # 动态缩放
        "scale_bits": (4, 3),     # FP8 缩放因子: 4 位指数 + 3 位尾数
    },
    "enable": True,
}
```

### NVFP4_MLP_ONLY_CFG

```python
_nvfp4_mlp_only_quant_cfg = {
    "*mlp*weight_quantizer": _nvfp4_quantizer,
    "*mlp*input_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*weight_quantizer": _nvfp4_quantizer,
    "*block_sparse_moe*input_quantizer": _nvfp4_quantizer,
    **_default_disabled_quantizer_cfg,   # 禁用 lm_head、路由器、default 等
}

NVFP4_MLP_ONLY_CFG = {
    "quant_cfg": _nvfp4_mlp_only_quant_cfg,
    "algorithm": "max",
}
```

**模式匹配**：`*mlp*` 匹配所有 MLP 子模块（gate_proj、up_proj、down_proj）。`*block_sparse_moe*` 匹配 MoE 专家权重。其余所有模块落入 `_default_disabled_quantizer_cfg`，量化被禁用。

### NVFP4_OMLP_ONLY_CFG

```python
NVFP4_OMLP_ONLY_CFG = {
    "quant_cfg": {
        "*o_proj*weight_quantizer": _nvfp4_quantizer,
        "*o_proj*input_quantizer": _nvfp4_quantizer,
        **_nvfp4_mlp_only_quant_cfg,   # 继承 MLP + MoE + 禁用默认
    },
    "algorithm": "max",
}
```

**关键区别**：在 MLP-only 配置基础上，额外添加 `*o_proj*`（attention 输出投影层）的量化。

### 默认禁用的量化器

以下层始终被跳过（定义在 `_default_disabled_quantizer_cfg` 中）：

| 模式 | 对应层 |
|------|--------|
| `*lm_head*` | 语言模型头 |
| `*block_sparse_moe.gate*` | MoE 路由器 gate |
| `*router*` | MoE 路由器 |
| `*mlp.gate.*` | MoE gate（不是 MLP 的 gate_proj） |
| `*mlp.shared_expert_gate.*` | 共享专家 gate |
| `*output_layer*` | 输出层 |
| `default` | 所有未匹配的量化器 |

---

## 使用方式

### 命令行 (hf_ptq.py)

```bash
# 仅量化 MLP 层
python hf_ptq.py \
    --pyt_ckpt_path <模型路径> \
    --qformat nvfp4_mlp_only \
    --export_path <输出路径>

# 量化 MLP + o_proj 层
python hf_ptq.py \
    --pyt_ckpt_path <模型路径> \
    --qformat nvfp4_omlp_only \
    --export_path <输出路径>
```

### Shell 脚本

```bash
scripts/huggingface_example.sh --model $HF_PATH --quant nvfp4_mlp_only --tp 1
scripts/huggingface_example.sh --model $HF_PATH --quant nvfp4_omlp_only --tp 1
```

### Python API

```python
import modelopt.torch.quantization as mtq

# 仅 MLP
model = mtq.quantize(model, mtq.NVFP4_MLP_ONLY_CFG, forward_loop)

# MLP + o_proj
model = mtq.quantize(model, mtq.NVFP4_OMLP_ONLY_CFG, forward_loop)
```

### 配合 KV Cache 量化

KV cache 量化通过 `--kv_cache_qformat` 独立控制（默认 `fp8_cast`）：

```bash
python hf_ptq.py \
    --pyt_ckpt_path <模型路径> \
    --qformat nvfp4_mlp_only \
    --kv_cache_qformat fp8_cast \
    --export_path <输出路径>
```

可用 KV cache 格式：`none`、`fp8_cast`（默认）、`fp8`、`fp8_affine`、`nvfp4_cast`、`nvfp4`、`nvfp4_affine`、`nvfp4_rotate`。

以 `_cast` 结尾的格式使用常量 amax（无需校准），其他格式使用数据驱动校准。

### 配合 AutoQuantize

两种格式均可用于 AutoQuantize 混合精度搜索：

```bash
scripts/huggingface_example.sh \
    --model $HF_PATH \
    --quant nvfp4_mlp_only,fp8 \
    --auto_quantize_bits 4.75
```

### 多节点 (FSDP2)

```bash
accelerate launch --config_file fsdp2.yaml \
    multinode_ptq.py \
    --pyt_ckpt_path <模型路径> \
    --qformat nvfp4_mlp_only \
    --export_path <输出路径>
```

---

## 精度与压缩的权衡

```
更高精度 ◄──────────────────────────────► 更高压缩率
                                                  
  不量化 > nvfp4_mlp_only > nvfp4_omlp_only > nvfp4（全量）
                                                  
  Attention QKV:    BF16          BF16              NVFP4
  Attention o_proj: BF16          NVFP4             NVFP4
  MLP 层:           NVFP4         NVFP4             NVFP4
```

**选择建议**：
- `nvfp4_mlp_only`：精度最高，NVFP4 PTQ 的推荐默认选择
- `nvfp4_omlp_only`：折中方案，比 mlp_only 压缩率略高
- `nvfp4`（全量）：最大压缩率，小模型上可能有明显精度损失

---

## 自定义配置

### 跳过额外的层

```python
import copy
import modelopt.torch.quantization as mtq

custom_cfg = copy.deepcopy(mtq.NVFP4_MLP_ONLY_CFG)
custom_cfg["quant_cfg"]["*某个敏感层*"] = {"enable": False}
model = mtq.quantize(model, custom_cfg, forward_loop)
```

### 添加更多层进行量化

```python
custom_cfg = copy.deepcopy(mtq.NVFP4_MLP_ONLY_CFG)
# 同时量化 q_proj 和 k_proj
custom_cfg["quant_cfg"]["*q_proj*weight_quantizer"] = mtq.NVFP4_MLP_ONLY_CFG["quant_cfg"]["*mlp*weight_quantizer"]
custom_cfg["quant_cfg"]["*q_proj*input_quantizer"] = mtq.NVFP4_MLP_ONLY_CFG["quant_cfg"]["*mlp*input_quantizer"]
```

---

## 关键约束

- **需要 Blackwell GPU**：NVFP4 推理需要 NVIDIA Blackwell GPU 和 TensorRT-LLM v0.17+
- **校准算法**：两者均使用 `"max"` 校准（简单快速）。如需更高精度，可考虑 `nvfp4_awq`（使用 `awq_lite`）
- **块大小**：NVFP4 使用 block_size=16（而 NVFP4_MLP_WEIGHT_ONLY_CFG 使用 block_size=32）
- **权重和激活同时量化**：这些配置同时量化权重和激活（input_quantizer 启用），与仅权重变体不同

---

## 详细文档

完整配置定义和模式匹配规则见：`modelopt/NVFP4_Selective_Quantization_Guide.md`

## 关键文件索引

```
modelopt/torch/quantization/config.py          # NVFP4_MLP_ONLY_CFG、NVFP4_OMLP_ONLY_CFG 定义（第 639-665 行）
examples/llm_ptq/hf_ptq.py                     # QUANT_CFG_CHOICES 映射（第 92-113 行）
examples/llm_ptq/example_utils.py              # build_quant_cfg() 函数（第 199 行）
examples/llm_ptq/scripts/huggingface_example.sh # Shell 脚本入口
examples/llm_ptq/README.md                     # 用户文档
examples/llm_ptq/multinode_ptq.py              # FSDP2 多节点 PTQ
```
