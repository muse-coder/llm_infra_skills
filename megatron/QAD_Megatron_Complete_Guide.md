# Megatron-LM QAD (Quantization-Aware Distillation) 完整指南

> 知识库版本: 2026-03-23 | 基于 Megatron-LM + nvidia-modelopt 实测总结
> 参考论文: arXiv:2601.20088 "Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery"

---

## 一、QAD 是什么

QAD = Quantization-Aware Distillation，核心思路：
1. 把 HF 模型量化为 NVFP4（Student），同时保留 BF16 全精度版本（Teacher）
2. 用 Teacher 的 logits 作为软标签，通过 KL 散度蒸馏训练 Student
3. Student 权重以 BF16 存储 + fake-quantization 模拟 FP4 行为（训练需要梯度）
4. 训练完成后 export 导出为真正的 FP4 权重，用于 TRT-LLM / vLLM / SGLang 部署

```
HF 权重 (BF16)
    ├─ quantize.sh ──→ Student (BF16 + NVFP4 fake-quant + modelopt_state)
    ├─ convert.sh  ──→ Teacher (BF16，纯 Megatron 格式)
    │
    ▼ finetune.sh (QAD 训练，KL 散度蒸馏)
训练后的 Student ckpt
    │
    ▼ export.sh (TP 强制为 1)
真正的 FP4 权重 → 部署
```

---

## 二、完整流程（以 Qwen3-30B-A3B 为例，8 卡单机）

### 前置条件
- 代码目录: `/workspace/mudi/Megatron-LM`
- 所有脚本必须从 `examples/post_training/modelopt/` 目录执行
- 需要模型配置文件: `conf/Qwen/Qwen3-30B-A3B.sh`

### Step 0: 环境变量（env.sh）

```bash
export MEGATRON_DIR=/workspace/mudi/Megatron-LM
export HF_MODEL_CKPT=/path/to/Qwen3-30B-A3B          # HuggingFace 格式模型
export WORK_DIR=/path/to/workspace/qad_qwen3_30b       # 工作目录

export STUDENT_CKPT=${WORK_DIR}/student-nvfp4
export TEACHER_CKPT=${WORK_DIR}/teacher-bf16
export QAD_SAVE=${WORK_DIR}/qad-output
export EXPORT_DIR=${WORK_DIR}/qad-export
export TEACHER_CONFIG=${WORK_DIR}/teacher_model_config.yaml
export TRAIN_DATA=/path/to/training/data               # HF dataset 路径或本地 JSONL

cd ${MEGATRON_DIR}/examples/post_training/modelopt
```

### Step 1: 量化生成 Student Checkpoint

```bash
TP=1 EP=1 \
HF_MODEL_CKPT=${HF_MODEL_CKPT} \
MLM_MODEL_SAVE=${STUDENT_CKPT} \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./quantize.sh Qwen/Qwen3-30B-A3B NVFP4_DEFAULT_CFG
```

**调用链**: `quantize.sh → arguments.sh → conf/Qwen/Qwen3-30B-A3B.sh → quantize.py`

**内部流程**:
1. `torchrun --nproc_per_node=N quantize.py`（N = TP×EP×PP）
2. 构建 Megatron GPT 模型（空壳，按 TP/EP 切分）
3. `import_mcore_gpt_from_hf()` 从 HF 权重导入到 Megatron 模型
4. `mtq.quantize()` 执行 NVFP4 PTQ 校准（用 cnn_dailymail 512 个样本前向）
5. `save_checkpoint()` 保存为 Megatron 分布式 checkpoint + `modelopt_state/`

**输出目录结构**:
```
student-nvfp4/
  ├─ iter_0000001/
  │    ├─ mp_rank_00/    # TP=0 的权重（TP=1 EP=1 时只有一个分片）
  │    └─ ...
  ├─ latest_checkpointed_iteration.txt → "1"
  └─ modelopt_state/     # 量化器配置、amax、scale 等
```

### Step 2: 转换生成 Teacher Checkpoint

```bash
TP=1 EP=8 \
HF_MODEL_CKPT=${HF_MODEL_CKPT} \
MLM_MODEL_SAVE=${TEACHER_CKPT} \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./convert.sh Qwen/Qwen3-30B-A3B
```

**注意**: Teacher 的 TP/EP 必须和后续 finetune 的 TP/EP 一致。

### Step 3: 创建 Teacher 模型配置 YAML

手动创建 `teacher_model_config.yaml`，NeMo 格式:

```yaml
# Qwen3-30B-A3B
num_layers: 48
hidden_size: 2048
ffn_hidden_size: 6144
num_attention_heads: 32
num_query_groups: 4
kv_channels: 128
num_moe_experts: 128
moe_ffn_hidden_size: 768
moe_router_topk: 8
max_position_embeddings: 40960
```

**来源**: 从 HuggingFace config.json 手动映射，字段名对应关系:
| HF config.json | Teacher YAML |
|----------------|--------------|
| num_hidden_layers | num_layers |
| hidden_size | hidden_size |
| intermediate_size | ffn_hidden_size |
| num_attention_heads | num_attention_heads |
| num_key_value_heads | num_query_groups |
| head_dim (或 hidden_size/num_attention_heads) | kv_channels |
| num_experts | num_moe_experts |
| moe_intermediate_size | moe_ffn_hidden_size |
| num_experts_per_tok | moe_router_topk |
| max_position_embeddings | max_position_embeddings |

### Step 4: 准备训练数据

Megatron SFTDataset 支持两种数据源:

**方式 A: HuggingFace Dataset（推荐）**
```bash
--finetune-hf-dataset nvidia/OpenCodeReasoning
```
已注册的 HF dataset（如 `Open-Orca/SlimOrca`）会自动做 ShareGPT → OpenAI 格式转换。

**方式 B: 本地 JSONL**
必须是 OpenAI 格式（`role`/`content`），不能是 ShareGPT 格式（`from`/`value`）。
如果你的数据是 ShareGPT 格式，需要预转换:
```python
# ShareGPT: {"conversations": [{"from": "User", "value": "..."}, ...]}
# OpenAI:   {"conversations": [{"role": "user", "content": "..."}, ...]}
```

**原因**: 对于本地 JSONL 路径（不在 `hf_dataset_to_conversation` 注册表里的），
`finetune.py` 会用 identity transformation，直接读 `role`/`content` 字段。
如果你的数据用的是 `from`/`value`，会在 `example[0]["role"]` 处报 KeyError。

### Step 5: QAD 训练（核心步骤）

```bash
TP=1 EP=8 \
MLM_MODEL_CKPT=${STUDENT_CKPT} \
MLM_MODEL_SAVE=${QAD_SAVE} \
MLM_DATA_ARGS=" \
    --train-samples 600000 \
    --lr-decay-samples 600000 \
    --lr-warmup-samples 5000 \
    --split 100,0,0 \
    --finetune-hf-dataset ${TRAIN_DATA} \
" \
MLM_OPTIM_ARGS=" \
    --lr 5.0e-6 \
    --min-lr 1.0e-7 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.010 \
    --use-distributed-optimizer \
" \
MLM_TRAIN_ARGS=" \
    --no-gradient-accumulation-fusion \
    --global-batch-size 64 \
    --micro-batch-size 1 \
    --reset-position-ids \
    --reset-attention-mask \
    --eod-mask-loss \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-check-for-nan-in-loss-and-grad \
" \
MLM_EVAL_ARGS=" \
    --eval-iters 1 \
    --eval-interval 9999999 \
    --save-interval 1000 \
    --log-interval 10 \
" \
MLM_EXTRA_ARGS=" \
    --export-kd-teacher-load ${TEACHER_CKPT} \
    --export-kd-teacher-model-config ${TEACHER_CONFIG} \
" \
./finetune.sh Qwen/Qwen3-30B-A3B
```

### Step 6: 导出部署格式

```bash
PP=1 \
HF_MODEL_CKPT=${HF_MODEL_CKPT} \
MLM_MODEL_CKPT=${QAD_SAVE} \
EXPORT_DIR=${EXPORT_DIR} \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./export.sh Qwen/Qwen3-30B-A3B
```

**注意**: `export.sh` 会强制 TP=1（第24-27行硬编码）。

---

## 三、并行度配置详解

### TP + EP 的限制

**modelopt 的 `QuantSequentialMLP` 不支持 TP>1 且 EP>1 同时使用。**

报错: `ValueError: TP+EP is not supported by QuantSequentialMLP. Set either TP or EP to 1!`

原因: `get_gpt_modelopt_spec()` 硬编码 `moe_grouped_gemm=False`，导致 MoE 层永远用 `SequentialMLP`。
`TEGroupedMLP` 的量化插件没有此限制，但在 modelopt spec 中被注释掉了。

**解决方案**: quantize 和 finetune 时，保持 TP=1 或 EP=1。
- 推荐 MoE 模型: **TP=1, EP=N**（N = GPU 数量）
- 推荐 Dense 模型: **TP=N, EP=1**

### GBS 计算公式

```
data_parallel_size = world_size / (TP × PP × CP)        # 注意：不除以 EP！
global_batch_size = micro_batch_size × data_parallel_size  # 不指定 --global-batch-size 时的默认值
gradient_accumulation = global_batch_size / (micro_batch_size × data_parallel_size)
```

**EP 不参与 data_parallel_size 计算！** EP 只影响 `expert_data_parallel_size`（MoE expert 参数的 DP 维度）。

例: TP=1, EP=8, PP=1, 8 GPU:
- `data_parallel_size = 8 / 1 = 8`
- 不设 `--global-batch-size` 时: `GBS = 1 × 8 = 8`
- 设 `--global-batch-size 64`: 梯度累积 = `64 / (1 × 8) = 8 次`

### Student 和 Teacher 的 TP/EP 必须一致

QAD 训练时 Student 和 Teacher 运行在同一组 GPU 上，并行度必须相同。
如果 Student 的 checkpoint 是不同 TP/EP 保存的，需要重新从 HF 量化（见下方 resharding 说明）。

### Checkpoint Resharding（改变 TP/EP）

**模型权重**: Megatron `dist_checkpointing`（`torch_dist` 格式）**原生支持** resharding。
保存 TP=1 的 checkpoint，可以用 TP=2 加载，`ShardedTensor` 自动处理切分。

**modelopt 量化状态**: `modelopt_state/` 目录**不支持** resharding。
`restore_sharded_modelopt_state()` 没有 reshard 能力，TP 不匹配会失败。

**结论**: 带量化的 checkpoint 不能直接改 TP/EP 加载。需要重新从 HF 跑 quantize。

---

## 四、超参数调优指南

### NVIDIA 论文推荐值

| 模型 | 学习率 | 训练 Token 数 | 说明 |
|------|--------|-------------|------|
| Nemotron 3 Nano 30B-A3B | 1e-5 | ~2.5B | RL 模型，LR 偏高 |
| Nemotron Nano 9B V2 | 1e-6 | ~6B | SFT 模型，LR 偏低 |
| Llama Nemotron Super 49B | 1e-6 | ~0.3B | 大模型收敛快 |
| Nemotron Nano 12B V2 VL | 2e-6 | ~0.5B | 多模态 |

### 学习率选择原则

- **SFT 训练的模型**: LR 应 ≤ 原始 SFT 的 LR，推荐 **1e-6 ~ 5e-6**
- **RL 训练的模型**: RL 偏移了分布，需要更高 LR，推荐 **1e-5**
- **Softmax 温度**: T=1（默认），精确匹配 Teacher 分布

### 训练量估算

```
总 token 数 ≈ train_samples × seq_length
```

以 seq_length=4096 为例:
- 0.2B tokens → train_samples ≈ 50,000（不够，效果差）
- 2.5B tokens → train_samples ≈ 600,000（推荐）
- 6B tokens → train_samples ≈ 1,500,000（充分训练）

### Global Batch Size

- 太小（如 GBS=8）→ 梯度噪声大，KL 散度 loss 方差高，优化不稳
- 推荐 GBS=64~256，通过 `--global-batch-size` 显式设置
- 通过梯度累积实现，不额外占显存
- `--global-batch-size` 必须能被 `micro_batch_size × data_parallel_size` 整除

### 效果差的排查清单

| 排查项 | 检查方法 | 常见问题 |
|--------|---------|---------|
| 训练量不够 | `train_samples × seq_length` 是否达到推荐值 | 差 10 倍以上 |
| GBS 太小 | 检查 log 里打印的 `global batch size` | 默认可能只有 DP 大小 |
| LR 过高 | SFT 模型不应超过 1e-5 | 导致遗忘 |
| 数据格式错误 | 检查是否 KeyError on "role" | ShareGPT 没转换 |
| Teacher ckpt 格式不对 | 检查 `teacher_model_config.yaml` 是否存在且正确 | 字段名映射错误 |
| 评估与训练域不匹配 | 代码数据训练 vs 通用知识评估 | 不代表效果真的差 |

---

## 五、知识蒸馏 Loss 机制

### 默认行为

```python
# modelopt 默认 DistillationConfig:
skip_lm_loss: True      # 跳过原始 LM 交叉熵 loss
kd_loss_scale: 1.0       # KD loss 缩放（skip_lm_loss=False 时才生效）
logit_kl_temperature: 1.0  # softmax 温度
```

**默认只用 KL 散度蒸馏 loss，不加原始 LM loss。**

### 自定义蒸馏配置

创建 `distill_config.yaml`:
```yaml
logit_layers: ["output_layer", "output_layer"]    # [student层名, teacher层名]
intermediate_layer_pairs:                          # 可选：中间层蒸馏
  - ["decoder.layers.0.input_layernorm", "decoder.layers.0.input_layernorm"]
skip_lm_loss: false          # 同时保留原始 LM loss
kd_loss_scale: 1.0           # KD loss 权重
logit_kl_temperature: 1.0    # 温度
```

使用: `--export-kd-distill-cfg distill_config.yaml`（加到 MLM_EXTRA_ARGS）

### Loss 类型
- **Logits**: KL Divergence Loss（默认）或 TopK KL Loss
- **中间层**: Cosine Similarity Loss
- **原始 LM**: Cross-Entropy Loss（默认跳过）

---

## 六、已知问题和 Workaround

### 1. TP+EP 不支持（量化时）
**现象**: `ValueError: TP+EP is not supported by QuantSequentialMLP`
**Workaround**: quantize/finetune 时 TP=1, EP=N

### 2. 训练时显存 OOM
**现象**: 蒸馏模型比普通训练多几 MB/microbatch 的开销
**Workaround**: 
- 不要用 `--manual-gc`
- 开启 recompute: `--recompute-granularity full --recompute-method block --recompute-num-layers N`

### 3. 训练速度慢 ~40%
**现象**: 每个 iteration 比不用 KD 时慢约 40%
**原因**: CUDA kernel 问题，Student 前向延迟在 Teacher 存在时被拉长
**状态**: NVIDIA 已知问题

### 4. export 强制 TP=1
**现象**: `export.sh` 第 24-27 行强制 `TP=1`
**影响**: 导出时必须单卡（或 PP>1），和训练时的 TP 无关

### 5. Interleaved Pipeline Parallel 不支持
**现象**: KD 不支持 interleaved PP
**Workaround**: 只用非 interleaved PP

### 6. modelopt_state 不支持 resharding
**现象**: 量化 ckpt 不能改 TP/EP 加载
**Workaround**: 从 HF 重新 quantize
**注释**: prune.py 第 230 行: "WAR till modelopt 0.39: Remove prune state to avoid converting again on restore which forces TP=1."

---

## 七、关键文件索引

```
examples/post_training/modelopt/
├── conf/
│   ├── arguments.sh          # 公共参数解析，source 模型配置
│   └── Qwen/
│       └── Qwen3-30B-A3B.sh  # 模型架构参数（MODEL_ARGS）
├── quantize.sh / quantize.py  # Step 1: HF → NVFP4 Student
├── convert.sh / convert_model.py  # Step 2: HF → BF16 Teacher
├── finetune.sh / finetune.py  # Step 5: QAD 训练
├── export.sh / export.py      # Step 6: 导出部署格式
├── distillation.md            # KD 文档
└── README.md                  # 总览 + 支持矩阵

megatron/
├── post_training/
│   ├── arguments.py           # --export-kd-* 参数定义
│   ├── model_builder.py       # 蒸馏模型构建（第 313-354 行）
│   ├── checkpointing.py       # load_modelopt_checkpoint / load_modelopt_state
│   └── loss_func.py           # KD loss 计算
├── training/
│   ├── arguments.py           # GBS/MBS 计算（第 562-568 行）
│   └── checkpointing.py       # save/load checkpoint
└── core/
    ├── dist_checkpointing/    # 分布式 checkpoint（支持 resharding）
    ├── parallel_state.py      # DP/EP 并行组计算
    └── post_training/modelopt/gpt/
        └── model_specs.py     # get_gpt_modelopt_spec()（moe_grouped_gemm=False 硬编码）

# modelopt 库（外部依赖）
modelopt/torch/quantization/plugins/megatron.py  # QuantSequentialMLP TP+EP 限制（第 558-563 行）
modelopt/torch/distill/plugins/megatron.py        # DistillationConfig 默认值
modelopt/torch/distill/losses.py                  # KL/Cosine loss 实现
```

---

## 八、快速 Checklist

开始 QAD 前，确认以下事项:

- [ ] 模型配置文件 `conf/Model/Model.sh` 存在且参数正确（对照 HF config.json）
- [ ] `teacher_model_config.yaml` 已创建（NeMo 格式，字段名对齐）
- [ ] 训练数据格式正确（OpenAI 格式: `role`/`content`，不是 ShareGPT 的 `from`/`value`）
- [ ] TP 和 EP 不同时 >1（modelopt 限制）
- [ ] Student 和 Teacher 的 TP/EP 一致
- [ ] `--global-batch-size` 显式设置（推荐 64~256）
- [ ] `--train-samples` 足够（seq_length × train_samples ≈ 推荐 token 数）
- [ ] LR 匹配模型类型（SFT: ≤1e-5, RL: 1e-5）
- [ ] 加了 `--no-gradient-accumulation-fusion`（如环境需要）
