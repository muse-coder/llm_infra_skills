# LM-Evaluation-Harness + Qwen3 评测完整指南

## 概述

本文档记录使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 框架对 Qwen3 系列模型（特别是 Qwen3-30B-A3B 及其量化变体）进行评测时的关键注意事项、最佳实践和踩坑记录。

适用场景：
- 使用 vLLM 后端进行 Qwen3 模型评测
- 评测任务包括 GSM8K、MMLU Pro、AIME25 等
- 涉及 NVFP4 量化模型的精度对比评测

---

## 1. Qwen3 Thinking 模式与采样参数

### 核心规则：不要使用 Greedy Decoding

Qwen3 官方明确指出：**thinking 模式下禁止使用 greedy decoding**，否则会导致性能下降和无限重复。

### 官方推荐采样参数

| 模式 | Temperature | TopP | TopK | MinP | 说明 |
|------|------------|------|------|------|------|
| **think 模式** (`enable_thinking=True`) | `0.6` | `0.95` | `20` | `0` | `generation_config.json` 中的默认值 |
| **no_think 模式** (`enable_thinking=False`) | `0.7` | `0.8` | `20` | `0` | 官方推荐值 |

来源：[Qwen3-30B-A3B ModelScope](https://modelscope.cn/models/Qwen/Qwen3-30B-A3B)、[Qwen3-32B HuggingFace](https://huggingface.co/Qwen/Qwen3-32B)

### 在 lm-evaluation-harness 中设置采样参数

采样参数通过 `--gen_kwargs` 传入，**不是** `--model_args`：

```bash
# think 模式
lm_eval ... --gen_kwargs "temperature=0.6,top_p=0.95,top_k=20,min_p=0"

# no_think 模式
lm_eval ... --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,min_p=0"
```

---

## 2. Chat Template 与 enable_thinking 控制

### --apply_chat_template 是否必要？

**对于 Qwen3 instruct 模型：必要。** 原因：

1. `enable_thinking` 的开关通过 chat template 的 `enable_thinking` 参数控制
2. `--chat_template_kwargs` 只有在 `--apply_chat_template` 开启时才生效
3. Qwen3 是 instruct 模型，按 `<|im_start|>...<|im_end|>` 格式训练，不加 chat template 会降低理解能力

### enable_thinking 的默认值

**Qwen3 的 chat template 默认 `enable_thinking=True`**。如果你想跑 no_think 模式，**必须显式传入 `enable_thinking=False`**，否则即使 `model_args` 里没有 `enable_thinking=true`，chat template 仍会默认开启 thinking。

### 正确的 think/no_think 切换方式

```bash
# think 模式
lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,...,enable_thinking=true,think_end_token=</think>" \
    --apply_chat_template \
    --chat_template_kwargs '{"enable_thinking": true}' \
    --gen_kwargs "temperature=0.6,top_p=0.95,top_k=20,min_p=0" \
    ...

# no_think 模式
lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL_PATH,..." \
    --apply_chat_template \
    --chat_template_kwargs '{"enable_thinking": false}' \
    --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,min_p=0" \
    ...
```

### 三层控制关系

| 层级 | 参数 | 作用 |
|------|------|------|
| **vLLM 层** | `model_args` 中的 `enable_thinking=true,think_end_token=</think>` | 告诉 vLLM 引擎处理 thinking token |
| **Chat Template 层** | `--chat_template_kwargs '{"enable_thinking": true/false}'` | 控制 tokenizer 是否在 prompt 中启用 thinking 格式 |
| **采样层** | `--gen_kwargs "temperature=0.6,..."` | 控制生成时的采样策略 |

---

## 3. vLLM 后端关键参数

### model_args 常用参数

```
pretrained=<模型路径>
dtype=auto                          # 自动选择精度
gpu_memory_utilization=0.85         # GPU 显存利用率
max_model_len=32768                 # 最大序列长度
tensor_parallel_size=1              # 张量并行数（单卡=1）
enforce_eager=True                  # 禁用 CUDA Graph，量化模型建议开启
enable_thinking=true                # 开启 thinking（仅 think 模式）
think_end_token=</think>            # thinking 结束 token（仅 think 模式）
quantization=modelopt_fp4           # 量化格式（仅量化模型）
```

### 量化模型注意事项

- NVFP4 量化模型需要传 `quantization=modelopt_fp4`
- 官方预量化模型（如 `Qwen3-30B-A3B-NVFP4`）**不需要**传 `quantization` 参数
- `enforce_eager=True` 建议在量化模型上开启，避免 CUDA Graph 兼容性问题

---

## 4. 多 GPU 并行评测

### 并行策略

8 张 GPU 同时评测不同模型，每个模型独占一张卡：

```bash
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# 通过 CUDA_VISIBLE_DEVICES 分配 GPU
CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval ...

# 后台并行启动
run_eval "$GPU_ID" "$MODEL_PATH" ... &
PIDS+=($!)

# 满一轮后等待
if [[ ${#PIDS[@]} -eq $NUM_GPUS ]]; then
    for PID in "${PIDS[@]}"; do wait "$PID"; done
    PIDS=()
fi
```

### 注意事项

- 不要设置全局 `export CUDA_VISIBLE_DEVICES=X`，改为在每个任务前动态指定
- 每个模型独占一张卡时 `tensor_parallel_size=1`
- 如果模型太大需要多卡，调整 `tensor_parallel_size` 并相应减少并行模型数

---

## 5. 环境变量

```bash
# 离线模式（无网络环境必须设置）
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# vLLM 优化参数（针对 NVFP4 量化）
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
export VLLM_USE_FLASHINFER_MOE_FP4=0
```

---

## 6. 评测任务说明

| 任务 | 类型 | 说明 |
|------|------|------|
| `gsm8k` | 数学推理 | 小学数学题，few-shot，exact_match 评估 |
| `mmlu_pro` | 知识理解 | 多学科选择题，custom-extract 评估 |
| `aime25` | 数学竞赛 | AIME 2025 竞赛题，exact_match 评估 |

---

## 7. 踩坑记录

### 坑 1：no_think 模式实际仍在 thinking

**现象**：no_think 模式的结果和 think 模式几乎一样。

**原因**：Qwen3 chat template 默认 `enable_thinking=True`，仅在 `model_args` 中不传 `enable_thinking` 并不能关闭 thinking。

**解决**：必须通过 `--chat_template_kwargs '{"enable_thinking": false}'` 显式关闭。

### 坑 2：不加 chat template 也能得高分

**现象**：旧脚本没有 `--apply_chat_template`，GSM8K 仍达到 89.54%。

**原因**：
1. vLLM 层面的 `enable_thinking=true` 独立于 chat template 生效
2. GSM8K 是 few-shot 任务，few-shot 示例本身提供了格式提示
3. 但这不是规范做法，建议始终加 `--apply_chat_template`

### 坑 3：Greedy decoding 导致无限重复

**现象**：模型输出无限重复相同内容，评测超时。

**原因**：thinking 模式下使用了 greedy decoding（默认行为）。

**解决**：必须设置 `--gen_kwargs "temperature=0.6,top_p=0.95,top_k=20,min_p=0"`。

---

## 8. 完整脚本模板

### think + no_think 双模式并行评测

```bash
#!/bin/bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
export VLLM_USE_FLASHINFER_MOE_FP4=0

OUTPUT_BASE_DIR="/workspace/output"
mkdir -p "$OUTPUT_BASE_DIR"

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
TASKS="gsm8k,mmlu_pro"

MODELS=(
    "/path/to/model1/|1|modelopt_fp4"
    "/path/to/model2/|0|"
)

run_eval() {
    local GPU_ID=$1 MODEL_PATH=$2 USE_QUANT=$3 QUANT=$4 RUN_ID=$5 MODE=$6
    local MODEL_NAME=$(basename "${MODEL_PATH%/}")
    local OUTPUT_DIR="$OUTPUT_BASE_DIR/${MODEL_NAME}/run${RUN_ID}/${MODE}"
    local LOG_FILE="$OUTPUT_BASE_DIR/${MODEL_NAME}_run${RUN_ID}_${MODE}.log"

    if [[ "$MODE" == "think" ]]; then
        local MODEL_ARGS="pretrained=$MODEL_PATH,dtype=auto,gpu_memory_utilization=0.85,max_model_len=32768,tensor_parallel_size=1,enforce_eager=True,enable_thinking=true,think_end_token=</think>"
        local GEN_KWARGS="temperature=0.6,top_p=0.95,top_k=20,min_p=0"
        local CHAT_TEMPLATE_KWARGS='{"enable_thinking": true}'
    else
        local MODEL_ARGS="pretrained=$MODEL_PATH,dtype=auto,gpu_memory_utilization=0.85,max_model_len=32768,tensor_parallel_size=1,enforce_eager=True"
        local GEN_KWARGS="temperature=0.7,top_p=0.8,top_k=20,min_p=0"
        local CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
    fi

    [[ "$USE_QUANT" == "1" ]] && MODEL_ARGS="$MODEL_ARGS,quantization=$QUANT"

    CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval \
        --model vllm \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASKS" \
        --batch_size auto \
        --apply_chat_template \
        --chat_template_kwargs "$CHAT_TEMPLATE_KWARGS" \
        --gen_kwargs "$GEN_KWARGS" \
        --output_path "$OUTPUT_DIR" \
        --log_samples \
        --trust_remote_code \
        2>&1 | tee "$LOG_FILE"
}

for MODE in no_think think; do
    GPU_IDX=0; PIDS=()
    for MODEL_CONFIG in "${MODELS[@]}"; do
        IFS='|' read -r MODEL_PATH USE_QUANT QUANT <<< "$MODEL_CONFIG"
        run_eval "${GPUS[$GPU_IDX]}" "$MODEL_PATH" "$USE_QUANT" "$QUANT" "1" "$MODE" &
        PIDS+=($!)
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
        if [[ ${#PIDS[@]} -eq $NUM_GPUS ]]; then
            for PID in "${PIDS[@]}"; do wait "$PID"; done; PIDS=()
        fi
    done
    for PID in "${PIDS[@]}"; do wait "$PID"; done
done
```

---

## 变更记录

- **2026-03-27**：初始版本，基于 Qwen3-30B-A3B 评测经验总结
