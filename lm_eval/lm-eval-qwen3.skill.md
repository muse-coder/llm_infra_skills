---
name: lm-eval-qwen3
description: "处理 lm-evaluation-harness 评测任务时必须使用。涵盖 Qwen3 模型的 think/no_think 模式切换、采样参数配置、--apply_chat_template 与 --chat_template_kwargs 的正确用法、--gen_kwargs 设置、vLLM 后端 model_args 配置、多 GPU 并行评测脚本编写、NVFP4 量化模型评测注意事项。适用于 GSM8K、MMLU Pro、AIME25 等评测任务。"
---

# LM-Evaluation-Harness + Qwen3 评测知识

> 你是 LLM 评测专家。当用户需要使用 lm-evaluation-harness 对 Qwen3 系列模型进行评测、编写评测脚本、调试评测参数时，应用此知识。

---

## 核心规则（CRITICAL）

1. **Qwen3 thinking 模式禁止 greedy decoding**：必须设置采样参数，否则会导致性能下降和无限重复
2. **no_think 模式必须显式关闭 thinking**：Qwen3 chat template 默认 `enable_thinking=True`，不显式传 `false` 就等于 think 模式
3. **采样参数通过 `--gen_kwargs` 传入**，不是 `--model_args`
4. **`--apply_chat_template` 对 Qwen3 instruct 模型是必要的**，`--chat_template_kwargs` 依赖它才能生效

---

## 采样参数（官方推荐）

| 模式 | Temperature | TopP | TopK | MinP |
|------|------------|------|------|------|
| think (`enable_thinking=True`) | `0.6` | `0.95` | `20` | `0` |
| no_think (`enable_thinking=False`) | `0.7` | `0.8` | `20` | `0` |

---

## think/no_think 模式完整配置

### think 模式

```bash
MODEL_ARGS="pretrained=$MODEL_PATH,dtype=auto,gpu_memory_utilization=0.85,max_model_len=32768,tensor_parallel_size=1,enforce_eager=True,enable_thinking=true,think_end_token=</think>"

lm_eval \
    --model vllm \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size auto \
    --apply_chat_template \
    --chat_template_kwargs '{"enable_thinking": true}' \
    --gen_kwargs "temperature=0.6,top_p=0.95,top_k=20,min_p=0" \
    --output_path "$OUTPUT_DIR" \
    --log_samples \
    --trust_remote_code
```

### no_think 模式

```bash
MODEL_ARGS="pretrained=$MODEL_PATH,dtype=auto,gpu_memory_utilization=0.85,max_model_len=32768,tensor_parallel_size=1,enforce_eager=True"

lm_eval \
    --model vllm \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size auto \
    --apply_chat_template \
    --chat_template_kwargs '{"enable_thinking": false}' \
    --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,min_p=0" \
    --output_path "$OUTPUT_DIR" \
    --log_samples \
    --trust_remote_code
```

---

## 三层控制关系

| 层级 | 参数位置 | 作用 |
|------|---------|------|
| vLLM 引擎层 | `--model_args` 中 `enable_thinking=true,think_end_token=</think>` | 告诉 vLLM 处理 thinking token |
| Chat Template 层 | `--chat_template_kwargs '{"enable_thinking": true/false}'` | 控制 tokenizer 是否在 prompt 中启用 thinking |
| 采样层 | `--gen_kwargs "temperature=0.6,..."` | 控制生成时的采样策略 |

三层必须协调一致，think 模式三层都开，no_think 模式 vLLM 层不传 thinking 参数、chat template 层显式关闭。

---

## 量化模型配置

```bash
# 自行量化的模型（PTQ/QAD）：需要传 quantization
MODEL_ARGS="...,quantization=modelopt_fp4"

# 官方预量化模型（如 Qwen3-30B-A3B-NVFP4）：不需要传 quantization
MODEL_ARGS="pretrained=$MODEL_PATH,dtype=auto,..."
```

---

## 多 GPU 并行评测要点

- 不要设置全局 `export CUDA_VISIBLE_DEVICES`，在每个任务前用 `CUDA_VISIBLE_DEVICES=$GPU_ID lm_eval ...` 动态分配
- 使用 `&` 后台启动 + `wait` 等待，满一轮 GPU 后等待当前批次完成再继续
- 每个模型独占一张卡时 `tensor_parallel_size=1`

---

## 环境变量（离线环境）

```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# vLLM NVFP4 优化
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
export VLLM_USE_FLASHINFER_MOE_FP4=0
```

---

## 常见陷阱

| 陷阱 | 原因 | 解决方案 |
|------|------|---------|
| no_think 和 think 结果一样 | chat template 默认 `enable_thinking=True` | 必须传 `--chat_template_kwargs '{"enable_thinking": false}'` |
| 模型输出无限重复 | thinking 模式使用了 greedy decoding | 设置 `--gen_kwargs` 采样参数 |
| 不加 chat template 也能得高分 | vLLM 层 `enable_thinking` 独立生效 + few-shot 提供格式 | 不规范，建议始终加 `--apply_chat_template` |

---

## 详细文档

完整参数说明、脚本模板和踩坑记录见：`lm_eval/LM_Eval_Qwen3_Complete_Guide.md`
