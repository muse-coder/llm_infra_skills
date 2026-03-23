---
name: megatron-qad
description: "Megatron-LM QAD (Quantization-Aware Distillation) expert. MUST USE for any task involving Megatron-LM quantization, knowledge distillation, NVFP4, QAD training, checkpoint conversion, or modelopt integration. Covers the full pipeline: quantize, convert, finetune (KD), export."
---

# Megatron-LM QAD (Quantization-Aware Distillation) Complete Knowledge

> You are a Megatron-LM QAD expert. Apply this knowledge when helping with quantization, distillation, checkpoint management, or training configuration in Megatron-LM.

---

## What is QAD

QAD = Quantization-Aware Distillation. The pipeline:
1. Quantize HF model to NVFP4 (Student) — weights stored as BF16 + fake-quant simulation
2. Convert HF model to BF16 Megatron format (Teacher)
3. KD training: Student learns from Teacher via KL divergence on logits
4. Export: convert fake-quant BF16 weights to real FP4 for deployment

```
HF weights (BF16)
    |-- quantize.sh --> Student (BF16 + NVFP4 fake-quant + modelopt_state/)
    |-- convert.sh  --> Teacher (BF16, pure Megatron format)
    |
    v finetune.sh (QAD training, KL divergence distillation)
Trained Student checkpoint
    |
    v export.sh (forces TP=1)
Real FP4 weights --> TRT-LLM / vLLM / SGLang deployment
```

---

## Complete Pipeline (6 Steps)

All scripts run from `examples/post_training/modelopt/` directory.

### Step 1: Quantize HF -> NVFP4 Student

```bash
TP=1 EP=1 \
HF_MODEL_CKPT=${HF_MODEL_CKPT} \
MLM_MODEL_SAVE=${STUDENT_CKPT} \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./quantize.sh <Model/Config> NVFP4_DEFAULT_CFG
```

Internal flow: `quantize.sh -> arguments.sh -> conf/<Model>.sh -> quantize.py`
- Builds empty Megatron model with TP/EP sharding
- `import_mcore_gpt_from_hf()` loads HF weights, auto-shards by TP/EP rank
- `mtq.quantize()` runs PTQ calibration (512 samples from cnn_dailymail)
- Saves Megatron distributed checkpoint + `modelopt_state/` directory

### Step 2: Convert HF -> BF16 Teacher

```bash
TP=1 EP=8 \
HF_MODEL_CKPT=${HF_MODEL_CKPT} \
MLM_MODEL_SAVE=${TEACHER_CKPT} \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./convert.sh <Model/Config>
```

**CRITICAL**: Teacher TP/EP MUST match finetune TP/EP.

### Step 3: Create Teacher Model Config YAML

Manual creation required. NeMo-format YAML:

```yaml
# Map from HF config.json to NeMo YAML:
# num_hidden_layers     -> num_layers
# hidden_size           -> hidden_size
# intermediate_size     -> ffn_hidden_size
# num_attention_heads   -> num_attention_heads
# num_key_value_heads   -> num_query_groups
# head_dim              -> kv_channels
# num_experts           -> num_moe_experts
# moe_intermediate_size -> moe_ffn_hidden_size
# num_experts_per_tok   -> moe_router_topk
# max_position_embeddings -> max_position_embeddings
```

Pass via `--export-kd-teacher-model-config <path>` or place as `model_config.yaml` in teacher checkpoint root.

### Step 4: Prepare Training Data

Megatron SFTDataset expects **OpenAI format** (`role`/`content`), NOT ShareGPT format (`from`/`value`).

- HF datasets registered in `hf_dataset_to_conversation` dict auto-convert ShareGPT
- Local JSONL files use identity transform — MUST be OpenAI format already
- ShareGPT data will cause `KeyError: 'role'` if not pre-converted

### Step 5: QAD Training (Core Step)

```bash
TP=1 EP=8 \
MLM_MODEL_CKPT=${STUDENT_CKPT} \
MLM_MODEL_SAVE=${QAD_SAVE} \
MLM_DATA_ARGS="--train-samples 600000 --lr-decay-samples 600000 --lr-warmup-samples 5000 --split 100,0,0 --finetune-hf-dataset ${TRAIN_DATA}" \
MLM_OPTIM_ARGS="--lr 5.0e-6 --min-lr 1.0e-7 --lr-decay-style cosine --clip-grad 1.0 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.010 --use-distributed-optimizer" \
MLM_TRAIN_ARGS="--no-gradient-accumulation-fusion --global-batch-size 64 --micro-batch-size 1 --reset-position-ids --reset-attention-mask --eod-mask-loss --attention-dropout 0.0 --hidden-dropout 0.0 --no-check-for-nan-in-loss-and-grad" \
MLM_EVAL_ARGS="--eval-iters 1 --eval-interval 9999999 --save-interval 1000 --log-interval 10" \
MLM_EXTRA_ARGS="--export-kd-teacher-load ${TEACHER_CKPT} --export-kd-teacher-model-config ${TEACHER_CONFIG}" \
./finetune.sh <Model/Config>
```

### Step 6: Export for Deployment

```bash
PP=1 \
HF_MODEL_CKPT=${HF_MODEL_CKPT} \
MLM_MODEL_CKPT=${QAD_SAVE} \
EXPORT_DIR=${EXPORT_DIR} \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./export.sh <Model/Config>
```

**Note**: `export.sh` forces TP=1 (hardcoded lines 24-27).

---

## Parallelism Rules

### TP + EP Constraint (CRITICAL)

**modelopt's `QuantSequentialMLP` does NOT support TP>1 AND EP>1 simultaneously.**

Error: `ValueError: TP+EP is not supported by QuantSequentialMLP. Set either TP or EP to 1!`

Root cause: `get_gpt_modelopt_spec()` in `megatron/core/post_training/modelopt/gpt/model_specs.py` hardcodes `moe_grouped_gemm=False` (line 146), forcing `SequentialMLP` instead of `TEGroupedMLP`.

**Solution**: For MoE models use TP=1 EP=N. For dense models use TP=N EP=1.

### GBS Calculation

```
data_parallel_size = world_size / (TP * PP * CP)   # EP is NOT in this formula!
default GBS = micro_batch_size * data_parallel_size  # when --global-batch-size not set
gradient_accumulation = GBS / (MBS * DP)
```

EP only affects `expert_data_parallel_size` (for MoE expert parameters).

Example: TP=1, EP=8, PP=1, 8 GPUs:
- data_parallel_size = 8/1 = 8
- Default GBS = 1*8 = 8 (too small for KD!)
- With `--global-batch-size 64`: gradient_accumulation = 64/(1*8) = 8 steps

**Always explicitly set `--global-batch-size`** (recommended 64-256).

### Checkpoint Resharding

- **Model weights**: Megatron `dist_checkpointing` (torch_dist format) natively supports resharding across different TP/PP/EP
- **modelopt_state (quantization state)**: Does NOT support resharding. `restore_sharded_modelopt_state()` has no reshard capability
- **Conclusion**: Quantized checkpoints cannot change TP/EP at load time. Must re-quantize from HF.

---

## Hyperparameter Guide (from NVIDIA paper arXiv:2601.20088)

### Recommended Values

| Model Type | Learning Rate | Training Tokens | Notes |
|-----------|--------------|----------------|-------|
| SFT models | 1e-6 ~ 5e-6 | Model-dependent | LR <= original SFT LR |
| RL models | 1e-5 | Model-dependent | Higher LR needed |
| 30B-A3B class | 5e-6 ~ 1e-5 | ~2.5B tokens | ~600K samples @ seq_len=4096 |
| 9B class | 1e-6 | ~6B tokens | |
| 49B class | 1e-6 | ~0.3B tokens | Large models converge faster |

### Token Count Estimation

```
total_tokens = train_samples * seq_length
```

At seq_length=4096:
- 50,000 samples = ~0.2B tokens (INSUFFICIENT)
- 600,000 samples = ~2.5B tokens (recommended for 30B)
- 1,500,000 samples = ~6B tokens (for thorough training)

### Common Hyperparameters

```
--lr 5.0e-6              # Conservative for SFT models
--min-lr 1.0e-7
--lr-decay-style cosine
--lr-warmup-samples 5000  # Scale with total training
--clip-grad 1.0
--weight-decay 0.01
--adam-beta1 0.9
--adam-beta2 0.95
--global-batch-size 64    # Use gradient accumulation
--micro-batch-size 1
```

---

## Knowledge Distillation Loss Details

### Default Behavior

```python
# modelopt default DistillationConfig:
skip_lm_loss = True           # Original LM cross-entropy loss SKIPPED by default
kd_loss_scale = 1.0           # Only used when skip_lm_loss=False
logit_kl_temperature = 1.0    # Softmax temperature
```

**Default: only KL divergence distillation loss, no original LM loss.**

Loss types:
- Logits: KL Divergence Loss (default) or TopK KL Loss
- Intermediate layers: Cosine Similarity Loss (optional)
- Original LM: Cross-Entropy Loss (skipped by default)

### Custom Distillation Config

Create `distill_config.yaml`:
```yaml
logit_layers: ["output_layer", "output_layer"]   # [student, teacher]
intermediate_layer_pairs:                          # Optional
  - ["decoder.layers.0.input_layernorm", "decoder.layers.0.input_layernorm"]
skip_lm_loss: false          # Include original LM loss
kd_loss_scale: 1.0           # KD loss weight
logit_kl_temperature: 1.0    # Temperature
```

Use: `--export-kd-distill-cfg distill_config.yaml`

---

## Troubleshooting

### Poor Training Results Checklist

| Issue | Check | Fix |
|-------|-------|-----|
| Insufficient training | train_samples * seq_length vs recommended tokens | Increase train-samples (10x+) |
| GBS too small | Log output "global batch size" | Set --global-batch-size 64+ |
| LR too high | SFT model with LR > 1e-5 | Lower to 5e-6 or 1e-6 |
| Data format wrong | KeyError on "role" | Convert ShareGPT to OpenAI format |
| Teacher config wrong | Check teacher_model_config.yaml exists | Verify field name mapping |
| Eval-domain mismatch | Code data training vs general eval | Not necessarily "bad" |

### Known Issues

1. **TP+EP not supported** (quantize/finetune): Use TP=1 for MoE models
2. **OOM with distillation**: Don't use `--manual-gc`. Enable recompute if needed
3. **~40% slower iterations**: Known CUDA kernel issue with KD, expected
4. **export.sh forces TP=1**: Hardcoded, cannot change
5. **Interleaved PP unsupported**: Use non-interleaved PP only
6. **modelopt_state no resharding**: Re-quantize from HF to change TP/EP. Comment in prune.py: "WAR till modelopt 0.39"

---

## Key File Index

```
examples/post_training/modelopt/
  conf/arguments.sh              # Common arg parsing, sources model config
  conf/Qwen/Qwen3-30B-A3B.sh    # Model architecture params (MODEL_ARGS)
  quantize.sh / quantize.py      # Step 1: HF -> NVFP4 Student
  convert.sh / convert_model.py  # Step 2: HF -> BF16 Teacher
  finetune.sh / finetune.py      # Step 5: QAD training
  export.sh / export.py          # Step 6: Export for deployment
  distillation.md                # KD documentation

megatron/
  post_training/arguments.py     # --export-kd-* flag definitions
  post_training/model_builder.py # Distillation model setup (lines 313-354)
  post_training/checkpointing.py # load_modelopt_checkpoint / load_modelopt_state
  post_training/loss_func.py     # KD loss computation
  training/arguments.py          # GBS/MBS calculation (lines 562-568)
  core/parallel_state.py         # DP/EP parallel group calculation
  core/post_training/modelopt/gpt/model_specs.py  # moe_grouped_gemm=False hardcode

# modelopt library (external)
modelopt/torch/quantization/plugins/megatron.py  # QuantSequentialMLP TP+EP limit (line 558)
modelopt/torch/distill/plugins/megatron.py        # DistillationConfig defaults
```
