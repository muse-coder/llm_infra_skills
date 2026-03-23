---
name: fsdp-qad
description: "HuggingFace FSDP2 QAD (Quantization-Aware Distillation) expert. MUST USE for any task involving HF-based QAT/QAD with accelerate, FSDP2, QATTrainer, QADTrainer, or modelopt HF integration. Covers quantization, distillation, FSDP2 config, and deployment export."
---

# HuggingFace FSDP2 QAD Complete Knowledge

> You are an HF FSDP2 QAD expert. Apply this knowledge when helping with HuggingFace-based quantization-aware training/distillation using accelerate, FSDP2, and modelopt.

---

## What is HF FSDP2 QAD

Alternative to the Megatron-LM QAD pipeline. Uses HuggingFace transformers + accelerate + modelopt for QAT/QAD on HF models directly, without converting to Megatron format.

```
HF model (BF16)
    |
    v  QATTrainer (lazy quantization on first training step)
    |  mtq.quantize() → calibrate → fake-quant student
    |
    |  QADTrainer adds: teacher model + KD loss (KL divergence)
    |  FSDP2 shards both student + teacher across GPUs
    |
    v  Training loop: student learns from teacher
    |
    v  export.py → TRT-LLM / vLLM / SGLang checkpoint
```

**When to use HF FSDP2 QAD vs Megatron QAD:**

| | HF FSDP2 QAD | Megatron QAD |
|---|---|---|
| Models | Dense HF models (Llama, Qwen dense) | Dense + MoE (includes EP support) |
| MoE support | Limited (no EP) | Native (TP=1 EP=N) |
| Setup complexity | Low (pip install + accelerate config) | High (Megatron-LM + model conversion) |
| Parallelism | FSDP2 (data parallelism) | TP + EP + PP + CP |
| Checkpoint format | HF safetensors | Megatron distributed checkpoint |

---

## Architecture

### Class Hierarchy

```
transformers.Trainer
    └── ModelOptHFTrainer          # modelopt/torch/opt/plugins/
        ├── QATTrainer             # quantization/plugins/transformers_trainer.py
        │   └── QADTrainer         # (inherits both QATTrainer + KDTrainer)
        └── KDTrainer              # distill/plugins/huggingface.py
```

`QADTrainer` = `QATTrainer` (quantization) + `KDTrainer` (distillation) via multiple inheritance.

### Key Components

| Component | File | Role |
|-----------|------|------|
| QATTrainer | `modelopt/torch/quantization/plugins/transformers_trainer.py` | Lazy quantization, FSDP2 patches, modelopt state save/load |
| QADTrainer | Same file (line 413) | Combines QAT + KD, wraps calibration with `hide_teacher_model()` |
| KDTrainer | `modelopt/torch/distill/plugins/huggingface.py` | `mtd.convert()`, KD loss, teacher save/restore |
| LMLogitsLoss | Same file (line 119) | KL divergence on logits with temperature, per-token masking |
| DistillationModel | `modelopt/torch/distill/distillation_model.py` | Meta-model wrapping student + teacher |

---

## Complete Pipeline

### Step 1: Fine-tune BF16 (Optional but Recommended)

```bash
./launch.sh --model meta-llama/Meta-Llama-3-8B \
    --num_epochs 2.0 --lr 1e-5 \
    --output_dir llama3-finetune
```

### Step 2: QAT (Quantize + Train)

```bash
./launch.sh --model llama3-finetune \
    --num_epochs 2.0 --lr 1e-5 \
    --quant_cfg NVFP4_DEFAULT_CFG \
    --output_dir llama3-qat
```

### Step 3: QAD (Quantize + Distill)

```bash
./launch.sh --model llama3-finetune \
    --num_epochs 3 --lr 4e-5 \
    --quant_cfg NVFP4_DEFAULT_CFG \
    --distill True \
    --output_dir llama3-qad
```

### Step 4: Export

```bash
python export.py --pyt_ckpt_path llama3-qad --export_path llama3-qad-deploy
```

---

## Internal Flow (What Happens Under the Hood)

### QADTrainer.__init__()

```
1. QATTrainer.__init__():
   - Resolves quant_cfg (string → mtq config dict)
   - Adds LoRA adapter if specified (before quantization)
   - Patches FSDP2: _patch_fsdp2_post_backward() + _patch_accelerate_for_fsdp2_fix()
   - Checks for existing modelopt_state_train.pth → restore or save

2. KDTrainer.__init__():
   - Asserts FSDP2 (FSDP1 raises ValueError)
   - mtd.convert(model, mode=[("kd_loss", distill_config)])
     → Wraps model as DistillationModel with teacher
     → Teacher is frozen (requires_grad_(False))
```

### First training_step() — Lazy Quantization

```
QATTrainer.training_step():
   if not is_quantized(model):
       QADTrainer._quantize_model():
           model = accelerator.unwrap_model(self.model)
           with model.hide_teacher_model(), model.only_student_forward():
               # Calibration: forward pass on calib_size samples (default 512)
               mtq.quantize(student, quant_cfg, forward_loop)
           # Save quantizer state for FSDP2 checkpoint recovery
           self._save_modelopt_state_with_weights()
```

### Training Loop — KD Loss

```
KDTrainer.train() sets compute_loss_func = _compute_kd_loss:
   1. Student forward: normal forward pass (with fake-quant)
   2. Teacher forward: torch.no_grad(), eval mode
   3. LMLogitsLoss:
      soft_log_probs = log_softmax(student_logits / T)
      soft_targets  = softmax(teacher_logits / T)
      kd_loss = T² × KL_div(soft_log_probs, soft_targets)
   4. Per-token masking: loss_mask = (labels != -100)
      final_loss = (kd_loss * loss_mask).sum() / loss_mask.sum()
```

---

## FSDP2 Specifics

### Required Config (accelerate_config/fsdp2.yaml)

```yaml
distributed_type: FSDP
fsdp_config:
  fsdp_version: 2
  fsdp_activation_checkpointing: true
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer  # Override per model
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
mixed_precision: bf16
```

### FSDP2 Patches (Applied Automatically by QATTrainer)

**Patch 1 — Post-backward gradient dtype fix:**
FSDP2 bf16 mixed precision upcasts params to fp32 for optimizer, but gradients stay bf16. PyTorch >= 2.6 rejects this dtype mismatch. The patch:
- Sets `grad_dtype=None` before reduction (allows bf16 assignment)
- Casts gradients to parameter dtype after reduction

**Patch 2 — Accelerate prepare fix for quantized models:**
FSDP2 assumes all buffers are sharded, but TensorQuantizer adds non-sharded buffers. The patch:
- Temporarily marks quantizer buffers as non-persistent during `accelerate.prepare()`
- Restores original buffer settings after wrapping

### FSDP2 Constraints

- **FSDP1 NOT supported for distillation** — `KDTrainer.__init__()` raises ValueError
- **Distillation disables `fsdp_cpu_ram_efficient_loading`** — launch.sh sets it to False
- **SHARDED_STATE_DICT during training** → **FULL_STATE_DICT for final save** (automatic)
- **QLoRA export not supported with FSDP2** — use DDP for QLoRA export

### Transformer Layer Class by Model

```
Llama:  LlamaDecoderLayer
Qwen3:  Qwen3DecoderLayer
```

Pass via: `--fsdp_transformer_layer_cls_to_wrap <ClassName>`

---

## Dataset Handling

### Daring-Anteater (Default Dataset)

Format: ShareGPT — `{"conversations": [{"from": "User", "value": "..."}, ...], "system": "...", "mask": "User"}`

`utils.py` `get_daring_anteater()` processes directly:
- Reads `from`/`value` fields, masks User turns (labels = -100), keeps Assistant turns
- Pads/truncates to `model_max_length`
- For local datasets: auto-detects `dataset_info.json` (load_from_disk) or `train.jsonl` (load_dataset)

### Local Dataset Support

```python
# In utils.py get_daring_anteater():
if os.path.exists(os.path.join(dataset_dir, "dataset_info.json")):
    dataset = datasets.load_from_disk(dataset_dir)      # save_to_disk format
elif os.path.exists(os.path.join(dataset_dir, "train.jsonl")):
    dataset = datasets.load_dataset("json", ...)["train"]  # raw JSONL
```

---

## Distillation Config

### Basic Usage

```python
from modelopt.torch.distill.plugins.huggingface import LMLogitsLoss

distill_config = {
    "teacher_model": teacher_model,          # AutoModelForCausalLM instance (bf16)
    "criterion": LMLogitsLoss(),             # Default: temperature=1.0, reduction="none"
}

trainer = QADTrainer(
    model=student_model,
    args=training_args,
    quant_args=quant_args,
    distill_config=distill_config,
    **data_module,
)
```

### Custom Temperature

```python
distill_config = {
    "teacher_model": teacher_model,
    "criterion": LMLogitsLoss(temperature=2.0),
}
```

### KDLossConfig Fields

```python
class KDLossConfig:
    teacher_model: ModelLike          # Required: teacher model instance
    criterion: Loss                   # Required: loss function (LMLogitsLoss)
    loss_balancer: Any | None         # Optional: multi-loss balancer
    expose_minimal_state_dict: bool   # Default True: hide teacher from checkpoint
                                      # Set False for FSDP if needed
```

---

## Quantization Configs

```python
import modelopt.torch.quantization as mtq

mtq.NVFP4_DEFAULT_CFG                   # NVFP4 weight + activation (recommended)
mtq.FP8_DEFAULT_CFG                     # FP8 per-tensor
mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG       # FP8 per-channel weight, per-token activation
mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG      # INT4 blockwise weight-only
mtq.INT8_DEFAULT_CFG                    # INT8 per-channel weight, per-tensor activation
mtq.MXFP8_DEFAULT_CFG                  # MXFP8
```

---

## Checkpoint Save/Restore

### Automatic (via `mto.enable_huggingface_checkpointing()`)

Called once at startup. Patches `save_pretrained()` / `from_pretrained()`:
- Save: writes `modelopt_state.pth` alongside model files
- Load: auto-restores modelopt state after model init

### FSDP2 Specific

QATTrainer saves `modelopt_state_train.pth` separately (includes quantizer weights):
```python
modelopt_state = mto.modelopt_state(model)
modelopt_state["modelopt_state_weights"] = get_quantizer_state_dict(model)
torch.save(modelopt_state, "modelopt_state_train.pth")
```

On resume: restores modelopt state + quantizer weights before training continues.

### save_model() Flow

```
QATTrainer.save_model():
  if FSDP and not FULL_STATE_DICT:
      switch to FULL_STATE_DICT temporarily
  KDTrainer.save_model():
      with model.hide_teacher_model(), model.hide_loss_modules():
          model.save_pretrained(output_dir)  # Only student weights saved
          tokenizer.save_pretrained(output_dir)
  update config.json dtype back to original (FSDP may upcast to fp32)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `FSDP1 is not supported for distillation` | Using FSDP1 backend | Set `--backend fsdp2` or use fsdp2.yaml |
| OOM during calibration | Calibration loads full model | Reduce `--calib_size` (default 512) |
| OOM during QAD training | Two models in memory | Enable gradient_checkpointing (auto in fsdp2.yaml) |
| RuntimeError: grad dtype mismatch | PyTorch >= 2.6 FSDP2 bug | QATTrainer auto-patches; update modelopt if missing |
| `fsdp_cpu_ram_efficient_loading` error | Distillation conflict | launch.sh auto-disables for distill mode |
| Perplexity not improving | LR too high or too few epochs | Try LR 1e-5, 2-3 epochs, larger batch size |
| Export fails with FSDP2 QLoRA | Not supported | Use DDP backend for QLoRA export |

---

## Key File Index

```
Model-Optimizer/
  examples/llm_qat/
    launch.sh                      # Entry point: accelerate launch
    main.py                        # Model loading, trainer setup, train/eval
    utils.py                       # Dataset loading (Daring-Anteater), LoRA config
    export.py                      # Export to TRT-LLM/vLLM/SGLang checkpoint
    accelerate_config/fsdp2.yaml   # FSDP2 accelerate config

  modelopt/torch/
    quantization/plugins/
      transformers_trainer.py      # QATTrainer, QADTrainer, FSDP2 patches
    distill/plugins/
      huggingface.py               # KDTrainer, LMLogitsLoss
    distill/
      distillation_model.py        # DistillationModel (student+teacher wrapper)
      losses.py                    # LogitsDistillationLoss (KL divergence base)
      config.py                    # KDLossConfig
    opt/plugins/
      huggingface.py               # enable_huggingface_checkpointing()
```
