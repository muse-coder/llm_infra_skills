# NVFP4 Dense Fusion 优化计划

> 场景: Qwen3.5-27B-Dense，NVFP4，TP=1，batch size=1，prefill token=8192。  
> 核心目标: 优化 FP4 GEMM 前的 producer 链路，而不是只优化单个 RMSNorm 或 quant kernel。

---

## 1. 总体思路

当前 NVFP4 Dense prefill 的最大热点是 FP4 GEMM，但 GEMM 前还有大量 producer 开销：

```text
RMSNorm / residual / gate / activation
  ↓
FP4 activation quant
  ↓
layout / copy / pack
  ↓
FP4 GEMM
```

优化目标是把这些 GEMM input producer 尽量融合成：

```text
FusedProducerQuantPack
  input:
    bf16 hidden_states / residual / gate / norm weight
  output:
    FP4 packed activation
    FP4 block scale
    layout already matches target GEMM backend
  ↓
FP4 GEMM
```

也就是说，主线不是“单独优化 `triton_red_fused_*`”，而是：

```text
norm + residual + activation/gate + quant + GEMM input packing
```

一起融合。

---

## 2. 当前收益池

当前 prefill window 约为：

```text
execute_context_1(8192)_generation_0(0)
  wall time:       366.56 ms
  GPU kernel sum:  356.53 ms
```

与 fusion 相关的主要成本：

| 模块 | 当前时间 | 说明 |
|---|---:|---|
| `triton_red_fused_*` | 18.43 ms | RMSNorm / residual / 部分 FP4Q reduction |
| standalone `cvt_fp16_to_fp4` | 10.48 ms | 独立 FP4 activation quant |
| `silu_mul_cvt_fp16_to_fp4` | 17.04 ms | MLP activation + down_proj 输入量化 |
| norm / gate / fused quant Triton | 12.10 ms | GDN output gated norm + quant 等 |
| layout / copy / small pointwise | 数 ms | reshape、copy、index、packing 边界 |

现实目标不是把这些都消掉，而是减少重复读写、减少 standalone quant、减少 GEMM 前 layout/copy。合理预期：

```text
保守收益:  prefill 1.5% - 2.5%
较好收益:  prefill 3% - 5%
更高收益:  需要同时改善 GEMM backend / tile，本计划不覆盖
```

---

## 3. 优先级计划

### P0. RMSNorm / Residual + FP4Q + Pack

这是最通用的融合点，覆盖 attention input、MLP input、下一层 input。

当前链路：

```text
hidden_states + optional residual
  ↓
GemmaRMSNorm
  ↓
FP4Q / scaled_fp4_quant
  ↓
layout / copy / pack
  ↓
qkv / qkvz / ba / gate_up GEMM
```

目标链路：

```text
FusedRMSNormResidualQuantPack
  input:
    hidden_states bf16
    optional residual bf16
    norm weight
  output:
    packed FP4 activation
    block scale
  ↓
FP4 GEMM
```

适用位置：

```text
1. input_layernorm → attention projection
2. post_attention_layernorm → MLP gate_up_proj
3. next layer input_layernorm → next attention projection
```

观察指标：

```text
1. triton_red_fused_* 总时间是否下降
2. standalone cvt_fp16_to_fp4 次数是否下降
3. GEMM 前 copy/layout 小 kernel 是否减少
4. FP4 GEMM 时间不能变差
```

---

### P1. GDN Output RMSNormGated + FP4Q + Pack

Linear attention output 是第二个重点。

当前链路：

```text
GDN core output
  ↓
state writeback / DtoD memcpy / layout
  ↓
RMSNormGated(core_attn_out, z)
  ↓
silu(z) gate
  ↓
FP4Q
  ↓
out_proj GEMM
```

目标链路：

```text
FusedGatedNormQuantPack
  input:
    core_attn_out
    z
    norm weight
  output:
    packed FP4 activation
    block scale
  ↓
out_proj GEMM
```

当前 trace 中，GDN core 到 fused norm path 之间有稳定边界：

```text
chunk_fwd_kernel_o -> triton_poi_fused_0
  avg gap: 149.9 us
  total:   3.60 ms
```

注意这段不能简单归因成 Python overhead，大部分是 state writeback / DtoD memcpy，但它说明 GDN output 到 out_proj input 之间仍有结构性边界。

优化方向：

```text
1. 让 GDN output layout 更接近 out_proj 输入 layout。
2. 把 RMSNormGated + silu(z) + FP4Q + pack 做成稳定 fused producer。
3. 如果可能，把部分 post-processing 下沉成 GDN epilogue。
```

---

### P2. MLP SiluMul + FP4Q + DownProj Pack

当前 MLP activation 已经有 fused kernel：

```text
gate_up_proj GEMM
  ↓
vllm::silu_mul_cvt_fp16_to_fp4
  ↓
down_proj GEMM
```

下一步不是再融合 SiLU，而是检查它输出的 FP4 layout 是否就是 `down_proj` GEMM 最优输入 layout。

目标链路：

```text
FusedSiluMulQuantPack
  input:
    gate_up output bf16
  output:
    packed FP4 activation
    block scale
    layout matches down_proj GEMM tile
  ↓
down_proj GEMM
```

观察指标：

```text
1. silu_mul_cvt_fp16_to_fp4 时间是否下降
2. down_proj 前是否还有额外 layout / copy kernel
3. down_proj GEMM 时间是否保持不变或下降
```

---

### P3. Shared Quant for Linear Attention Input Projections

Linear attention input 有两个 projection：

```text
normalized hidden_states
  ├─ in_proj_qkvz
  └─ in_proj_ba
```

如果两路各自做 activation quant，则可以考虑共享 producer：

```text
FusedNormQuantPack once
  ↓
shared FP4 activation + scale
  ├─ in_proj_qkvz GEMM
  └─ in_proj_ba GEMM
```

如果两路 projection 当前同 stream 串行，还可以评估 dual-stream overlap：

```text
stream 0: in_proj_qkvz GEMM
stream 1: in_proj_ba GEMM
sync before GDN core
```

注意：`in_proj_ba` 本身是 tiny GEMM，收益上限不高；这个方向主要用于减少重复 quant 和 critical path 小开销。

---

### P4. QK Norm + Partial RoPE Fusion

Full attention 中还有一个小的 fusion gap：

```text
qkv_proj GEMM
  ↓
split q / k / v / gate
  ↓
QK norm
  ↓
partial RoPE
  ↓
FlashAttention
```

当前 trace 可见多个小 kernel：

```text
triton_red_fused_7
triton_poi_fused__...rms_norm...split...cat...
```

聚合约：

```text
QK norm / RoPE-ish kernels:
  total: 2.22 ms
  count: 36
```

目标：

```text
FusedQKNormPartialRoPE
  input:
    q, k
    q_norm / k_norm weight
    cos / sin
  output:
    q_rope
    k_rope
```

这个方向收益较小，但工程边界清晰。重点检查：

```text
1. QKNormRoPEFusionPass 是否启用。
2. partial_rotary_factor=0.25 是否被 pattern 覆盖。
3. attn_output_gate=True 导致的 split/view/cat 是否破坏 pattern match。
```

---

### P5. GDN Chunk 内部 Fusion

GDN core 内部是一串严格串行 kernel：

```text
_causal_conv1d_fwd_kernel
  ↓
_fused_post_conv_kernel
  ↓
chunk_scaled_dot_kkt_fwd_kernel
  ↓
merge_16x16_to_64x64_inverse_kernel
  ↓
recompute_w_u_fwd_kernel
  ↓
chunk_gated_delta_rule_fwd_kernel_h_blockdim64
  ↓
chunk_fwd_kernel_o
```

可考虑的融合：

```text
1. post_conv + chunk input prep
2. kkt + merge + recompute_w_u
3. chunk_gated_delta_rule + chunk output epilogue
```

这类改动更底层，风险高于 P0-P2。建议等 GEMM input producer fusion 做出稳定收益后再推进。

---

## 4. 推荐实施顺序

```text
Step 1: 做 RMSNormResidualQuantPack 原型
  覆盖 attention input 和 MLP input。
  目标是减少 standalone cvt_fp16_to_fp4 和 GEMM 前 layout/copy。

Step 2: 做 GDN GatedNormQuantPack
  覆盖 linear attention out_proj 前的 RMSNormGated + silu(z) + FP4Q。
  同时观察 GDN core 后 gap 是否下降。

Step 3: 检查 SiluMulQuantPack 输出 layout
  不一定重写 silu_mul kernel，先确认 down_proj 前是否仍有额外 packing/copy。

Step 4: 处理 shared quant / dual-stream projection
  只在确认 in_proj_qkvz / in_proj_ba 存在重复 quant 或串行 critical path 后再做。

Step 5: 修 QK Norm + RoPE fusion
  作为 full attention 的小收益优化。

Step 6: 再考虑 GDN chunk 内部 fusion
  收益可能更高，但风险和验证成本也更高。
```

---

## 5. 验证方式

每个 fusion 实验都需要同时看端到端和 trace：

```text
端到端:
  - TTFT
  - prefill wall time
  - decode TPOT 是否回退

trace group:
  - triton_red_fused_* total time
  - cvt_fp16_to_fp4 count / time
  - silu_mul_cvt_fp16_to_fp4 time
  - norm / gate / fused quant Triton time
  - layout / copy / triton_poi 小 kernel count
  - FP4 GEMM total time

正确性:
  - 输出 token 稳定
  - FP4 GEMM 没有 fallback 到 bf16
  - quant scale 和原路径数值一致或在可接受误差内
```

一个 fusion 被认为有效，至少要满足：

```text
1. prefill wall time 稳定下降。
2. 对应 producer kernel 和 standalone quant 时间下降。
3. FP4 GEMM 时间不变差。
4. 没有新增更多 copy/layout kernel 抵消收益。
```

---

## 6. 简短结论

最优先做的是：

```text
RMSNorm / residual / gate / activation
  + FP4 quant
  + GEMM input packing
```

也就是 **GEMM-input producer fusion**。

推荐先做：

```text
P0: RMSNormResidualQuantPack
P1: GDN GatedNormQuantPack
P2: SiluMulQuantPack layout check
```

这三项覆盖了 NVFP4 Dense prefill 中最常见的 GEMM 前 producer 边界。现实收益目标可以先定在 **prefill 2-4%**；如果 packing layout 与 GEMM backend 协同做得好，再尝试冲击 **5%** 左右。
