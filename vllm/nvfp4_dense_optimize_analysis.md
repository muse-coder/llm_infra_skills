# NVFP4 Dense Prefill 优化分析

> 场景: Qwen3.5-27B-Dense，NVFP4 量化，TP=1，batch size=1，prefill token=8192，decode token=1。
> Trace: `rank0.1778566047660882823.pt.trace.json` 为 torch.compile / CUDA graph 路径，`eager.1778572740212119173.pt.trace.json` 为 eager 对照路径。

---

## 1. Prefill 热点概览

本次分析以 compiled trace 中的 prefill window 为准：

```text
execute_context_1(8192)_generation_0(0)
  wall time:       366.56 ms
  GPU kernel sum:  356.53 ms
  GPU memcpy sum:    3.27 ms
```

prefill 阶段 GPU 时间主要集中在 5 类路径：

| 优先级 | 模块 | GPU 时间 | 占 prefill wall time | 典型 kernel |
|---|---:|---:|---:|---|
| P0 | FP4 GEMM / Linear / MLP / Proj | 203.63 ms | 55.5% | `_ZN7cutlass...GemmUniversal...`, `vllm::flashinfer_mm_fp4` |
| P1 | GDN linear attention core / conv | 55.36 ms | 15.1% | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o`, `_causal_conv1d_fwd_kernel` |
| P2 | Full Attention / FlashAttention | 31.92 ms | 8.7% | `flash::flash_fwd_splitkv_kernel` |
| P3 | FP4 activation quant | 26.43 ms | 7.2% | `vllm::cvt_fp16_to_fp4`, `scaled_fp4_quant` |
| P4 | MLP activation + FP4 quant | 17.04 ms | 4.6% | `vllm::silu_mul_cvt_fp16_to_fp4` |
| P5 | Norm / gate / fused quant Triton | 12.10 ms | 3.3% | `triton_*rms_norm*`, `triton_per_fused_1` |

结论：prefill 的第一优化目标不是 attention，而是 **NVFP4 Dense Linear/GEMM 路径**。GDN linear attention 和 FP4 activation quant 是第二梯队，Full Attention 在 8192 prefill 下有成本，但不是最大头。

---

## 2. P0: FP4 GEMM 路径

### 2.1 现象

compiled trace 中 FP4 GEMM 累计：

```text
CUTLASS / FlashInfer FP4 GEMM
  count:     149
  total:     203.63 ms
  avg:       1.37 ms / launch
  share:     55.5% of prefill wall time
```

这些 GEMM 覆盖以下代码逻辑：

```text
Linear Attention:
  in_proj_qkvz
  in_proj_ba
  out_proj

Full Attention:
  qkv_proj
  o_proj

Dense MLP:
  gate_up_proj
  down_proj

Logits path:
  final norm / lm_head 相关路径不属于主要 prefill 热点
```

NVFP4 配置中 `targets=["Linear"]`，因此主要 Linear 输入会走：

```text
bf16 activation
  ↓
FP4 activation quant + scale
  ↓
FP4 weight + FP4 activation GEMM
  ↓
bf16 / fp32 accumulator output
```

### 2.2 优化方向

优先做 backend 与 shape 维度的 A/B，而不是先改上层模型逻辑：

```text
候选方向:
  1. 对比 flashinfer_cutlass / flashinfer_b12x / altrex 的 GEMM kernel time
  2. 固定 input_len=8192、output_len=2，多轮重复，避免单次 TTFT 抖动误判
  3. 分别统计 prefill window 内:
       - FP4 GEMM total time
       - FP4 GEMM launch count
       - avg / p50 / p90 GEMM duration
  4. 关注 MLP gate_up_proj / down_proj 的大矩阵 shape 是否命中最优 tile
```

当前已有单次 benchmark 显示：

| backend | TTFT |
|---|---:|
| `flashinfer_cutlass` | 791.55 ms |
| `atrex` | 795.23 ms |
| `flashinfer_b12x` | 824.28 ms |

这说明在当前 workload 下，`flashinfer_cutlass` 并不比其它 backend 差，甚至单次 TTFT 最好。后续应该基于多轮 trace 里的 GEMM kernel time 做判断，不能只凭 backend 名称预设结论。

### 2.3 可验证实验

```text
实验 A: 固定 workload 多轮 backend 对比
  input_len:       8192
  output_len:      2
  concurrency:     1
  backend:         flashinfer_cutlass / flashinfer_b12x / altrex
  观察指标:
    - TTFT
    - prefill window wall time
    - FP4 GEMM total time
    - FP4 activation quant total time

实验 B: 按 Linear 类型拆 GEMM
  目标:
    区分 MLP GEMM、attention projection GEMM、out_proj GEMM 的贡献
  方法:
    使用 kernel 相邻关系或 Python stack / cpu_op external id 归因
  观察:
    如果 MLP GEMM 占绝对大头，应优先看 gate_up/down 的 tile 和 quant packing。
```

---

## 3. P1: GDN Linear Attention Core

### 3.1 现象

linear attention 的 GDN core 在 prefill 中累计约 55.36 ms。单层 linear attention 的核心链路大致是：

```text
mixed_qkvz, ba = FP4 GEMM(hidden_states)
  ↓
split / reshape:
  q, k, v, z
  b, a
  ↓
_causal_conv1d_fwd_kernel
  ↓
_fused_post_conv_kernel
  ↓
chunk_scaled_dot_kkt_fwd_kernel
chunk_gated_delta_rule_fwd_kernel_h_blockdim64
recompute_w_u_fwd_kernel
merge_16x16_to_64x64_inverse_kernel
chunk_fwd_kernel_o
  ↓
RMSNormGated(core_attn_out, z)
  ↓
FP4 quant
  ↓
out_proj FP4 GEMM
```

对应代码位置：

```text
vllm/model_executor/layers/mamba/gdn_linear_attn.py

forward_cuda():
  Part 1: Input Projection
  Part 2: Core Attention (torch.ops.vllm.gdn_attention_core)
  Part 3: Output Projection
```

### 3.2 主要成本

prefill 中较大的 GDN kernel：

| kernel | GPU 时间 | count | 说明 |
|---|---:|---:|---|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 12.07 ms | 24 | GDN chunk rule 主体 |
| `chunk_fwd_kernel_o` | 10.33 ms | 24 | 输出累积 |
| `merge_16x16_to_64x64_inverse_kernel` | 9.46 ms | 24 | layout / block merge |
| `recompute_w_u_fwd_kernel` | 8.84 ms | 24 | 中间量 recompute |
| `_fused_post_conv_kernel` | 7.47 ms | 24 | conv 后 q/k/v/g/beta 准备 |
| `_causal_conv1d_fwd_kernel` | 7.19 ms | 24 | linear attention 的短卷积 |

这里的优化空间主要不是单个小 pointwise，而是 **减少 GDN prefill core 的 kernel 链长度和中间 memory traffic**。

### 3.3 实际执行序列与串行边界

GDN prefill core 不是一个单 kernel，而是一串严格串行的 kernel。当前 trace 中典型顺序如下：

```text
_causal_conv1d_fwd_kernel
  ↓
_fused_post_conv_kernel
  ↓
chunk_local_cumsum_scalar_kernel
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
  ↓
state writeback / index scatter / DtoD memcpy
  ↓
triton_poi_fused_0
triton_per_fused_1
triton_poi_fused__...rms_norm...scaled_fp4_quant_silu..._2
  ↓
out_proj 前后的 FP4Q / GEMM
```

这里的关键点是：`merge_16x16_to_64x64_inverse_kernel`、`recompute_w_u_fwd_kernel`、`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`、`chunk_fwd_kernel_o` 之间没有明显并行空间，优化方向应该是 **减少中间写回 / layout transform / recompute**，而不是期望它们天然 overlap。

trace 中也能看到 GDN core 输出后到 `RMSNormGated + silu(z) + FP4Q` 之间存在一个稳定间隔：

```text
chunk_fwd_kernel_o
  ↓
state writeback / DtoD memcpy     ~125-139 us
  ↓
small stream gap                  ~1-2 us
  ↓
triton_poi_fused_0                RMSNormGated fused path 开始
```

按当前 trace 的可验证统计：

```text
GDN post gap samples: 24
chunk_fwd_kernel_o -> triton_poi_fused_0:
  avg gap: 149.9 us
  p50 gap: 147.4 us
  total:   3.60 ms

注意:
  这个 gap 里大部分是 state writeback / DtoD memcpy，
  不能简单全部归因为 Python launch overhead。
```

这仍然是结构性优化点：如果能让 GDN custom op 的输出 layout 更接近 `RMSNormGated` / `out_proj` 需要的输入，或者把部分 post-processing 做成 GDN epilogue，就有机会减少这段串行尾巴。

### 3.4 GDN Input Projection Overlap

linear attention 的输入 projection 有两个分支：

```text
normalized hidden_states
  ├─ in_proj_qkvz  → q / k / v / z
  └─ in_proj_ba    → b / a
```

这两个分支吃同一个 input，理论上存在 dual-stream overlap 机会。尤其是 `in_proj_ba` 的输出维度很小，相比 `in_proj_qkvz` 是 tiny GEMM；如果当前路径把两者串行执行，就会把一个小 GEMM 和相关 quant/layout 准备暴露在关键路径上。

优化方向：

```text
1. 检查当前 graph 中 in_proj_qkvz 和 in_proj_ba 是否同 stream 串行。
2. 如果串行，评估 dual-stream overlap:
   - stream 0: in_proj_qkvz
   - stream 1: in_proj_ba
   - GDN core 前同步两个输出
3. 同时检查两者是否重复做同一个 normalized hidden_states 的 FP4 activation quant。
```

这个方向与已有的 GDN input projection overlap 思路一致，适合放在 GDN 结构优化中验证，但收益上限受 `in_proj_ba` 本身很小限制，应通过 trace 中 GEMM 和 quant 的 critical path 来确认。

### 3.5 优化方向

```text
方向 1: GDN post-conv 与 chunk 前处理融合
  当前:
    causal_conv1d -> fused_post_conv -> chunk kernels
  目标:
    减少 q/k/v/g/beta 准备阶段的中间写回
  风险:
    GDN 数学路径复杂，需严格对齐数值。

方向 2: chunk kernel 链路合并或减少 layout transform
  当前:
    chunk_scaled_dot_kkt
    chunk_gated_delta_rule
    recompute_w_u
    merge_16x16_to_64x64_inverse
    chunk_fwd_kernel_o
  目标:
    重点看 merge/recompute 是否可避免或延后。
  风险:
    这些 kernel 可能来自 FLA 算法分解，改动会触及底层算子。

方向 3: 减少 GDN core 后的 DtoD copy / index_put
  trace 现象:
    GDN core 后面经常跟随 index_put、DtoD memcpy、Triton post-norm/gate kernel。
  目标:
    让 core_attn_out 的 layout 更接近 out_proj 输入需要的 layout。

方向 4: GDN input projection dual-stream overlap
  当前:
    in_proj_qkvz 和 in_proj_ba 都依赖同一个 normalized hidden_states。
  目标:
    如果两路 projection 当前在关键路径上串行，则让 tiny in_proj_ba 与 in_proj_qkvz overlap。
```

---

## 4. P2: Full Attention / FlashAttention

### 4.1 现象

Qwen3.5-27B-Dense 的层类型是：

```text
[linear, linear, linear, full] × 16
```

因此 prefill 中 full attention 只有 16 层，不是每层都有。trace 中主要 FlashAttention kernel：

```text
flash::flash_fwd_splitkv_kernel
  total: 31.92 ms
  count: 7
  avg:   4.56 ms
```

这里 count 小于 full attention 层数，说明 compiled / CUDA graph / backend 可能把部分路径以不同 launch 方式体现，不能简单按 kernel count 等于层数解读。

### 4.2 优化方向

```text
方向 1: splitkv 参数检查
  对 bs=1, seq=8192 的 prefill，splitkv 不一定总是最优。
  需要对比不同 backend / metadata 配置下 FlashAttention kernel time。

方向 2: KV cache 写入成本检查
  reshape_and_cache_flash_kernel 在本次 prefill 不是大头，但 full attention 层数增加时会变重要。

方向 3: full attention 与 linear attention 分开看
  不要用总 TTFT 直接判断 FlashAttention 优劣。
  需要只统计 full attention window 或相关 kernel。

方向 4: FA3 / FlashInfer backend 验证
  当前 trace 主要看到 FA2 splitkv 形态。
  Qwen3.5 full attention 的 head_dim=256，属于 FA3/TMA pipeline 可能受益的场景。
  但当前 FlashAttention 总时间约 31.92 ms，因此 20-30% 的 FA 提升对应整体约 6-10 ms。
```

当前 Full Attention 约占 8.7%，优化价值有，但低于 GEMM 和 GDN core。

### 4.3 QK Norm + RoPE Fusion 检查

full attention 分支除了 FlashAttention 主 kernel，还有 QK norm、partial RoPE、split/cat/view 等附属 kernel。当前 trace 中能看到这一类 Triton kernel：

```text
triton_red_fused_7
triton_poi_fused__...clone_rms_norm...view_9
triton_poi_fused__...clone_mul_rms_norm...view_8
triton_poi_fused__...cat_rms_norm...view_11
triton_poi_fused__...cat_mul_rms_norm...view_10
```

当前 trace 中这组 kernel 的可验证聚合约为：

```text
QK norm / RoPE-ish kernels:
  total: 2.22 ms
  count: 36
```

这说明 QK norm + RoPE 没有完全变成单个 fused path，仍然存在多个 Triton pointwise/reduction kernel。vLLM 中已有 `QKNormRoPEFusionPass` / `enable_qk_norm_rope_fusion` 相关能力，后续需要确认为什么当前 Qwen3.5-Dense 形态没有完全命中。

可能原因：

```text
1. partial_rotary_factor=0.25，实际 rot_dim=64，只对 head_dim=256 的前 64 维做 RoPE。
2. attn_output_gate=True，q projection 中还带 gate split。
3. compiled graph 中 split / cat / view 形态与 fusion pass pattern 不完全匹配。
```

优化方向：

```text
1. 检查 QKNormRoPEFusionPass 是否启用。
2. 对 Qwen3.5 的 partial RoPE + output gate pattern 扩展匹配。
3. 用 trace 验证相关 triton_poi / triton_red kernel 数量和总时长是否下降。
```

---

## 5. P3: FP4 Activation Quant 边界

### 5.1 现象

独立 FP4 activation quant kernel：

```text
void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>
  total: 10.48 ms
  count: 118
  avg:   88.85 us
```

如果把 fused quant 也算进去：

```text
standalone cvt_fp16_to_fp4
  + silu_mul_cvt_fp16_to_fp4
  + rms_norm_scaled_fp4_quant
  + triton fused scaled_fp4_quant
≈ 50 ms 级别
```

这说明 NVFP4 的收益和成本都很集中：GEMM 算得更快，但 Linear 输入前需要频繁做 activation quant。

### 5.2 优化方向

```text
方向 1: 把 norm / activation / quant / layout 合并
  已经出现的融合:
    triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3
    triton_poi_fused__...rms_norm...scaled_fp4_quant...silu...
    vllm::silu_mul_cvt_fp16_to_fp4

  继续检查:
    是否仍存在 bf16 intermediate 写回后再 cvt_fp16_to_fp4。

方向 2: 减少重复 quant
  检查同一个 hidden_states 是否被多个 Linear 分支分别 quant。
  例如 linear attention 中 in_proj_qkvz 和 in_proj_ba 都吃同一个 normalized hidden_states。

方向 3: quant scale 生成和 GEMM input packing 协同
  当前 quant 输出包括 FP4 activation 和 block scale。
  如果后续 GEMM backend 需要特定 layout，可以考虑 quant 阶段直接产出更接近 GEMM 消费的 layout。
```

### 5.3 重点检查点

```text
Attention input:
  GemmaRMSNorm(hidden_states)
    ↓
  FP4Q
    ↓
  qkv / qkvz / ba projection GEMM

MLP input:
  post_attention_layernorm(hidden_states)
    ↓
  FP4Q
    ↓
  gate_up_proj GEMM

MLP activation:
  gate_up output bf16
    ↓
  silu(gate) * up + FP4Q
    ↓
  down_proj GEMM

Linear attention output:
  GDN core output + z
    ↓
  RMSNormGated + silu(z) + FP4Q
    ↓
  out_proj GEMM
```

---

## 6. P4: Dense MLP 路径

### 6.1 现象

Dense MLP 每层都有，且中间维度较大：

```text
hidden_size:        5120
intermediate_size:  17408
gate_up:            34816
```

MLP 主要路径：

```text
hidden_states [T, 5120]
  ↓ FP4Q
gate_up_proj FP4 GEMM:
  [T, 5120] x [5120, 34816]
  ↓
gate, up split
  ↓
silu(gate) * up + FP4Q
  ↓
down_proj FP4 GEMM:
  [T, 17408] x [17408, 5120]
```

trace 中 `silu_mul_cvt_fp16_to_fp4`：

```text
total: 17.04 ms
count: 31
avg:   549.72 us
```

count 接近 MLP 路径中被捕获的 fused activation quant 次数。它已经是 fused kernel，不是 eager 下拆散的 `silu + mul + quant`。

### 6.2 优化方向

```text
方向 1: gate_up/down GEMM tile 调优
  MLP GEMM 形状大，可能是 FP4 GEMM 最大贡献者。
  需要把 GEMM 按 MLP / attention projection 归因。

方向 2: silu_mul_cvt_fp16_to_fp4 与 down_proj 衔接
  检查 fused activation quant 输出 layout 是否最适合 down_proj GEMM。

方向 3: 避免 split/view/copy 额外开销
  gate_up 输出后需要 split gate/up。
  如果 trace 里存在额外 copy/view kernel，应看是否能由 fused activation kernel 消化。
```

---

## 7. P5: RMSNorm / Gate / Triton Fused Kernel

### 7.1 现象

compiled trace 中很多 RMSNorm 和 quant 已经被 Inductor 融合：

```text
triton_red_fused__to_copy_add_copy__rms_norm_4
triton_red_fused__to_copy_add_rms_norm_scaled_fp4_quant_zeros_3
triton_poi_fused__to_copy__unsafe_view_add_clone_mean_mul_pow_rsqrt_scaled_fp4_quant_silu_view_zeros_2
triton_per_fused_1
```

其中：

```text
triton_poi_fused__...rms_norm...scaled_fp4_quant...silu..._2
```

对应 linear attention 中：

```text
core_attn_out = self.norm(core_attn_out, z)
core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
output[:num_tokens], _ = self.out_proj(core_attn_out)
```

也就是：

```text
RMSNormGated(core_attn_out, z)
  = norm(core_attn_out) * silu(z)
  ↓
FP4 activation quant for out_proj
```

### 7.2 优化方向

这部分已经有较多融合，单独收益不会最大。后续重点是检查它和上下游的边界：

```text
1. GDN core 输出 layout 是否导致额外 copy / index_put
2. RMSNormGated fused quant 输出是否直接喂给 out_proj GEMM
3. 是否还有 standalone cvt_fp16_to_fp4 可以被并入 norm/gate kernel
```

---

## 8. 建议执行顺序

```text
Step 1: 建立稳定 profiling 基线
  - 同一 backend 至少跑 3-5 次
  - 固定 input_len=8192, output_len=2, max_concurrency=1
  - 记录 TTFT、prefill wall time、kernel group time

Step 2: 拆分 FP4 GEMM 归因
  - 区分 MLP GEMM、attention projection GEMM、out_proj GEMM
  - 找出最大 GEMM shape
  - 对最大 shape 做 backend / tile 对比

Step 3: 分析 GDN prefill kernel 链
  - 统计 48 个 linear attention 层内每类 GDN kernel 时间
  - 检查 core 后的 copy/index_put/DtoD memcpy
  - 统计 chunk_fwd_kernel_o -> RMSNormGated fused kernel 的 gap
  - 检查 in_proj_qkvz / in_proj_ba 是否可 dual-stream overlap
  - 判断是否有 layout transform 可避免

Step 4: 检查 FP4 quant 边界
  - 列出 standalone cvt_fp16_to_fp4 的调用位置
  - 标记哪些已经被 rms_norm / silu 融合
  - 优先处理高频、高耗时、紧邻 GEMM 的 quant

Step 5: 再看 Full Attention
  - 只在 P0-P3 有清晰结论后再调 FlashAttention
  - 对 bs=1, seq=8192 检查 splitkv 是否合理
  - 验证 FA3 / FlashInfer 对 head_dim=256 的收益
  - 检查 QKNormRoPEFusionPass 是否命中 partial RoPE + output gate pattern
```

---

## 9. 最可能有收益的具体方向

### 9.1 GEMM backend / tile 精调

收益潜力最高，因为 FP4 GEMM 已经占 prefill 过半。

```text
目标:
  降低 CUTLASS / FlashInfer FP4 GEMM total time

重点:
  - MLP gate_up_proj
  - MLP down_proj
  - linear attention in_proj_qkvz / in_proj_ba
  - linear attention out_proj

验证:
  用 trace kernel group time 判断，不只看 benchmark TTFT。
```

### 9.2 GDN core 减少 kernel 链和 layout traffic

收益潜力中高，因为 GDN core 是第二大头。

```text
目标:
  减少 linear attention prefill 的 chunk kernel 总时长

重点:
  - merge_16x16_to_64x64_inverse_kernel
  - recompute_w_u_fwd_kernel
  - chunk_fwd_kernel_o
  - GDN core 后 index_put / DtoD memcpy
  - chunk_fwd_kernel_o -> RMSNormGated fused path 的 ~150us 串行间隔
  - in_proj_qkvz / in_proj_ba dual-stream overlap
```

### 9.3 Quant + activation + norm 继续融合

收益潜力中等，但对 NVFP4 Dense 很关键。

```text
目标:
  减少 standalone cvt_fp16_to_fp4

重点:
  - attention input norm 后的 FP4Q
  - MLP input norm 后的 FP4Q
  - GDN output RMSNormGated 后的 FP4Q
  - silu_mul_cvt_fp16_to_fp4 到 down_proj 的 layout
```

### 9.4 Full Attention backend 检查

收益潜力中等偏低。

```text
目标:
  降低 flash_fwd_splitkv_kernel time

重点:
  - bs=1, seq=8192 下 splitkv 参数
  - head_dim=256 下 FA3 / TMA pipeline 是否比当前 FA2 splitkv 更合适
  - full attention 层的 qkv/o projection GEMM 与 FlashAttention 分开归因
  - QK norm + partial RoPE fusion pass 是否生效
```

### 9.5 QK Norm + RoPE Fusion 修复

收益潜力中等偏低，但这是比较明确的 fusion gap。

```text
目标:
  减少 full attention 中 QK norm / partial RoPE / split-cat-view 的 Triton 小 kernel。

当前 trace 可见:
  triton_red_fused_7
  triton_poi_fused__...rms_norm...split...cat...
  total: 约 2.22 ms

重点:
  - partial_rotary_factor=0.25
  - rot_dim=64, head_dim=256
  - attn_output_gate=True
  - fusion pass pattern 是否覆盖这个组合
```

---

## 10. 判断优化是否有效的标准

不要只看单次 TTFT。建议每个实验至少记录：

```text
整体指标:
  - mean TTFT
  - p50 / p90 TTFT
  - prefill window wall time

kernel group 指标:
  - FP4 GEMM total time
  - GDN core total time
  - FlashAttention total time
  - FP4 activation quant total time
  - MLP silu_mul_cvt_fp16_to_fp4 total time

结构指标:
  - kernel launch count
  - DtoD memcpy count / time
  - standalone cvt_fp16_to_fp4 count
```

一个优化可以认为有效，至少应满足：

```text
1. prefill wall time 稳定下降
2. 对应 kernel group time 明确下降
3. 没有把成本转移到其它 group
4. decode TPOT 没有明显回退
5. 输出数值和原路径一致，或误差在 NVFP4 预期范围内
```

---

## 11. 当前结论

```text
第一优先级:
  FP4 GEMM backend / tile / shape 归因

第二优先级:
  GDN linear attention prefill kernel 链优化

第三优先级:
  FP4 activation quant 与 norm / silu / layout 的边界融合

第四优先级:
  Full Attention splitkv / backend 参数检查
```

这份 trace 的 prefill 性能主要是 Dense NVFP4 GEMM 驱动的。GDN 和 quant 边界是后续最有价值的结构性优化点，而 Full Attention 当前不是首要瓶颈。
