# FlashInfer CUTLASS MoE 后端：Qwen3.5-35B-A3B GPU Kernel 逐层解析

> **模型**: Qwen3.5-35B-A3B（40 个 MoE 层，top-8 routing，256 experts，H=2048，I=512）
>
> **MoE 后端**: `FLASHINFER_CUTLASS`（FlashInfer 调度 + TRT-LLM CUTLASS Grouped GEMM）
>
> **数据来源**: `qwen3_5_35b_a3b/origin_profile_8192` trace，Blackwell GPU
>
> **Trace 配置**: `max_num_batched_tokens=8192, enable_prefix_caching=True, enable_chunked_prefill=True`

---

## 1. 单个 MoE 层的 GPU Kernel 执行序列

从 trace 中提取一个完整 MoE 层（7392 token prefill，warm iteration）的 GPU kernel 序列：

```text
 #   Offset(us)   Dur(us)   Kernel
───  ──────────   ────────   ──────────────────────────────────
 1        0.0      14.46    topkGating
 2       66.1       0.90    elementwise (topk_ids cast)
 3       78.3      28.00    cvt_fp16_to_fp4
 4      376.7      31.71    blockExpertPrefixSum<1024>
 5      408.4       2.05    globalExpertPrefixSumLarge<1024>
 6      410.3      13.09    mergeExpertPrefixSum
 7      422.9     236.93    expandInputRows
 8      659.8       2.43    computeStridesTmaWarp
 9      662.2     500.25    CUTLASS GroupedGEMM  (GEMM1: gate+up)
10     1162.3     218.81    doActivation  (SiLU + FP4 requant)
11     1377.4     372.93    CUTLASS GroupedGEMM  (GEMM2: down)
12     1750.2     213.53    finalizeMoeRouting
                  ────────
        total    ~1964 us   (单层 MoE routed expert GPU 总耗时)
```

- **Offset(us)**：该 kernel 相对于本层第一个 kernel（`topkGating`）的 GPU 启动时间偏移。
- **Dur(us)**：该 kernel 单次调用在 GPU 上的实际执行耗时。

这 12 个 kernel 可分为 4 个阶段：

```text
阶段 1  Routing:         topkGating
阶段 2  Dispatch 准备:   cvt_fp16_to_fp4 → blockExpertPrefixSum → globalExpertPrefixSum
                          → mergeExpertPrefixSum → expandInputRows → computeStridesTmaWarp
阶段 3  Expert 计算:     CUTLASS GEMM1 → doActivation → CUTLASS GEMM2
阶段 4  Finalize:        finalizeMoeRouting
```

下面逐个 kernel 解析。

---

## 2. topkGating

```text
kernel: void vllm::moe::topkGating<8, 256, 4, 16, 32, int, __nv_bfloat16, ScoringFunc::SOFTMAX>(...)
```

**作用**：接收 router gate 的 logits `[M, E]`，计算 softmax 并选出每个 token 的 top-K expert。

**输入**：
- `router_logits`: `[M, 256]` BF16，由 `gate` 线性层产出

**输出**：
- `topk_ids`: `[M, 8]` int32，每个 token 被路由到的 8 个 expert 编号
- `topk_weights`: `[M, 8]` float32，对应的 softmax 权重

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[1848, 1, 1]` |
| block | `[32, 4, 1]` |

grid.x = ceil(7392 * 8 / 32) = 1848，每个 warp 处理一组 token-expert 的 softmax + topk 选择。

**耗时对比**：

| M | dur |
|---:|---:|
| 7392 | 14.5 us |
| 800 | 3.9 us |

接近线性 scale，routing 计算本身开销不大。

---

## 3. cvt_fp16_to_fp4（Input Quantization）

```text
kernel: void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>(...)
```

**作用**：将 BF16 hidden_states 量化为 NVFP4 packed 格式，供后续 `expandInputRows` 和 GEMM1 使用。

这是 `_C::scaled_fp4_quant` CPU op 发起的底层 CUDA kernel。quantization 公式：

```text
fp4_value = round_to_fp4(bf16_input * global_scale * block_scale)
```

每 2 个 FP4 值 pack 到 1 个 uint8。

**在 MoE 层中出现 1 次**：对 routed input `[M, H]` 做整体量化。

**耗时**：7392 tokens 时约 28 us，800 tokens 时约 3 us。

> 注意：trace 中 `cvt_fp16_to_fp4` 总计 3220 次 / 9 个 prefill phase = 每 phase 约 230 次。其中 40 次属于 MoE 层（每层 1 次 input quant），其余来自 dense attention 投影的 NVFP4 GEMM（`flashinfer_mm_fp4` 路径）。

---

## 4. blockExpertPrefixSum

```text
7392 tokens: void tensorrt_llm::..::blockExpertPrefixSumKernel<1024>(...)
 800 tokens: void tensorrt_llm::..::blockExpertPrefixSumKernel<512>(...)
```

**作用**：统计每个 expert 被分配了多少 token，并在 block 内计算 prefix sum。这是 MoE routing 的第一步物理准备——从 `topk_ids` 中统计 expert-token 分布。

**为什么 template 参数不同**：TRT-LLM 根据 token 数量 M 选择不同的 block size 特化：
- M 较大（7392）→ `<1024>` 线程块，处理更多 token-expert pair
- M 较小（800）→ `<512>` 线程块，减少空闲线程浪费

**输入**：`topk_ids [M, 8]`

**输出**：每个 expert 的 token 计数和 block-local prefix sum

**耗时**：

| M | variant | dur |
|---:|---|---:|
| 7392 | `<1024>` | 31.5 us |
| 800 | `<512>` | 3.8 us |

---

## 5. globalExpertPrefixSum / globalExpertPrefixSumLarge

```text
7392 tokens: void tensorrt_llm::..::globalExpertPrefixSumLargeKernel<1024>(...)
 800 tokens: void tensorrt_llm::..::globalExpertPrefixSumKernel<512>(...)
```

**作用**：将 `blockExpertPrefixSum` 输出的 block-local prefix sum 合并为全局 prefix sum。确定每个 expert 在展开后 buffer 中的全局起始偏移。

**这也是 800 和 7392 phase 之间的主要 kernel 差异之一**：
- 小 M 用 `globalExpertPrefixSumKernel<512>`（简单单 pass）
- 大 M 用 `globalExpertPrefixSumLargeKernel<1024>`（可能多 pass 或分层 reduce）

**耗时**：约 1.3 ~ 2.0 us，非常快，因为只是对 256 个 expert 的计数做 prefix sum。

---

## 6. mergeExpertPrefixSum

```text
kernel: tensorrt_llm::..::mergeExpertPrefixSumKernel(int const*, int const*, int const*, int*, int*, int*, int)
```

**作用**：将 block-level 和 global-level 的 prefix sum 合并，生成最终的 expert-to-token mapping。这个 mapping 告诉后续 kernel：expert `e` 的第 `i` 个 token 在展开 buffer 中的位置是什么。

**耗时**：约 2 ~ 13 us（随 M 线性增长）。

> 注意：`mergeExpertPrefixSum` 只在 prefill（800 和 7392 phase）中出现，decode 不需要（decode 用 `fusedBuildExpertMapsSortFirstTokenKernel` 替代整个 routing 链）。

---

## 7. expandInputRows ★

```text
kernel: void tensorrt_llm::..::expandInputRowsKernel
    <__nv_fp4_e2m1, __nv_fp4_e2m1, FpXBlockScalingType::1, false>(...)
```

**作用**：根据 routing mapping，将 FP4 quantized input 从 `[M, K_packed]` 展开为 `[M*topK, K_packed]`，使得被路由到同一个 expert 的所有 token 在内存中连续排列。这是 grouped GEMM 的前置数据准备。

**具体操作**：

```text
输入: fp4_input[M, K/2]     (packed FP4, K=2048 → K/2=1024 bytes per token)
      routing_map[M*topK]    (每个 expanded row 对应哪个原始 token)

输出: expanded[M*topK, K/2]  (按 expert 分组、连续排列)
      expanded_scale[...]     (对应的 block scale 也同步展开)
```

对 Qwen3.5（top-8 routing）：
- 7392 tokens → 7392 * 8 = 59136 expanded rows
- 800 tokens → 800 * 8 = 6400 expanded rows

**Launch 配置**：

| 项 | 7392 tokens | 800 tokens |
|---|---:|---:|
| grid | `[880, 1, 1]` | `[880, 1, 1]` |
| block | `[256, 1, 1]` | `[256, 1, 1]` |
| regs/thread | 40 | 40 |
| shared mem | 0 | 0 |
| blocks per SM | 8 | 8 |

grid 大小相同（880）是因为 grid 是按 expert 数量 * 某个 tile factor 决定的，与 token 数无关。但实际工作量不同，体现在每个 block 内的循环次数。

**耗时**：

| M | dur | 数据量 |
|---:|---:|---|
| 7392 | 237 us | 59136 rows × 1024 bytes = ~57 MB scatter |
| 800 | 30 us | 6400 rows × 1024 bytes = ~6.2 MB scatter |

耗时比约 7.9x，接近 token 数之比 9.2x。差异来自 kernel launch 固定开销和 memory bandwidth 利用率差异。

**性能特征**：纯 memory-bound scatter/gather 操作。无算术计算，只做数据搬运。FP4 packed 格式让数据量比 BF16 小 4x，是选择在量化后做 expand 而非量化前的原因。

---

## 8. computeStridesTmaWarp

```text
kernel: void tensorrt_llm::..::computeStridesTmaWarpSpecializedKernel
    <__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>(...)
```

**作用**：为 CUTLASS TMA Warp-Specialized Grouped GEMM 计算每个 expert group 的 stride 参数和 TMA descriptor。

CUTLASS grouped GEMM 需要知道每个 expert 子矩阵的：
- 起始指针（A: expanded input，B: expert weight，C: output）
- M 维大小（该 expert 分配到的 token 数）
- stride layout

这个 kernel 根据 `expandInputRows` 产生的 expert grouping 信息，一次性为所有 expert 生成这些参数，供后续 GEMM kernel 的 TMA load 使用。

**耗时**：约 2.4 ~ 3.0 us，非常快。只涉及少量元数据计算（256 个 expert 的 pointer/stride）。

---

## 9. CUTLASS Grouped GEMM — GEMM1（Gate+Up Projection）

```text
kernel: _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalI
    NS1_17GroupProblemShapeIN4cute5tupleIJlllEEEEE...
```

**作用**：执行所有 expert 的第一层投影 W13（gate + up fused），即：

```text
u[t, e] = expanded_fp4_input[t] @ W13[e]    # [K] × [K, 2I] → [2I]
```

对 Qwen3.5：K=2048, I=512, 所以 W13 shape 是 `[K, 2*I] = [2048, 1024]`（FP4 packed: `[1024, 512]`）。

这是一个 **Grouped GEMM**：256 个 expert 的矩阵乘法被组织为一个 CUTLASS GroupProblemShape，每个 group 的 M 维不同（取决于该 expert 被分配了多少 token），N 和 K 维相同。

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[1, 110, 1]` |
| block | `[384, 1, 1]` |
| regs/thread | 168 |
| shared mem | 89088 bytes (87 KB) |

grid.y = 110 对应 256 个 expert 分组后的 tile 数量。block=384 是 CUTLASS warp-specialized kernel 的标准配置（通常 3-4 个 warp）。

**耗时**：

| M | dur |
|---:|---:|
| 7392 | 500 us |
| 800 | ~70 us (avg from CUTLASS GEMM pool) |

GEMM1 是单个 MoE 层中最耗时的 kernel，占约 25% 的 MoE 层 GPU 时间。

---

## 10. doActivation（SiLU + FP4 Requantization）

```text
kernel: void tensorrt_llm::..::doActivationKernel
    <__nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16, GLUAdaptor<SiLu>, FpXBlockScalingType::1>(...)
```

**作用**：对 GEMM1 输出执行 SwiGLU 激活，并将结果重新量化为 FP4 供 GEMM2 使用。

```text
gemm1_out: [M*topK, 2I] BF16 (GEMM1 accumulator 输出)

gate = gemm1_out[:, I:]       # 后半部分
up   = gemm1_out[:, :I]       # 前半部分
mid  = silu(gate) * up         # [M*topK, I]

fp4_mid = quantize_to_fp4(mid, a2_global_scale)   # 重新量化为 FP4
fp4_mid_scale = compute_block_scale(mid)
```

> 注意：gate/up 的前后半顺序取决于权重布局。FlashInfer CUTLASS 路径使用 `[up, gate]` 排列（与 ATREX 相同），所以 `up = out[:, :I]`, `gate = out[:, I:]`。

**关键设计**：这个 kernel 融合了三个操作：
1. SiLU 激活函数
2. Gate-Up 乘法（GLU）
3. FP4 动态量化（用于 GEMM2 输入）

融合避免了中间 BF16 tensor `[M*topK, I]` 的写回和重读。

**耗时**：

| M | dur |
|---:|---:|
| 7392 | 219 us |
| 800 | 25 us |

---

## 11. CUTLASS Grouped GEMM — GEMM2（Down Projection）

与 GEMM1 相同的 CUTLASS GroupedGEMM kernel，执行第二层投影 W2：

```text
y[t, e] = fp4_mid[t] @ W2[e]    # [I] × [I, K] → [K]
```

W2 shape: `[512, 2048]`（FP4 packed: `[256, 1024]`）。

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[1, 110, 1]` |
| block | `[384, 1, 1]` |
| regs/thread | 168 |
| shared mem | 98304 bytes (96 KB) |

GEMM2 shared memory（98304）比 GEMM1（89088）大，因为 W2 的 N 维（K=2048）大于 W13 的 N 维（2I=1024）。

**耗时**：

| M | dur |
|---:|---:|
| 7392 | 373 us |
| 800 | ~55 us |

GEMM2 比 GEMM1 快约 25%，因为 GEMM2 的 K 维更小（I=512 vs K=2048），计算量更小。

---

## 12. finalizeMoeRouting

```text
kernel: void tensorrt_llm::..::finalizeMoeRoutingKernel
    <__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, ScaleMode::1>(...)
```

**作用**：将 GEMM2 的 expert-major 输出 unpermute 回 token-major，并执行 topk 加权求和：

```text
输入: expert_output[M*topK, K]  BF16  (按 expert 排列)
      topk_weights[M, topK]     float32
      routing_map               (expanded_row → original_token 的映射)

输出: routed_out[M, K]  BF16

routed_out[t] = sum_{j=0}^{topK-1} topk_weights[t, j] * expert_output[mapping[t, j]]
```

这个 kernel 同时完成 unpermute + weighted reduce，避免了显式的 `moe_unpermute` + `moe_fused_mul_sum` 两个 kernel。

**耗时**：

| M | dur |
|---:|---:|
| 7392 | 210 us |
| 800 | 16 us |

> 注意：`finalizeMoeRouting` 只在 prefill 中出现（每 phase 40 次 = 40 MoE 层）。decode 阶段的 finalize 走不同的路径。

---

## 13. Decode 阶段的 Routing 差异

Decode 阶段（M=1）的 MoE routing 使用完全不同的 kernel：

```text
prefill routing 链 (3 kernels):
  blockExpertPrefixSum → globalExpertPrefixSum → mergeExpertPrefixSum

decode routing (1 kernel):
  fusedBuildExpertMapsSortFirstTokenKernel<32, 8, 9>(...)
```

Decode 用一个融合 kernel 替代了 prefill 的 3 步 routing，因为 M=1 时只有 8 个 token-expert pair，prefix sum 可以在一个 kernel 内完成排序和 map 构建。

类似地，decode 阶段没有 `finalizeMoeRouting` 和 `mergeExpertPrefixSum`（各 0 次），因为小 M 的 weighted reduce 融合在了其他步骤中。

---

## 14. 完整 Kernel 统计汇总

以下是所有 MoE 相关 GPU kernel 在不同 phase 中的分布和耗时统计。

### 14.1 每 phase 的 kernel 调用次数（每次 = 40 MoE 层各调用 1 次）

| Kernel | per 800 | per 7392 | per decode |
|---|:---:|:---:|:---:|
| topkGating | 40 | 40 | 40 |
| cvt_fp16_to_fp4 (MoE input quant) | 40 | 40 | 40 |
| blockExpertPrefixSum`<512>` | 40 | — | — |
| blockExpertPrefixSum`<1024>` | — | 40 | — |
| globalExpertPrefixSum`<512>` | 40 | — | — |
| globalExpertPrefixSumLarge`<1024>` | — | 40 | — |
| fusedBuildExpertMaps | — | — | 40 |
| mergeExpertPrefixSum | 40 | 40 | — |
| expandInputRows | 40 | 40 | 40 |
| computeStridesTmaWarp | 40 | 40 | 40 |
| CUTLASS GEMM1 | 40 | 40 | 40 |
| doActivation | 40 | 40 | 40 |
| CUTLASS GEMM2 | 40 | 40 | 40 |
| finalizeMoeRouting | 40 | 40 | — |

### 14.2 单层 MoE 平均耗时对比（warm iteration, 40 层平均）

| Kernel | 7392 tokens | 800 tokens | 倍数 |
|---|---:|---:|---:|
| topkGating | 14.5 us | 3.9 us | 3.7x |
| cvt_fp16_to_fp4 | 28.0 us | 3.0 us | 9.3x |
| blockExpertPrefixSum | 31.5 us | 3.8 us | 8.3x |
| globalExpertPrefixSum | 2.0 us | 1.3 us | 1.6x |
| mergeExpertPrefixSum | 13.1 us | 2.0 us | 6.5x |
| expandInputRows | 237.0 us | 30.2 us | 7.8x |
| computeStridesTmaWarp | 3.0 us | 2.9 us | 1.0x |
| CUTLASS GEMM1 | 500.3 us | ~70 us | ~7x |
| doActivation | 219.0 us | 24.7 us | 8.9x |
| CUTLASS GEMM2 | 372.9 us | ~55 us | ~7x |
| finalizeMoeRouting | 213.5 us | 16.4 us | 13.0x |
| **单层总计** | **~1964 us** | **~280 us** | **~7x** |

### 14.3 单层 MoE 耗时占比（7392 tokens）

```text
CUTLASS GEMM1 (gate+up)  ████████████████████████████  500 us  (25.5%)
CUTLASS GEMM2 (down)      ████████████████████         373 us  (19.0%)
expandInputRows            ████████████                 237 us  (12.1%)
doActivation (SiLU+quant)  ███████████                  219 us  (11.1%)
finalizeMoeRouting          ███████████                  214 us  (10.9%)
blockExpertPrefixSum        ██                            32 us   (1.6%)
cvt_fp16_to_fp4             ██                            28 us   (1.4%)
topkGating                  █                             15 us   (0.7%)
mergeExpertPrefixSum        █                             13 us   (0.7%)
computeStridesTmaWarp                                      3 us   (0.2%)
globalExpertPrefixSum                                      2 us   (0.1%)
                                                     ─────────
                                                     ~1964 us
```

**总结**：
- **Compute-bound**（44.5%）：GEMM1 + GEMM2 是主体计算。
- **Memory-bound scatter/gather/reduce**（34.1%）：`expandInputRows`、`doActivation`、`finalizeMoeRouting` 各占 10-12%，涉及数据重排和中间结果搬运。
- **Routing 准备**（2.4%）：prefix sum + merge 开销很小。
- **Metadata 计算**（0.2%）：`computeStridesTmaWarp` 几乎可以忽略。

---

## 15. 端到端 MoE 数据流总结

```text
hidden_states [M, 2048] BF16
  │
  ├─ gate linear → router_logits [M, 256]
  │
  ├─ topkGating → topk_ids [M, 8], topk_weights [M, 8]
  │
  ├─ cvt_fp16_to_fp4 → fp4_input [M, 1024] uint8 + block_scale
  │
  ├─ blockExpertPrefixSum ──┐
  ├─ globalExpertPrefixSum ─┤ → expert routing map
  ├─ mergeExpertPrefixSum ──┘
  │
  ├─ expandInputRows → expanded_fp4 [M*8, 1024] uint8 (expert-major)
  │
  ├─ computeStridesTmaWarp → GEMM metadata (per-expert pointers/strides)
  │
  ├─ CUTLASS GEMM1 → gemm1_out [M*8, 1024] BF16  (W13 @ expanded_fp4)
  │
  ├─ doActivation → fp4_mid [M*8, 256] uint8  (SiLU + FP4 requant)
  │
  ├─ CUTLASS GEMM2 → expert_out [M*8, 2048] BF16  (W2 @ fp4_mid)
  │
  └─ finalizeMoeRouting → routed_out [M, 2048] BF16  (unpermute + weighted sum)
```
