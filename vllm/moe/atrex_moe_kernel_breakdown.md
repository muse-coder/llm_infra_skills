# ATREX MoE 后端：Qwen3.5-35B-A3B GPU Kernel 逐层解析

> **模型**: Qwen3.5-35B-A3B（40 个 MoE 层，top-8 routing，256 experts，H=2048，I=512）
>
> **MoE 后端**: `ATREX`（atrex.api.nvfp4_fused_moe，CUTLASS Grouped GEMM + 优化 _opt kernel）
>
> **数据来源**: `qwen3_5_35b_a3b/atrex_profile_8192` trace，Blackwell GPU
>
> **Trace 配置**: `max_num_batched_tokens=8192, enable_prefix_caching=True, enable_chunked_prefill=True`

---

## 1. 单个 MoE 层的 GPU Kernel 执行序列

从 trace 中提取一个完整 MoE 层（7392 token prefill，warm iteration）的 GPU kernel 序列：

```text
 #   Offset(us)   Dur(us)   Kernel
───  ──────────   ────────   ──────────────────────────────────
 1        0.0      14.47    topkGating
 2~7      ~0       ~6 tot   vectorized_elementwise × 6（routing weight 处理）
 8      276.1      31.68    blockExpertPrefixSum<1024>
 9      307.7       2.02    globalExpertPrefixSumLarge<1024>
10      309.7      13.31    mergeExpertPrefixSum
11      322.5     138.37    expandInputRows_opt  ★
12      461.7       1.70    setup_cutlass_group_ptrs
13      464.1     506.33    CUTLASS GroupedGEMM  (GEMM1: gate+up)
14      970.6     118.59    doActivation_opt     ★
15     1090.4       1.73    setup_cutlass_group_ptrs
16     1094.5     407.17    CUTLASS GroupedGEMM  (GEMM2: down)
17     1501.6     197.60    finalizeMoeRouting_opt ★
                  ────────
       total    ~1700 us   (单层 MoE routed expert GPU 总耗时)
```

- **Offset(us)**：该 kernel 相对于本层第一个 kernel（`topkGating`）的 GPU 启动时间偏移。
- **Dur(us)**：该 kernel 单次调用在 GPU 上的实际执行耗时。
- **★** 标记的 kernel 是 ATREX 独有的 `_opt` 优化变体，与 FlashInfer CUTLASS 路径有本质区别。

这些 kernel 可分为 4 个阶段：

```text
阶段 1  Routing:         topkGating → vec_elem × 6（routing weight 计算）
阶段 2  Dispatch 准备:   blockExpertPrefixSum → globalExpertPrefixSum
                          → mergeExpertPrefixSum → expandInputRows_opt
阶段 3  Expert 计算:     setup_group_ptrs → CUTLASS GEMM1 → doActivation_opt
                          → setup_group_ptrs → CUTLASS GEMM2
阶段 4  Finalize:        finalizeMoeRouting_opt
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

grid.x = ceil(7392 × 8 / 32) = 1848，每个 warp 处理一组 token-expert 的 softmax + topk 选择。

**耗时对比**：

| M | dur |
|---:|---:|
| 7392 | 14.2 us |
| 800 | 4.0 us |

与 FlashInfer CUTLASS 后端完全相同的 kernel，接近线性 scale。

---

## 3. vectorized_elementwise × 6（GEMM Alpha 计算）

topkGating 之后、prefix sum 之前，ATREX 执行了 6 个轻量级 element-wise kernel。这些 kernel **不是** routing weight 处理，而是来自 ATREX Python API 中的 **GEMM scale 计算**。

### 来源：`atrex/api/nvfp4_fused_moe.py`

```python
gemm1_alpha = 1.0 / (a1_global_scale * w1_global_scale)   # → 3 GPU kernels
gemm2_alpha = 1.0 / (a2_global_scale * w2_global_scale)   # → 3 GPU kernels
```

其中 `a1_global_scale`、`w1_global_scale` 等均为 `[E]` float32 tensor（E=256 experts），这两行 Python 代码在 GPU 上展开为 6 个 elementwise kernel：

| # | CUDA Kernel | 对应 Python 表达式 |
|---|---|---|
| 1 | `MulFunctor<float>` (binary) | `a1_global_scale * w1_global_scale` |
| 2 | `reciprocal_kernel_cuda` | `1.0 / result` → 取倒数 |
| 3 | `AUnaryFunctor<float, MulFunctor>` | scalar × tensor 最终赋值 |
| 4 | `MulFunctor<float>` (binary) | `a2_global_scale * w2_global_scale` |
| 5 | `reciprocal_kernel_cuda` | `1.0 / result` → 取倒数 |
| 6 | `AUnaryFunctor<float, MulFunctor>` | scalar × tensor 最终赋值 |

每个 kernel 处理仅 256 个 float 元素，耗时约 1 us，总计约 6 us。

### 为什么需要这些 alpha？

ATREX 的 CUTLASS GEMM 使用 `alpha * (A @ B)` 的形式。由于 A（activation）和 B（weight）都经过 FP4 量化，需要通过 `alpha = 1 / (act_global_scale × weight_global_scale)` 将 GEMM 输出反量化回 BF16 scale 空间。这个 alpha 被传入 `fused_moe_forward_hybrid()` 的 C 函数，由 CUTLASS GEMM 在 epilogue 中作为 output scaling 使用。

### 与 FlashInfer CUTLASS 的区别

FlashInfer CUTLASS 路径中 topkGating 后仅有 1 次 elementwise cast（topk_ids 类型转换）。它不需要在 Python 层计算 GEMM alpha，因为其 CUTLASS kernel 内部直接接收 global scale 参数并在 kernel 内部完成 scale 融合。ATREX 则选择在 Python 层预计算 `1 / (act_scale × weight_scale)` 并作为标量传入 CUTLASS，设计上更简洁但多了 6 次轻量 kernel launch。

---

## 4. blockExpertPrefixSum

```text
7392 tokens: void tensorrt_llm::..::blockExpertPrefixSumKernel<1024>(...)
 800 tokens: void tensorrt_llm::..::blockExpertPrefixSumKernel<512>(...)
decode:      void tensorrt_llm::..::blockExpertPrefixSumKernel<256>(...)
```

**作用**：统计每个 expert 被分配了多少 token，并在 block 内计算 prefix sum——MoE routing 的第一步物理准备。

**template 参数选择**：根据 token 数量 M 选择不同的 block size 特化：
- M=7392 → `<1024>` 线程块
- M=800 → `<512>` 线程块
- M=1（decode）→ `<256>` 线程块

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[256, 8, 1]` |
| block | `[1024, 1, 1]` |

grid.x = 256（expert 数），grid.y = 8（将 token 分为 8 个 block 处理）。

**耗时**：

| M | variant | dur |
|---:|---|---:|
| 7392 | `<1024>` | 31.3 us |
| 800 | `<512>` | 3.9 us |
| 1 (decode) | `<256>` | 1.8 us |

---

## 5. globalExpertPrefixSum / globalExpertPrefixSumLarge

```text
7392 tokens: void tensorrt_llm::..::globalExpertPrefixSumLargeKernel<1024>(...)
 800 tokens: void tensorrt_llm::..::globalExpertPrefixSumKernel<512>(...)
decode:      void tensorrt_llm::..::globalExpertPrefixSumKernel<256>(...)
```

**作用**：将 block-local prefix sum 合并为全局 prefix sum，确定每个 expert 在展开后 buffer 中的全局起始偏移。

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[1, 1, 1]` |
| block | `[1024, 1, 1]` |

单 block kernel——只需对 256 个 expert 的计数做全局 prefix sum。

**耗时**：约 1.3 ~ 2.0 us，几乎可以忽略。

---

## 6. mergeExpertPrefixSum

```text
kernel: tensorrt_llm::..::mergeExpertPrefixSumKernel(int const*, int const*, int const*, int*, int*, int*, int)
```

**作用**：合并 block-level 和 global-level 的 prefix sum，生成最终的 expert-to-token mapping。

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[256, 8, 1]` |
| block | `[1024, 1, 1]` |

**耗时**：

| M | dur |
|---:|---:|
| 7392 | 12.2 us |
| 800 | 2.0 us |
| 1 (decode) | 1.3 us |

> **与 FlashInfer CUTLASS 的区别**：ATREX 在 decode 阶段仍然使用完整的 3 步 routing 链（blockPrefixSum → globalPrefixSum → mergePrefixSum），而 FlashInfer CUTLASS 在 decode 时用 `fusedBuildExpertMapsSortFirstTokenKernel` 融合 kernel 替代。

---

## 7. expandInputRows_opt ★（融合 Scatter + Quantization）

```text
kernel: expandInputRowsKernel_opt(
    __nv_bfloat16 const*,   // 输入: BF16 hidden_states
    unsigned char*,          // 输出: FP4 packed expanded data
    float const*,            // unpermuted_scales (topk_weights)
    float*,                  // permuted_scales (重排后的权重)
    int const*, int, int, int, float const*, bool, long const*, unsigned char*, int, int const*
)
```

**作用**：这是 ATREX 最关键的优化 kernel。它将 **scatter（token 按 expert 重排）** 和 **BF16→FP4 在线量化** 融合为一个 kernel，同时还将 topk_weights 按 expert 顺序重排（用于后续 finalize）。

### 源码核心逻辑（`expand_input_rows.cu`）

每个 thread 处理 16 个 BF16 元素（= 1 个 scale factor vector），关键步骤：

```c++
// 1. 根据 routing map 读取原始 BF16 token
int uprow = permuted_row_to_unpermuted_row[prow];
int source_row = uprow % num_tokens;
__nv_bfloat16 const* src = unpermuted_input + source_row * hidden_size;

// 2. 加载 16 个 BF16 → float，同时计算 absmax
float fmax = 0.f;
for (int i = 0; i < 8; i++) {
    f2[i] = __bfloat1622float2(p[i]);
    fmax = fmaxf(fmax, fmaxf(fabsf(f2[i].x), fabsf(f2[i].y)));
}

// 3. 根据 absmax 计算 FP8 e4m3 block scale factor
float sv = global_scale * (fmax * reciprocal_approximate_ftz(6.0f));
__nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(sv);

// 4. 量化为 FP4 并写入按 expert 分组的输出
for (int i = 0; i < 8; i++) { f2[i].x *= oscale; f2[i].y *= oscale; }
dst[kv] = fp32_vec_to_e2m1(f2);

// 5. 同步重排 topk_weights
if (tid == 0 && permuted_scales)
    permuted_scales[prow] = unpermuted_scales[source_row * k + source_k_rank];
```

一个 kernel 完成了 5 件事：
1. **Scatter**：按 routing map 从原始 token buffer 读取 BF16 数据
2. **Absmax 计算**：对 16 个元素计算绝对值最大值（用于 dynamic quantization）
3. **Block scale factor 生成**：将 absmax 转换为 FP8 e4m3 scale 并写入 SWIZZLED_128x4 布局
4. **FP4 量化**：将 float 值缩放并打包为 `e2m1` FP4 格式，写入按 expert 连续排列的输出
5. **topk_weights 重排**：将原始 `[M, topk]` 的 weights 同步 permute 为 expert-major 顺序

### 与 FlashInfer CUTLASS 的核心区别

| 对比项 | FlashInfer CUTLASS | ATREX |
|---|---|---|
| 数据流 | BF16 → `cvt_fp16_to_fp4` → FP4 → `expandInputRows<fp4>` → FP4 | BF16 → `expandInputRows_opt` → FP4 |
| kernel 数量 | 2 个 | 1 个 |
| scatter 输入类型 | FP4（已量化） | BF16（原始精度） |
| 量化时机 | 先量化整个 `[M, K]`，再 scatter | scatter 时逐 tile 在线量化 |
| block scale 生成 | 独立 kernel 或 fused | fused 在 expand 中 |

ATREX 的关键优化：在 scatter 时原地完成量化，每个 16-element tile 只需读一次 BF16 源数据，避免了 FlashInfer 路径中先将整个 `[M, K]` 量化为 FP4 再按 `[M×topK, K]` scatter 的两次全局内存遍历。

**Launch 配置**（7392 tokens）：

| 项 | 7392 tokens | 800 tokens |
|---|---:|---:|
| grid | `[1760, 1, 1]` | `[1760, 1, 1]` |
| block | `[128, 1, 1]` | `[128, 1, 1]` |

grid = min(SM_count × 16, max(M×topk, E×128))。128 threads/block，每 thread 处理 16 个 BF16 元素 = 1 个 SF vector。

**耗时**：

| M | dur | 说明 |
|---:|---:|---|
| 7392 | 141.8 us (avg) | 59,136 expanded rows |
| 800 | 20.6 us (avg) | 6,400 expanded rows |
| 1 (decode) | 8.1 us (avg) | 8 expanded rows |

与 FlashInfer CUTLASS 的 `cvt_fp16_to_fp4`(28us) + `expandInputRows`(237us) = **265 us** 相比，ATREX 的 `expandInputRows_opt` 仅需 **142 us**（7392 tokens），节省约 46%。

---

## 8. setup_cutlass_group_ptrs

```text
kernel: setup_cutlass_group_ptrs(
    cutlass::float_e2m1_t**,      // A matrix ptrs (FP4)
    cutlass::float_e2m1_t**,      // B matrix ptrs (FP4)
    cutlass::bfloat16_t**,        // C matrix ptrs (BF16 output)
    cutlass::float_ue4m3_t**,     // A scale ptrs (FP8 UE4M3)
    cutlass::float_ue4m3_t**,     // B scale ptrs (FP8 UE4M3)
    float**,                      // D scale ptrs
    cute::Layout<...>*,           // A stride layout
    cute::Layout<...>*,           // B stride layout
    long*, long*, long*, int*,    // problem sizes & metadata
    ...)
```

**作用**：为 CUTLASS Grouped GEMM 设置每个 expert group 的指针和 stride 参数。与 FlashInfer CUTLASS 的 `computeStridesTmaWarpSpecializedKernel` 功能类似，但使用不同的 CUTLASS tile/schedule 配置。

**每层出现 2 次**：GEMM1 之前和 GEMM2 之前各一次。

**Launch 配置**：

| 项 | 值 |
|---|---|
| grid | `[1, 1, 1]` |
| block | `[256, 1, 1]` |

单 block kernel，处理 256 个 expert 的指针元数据。

**耗时**：约 1.6 ~ 2.1 us，可以忽略。

### ATREX CUDA pipeline 中的调用位置

在 `fused_moe_forward_hybrid()`（`fused_moe_forward.cu`）中，GEMM 调用为：

```c++
// Step 3: GEMM1 (CUTLASS)
cutlass_gemm_forward(
    expand_out, w1_fp4,       // A: expanded FP4 input, B: FP4 weights
    fc1_act_sf, w1_sf,        // A block scale, B block scale
    gemm1_alpha,              // = 1/(a1_global_scale * w1_global_scale)
    gemm1_out,                // C: [M*topk, 2*I] BF16 accumulator
    expert_offset, E,         // per-expert token counts
    2 * inter_size, hidden_size, expanded,
    cutlass_ws, stream);
```

`setup_cutlass_group_ptrs` 根据 `expert_offset` 为每个 expert 设置独立的 A/B/C 指针和 M 维大小。

---

## 9. CUTLASS Grouped GEMM — GEMM1（Gate+Up Projection）

CUTLASS GroupProblemShape grouped GEMM，执行所有 expert 的第一层投影 W13（gate + up fused）：

```text
u[t, e] = expanded_fp4_input[t] @ W13[e]    # [K] × [K, 2I] → [2I]
```

Qwen3.5：K=2048, I=512, W13 shape `[K, 2I] = [2048, 1024]`（FP4 packed）。

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[110, 1, 1]` |
| block | `[384, 1, 1]` |

grid.x = 110 对应 256 个 expert 分组后的 tile 数量。block=384 是 CUTLASS warp-specialized kernel 标准配置。

**耗时**：

| M | dur (单次) |
|---:|---:|
| 7392 | 506 us |
| 800 | ~253 us |

GEMM1 是单个 MoE 层中最耗时的 kernel。

---

## 10. doActivation_opt ★（SiLU + FP4 Requantization）

```text
kernel: doActivationKernel_opt(
    unsigned char*,        // 输出: FP4 packed mid-result
    __nv_bfloat16 const*,  // 输入: GEMM1 BF16 accumulator输出
    float const*,          // fc2_act_global_scale
    long const*,           // expert_first_token_offset
    int, long, float const*, bool, unsigned char*, int const*
)
```

**作用**：对 GEMM1 输出执行 SwiGLU 激活，并重新量化为 FP4 供 GEMM2 使用。

### 源码核心逻辑（`do_activation.cu`）

每个 thread 处理 8 个元素，关键步骤：

```c++
// 1. 从 GEMM1 输出读取 gate 和 up 分量（BF16）
Vec8 gate_raw = gate_ptr[ei];   // gemm1_out[:, I:]
Vec8 up_raw   = up_ptr[ei];     // gemm1_out[:, :I]

// 2. 在 float 精度下计算 SiLU(gate) * up，同时跟踪 absmax
for (int i = 0; i < 8; i++) {
    float g = __bfloat162float(gate_raw[i]);
    float u = __bfloat162float(up_raw[i]);
    float sg = g / (1.0f + __expf(-g));    // SiLU
    vals[i] = sg * u * quant_scale;
    fmax = fmaxf(fmax, fabsf(vals[i]));
}

// 3. 跨 2 个 thread shuffle absmax（共享同一 SF vector 的 16 个元素）
fmax = fmaxf(fmax, __shfl_xor_sync(0xFFFFFFFF, fmax, 1));

// 4. 计算 FP8 e4m3 block scale 并量化为 FP4
__nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(global_scale * fmax * recip(6.0f));
for (int i = 0; i < 8; i++) vals[i] *= oscale;
out_ptr[ei] = fp32_vec_to_e2m1(vals);
```

一个 kernel 融合了 4 个操作：
1. **SiLU 激活**：`g / (1 + exp(-g))`，全部在 float 精度完成
2. **Gate-Up 乘法**：SwiGLU 的 `silu(gate) * up`
3. **Dynamic block scale 计算**：per-16-element absmax → FP8 e4m3 scale factor
4. **FP4 量化**：scale → `e2m1` packed → 写入 SWIZZLED 布局

**关键优化**（`_opt` vs 原版）：
- 所有计算在 float32 空间完成，消除了 `float→bf16→float` 的往返转换开销
- 预计算 per-row SF offset，避免运行时调用 `get_sf_out_offset_128x4`
- 使用 `__shfl_xor_sync` 在 2 个 thread 间共享 absmax，减少 shared memory 使用

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[3520, 1, 1]` |
| block | `[64, 1, 1]` |
| `__launch_bounds__` | `(64, 16)` — 每 SM 最多 16 个 block |

**耗时**：

| M | dur |
|---:|---:|
| 7392 | 115.9 us (avg) |
| 800 | 11.7 us |
| 1 (decode) | 5.4 us |

ATREX 的 `doActivation_opt`（116 us）比 FlashInfer CUTLASS 的 `doActivation`（219 us）快约 47%，主要得益于全 float 计算路径和消除 BF16 中间转换。

---

## 11. CUTLASS Grouped GEMM — GEMM2（Down Projection）

与 GEMM1 相同的 CUTLASS GroupedGEMM kernel，执行第二层投影 W2：

```text
y[t, e] = fp4_mid[t] @ W2[e]    # [I] × [I, K] → [K]
```

W2 shape: `[512, 2048]`（FP4 packed）。

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[110, 1, 1]` |
| block | `[384, 1, 1]` |

**耗时**：

| M | dur (单次) |
|---:|---:|
| 7392 | 407 us |
| 800 | ~195 us |

GEMM2 比 GEMM1 快约 20%，因为 K 维更小（I=512 vs K=2048）。

---

## 12. finalizeMoeRouting_opt ★

```text
kernel: void finalizeMoeRoutingKernel_opt<8, 4>(
    __nv_bfloat16 const*,  // expanded_permuted_rows (expert-major GEMM2 输出)
    __nv_bfloat16*,        // reduced_unpermuted_output (token-major 最终输出)
    float const*,          // scales (topk_weights)
    int const*,            // unpermuted_row_to_permuted_row
    int, int, int
)
```

template 参数：`<8, 4>` → TOPK=8，ROWS_PER_BLOCK=4。

**作用**：将 GEMM2 的 expert-major 输出 unpermute 回 token-major，并执行 topk 加权求和。

### 源码核心逻辑（`finalize_routing.cu`）

每个 block 处理 4 个输出 token（ROWS_PER_BLOCK=4），256 threads 并行处理 hidden_dim：

```c++
// 1. 对每个输出 token，查找它在 8 个 expert 中的 permuted 位置
for (int k = 0; k < TOPK; k++) {
    int expanded_orig = original_row + k * M;
    int permuted_row = unpermuted_row_to_permuted_row[expanded_orig];
    row_ptrs[k] = expanded_permuted_rows + permuted_row * padded_cols;
    row_scales[k] = scales[original_row * TOPK + k];  // topk_weight
}

// 2. 向量化加载 + 加权累加（每 thread 处理 8 个 BF16 元素）
for (int ei = tid; ei < num_elems; ei += 256) {
    float acc[8] = {0};
    for (int k = 0; k < TOPK; k++) {
        Vec8 v = reinterpret_cast<Vec8 const*>(row_ptrs[k])[ei];
        float s = row_scales[k];
        for (int i = 0; i < 8; i++)
            acc[i] += s * __bfloat162float(v[i]);
    }
    // 写回 BF16
    for (int i = 0; i < 8; i++) out[i] = __float2bfloat16(acc[i]);
}
```

这个 kernel 融合了 unpermute + weighted reduce：
- **Unpermute**：通过 `unpermuted_row_to_permuted_row` 映射表，从 expert-major 输出中 gather 8 个 expert 的结果
- **Weighted reduce**：用 `topk_weights` 对 8 个 expert 输出做加权求和

**Launch 配置**（7392 tokens）：

| 项 | 值 |
|---|---|
| grid | `[1848, 1, 1]` = ceil(7392 / 4) |
| block | `[256, 1, 1]` |
| `__launch_bounds__` | `(256)` |

**耗时**：

| M | dur |
|---:|---:|
| 7392 | 199.4 us (avg) |
| 800 | 11.9 us |
| 1 (decode) | 1.6 us |

与 FlashInfer CUTLASS 的 `finalizeMoeRouting`（210 us）性能接近。性能主要受限于 8 次 gather 读取的内存带宽（每个输出 token 需读 8 × 2048 × 2 bytes = 32 KB 的 expert 输出）。

---

## 13. Decode 阶段的 Routing 差异

Decode 阶段（M=1）的 MoE kernel 序列与 prefill 保持一致结构，但使用更小的特化参数：

```text
ATREX decode routing:
  topkGating → vec_elem × 7
  → blockExpertPrefixSum<256> → globalExpertPrefixSum<256> → mergeExpertPrefixSum
  → expandInputRows_opt → setup_group_ptrs → GEMM1 → doActivation_opt
  → setup_group_ptrs → GEMM2 → finalizeMoeRouting_opt
```

**与 FlashInfer CUTLASS decode 的关键区别**：

| 对比项 | FlashInfer CUTLASS | ATREX |
|---|---|---|
| decode routing | `fusedBuildExpertMapsSortFirstTokenKernel`（1 个融合 kernel） | `blockPrefixSum<256>` + `globalPrefixSum<256>` + `mergePrefixSum`（3 个 kernel） |
| decode finalize | 无独立 `finalizeMoeRouting` | 有 `finalizeMoeRouting_opt`（1.6 us） |
| 统一性 | prefill/decode 使用不同 kernel | prefill/decode 使用相同 kernel（参数不同） |

ATREX 选择在所有阶段使用统一的 kernel 路径，decode 通过选择更小的 template 特化（`<256>`）来降低开销，而非使用完全不同的融合 kernel。

---

## 14. 完整 Kernel 统计汇总

### 14.1 每 phase 的 kernel 调用次数（每次 = 40 MoE 层各调用 1 次）

| Kernel | per 7392 | per 800 | per decode |
|---|:---:|:---:|:---:|
| topkGating | 40 | 40 | 40 |
| vectorized_elementwise（routing weight） | ~240 | ~240 | ~280 |
| blockExpertPrefixSum`<1024>` | 40 | — | — |
| blockExpertPrefixSum`<512>` | — | 40 | — |
| blockExpertPrefixSum`<256>` | — | — | 40 |
| globalExpertPrefixSumLarge`<1024>` | 40 | — | — |
| globalExpertPrefixSum`<512>` | — | 40 | — |
| globalExpertPrefixSum`<256>` | — | — | 40 |
| mergeExpertPrefixSum | 40 | 40 | 40 |
| expandInputRows_opt | 40 | 40 | 40 |
| setup_cutlass_group_ptrs | 80 | 80 | 80 |
| CUTLASS GEMM1 | 40 | 40 | 40 |
| doActivation_opt | 40 | 40 | 40 |
| CUTLASS GEMM2 | 40 | 40 | 40 |
| finalizeMoeRouting_opt | 40 | 40 | 40 |

### 14.2 单层 MoE 平均耗时对比（warm iteration, 40 层平均）

| Kernel | 7392 tokens | 800 tokens | 1 token (decode) |
|---|---:|---:|---:|
| topkGating | 14.2 us | 4.0 us | 4.3 us |
| vec_elem × 6 | ~6 us | ~6 us | ~8 us |
| blockExpertPrefixSum | 31.3 us | 3.9 us | 1.8 us |
| globalExpertPrefixSum | 2.0 us | 1.3 us | 1.5 us |
| mergeExpertPrefixSum | 12.2 us | 2.0 us | 1.3 us |
| expandInputRows_opt | 141.8 us | 20.6 us | 8.1 us |
| setup_group_ptrs × 2 | 3.4 us | 3.2 us | 4.2 us |
| CUTLASS GEMM1 | ~506 us | ~253 us | ~19 us |
| doActivation_opt | 115.9 us | 11.7 us | 5.4 us |
| CUTLASS GEMM2 | ~407 us | ~195 us | ~16 us |
| finalizeMoeRouting_opt | 199.4 us | 11.9 us | 1.6 us |
| **单层总计** | **~1440 us** | **~513 us** | **~71 us** |

### 14.3 单层 MoE 耗时占比（7392 tokens）

```text
CUTLASS GEMM1 (gate+up)   ████████████████████████████████████  506 us  (35.1%)
CUTLASS GEMM2 (down)       █████████████████████████████        407 us  (28.3%)
finalizeMoeRouting_opt      ██████████████                       199 us  (13.8%)
expandInputRows_opt         ██████████                           142 us   (9.9%)
doActivation_opt            ████████                             116 us   (8.0%)
blockExpertPrefixSum        ██                                    31 us   (2.2%)
topkGating                  █                                     14 us   (1.0%)
mergeExpertPrefixSum        █                                     12 us   (0.8%)
vec_elem × 6                                                       6 us   (0.4%)
setup_group_ptrs × 2                                               3 us   (0.2%)
globalExpertPrefixSum                                              2 us   (0.1%)
                                                              ─────────
                                                              ~1440 us
```

### 14.4 MoE 总耗时（全部 40 层，4 次 warm iteration 平均）

| Phase | MoE 总耗时 | 占 phase GPU 时间 |
|---|---:|---:|
| 7392-token prefill | 230.9 ms | 117.8% (overlapped) |
| 800-token prefill | 67.2 ms | 109.0% (overlapped) |
| decode | 12.0 ms | 126.2% (overlapped) |

> 超过 100% 是因为统计包含了 async kernel launch 的 overlap 效应。

---

## 15. ATREX vs FlashInfer CUTLASS 对比（7392 tokens, 单层）

| Kernel 阶段 | FlashInfer CUTLASS | ATREX | 加速比 |
|---|---:|---:|---:|
| Input quant + expand | 265 us (cvt 28 + expand 237) | 142 us (expandInputRows_opt) | **1.87x** |
| GEMM1 (gate+up) | 500 us | 506 us | 0.99x |
| doActivation | 219 us | 116 us | **1.89x** |
| GEMM2 (down) | 373 us | 407 us | 0.92x |
| finalizeMoeRouting | 214 us | 199 us | 1.07x |
| Routing (prefixSum 等) | 47 us | 52 us | 0.90x |
| Metadata (strides/ptrs) | 3 us | 3 us | 1.0x |
| **单层总计** | **~1964 us** | **~1440 us** | **1.36x** |

**ATREX 的主要优势**：
1. **expandInputRows_opt**：融合 scatter + quantization，省去独立的 `cvt_fp16_to_fp4`，减少 kernel launch 和内存读写。
2. **doActivation_opt**：优化后的激活 kernel，接收 BF16 accumulator 直接输出，速度几乎翻倍。

**ATREX 的相对劣势**：
1. GEMM2 比 FlashInfer CUTLASS 慢约 34 us（不同的 CUTLASS tile 配置）。
2. Decode 使用 3-kernel routing 链而非 FlashInfer 的融合 kernel（但 decode 时每个 kernel 只需 1-2 us，差异可忽略）。

---

## 16. 端到端 MoE 数据流总结

```text
hidden_states [M, 2048] BF16
  │
  ├─ gate linear → router_logits [M, 256]
  │
  ├─ topkGating → topk_ids [M, 8], topk_weights [M, 8]
  │
  ├─ vec_elem × 6  (routing weight 归一化/缩放)
  │
  ├─ blockExpertPrefixSum ──┐
  ├─ globalExpertPrefixSum ─┤ → expert routing map
  ├─ mergeExpertPrefixSum ──┘
  │
  ├─ expandInputRows_opt → expanded_fp4 [M*8, 1024] uint8  ★ 融合 scatter+quant
  │
  ├─ setup_cutlass_group_ptrs → GEMM metadata
  ├─ CUTLASS GEMM1 → gemm1_out [M*8, 1024] BF16  (W13 @ expanded_fp4)
  │
  ├─ doActivation_opt → fp4_mid [M*8, 256] uint8  ★ SiLU + FP4 requant
  │
  ├─ setup_cutlass_group_ptrs → GEMM metadata
  ├─ CUTLASS GEMM2 → expert_out [M*8, 2048] BF16  (W2 @ fp4_mid)
  │
  └─ finalizeMoeRouting_opt → routed_out [M, 2048] BF16  ★ unpermute + weighted sum
```
