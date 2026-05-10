# FlashInfer CUTLASS MoE 后端全流程解析

> 以 Qwen3.5-35B-A3B（NVFP4 量化，SM120 Blackwell）为例，从 Python API 到每一个 GPU kernel，完整讲清楚 FlashInfer CUTLASS MoE 后端的架构与执行流程。
>
> **Trace 来源**：单卡 Blackwell GPU，`max_num_batched_tokens=8192, enable_prefix_caching=True, enable_chunked_prefill=True`
>
> **Profiling 数据**：包含 7392 token prefill 和 800 token prefill 的 warm iteration 实测耗时。

---

## 1. MoE Expert 基本原理

MoE 的核心：每个 token 不经过所有 FFN 参数，而是由 router 选择少量 expert 计算后加权求和。

Qwen3.5-35B-A3B 参数：

| 项 | 值 |
|---|---:|
| hidden size `H` | 2048 |
| routed experts `E` | 256 |
| experts per token `top_k` | 8 |
| expert intermediate size `I` | 512 |

每个 expert 是 SwiGLU MLP：

```
u = W13[e] @ x          # [2*I] = [1024], gate/up fused
gate, up = split(u)     # [512], [512]
mid = silu(gate) * up   # [512]
y = W2[e] @ mid         # [H] = [2048]
```

全模型 routed expert 权重 shape：

```
W13: [256, 1024, 2048]   (NVFP4 packed: [256, 1024, 1024] uint8)
W2 : [256, 2048, 512]    (NVFP4 packed: [256, 2048, 256]  uint8)
```

NVFP4 block scale（group size 16）：

```
W13 scale: [256, 1024, 128]
W2  scale: [256, 2048, 32]
```

---

## 2. 整体架构：从 Python 到 GPU Kernel

### 2.1 调用链总览

```
Python: cutlass_fused_moe(input, ...)           # flashinfer/fused_moe/core.py
  |
  +-> get_cutlass_fused_moe_module("120")        # @functools.cache, JIT 编译
  |     +-> gen_cutlass_fused_moe_sm120_module() # flashinfer/jit/fused_moe.py
  |     |     +-> generate_gemm_operations()     # 生成 CUTLASS kernel 实例化代码
  |     |     +-> gen_jit_spec(sources, flags)   # 返回 JitSpec
  |     +-> jit_spec.build_and_load()            # nvcc 编译 -> .so -> TVM-FFI 加载
  |
  +-> MoERunner.init(dtypes)                     # 创建 C++ FusedMoeRunner
  |     +-> CutlassMoeFCRunner<fp4,fp4,bf16>()   # 实例化 CUTLASS runner
  |     +-> MoeGemmRunner 检测 SM120             # 查询可用 tile configs
  |
  +-> AutoTuner.choose_one()                     # GEMM1/GEMM2 分别自动调优
  |
  +-> FusedMoeRunner.runMoe(...)                 # C++ 主执行路径
        +-> configureWsPtrs()                    # 切分 workspace buffer
        +-> Expert Map Building                  # token-expert 排序
        +-> expandInputRowsKernel                # 输入行复制+排列+scale搬运
        +-> computeStridesTmaWarpSpecializedKernel # 计算 per-expert TMA 元数据
        +-> GEMM1 (CUTLASS Grouped GEMM)         # FC1 矩阵乘
        +-> doActivationKernel                   # SwiGLU + FP4 重量化
        +-> GEMM2 (CUTLASS Grouped GEMM)         # FC2 矩阵乘
        +-> finalizeMoeRoutingKernel             # 反排列 + 加权求和
```

### 2.2 vLLM 的接入点

vLLM 通过 `FlashInferExperts.apply()` 调用 FlashInfer：

```
Qwen3NextSparseMoeBlock.forward
  -> FusedMoE.forward
    -> MoERunner._forward_impl
      -> router.select_experts(...)              # vLLM 负责 topk routing
      -> modular_kernel.apply()
        -> prepare (input 量化)                  # no_dp_ep.py
        -> FlashInferExperts.apply(...)           # -> cutlass_fused_moe()
        -> finalize (通常已融合)
```

vLLM 传给 FlashInfer 的关键参数：

```python
cutlass_fused_moe(
    input                  = a1q,                # bf16 或 NVFP4 packed（取决于 prepare 是否提前量化）
    token_selected_experts = topk_ids,           # [T, 8] int32
    token_final_scales     = topk_weights,       # [T, 8] fp32
    fc1_expert_weights     = w13.view(torch.long),
    fc2_expert_weights     = w2.view(torch.long),
    output                 = output,
    output_dtype           = torch.bfloat16,
    quant_scales           = [
        gemm1_act_global_scale,
        gemm1_weight_block_scales,
        gemm1_dequant_alpha,
        gemm2_act_global_scale,
        gemm2_weight_block_scales,
        gemm2_dequant_alpha,
    ],
)
```

---

## 3. JIT 编译系统

### 3.1 模块生成

`gen_cutlass_fused_moe_sm120_module()`（`flashinfer/jit/fused_moe.py:58`）做两件事：

**第一步：代码生成。** 调用 `generate_gemm_operations(output_dir, "120")`，在 `output_dir` 下生成 `.generated.cu` 文件，包含所有 CUTLASS kernel 实例化宏调用（`INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(...)`）。

SM120 NVFP4 W4A4 的 GEMM 配置：

| Activation dtype | Weight dtype | Output dtype |
|---|---|---|
| `e2m1` (FP4) | `e2m1` (FP4) | `bf16` / `f16` |

SM120 NVFP4 MoE 当前使用的 CTA tile shapes（MNK 格式）：

```
[128, 128, 128]
[128, 128, 256]
[256, 128, 128]
[128, 256, 128]
```

Cluster shape 固定为 `1x1x1`（SM120 不使用 multi-CTA clustering）。

每个 tile shape 都会生成 `NONE`（普通 GEMM）和 `FINALIZE`（GEMM2 epilogue 融合 finalize）两种 epilogue 变体，以及 `SwapAB=True/False` 两种矩阵布局。

**第二步：构建 JitSpec。** 列出所有需要编译的源文件：

```
csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu   # TVM-FFI 绑定
csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu        # 模板实例化
csrc/nv_internal/.../moe_gemm/moe_gemm_kernels_*.cu                     # CUTLASS kernel 实现
<gen_dir>/*.generated.cu                                                 # 代码生成的实例化
```

关键编译 flags：

```
-DCOMPILE_BLACKWELL_TMA_GEMMS
-DCOMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS
-DENABLE_BF16 -DENABLE_FP8 -DENABLE_FP4
-DUSING_OSS_CUTLASS_MOE_GEMM
```

### 3.2 自动调优

FlashInfer 对 GEMM1 和 GEMM2 分别自动调优（`core.py:486-515`）。每个 GEMM 独立选择最优 tile config（CTA shape + schedule），调优过程通过 `runGemmProfile()` 在 GPU 上实际运行候选 config 并计时。

---

## 4. C++ 绑定层

### 4.1 FusedMoeRunner

`flashinfer_cutlass_fused_moe_binding.cu` 定义了 `FusedMoeRunner`（继承 `tvm::ffi::ModuleObj`），是 Python 和 C++ 之间的桥梁。

构造时根据 dtype 组合实例化对应的 `CutlassMoeFCRunner` 模板：

```cpp
// NVFP4 W4A4 路径
CutlassMoeFCRunner<__nv_fp4_e2m1,   // activation type
                   __nv_fp4_e2m1,   // weight type
                   __nv_bfloat16,   // output type
                   ...>
```

通过 TVM-FFI 导出的方法：

| 方法名 | 功能 |
|---|---|
| `run_moe` | 主执行入口 |
| `run_moe_min_latency` | 导出接口；Blackwell `cutlass_fused_moe(min_latency_mode=True)` 当前未实现 |
| `run_gemm_profile` | 单个 GEMM config 计时（自动调优用） |
| `get_gemm1_tactic_count` | 获取 GEMM1 候选 config 数 |
| `get_gemm2_tactic_count` | 获取 GEMM2 候选 config 数 |
| `get_tactic_occupancy` | 获取某 config 的 SM 占用率 |

### 4.2 runMoe 入口

`FusedMoeRunner::runMoe()`（`binding.cu:245`）做参数校验、构造 `QuantParams`，然后调用 `CutlassMoeFCRunner::runMoe()`。

对 NVFP4，`QuantParams` 包含 6 个量化参数：

```
fp4.fc1.act_global_scale        # GEMM1 activation 全局 scale
fp4.fc1.weight_block_scales     # GEMM1 weight block scales
fp4.fc1.alpha                   # GEMM1 dequant alpha
fp4.fc2.act_global_scale        # GEMM2 activation 全局 scale
fp4.fc2.weight_block_scales     # GEMM2 weight block scales
fp4.fc2.alpha                   # GEMM2 dequant alpha
```

---

## 5. 数据流图

```
hidden_states [T, H] bf16
    │
    ▼ ③ cvt_fp16_to_fp4 (vLLM prepare)
    │
a1q [T, H/2] FP4 packed + a1q_scale [T, H/16]
    │
    ▼ ④⑤⑥ Expert Map Building (3-step sort)
    │                    ↓
    │     permuted_row_to_unpermuted_row [T*top_k]
    │     expert_first_token_offset [E+1]
    │
    ▼ ⑦ expandInputRowsKernel
    │
permuted_input [T*top_k, H/2] FP4 + permuted_act_scale + permuted_router_scales
    │
    ▼ ⑧ computeStridesTmaWarpSpecializedKernel
    │     (per-expert shapes, pointers, strides)
    │
    ▼ ⑨ CUTLASS Grouped GEMM (GEMM1 / FC1)
    │     A: permuted_input FP4   ×   B: W13[e] FP4   →   D: fc1_result bf16
    │     [M_e, 2048]                 [1024, 2048]         [M_e, 1024]
    │
fc1_result [T*top_k, 1024] bf16  (gate | up)
    │
    ▼ ⑩ doActivationKernel (SwiGLU + FP4 requant)
    │     silu(gate) * up → quantize to FP4
    │
intermediate [T*top_k, 256] FP4 packed + inter_scale [T*top_k, 32]
    │
    ▼ ⑪ CUTLASS Grouped GEMM (GEMM2 / FC2)
    │     A: intermediate FP4   ×   B: W2[e] FP4   →   D: fc2_result bf16
    │     [M_e, 512]                [2048, 512]         [M_e, 2048]
    │
fc2_result [T*top_k, 2048] bf16
    │
    ▼ ⑫ finalizeMoeRoutingKernel (unpermute + weighted sum)
    │     output[t] = Σ_k topk_weight[t,k] * fc2_result[permuted_row(t,k)]
    │
routed_output [T, H] bf16
```

---

## 6. GPU Kernel 序列（逐个解析）

从 trace 中观察到的一个 MoE 层的完整 GPU kernel 序列：

```
 #   Offset(us)   Dur(us)   Kernel                                    阶段
───  ──────────   ────────   ──────────────────────────────────        ────────
 ①        0.0      14.46    topkGating                                Routing
 ②       66.1       0.90    FillFunctor<int> (清零 scale buffer)      Prepare
 ③       78.3      28.00    cvt_fp16_to_fp4                           Prepare
 ④      376.7      31.71    blockExpertPrefixSum<1024>                 Dispatch
 ⑤      408.4       2.05    globalExpertPrefixSumLarge<1024>           Dispatch
 ⑥      410.3      13.09    mergeExpertPrefixSum                      Dispatch
 ⑦      422.9     236.93    expandInputRows                           Dispatch
 ⑧      659.8       2.43    computeStridesTmaWarp                     Dispatch
 ⑨      662.2     500.25    CUTLASS GroupedGEMM  (GEMM1: gate+up)     Compute
 ⑩     1162.3     218.81    doActivation  (SiLU + FP4 requant)        Compute
 ⑪     1377.4     372.93    CUTLASS GroupedGEMM  (GEMM2: down)        Compute
 ⑫     1750.2     213.53    finalizeMoeRouting                        Finalize
                  ────────
        total    ~1964 us   (单层 MoE routed expert GPU 总耗时, M=7392)
```

- **Offset(us)**：该 kernel 相对于本层第一个 kernel（`topkGating`）的 GPU 启动时间偏移。
- **Dur(us)**：该 kernel 单次调用在 GPU 上的实际执行耗时（warm iteration, M=7392）。

其中 ①②③ 由 vLLM 侧发起（router + prepare），④-⑫ 由 FlashInfer `CutlassMoeFCRunner::runMoe()` 发起。

下面逐个解析。

---

### 6.1 Kernel ①：topkGating（vLLM Router）

```
void vllm::moe::topkGating<8, 256, 4, 16, 32, int, __nv_bfloat16, ScoringFunc::0>(...)
```

**源码位置**：vLLM `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`

**模板参数含义**：`<top_k=8, num_experts=256, tokens_per_thread=4, thread_group_size=16, warp_size=32, int, bf16, softmax>`

**功能**：对 router logits 做 softmax + top-k 选择。

**输入**：
- `router_logits`: `[T, 256]` bf16 — gate linear 的输出

**输出**：
- `topk_weights`: `[T, 8]` fp32 — softmax 后的 top-k 权重
- `topk_ids`: `[T, 8]` int32 — 选中的 expert ID

**要点**：这是 vLLM 的 kernel，不属于 FlashInfer。FlashInfer 接收的是已经计算好的 `topk_ids` 和 `topk_weights`。

---

### 6.2 Kernel ②③：Input NVFP4 量化（vLLM Prepare）

```
② void FillFunctor<int>(...)                  # 清零 scale output tensor
③ void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>(...)
```

**源码位置**：vLLM `vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py` → `moe_kernel_quantize_input` → `scaled_fp4_quant`

**功能**：将 bf16 hidden_states 量化为 NVFP4 packed activation + block scale。

**输入**：
- `hidden_states`: `[T, H]` bf16
- `act_global_scale`: fp32 — NVFP4 全局 scale

**输出**：
- `a1q`: `[T, H/2]` uint8 — packed FP4 activation（两个 FP4 pack 到一个 uint8）
- `a1q_scale`: `[T, H/16]` — block scale（group size 16）

**要点**：
- 这一步发生在 vLLM 的 `prepare` 阶段（`defer_input_quant=False`）。
- FlashInfer 的 `FlashInferExperts` 声明 `expects_unquantized_inputs=False`（对 NVFP4），所以 vLLM prepare 提前量化。
- 提前量化的好处：在 EP/all-to-all 场景下可以减少通信量（传输 packed FP4 而非 bf16）。
- 量化完成后，传给 FlashInfer 的 `input` 已经是 NVFP4 packed 格式。

---

### 6.3 Kernel ④⑤⑥：Expert Map Building（Token-Expert 排序）

这三个 kernel 将 `[T, top_k]` 的 token-expert pair 按 expert 分组排列，生成 permutation mapping 供后续 grouped GEMM 使用。

**源码位置**：`cutlass_fused_moe_kernels.cuh` 的 `threeStepBuildExpertMapsSortFirstToken()`

> 注：当 `num_tokens <= 256` 时会尝试使用 `fusedBuildExpertMapsSortFirstTokenKernel`（单个 CTA，用 CUB BlockRadixRank 完成全部排序）。超过 256 tokens 时 fallback 到下面的三步流程。

#### Kernel ④：blockExpertPrefixSumKernel

```
void tensorrt_llm::kernels::cutlass_kernels::blockExpertPrefixSumKernel<1024>(...)
```

**功能**：Phase 1 — 按 block 分段，统计每个 expert 在每个 block 中收到多少 token。

**Grid**: `[num_experts_per_node, num_blocks_per_seq]`，每个 CTA 1024 threads。

**输入**：
- `token_selected_experts`: `[T * top_k]` — 展开后的 expert ID 列表

**输出**：
- `blocked_expert_counts`: `[E_local, num_blocks]` — 每个 expert 在每个 block 中的 token 计数
- `blocked_row_to_unpermuted_row`: `[E_local, T]` — block 级别的局部紧凑索引

**内部逻辑**：
1. 每个 thread 检查自己负责的 token 是否选择了当前 expert
2. 使用 `cub::BlockScan::ExclusiveSum` 计算 block 内偏移
3. 匹配的 token 将其原始行号写入 scratch buffer
4. 最后一个 thread 写入当前 (expert, block) 的总计数

#### Kernel ⑤：globalExpertPrefixSumLargeKernel

```
void tensorrt_llm::kernels::cutlass_kernels::globalExpertPrefixSumLargeKernel<1024>(...)
```

**功能**：Phase 2 — 对 `blocked_expert_counts` 做全局前缀和，产生每个 expert 的起始偏移。

**Grid**: `[1]`，单个 CTA，1024 threads。

**输入**：
- `blocked_expert_counts`: `[E_local * num_blocks]`

**输出**：
- `blocked_expert_counts_cumsum`: `[E_local * num_blocks]` — 累计和
- `expert_first_token_offset`: `[E_local + 1]` — 每个 expert 的第一个 token 在排列后数组中的偏移

**内部逻辑**：
1. 每个 thread 负责多个元素，先局部求和
2. `cub::BlockScan::ExclusiveSum` 计算 thread 级别的前缀
3. 回写每个元素的全局累计和
4. 在 expert 边界处（`offset % num_blocks == 0`）写入 `expert_first_token_offset`

#### Kernel ⑥：mergeExpertPrefixSumKernel

```
void tensorrt_llm::kernels::cutlass_kernels::mergeExpertPrefixSumKernel(...)
```

**功能**：Phase 3 — 合并 block 级别结果，生成全局 permutation mapping。

**Grid**: `[num_experts_per_node, num_blocks_per_seq]`

**输入**：
- `blocked_expert_counts`、`blocked_expert_counts_cumsum`、`blocked_row_to_unpermuted_row`

**输出**：
- `permuted_row_to_unpermuted_row`: `[expanded_num_rows]` — 排列后行号 → 原始行号
- `unpermuted_row_to_permuted_row`: `[expanded_num_rows]` — 原始行号 → 排列后行号

**内部逻辑**：
1. 每个 CTA 读取自己 block 的计数和累计偏移
2. `permuted_row = cumsum_offset + threadIdx.x`
3. 写入双向映射

#### 排序结果示意

假设 T=4, top_k=2, E=4：

```
token_selected_experts = [[0,2], [1,3], [0,1], [2,3]]

排序后:
expert 0: token 0, token 2
expert 1: token 1, token 2
expert 2: token 0, token 3
expert 3: token 1, token 3

permuted_row_to_unpermuted_row = [0, 4, 3, 5, 2, 7, 1, 6]
                                  ^e0  ^e1  ^e2    ^e3
```

---

### 6.4 Kernel ⑦：expandInputRowsKernel（输入行复制 + 排列 + scale 搬运）

```
void tensorrt_llm::kernels::cutlass_kernels::expandInputRowsKernel<
    __nv_fp4_e2m1, __nv_fp4_e2m1, BlockScalingType::NVFP4, false>(...)
```

**源码位置**：`cutlass_fused_moe_kernels.cuh:1366`

**功能**：按 permutation mapping 复制并排列输入行。如果一个 token 被路由到 K 个 expert，它的输入行会被复制 K 次到对应的 permuted 位置。同时处理 NVFP4 block scale 的搬运和 router scale 的排列。

**Grid**: `min(smCount * 8, num_valid_tokens)` blocks, 256 threads each。

**模板参数**：
- `InputActivationsType = __nv_fp4_e2m1`：输入已是 NVFP4 packed
- `ExpandedActivationsType = __nv_fp4_e2m1`：输出也是 NVFP4 packed
- `BlockScalingType = NVFP4`：需要搬运 block scale
- `PRE_QUANT_AWQ = false`：不需要 AWQ pre-quant

**输入**：
- `unpermuted_input`: `[T, H/2]` uint8 — NVFP4 packed activation
- `unpermuted_scales`: `[T * top_k]` fp32 — router softmax 权重
- `permuted_row_to_unpermuted_row`: permutation mapping
- `input_sf`: `[T, H/16]` — activation block scale
- `fc1_act_global_scale`: fp32

**输出**：
- `permuted_output`: `[expanded_num_rows, H/2]` uint8 — 排列后的 NVFP4 packed activation
- `permuted_scales`: `[expanded_num_rows]` fp32 — 排列后的 router 权重
- `fc1_act_sf_flat`: 排列后的 activation block scale

**内部逻辑**：
1. 遍历 valid token 行（grid-strided loop）
2. 对每个 permuted row，查 `permuted_row_to_unpermuted_row` 得到原始 token 和 k_idx
3. 以 128-bit 向量化方式复制输入行
4. 对 NVFP4：同时复制 block scale factor
5. Thread 0 负责排列 router scale：`permuted_scales[row] = unpermuted_scales[k_idx]`

**要点**：这个 kernel 在 decode（T=2）时处理 `2*8=16` 个 expanded rows。对 prefill（T=4096）处理 `4096*8=32768` 个 expanded rows。

---

### 6.5 Kernel ⑧：computeStridesTmaWarpSpecializedKernel（TMA 元数据计算）

```
void tensorrt_llm::kernels::cutlass_kernels::computeStridesTmaWarpSpecializedKernel<
    __nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>(...)
```

**源码位置**：`cutlass_fused_moe_kernels.cuh:1228`

**功能**：为 CUTLASS TMA warp-specialized grouped GEMM 计算 per-expert 的元数据。CUTLASS 3.x 的 grouped GEMM 需要知道每个 group（expert）的 problem shape、数据指针、stride 和 scale 指针。

**Grid**: `ceil(E_local / 32)` blocks, `min(32, E_local)` threads — 一个 thread 负责一个 expert。

**输入**：
- `expert_first_token_offset`: `[E_local + 1]` — 每个 expert 的 token 偏移
- Weight pointers、activation pointers、scale pointers
- GEMM 维度：`gemm1_n`, `gemm1_k`, `gemm2_n`, `gemm2_k`

**输出**（写入 `TmaWarpSpecializedGroupedGemmInput` 的 device 数组）：
- `problem_shapes[expert]`: `(M, N, K)` — 其中 M = 当前 expert 的 token 数
- Stride arrays（A, B, D 矩阵的 leading dimension）
- Pointer arrays（input, weight, output 的 per-expert base pointer）
- Block scaling factor pointer arrays（NVFP4 weight scale, act scale, alpha scale）

**内部逻辑**（per expert）：
1. `num_tokens_before = expert_first_token_offset[expert]`
2. `num_tokens_to = expert_first_token_offset[expert + 1]`
3. `gemm_m = num_tokens_to - num_tokens_before`
4. 如果 `gemm_m == 0`，设 shape 为 0 后返回
5. 设置 `problem_shapes[expert] = (M, N, K)`（考虑 `swap_ab` 时交换 M 和 N）
6. 计算 input/output pointer = base + `num_tokens_before * stride`
7. 计算 weight pointer = base + `expert * weight_stride`
8. 设置 FP4 block scaling factor 指针

**要点**：这个 kernel 同时为 GEMM1 和 GEMM2 准备元数据（通过两个 `TmaWarpSpecializedGroupedGemmInput` 结构体）。

---

### 6.6 Kernel ⑨：CUTLASS Grouped GEMM（GEMM1 / FC1）

```
cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<
    GroupProblemShape<cute::tuple<l,l,l>>,
    CollectiveMma<MainloopSm120ArrayTmaWarpSpecializedBlockScaled<...>>,
    ...>>
```

**源码位置**：
- Dispatch: `moe_gemm_template_dispatch_tma_ws.h:495-512`（SM120 分支）
- Launcher: `moe_gemm_tma_ws_launcher.inl`
- Kernel: CUTLASS 3.x `GemmGrouped::run()`

**功能**：执行 FC1 的 grouped GEMM。每个 expert 是一个独立的 GEMM problem，CUTLASS 内部通过 persistent kernel 调度多个 problem 到不同的 CTA tile 上。

**计算语义**：

```
对每个 expert e:
  M_e = expert 收到的 token 数
  fc1_out[e] = permuted_input[e] @ W13[e].T
  # shape: [M_e, H/2] x [1024, H/2].T -> [M_e, 1024]
  # 实际是 NVFP4 block-scaled GEMM
```

**GEMM 规格（NVFP4 W4A4）**：
- A (activation): `[M, K]` = `[M_e, 2048]` NVFP4 packed + block scale
- B (weight): `[N, K]` = `[1024, 2048]` NVFP4 packed + block scale
- D (output): `[M, N]` = `[M_e, 1024]` bf16/fp32 accumulator

**SM120 Block-Scaled TMA Warp-Specialized**：
- 使用 CUTLASS 3.x 的 `MainloopSm120ArrayTmaWarpSpecializedBlockScaled` mainloop
- TMA（Tensor Memory Accelerator）用于异步全局内存到共享内存的数据搬运
- Block scaling 在 MMA 指令级别融合（`SM120_16x8x64_TN_VS` MMA atom）
- Warp specialization：producer warps 负责 TMA 搬运，consumer warps 负责 MMA 计算

**SM120 NVFP4 MoE 可用 tile configs（自动调优选择，MNK 格式）**：

```
[128, 128, 128]
[128, 128, 256]
[256, 128, 128]
[128, 256, 128]
```


**要点**：
- 输出是 bf16 的中间结果 `fc1_result`，包含 gate 和 up 两部分（维度 = `inter_size * 2 = 1024`）
- 不在 GEMM kernel 内做 activation，activation 由下一个独立 kernel 处理

---

### 6.7 Kernel ⑩：doActivationKernel（SwiGLU + FP4 重量化）

```
void tensorrt_llm::kernels::cutlass_kernels::doActivationKernel<
    __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16,
    GLUAdaptor<SiLu>, BlockScalingType::NVFP4>(...)
```

**源码位置**：`cutlass_fused_moe_kernels.cuh:2042`

**功能**：对 GEMM1 输出执行 SwiGLU 激活，并将结果重量化为 NVFP4 作为 GEMM2 的输入。

**Grid**: `min(smCount * 8, expanded_num_tokens)` blocks, 256 threads。

**模板参数**：
- `T = __nv_fp4_e2m1`：输出类型（FP4 用于 GEMM2 输入）
- `GemmOutputType = __nv_bfloat16`：GEMM1 输出类型
- `ActFn = GLUAdaptor<SiLu>`：SwiGLU 激活
- `BlockScalingType = NVFP4`：输出做 NVFP4 block-scale 量化

**输入**：
- `gemm_result`: `[expanded_num_rows, 1024]` bf16 — GEMM1 输出，前 512 列是 gate，后 512 列是 up（对应 vLLM `gate_up_proj` 的 `[gate_proj, up_proj]` 排列）
- `expert_first_token_offset`: 用于确定每个 token 属于哪个 expert（二分查找）
- `fc2_act_global_scale`: GEMM2 activation 全局 scale

**输出**：
- `output`: `[expanded_num_rows, 256]` uint8 — NVFP4 packed intermediate（512 个 FP4 pack 成 256 个 uint8）
- `fc2_act_sf_flat`: `[expanded_num_rows, 512/16]` — GEMM2 activation block scale

**内部逻辑**（per token per chunk）：
1. 通过二分查找 `expert_first_token_offset` 确定当前 token 的 expert
2. 加载 GEMM1 输出的 gate 部分（前半，offset = 0）和 up 部分（后半，offset = `inter_size`）
3. 执行 SwiGLU：`mid = silu(gate) * up`
4. 乘以 FP8 quant scale（如果有）
5. **FP4 重量化**：调用 `quantizePackedFPXValue()` 将 float 值量化为 packed FP4 + block scale
6. 写入 packed FP4 输出和 block scale factor

**SwiGLU 激活函数实现**：

```cpp
// GLUAdaptor<SiLu>
output = SiLu(gate) * up
       = gate * sigmoid(gate) * up
```

**FP4 重量化过程**：

```
float mid = silu(gate) * up;                // bf16 -> float 计算
float scaled = mid / fc2_act_global_scale;  // 应用全局 scale
fp4_value = quantize_to_e2m1(scaled);       // 量化到 FP4
block_scale = max(abs(chunk)) / max_fp4;    // 计算 block scale
```

**要点**：
- 这个 kernel 把 GEMM1 后的两步操作融合了：SwiGLU activation + GEMM2 input quantization
- 避免了中间 bf16 结果的一次全局内存写入和读取
- 输出已经是 GEMM2 ready 的 NVFP4 packed 格式

---

### 6.8 Kernel ⑪：CUTLASS Grouped GEMM（GEMM2 / FC2）

```
cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<
    GroupProblemShape<cute::tuple<l,l,l>>,
    CollectiveMma<MainloopSm120ArrayTmaWarpSpecializedBlockScaled<...>>,
    ...>>
```

**功能**：执行 FC2 的 grouped GEMM。

**计算语义**：

```
对每个 expert e:
  expert_output[e] = intermediate[e] @ W2[e].T
  # shape: [M_e, I/2] x [H, I/2].T -> [M_e, H]
  # 即: [M_e, 256] x [2048, 256].T -> [M_e, 2048]
  # NVFP4 block-scaled GEMM
```

**GEMM 规格**：
- A: `[M_e, 512]` NVFP4 packed + block scale（来自 doActivation 输出）
- B: `[2048, 512]` NVFP4 packed + block scale（W2 权重）
- D: `[M_e, 2048]` bf16

**两种 Epilogue 模式**：

1. **NONE（本 trace 中观察到的）**：GEMM2 只输出 `[expanded_num_rows, H]` 的 bf16 中间结果到 `fc2_result`，后续由独立的 `finalizeMoeRoutingKernel` 做加权求和。

2. **FINALIZE（可选融合路径）**：GEMM2 的 epilogue 直接融合 unpermute + weighted reduction + atomicAdd。需要先 `cudaMemsetAsync(final_output, 0)`，然后 GEMM2 epilogue 内部做：
   ```
   token_id = permuted_row_to_unpermuted_row[row] / top_k
   weight = router_scales[row]
   atomicAdd(output[token_id, :], weight * gemm_output)
   ```
   这种模式下不需要独立的 finalize kernel。

**选择逻辑**：`use_fused_finalize_` 默认为 true，但在某些条件下（LoRA、W4 groupwise）会 fallback 到独立 finalize。自动调优时也会尝试两种变体。

---

### 6.9 Kernel ⑫：finalizeMoeRoutingKernel（反排列 + 加权求和）

```
void tensorrt_llm::kernels::cutlass_kernels::finalizeMoeRoutingKernel<
    __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, ScaleMode::DEFAULT>(...)
```

**源码位置**：`cutlass_fused_moe_kernels.cuh:1658`

**功能**：将 permuted layout 的 GEMM2 输出反排列回原始 token 顺序，乘以 router 权重，对同一 token 的 top_k 个 expert 输出求和。

**Grid**: `num_rows`（每个原始 token 一个 block），256 threads。

**模板参数**：
- `OutputType = __nv_bfloat16`
- `GemmOutputType = __nv_bfloat16`
- `ScaleMode = DEFAULT`：乘以 router scale

**输入**：
- `expanded_permuted_rows`: `[expanded_num_rows, H]` bf16 — GEMM2 输出
- `scales`: `[T * top_k]` fp32 — router 权重（unpermuted）
- `unpermuted_row_to_permuted_row`: `[T * top_k]` — 反排列映射
- `token_selected_experts`: `[T * top_k]` — expert ID

**输出**：
- `reduced_unpermuted_output`: `[T, H]` bf16 — 最终 MoE 输出

**内部逻辑**（per original token, one block）：
1. 初始化 `thread_output = 0`
2. 遍历 `k_idx` = 0 到 `top_k - 1`：
   a. 检查 `expert_id` 是否在当前节点（EP 场景）
   b. 查 `unpermuted_row_to_permuted_row` 得到 permuted row index
   c. 以 128-bit 向量化方式读取 GEMM2 输出行
   d. `thread_output += router_scale * gemm_output`
3. 转换回 `OutputType` 并写入 `reduced_unpermuted_output`

**ScaleMode**：
- `DEFAULT`：`output[t] = Σ_k topk_weights[t,k] * expert_output[t,k]`
- `NO_SCALE`：`output[t] = Σ_k expert_output[t,k]`（router weight = 1.0）

**EP 变体**：当 `ep_size > 1` 且 alltoall 启用时，使用 `finalizeMoeRoutingNoFillingKernel`，不填充无效行，grid-strided loop 只处理有效 token。

---

## 7. NVFP4 量化体系总结

一个 MoE 层中有 **三处** NVFP4 量化：

| 位置 | 发生时机 | 谁做的 | 输入 | 输出 |
|---|---|---|---|---|
| GEMM1 input quant | Kernel ③ | vLLM prepare | bf16 activation | FP4 packed + block scale |
| GEMM2 input quant | Kernel ⑩ | doActivationKernel | SwiGLU bf16 result | FP4 packed + block scale |
| Weight quant | 离线/加载时 | 模型量化工具 | 原始权重 | FP4 packed + block scale |

每处量化都涉及三类 scale：

```
global_scale   — 单个 fp32 标量，将 activation/weight 的动态范围压缩到 FP4 可表示范围
block_scale    — 每 16 个元素一个 scale，细粒度补偿
alpha          — GEMM 输出的 dequant scale，恢复到目标数值范围
```

NVFP4 GEMM 的完整数值流程：

```
D = alpha * MMA(
    quantize(A, act_global_scale, act_block_scale),
    quantize(B, weight_global_scale, weight_block_scale)
)
```

其中 `MMA` 在硬件级别使用 block scale 做 block-scaled matrix multiply。

---

## 8. Prefill vs Decode 的 Kernel 差异

> 详细实测耗时数据见 [第 10 节](#10-profiling-实测数据)。

| 维度 | Prefill (T=4096) | Decode (T=1~2) |
|---|---|---|
| Expanded rows | 32768 | 8~16 |
| Expert sort | 三步 sort（tokens > 256） | fused 单 CTA sort |
| 每个 expert 的 M | 平均 ~128 | 0~几个 |
| GEMM 特征 | 计算密集，tile 利用率高 | 极小 M，内存带宽受限 |
| expandInputRows | 大量数据搬运 | 少量搬运但 kernel launch 开销相对大 |
| GEMM 性能 | TMA + warp specialization 发挥作用 | 多数 expert 的 GEMM M 极小，效率低 |
| Finalize | 大量 token 求和 | 少量 token，简单 |

### 8.1 Decode 阶段的 Routing Kernel 差异

Decode 阶段（M=1~2）的 MoE routing 使用完全不同的 kernel 组合：

```
Prefill routing 链 (3 kernels):
  blockExpertPrefixSum → globalExpertPrefixSum → mergeExpertPrefixSum

Decode routing (1 kernel):
  fusedBuildExpertMapsSortFirstTokenKernel<32, 8, 9>(...)
```

Decode 用一个融合 kernel 替代了 prefill 的 3 步 routing，因为 M=1 时只有 8 个 token-expert pair，排序和 map 构建可以在单个 CTA 内用 `CUB::BlockRadixRank` 一步完成。

模板参数含义：`<BLOCK_SIZE=32, EXPERTS_PER_TOKEN=8, LOG2_NUM_EXPERTS=9>`

关于 decode 阶段是否出现独立的 `finalizeMoeRouting`：这取决于 GEMM2 autotune 是否选择了 FINALIZE epilogue fusion。如果 autotune 选择了 `EpilogueFusion::FINALIZE`，GEMM2 epilogue 内部直接完成 unpermute + weighted reduction，不会出现独立的 `finalizeMoeRoutingKernel`；否则仍可能有独立 finalize。

### 8.2 blockExpertPrefixSum 的模板参数自适应

TRT-LLM 根据 token 数量 M 选择不同的 block size 特化：

```
M 较大（7392）→ blockExpertPrefixSumKernel<1024>
M 较小（800） → blockExpertPrefixSumKernel<512>
```

类似地，全局 prefix sum 也有两个变体：

```
M 较大（elements > 1024）→ globalExpertPrefixSumLargeKernel<1024>（多 pass）
M 较小（elements <= 1024）→ globalExpertPrefixSumKernel<512>（单 pass）
```

---

## 9. 与 CuteDSL MoE 路径的对比

FlashInfer 还有一条 Blackwell CuteDSL MoE 路径（`flashinfer/fused_moe/cute_dsl/`），支持 SM100 和 SM103（`@supported_compute_capability([100, 103])`）。

| 维度 | CUTLASS 后端（本文） | CuteDSL 后端 |
|---|---|---|
| vLLM 当前集成 | 是（`FlashInferExperts`） | 否（FlashInfer 独立 API，无 vLLM adapter） |
| Expert sorting | `threeStepBuildExpertMaps`（3 kernel）或 `fusedBuildExpertMaps`（1 kernel） | `moe_sort`（1~3 kernel，取决于 token 数） |
| Input permute | `expandInputRowsKernel` 显式搬运 | GEMM1 kernel 内通过 LDGSTS + `token_id_mapping` gather（无显式搬运） |
| GEMM1 | CUTLASS Grouped GEMM | CuteDSL `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4` |
| Activation + requant | 独立 `doActivationKernel` | 融合在 GEMM1 epilogue（SwiGLU + FP4 requant 在同一 kernel 的 epilogue warp 中） |
| GEMM2 | CUTLASS Grouped GEMM | CuteDSL `blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4` |
| Finalize | 独立 `finalizeMoeRoutingKernel`（或 GEMM2 epilogue 融合） | GEMM2 epilogue 内 `vectorized_atomic_add` / `blk_reduce` scatter-add |
| Kernel 数量 | 8~9 个 GPU kernel | 3~5 个 GPU kernel + 1 个 cudaMemsetAsync |
| 中间 buffer | 显式 permuted input + fc1_result + fc2_result | 只有 intermediate FP4（`gemm1_output` uint8 + scale） |

### 9.1 CuteDSL `_moe_core_impl` 的 4 步流程

```python
# flashinfer/fused_moe/cute_dsl/fused_moe.py: _moe_core_impl()

# Step 1: moe_sort — 1~3 个 CUDA kernel
#   小 M: 单 CTA kernel（1 kernel）
#   中 M: initExpertCounts + coopKernel（2 kernels）
#   大 M: initExpertCounts + histogram + offsets（3 kernels）
moe_sort(token_selected_experts, ...) -> permuted_idx_to_expanded_idx, ...

# Step 2: GEMM1 + gather + SwiGLU + FP4 requant — 1 个 CUDA kernel
#   输入: 未排列的 FP4 activation（通过 token_id_mapping gather）
#   输出: FP4 packed intermediate + scale
blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4(
    a=x, token_id_mapping=permuted_idx_to_expanded_idx, ...)

# Step 3: zero output — cudaMemsetAsync（非自定义 kernel）
moe_output.zero_()

# Step 4: GEMM2 + weighted scatter-add finalize — 1 个 CUDA kernel
#   epilogue 内直接 atomicAdd 到 output[token_id]
blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
    a=intermediate, token_final_scales=...,
    permuted_idx_to_expanded_idx=..., scatter_out=moe_output, ...)
```

### 9.2 关键设计差异

**为什么 CuteDSL 不需要 `expandInputRows`**：GEMM1 kernel 内部用 LDGSTS（Load Global Store Shared）指令直接从未排列的输入中按 `token_id_mapping` gather 到 shared memory，省去了一个显式搬运 kernel。

**为什么 CuteDSL 不需要独立的 `finalizeMoeRouting`**：GEMM2 的 epilogue 在每个 CTA 完成局部 GEMM 后，直接将 `router_scale * gemm_output` 通过 `vectorized_atomic_add_bf16x8`（或 TMA block reduce `blk_reduce_bf16`）原子加到 `output[token_id]`。这要求 output 预先清零（Step 3）。

**为什么 CuteDSL 不需要 `computeStridesTmaWarp`**：CuteDSL 使用 contiguous layout（每个 expert 的 token 在排列后的数组中连续），tile-to-expert 的映射通过 `tile_idx_to_expert_idx` 直接索引，不需要为每个 expert 计算独立的 strides/pointers。

两条路径数学语义完全一致：

```
out[t] = Σ_k topk_weights[t,k] * W2[e_k] @ silu_gate(W13[e_k] @ x[t])
```

差异在于 kernel 划分边界和融合策略。CuteDSL 路径融合更激进（更少 kernel launch、更少中间 buffer），CUTLASS 路径更通用（支持更多 dtype 组合和更多 SM 架构）。

---

## 10. Profiling 实测数据

> 数据来源：Blackwell GPU 单卡，Qwen3.5-35B-A3B NVFP4，`max_num_batched_tokens=8192`，warm iteration。
>
> 模型共 40 个 MoE 层，每层各 kernel 调用 1 次。

### 10.1 单层 MoE Kernel 执行时序（M=7392 prefill）

```
 #   Offset(us)   Dur(us)   Kernel
───  ──────────   ────────   ──────────────────────────────────
 ①        0.0      14.46    topkGating
 ②       66.1       0.90    elementwise (topk_ids cast)
 ③       78.3      28.00    cvt_fp16_to_fp4
 ④      376.7      31.71    blockExpertPrefixSum<1024>
 ⑤      408.4       2.05    globalExpertPrefixSumLarge<1024>
 ⑥      410.3      13.09    mergeExpertPrefixSum
 ⑦      422.9     236.93    expandInputRows
 ⑧      659.8       2.43    computeStridesTmaWarp
 ⑨      662.2     500.25    CUTLASS GroupedGEMM  (GEMM1: gate+up)
 ⑩     1162.3     218.81    doActivation  (SiLU + FP4 requant)
 ⑪     1377.4     372.93    CUTLASS GroupedGEMM  (GEMM2: down)
 ⑫     1750.2     213.53    finalizeMoeRouting
                  ────────
        total    ~1964 us
```

### 10.2 单层 MoE 平均耗时对比（40 层平均）

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

### 10.3 单层 MoE 耗时占比（M=7392）

```
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

**耗时分布**：
- **GEMM 主体**（44.5%）：GEMM1 + GEMM2 合计占比最大。
- **数据重排与激活**（34.1%）：`expandInputRows`（scatter/gather）、`doActivation`（SwiGLU + FP4 requant）、`finalizeMoeRouting`（unpermute + weighted reduce）各占 10-12%。注意 `doActivation` 不是纯搬运操作，还包含 SiLU 激活和 FP4 量化计算。
- **Routing 准备**（2.4%）：prefix sum + merge 开销很小。
- **Metadata 计算**（0.2%）：`computeStridesTmaWarp` 几乎可以忽略。

### 10.4 每 phase 的 kernel 调用次数

| Kernel | per 800 prefill | per 7392 prefill | per decode |
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
| finalizeMoeRouting | 40 | 40 | 本 trace 为 —；若 GEMM2 未选 FINALIZE epilogue 仍可能出现 |

### 10.5 GEMM Launch 配置（M=7392）

#### GEMM1

| 项 | 值 |
|---|---|
| grid | `[1, 110, 1]` — 256 个 expert 分组后的 tile 数 |
| block | `[384, 1, 1]` — warp-specialized（3-4 warps） |
| regs/thread | 168 |
| shared mem | 89088 bytes (87 KB) |

#### GEMM2

| 项 | 值 |
|---|---|
| grid | `[1, 110, 1]` |
| block | `[384, 1, 1]` |
| regs/thread | 168 |
| shared mem | 98304 bytes (96 KB) |

GEMM2 shared memory（96 KB）比 GEMM1（87 KB）大，因为 W2 的 N 维（H=2048）大于 W13 的 N 维（2I=1024）。

#### expandInputRows

| 项 | 7392 tokens | 800 tokens |
|---|---:|---:|
| grid | `[880, 1, 1]` | `[880, 1, 1]` |
| block | `[256, 1, 1]` | `[256, 1, 1]` |
| regs/thread | 40 | 40 |
| shared mem | 0 | 0 |

Grid 大小相同（880 = `smCount * 8`），实际工作量差异体现在每个 block 内的 grid-strided loop 迭代次数。纯 memory-bound scatter 操作，FP4 packed 格式让数据量比 BF16 小 4x。
