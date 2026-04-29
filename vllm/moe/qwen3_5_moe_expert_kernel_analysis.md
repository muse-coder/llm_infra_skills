# Qwen3.5 MoE Expert Kernel 层级解析

> 目标：单独解析 Qwen3.5-35B-A3B 中 routed expert 的计算原理、vLLM 的 MoE 封装，以及 FlashInfer 对 NVFP4 MoE 的 kernel 实现方式。
>
> 本文暂不展开 `AtrexNvFp4Experts`。

---

## 1. MoE Expert 的基本原理

MoE（Mixture of Experts）的核心思想是：每个 token 不经过所有 FFN 参数，而是由 router 选择少量专家计算，再把这些专家输出加权求和。

以 Qwen3.5-35B-A3B 为例：

| 项 | 值 |
|---|---:|
| hidden size `H` | 2048 |
| routed experts `E` | 256 |
| experts per token `top_k` | 8 |
| expert intermediate size `I` | 512 |
| shared expert intermediate size | 512 |

对每个 token `x[t]`，routed expert 的语义计算是：

```text
router_logits[t] = gate(x[t])                         # [256]
topk_ids[t], topk_weights[t] = topk(router_logits[t]) # [8], [8]

routed_out[t] = sum_j topk_weights[t, j] * Expert[topk_ids[t, j]](x[t])
```

每个 expert 是一个 SwiGLU MLP：

```text
u = W13[e] @ x          # [2 * I] = [1024], gate/up fused
gate, up = split(u)     # [512], [512]
m = silu(gate) * up     # [512]
y = W2[e] @ m           # [H] = [2048]
```

所以单个 routed expert 的逻辑权重 shape 是：

```text
W13[e]: [2 * I, H] = [1024, 2048]
W2[e] : [H, I]     = [2048, 512]
```

全部 256 个 routed experts 的逻辑 shape：

```text
W13: [256, 1024, 2048]
W2 : [256, 2048, 512]
```

对 batch/sequence 维度展开后：

| 阶段 | Prefill `T=4096` | Decode `T=1` |
|---|---:|---:|
| router logits | `[4096, 256]` | `[1, 256]` |
| top-k pairs | `4096 * 8 = 32768` | `1 * 8 = 8` |
| routed output | `[4096, 2048]` | `[1, 2048]` |

prefill 阶段 token-expert pair 较多，适合 grouped GEMM；decode 阶段只有 8 个 token-expert pair，计算量很小，但需要读取 8 个 expert 的权重，通常更受内存带宽和 kernel launch 开销影响。

---

## 2. Shared Expert 与 Routed Experts

Qwen3.5 MoE block 实际包含两条分支：

```text
input x [T, 2048]
  ├─ shared expert: 所有 token 都计算一次 dense MLP
  └─ routed experts: router 选择 top-8 experts

output = shared_out + routed_out
```

shared expert 不是 router 的 top-k 结果之一，而是一个额外 dense MLP：

```text
gate_up = x @ W_shared_gate_up.T       # [T, 1024]
gate, up = split(gate_up)              # [T, 512], [T, 512]
mid = silu(gate) * up                  # [T, 512]
shared = mid @ W_shared_down.T         # [T, 2048]
shared = sigmoid(shared_expert_gate(x)) * shared
```

本文重点是 routed expert kernel。shared expert 在 vLLM 中可以和 routed expert 分支重叠执行，但语义上就是一个普通 dense SwiGLU MLP。

---

## 3. vLLM 中的 MoE 封装层级

Qwen3.5 MoE 使用 `Qwen3NextSparseMoeBlock` 作为 sparse block 实现，关键对象是：

```text
Qwen3NextSparseMoeBlock
  ├─ gate: ReplicatedLinear(2048 -> 256)
  ├─ shared_expert_gate: ReplicatedLinear(2048 -> 1)
  ├─ shared_expert: Qwen3NextMLP
  └─ experts: FusedMoE
```

对应文件：

```text
vllm/model_executor/models/qwen3_next.py
vllm/model_executor/models/qwen3_5.py
vllm/model_executor/layers/fused_moe/layer.py
vllm/model_executor/layers/fused_moe/runner/moe_runner.py
vllm/model_executor/layers/fused_moe/modular_kernel.py
```

vLLM 的 routed expert 调用链可以概括为：

```text
Qwen3NextSparseMoeBlock.forward
  -> FusedMoE.forward
    -> MoERunner.forward
      -> torch.ops.vllm.moe_forward / moe_forward_shared
        -> MoERunner._forward_impl
          -> router.select_experts(...)
          -> quant_method.apply(...)
            -> FusedMoEKernel.apply(...)
              -> prepare
              -> fused_experts.apply(...)
              -> finalize
```

### 3.1 Router 选择专家

Qwen3.5 这里通常是普通 top-k softmax 路由：

```text
router_logits = gate(hidden_states)  # [T, 256]
topk_weights, topk_ids = topk_softmax(router_logits, k=8)
```

在 vLLM 中，默认路由在：

```text
vllm/model_executor/layers/fused_moe/router/fused_topk_router.py
```

`FusedTopKRouter` 调用 `ops.topk_softmax`，输出：

```text
topk_weights: [T, 8] fp32
topk_ids    : [T, 8] int32
```

这些 `topk_weights` 通常不会提前乘到 input 上，而是在 expert 输出阶段做 weighted reduce：

```text
out[t] = sum_j topk_weights[t, j] * y[t, j]
```

### 3.2 FusedMoEKernel 的三段抽象

vLLM 的 modular MoE kernel 把 routed expert 拆成三段：

```text
prepare -> fused_experts -> finalize
```

#### prepare

职责：

- 根据 `topk_ids` 准备 expert dispatch 元数据。
- 在 EP/all2all 场景下把 token 发送到拥有对应 expert 的 rank。
- 根据 backend 需要决定是否提前量化 routed input。
- 返回 expert kernel 需要的输入张量、scale、metadata。

抽象输出类似：

```text
a1q                # expert kernel 的输入，可能仍是 bf16/fp16，也可能已量化
a1q_scale          # input scale；只有 prepare 提前量化时才通常存在
expert_tokens_meta # expert token metadata，可选
topk_ids
topk_weights
```

这里 `a1q` 这个名字容易误导。它在接口上叫 quantized activation，但实际是否量化由 backend 决定：

```text
if defer_input_quant == False:
    prepare 里做 input quantization
    a1q       = packed/quantized input
    a1q_scale = input block scale / token scale

if defer_input_quant == True:
    prepare 不量化 input
    a1q       = 原始 bf16/fp16 hidden_states
    a1q_scale = None
```

vLLM 里控制这个行为的是：

```text
defer_input_quant = self.fused_experts.expects_unquantized_inputs
```

也就是说，**如果 expert kernel 声明自己希望拿未量化输入**，prepare 就把量化延后，让 `fused_experts.apply` 内部处理 input quantization。反过来，如果 expert kernel 希望拿已经量化好的 input，prepare 就提前调用 `moe_kernel_quantize_input`。

### prepare 为什么有时要提前量化，有时要延迟量化

这是因为 MoE backend 的边界不同。

#### 情况 A：prepare 提前量化

prepare 提前量化时，数据流是：

```text
hidden_states bf16/fp16
  -> prepare: quantize to FP4/FP8 + produce scale
  -> fused_experts: directly consume quantized input + input scale
```

这样做的好处：

- expert kernel 可以假设输入已经是目标量化格式。
- 在 EP/all2all 场景下，如果先量化再通信，可以减少通信量。
- `fused_experts` 可以专注于 grouped GEMM/activation/finalize。

代价：

- 量化本身可能是额外 kernel 或额外步骤。
- 对 NVFP4 这类有特殊 scale layout 的格式，prepare 需要处理 scale swizzle/layout。
- 如果 backend 本来可以把 input quant 融进 GEMM1，提前量化反而可能多一次读写。

#### 情况 B：fused_experts 内部量化

延迟量化时，数据流是：

```text
hidden_states bf16/fp16
  -> prepare: 不量化，只处理 dispatch/topk/all2all
  -> fused_experts: 在 expert kernel 内做 input quant + GEMM
```

这样做的好处：

- input quant 可以和 GEMM1 融合，减少中间张量写回。
- 对 decode 这种 `T=1` 的小 M 场景，可以减少 kernel launch 和固定开销。
- backend 可以按自己的 kernel layout 直接生成最合适的 scale 格式。

代价：

- 如果有 EP/all2all，通信的可能是 bf16/fp16 原始激活，通信量更大。
- expert kernel 需要承担更多职责，接口更 backend-specific。

### NVFP4 case 下 prepare 的具体差异

NVFP4 MoE 有两种常见边界：

```text
prepare quantizes input:
  hidden_states bf16
    -> prepare 量化成 NVFP4 packed input + scale
    -> fused_experts 做 NVFP4 GEMM1/GEMM2

fused_experts quantizes input:
  hidden_states bf16
    -> prepare 只传 bf16 input
    -> fused_experts 内部做 input NVFP4 quant + GEMM1
```

在 vLLM 的 no-DP/no-EP prepare 里，逻辑接近：

```text
if defer_input_quant:
    return a1, None
else:
    input_sf = quant_config.a1_gscale if NVFP4 else quant_config.a1_scale
    a1q, a1q_scale = moe_kernel_quantize_input(
        a1,
        input_sf,
        quant_dtype=quant_config.quant_dtype,
        block_shape=quant_config.block_shape,
        is_fp4_scale_swizzled=quant_config.is_nvfp4_scale_swizzled,
    )
    return a1q, a1q_scale
```

在 FlashInfer NVLink all-to-all prepare 中还有一个额外细节：如果不延迟量化，会先量化 input，再 all-to-all 传输 packed activation 和 scale；但 NVFP4 scale 的 swizzle 会影响 all-to-all 的 shape，因此代码里会先用 non-swizzled scale 通信，再在通信后做 `nvfp4_block_scale_interleave`。

这就是为什么同样是 NVFP4，`prepare` 的输出可能有两种形态：

```text
提前量化:
  a1q       = NVFP4 packed activation
  a1q_scale = NVFP4 block scale

延迟量化:
  a1q       = bf16/fp16 activation
  a1q_scale = None
```

#### fused_experts

职责：

- 执行 expert 的 `W13 -> activation -> W2` 主体计算。
- 消费 `prepare` 传来的 `a1q/a1q_scale/topk_ids/topk_weights`。
- 根据 backend 的融合边界，决定自己是否还要处理 input quant、expert grouping、top-k weighted reduce、finalize。

调用入口是：

```text
fused_experts.apply(
    output=fused_out,
    hidden_states=a1q,
    w1=w13_weight,
    w2=w2_weight,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    ...
)
```

逻辑上，`fused_experts` 要完成的是：

```text
for each token t:
  for each selected expert e in topk_ids[t]:
    u = W13[e] @ x[t]
    gate, up = split(u)
    mid = silu(gate) * up
    y = W2[e] @ mid
```

但不同 backend 暴露给 vLLM 的边界差异很大。

### fused_experts 可能只做“两层 GEMM + activation”

较传统的 modular kernel 会让 `prepare` 负责 input quant/dispatch，让 `finalize` 负责 top-k weight 和 reduce。此时 `fused_experts` 更像：

```text
input: expert-ready a1q, topk_ids

1. 根据 topk_ids/metadata 组织 expert-major GEMM
2. GEMM1: a1q @ W13
3. activation: silu(gate) * up
4. GEMM2: mid @ W2

output: per-token-per-expert result
```

随后由 `finalize` 做：

```text
output[t] = sum_j topk_weights[t, j] * expert_result[t, j]
```

这种边界清晰，但 kernel 数和中间 buffer 可能更多。

### fused_experts 可能融合 quantization

如果 backend 的 `expects_unquantized_inputs=True`，`fused_experts` 收到的是 bf16/fp16 input。此时 expert kernel 内部会做：

```text
1. load bf16/fp16 x
2. quantize x -> FP4/FP8 + scale
3. GEMM1
4. activation
5. GEMM2
```

对 NVFP4 来说，这种设计通常是为了把 input quantization 和 GEMM1 尽量融合，减少一次中间写回。

### fused_experts 可能融合 finalize

有些 backend 不把 `[T, top_k, H]` 的中间 expert output 暴露给 vLLM，而是在 GEMM2 epilogue 里直接做：

```text
expert_output = W2[e] @ mid
weighted = topk_weights[t, slot] * expert_output
atomicAdd(output[t], weighted)
```

这种情况下：

- `fused_experts.apply` 写出的 `output` 已经接近最终 routed output。
- `finalize_weight_and_reduce_impl()` 可能返回 no-op。
- vLLM 的 `finalize` 阶段只需要接受这个结果，或者做通信 combine。

FlashInfer CuteDSL 的 GEMM2 finalize fusion 就是这种形态。

### fused_experts 可能融合 dispatch/gather

为了避免显式 `moe_permute`，kernel 可以不提前搬运 activation，而是使用 `moe_sort` 生成的 mapping，在 GEMM1 中按 mapping gather：

```text
permuted row -> expanded token-expert pair -> original token id
```

这样 GEMM1 kernel 内部同时完成：

```text
gather input + grouped GEMM1
```

FlashInfer CuteDSL 的 GEMM1 kernel 就是这个思路。

#### finalize

职责：

- 将 expert-major 或 permuted layout 的结果 unpermute 回 token-major。
- 乘 `topk_weights`。
- 对同一 token 的 top-k expert 输出求和。
- 在 TP/EP 场景下做必要的 reduce。

普通 unfused 语义是：

```text
fused_out: [T, top_k, H]
output[t] = sum_j topk_weights[t, j] * fused_out[t, j]
```

但在 FlashInfer 这类 backend 中，finalize 往往已经融合进 GEMM2 kernel。

---

## 4. NVFP4 Expert 权重与 Scale

NVFP4 是 4-bit floating-point，两个 FP4 元素 packed 到一个 `uint8`。所以逻辑 weight shape 和物理 packed shape 不同。

对 Qwen3.5 MoE：

```text
W13 logical: [256, 1024, 2048]
W13 packed : [256, 1024, 1024] uint8

W2 logical : [256, 2048, 512]
W2 packed  : [256, 2048, 256] uint8
```

每组 FP4 权重还有 block scale。Qwen3.5 配置中 group size 是 16，因此 scale 维度大致是：

```text
W13 scale: [256, 1024, 2048 / 16] = [256, 1024, 128]
W2 scale : [256, 2048, 512 / 16]  = [256, 2048, 32]
```

此外还会有 global scale / alpha，用来把 block-scaled MMA 的结果恢复到目标数值范围。

需要注意：

- 权重是 packed NVFP4。
- 输入激活可能由 kernel 内部量化，也可能由 vLLM prepare 阶段量化。
- GEMM 输出通常回到 bf16/fp16，或者中间激活再次量化为 FP4 供 GEMM2 使用。
- `lm_head`、router `gate`、`shared_expert_gate`、linear attention `conv1d` 等通常不走 NVFP4 MoE expert kernel。

---

## 5. vLLM 中 FlashInfer CUTLASS MoE 接入

vLLM 的 FlashInfer CUTLASS MoE backend 在：

```text
vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py
```

`FlashInferExperts.apply` 最终调用：

```text
flashinfer.fused_moe.core.cutlass_fused_moe(...)
```

对 NVFP4，vLLM 传入的关键参数是：

```text
input                  = hidden_states 或 a1q
token_selected_experts = topk_ids
token_final_scales     = topk_weights
fc1_expert_weights     = w1.view(torch.long)
fc2_expert_weights     = w2.view(torch.long)
output                 = output
output_dtype           = bf16/fp16
```

NVFP4 的 `quant_scales` 是 6 个张量：

```text
quant_scales = [
    gemm1 activation global scale,
    gemm1 weight block scales,
    gemm1 dequant/global alpha,
    gemm2 activation global scale,
    gemm2 weight block scales,
    gemm2 dequant/global alpha,
]
```

这说明 vLLM 在调用 FlashInfer CUTLASS 时，已经完成了 router/top-k；FlashInfer 接收的是 `topk_ids/topk_weights`，不负责从 router logits 中选专家。

这里的 `input` 具体是 bf16 还是 NVFP4 packed，取决于前面 `prepare` 是否已经量化：

```text
prepare 提前量化:
  input    = NVFP4 packed activation
  input_sf = a1q_scale

prepare 延迟量化:
  input    = bf16/fp16 activation
  input_sf = None
```

FlashInfer CUTLASS API 自身支持 NVFP4 的两种输入边界：既可以接收已量化 packed input，也可以接收未量化 input，并在内部完成量化。因此从 vLLM 视角看，`FlashInferExperts.apply` 只负责把 `a1q/a1q_scale` 和 `quant_scales` 传进去；量化实际发生在 prepare 还是 FlashInfer kernel 内，由 `defer_input_quant` 和 backend 能力共同决定。

FlashInfer CUTLASS 的 public API 在：

```text
/Users/moudi/Desktop/llm_infra/infer/flashinfer/flashinfer/fused_moe/core.py
```

接口语义：

```text
cutlass_fused_moe(
    input,
    token_selected_experts,
    token_final_scales,
    fc1_expert_weights,
    fc2_expert_weights,
    output_dtype,
    quant_scales,
    input_sf=None,
    ...
)
```

它会根据 GPU 架构加载对应 JIT module：

```text
SM90  -> gen_cutlass_fused_moe_sm90_module
SM100 -> gen_cutlass_fused_moe_sm100_module
SM103 -> gen_cutlass_fused_moe_sm103_module
SM120 -> gen_cutlass_fused_moe_sm120_module
```

然后进入 C++/CUDA extension 的 `cutlass_fused_moe` 实现。

---

## 6. FlashInfer CuteDSL NVFP4 MoE 实现

FlashInfer 里还有一条更显式的 CuteDSL NVFP4 MoE 路径：

```text
/Users/moudi/Desktop/llm_infra/infer/flashinfer/flashinfer/fused_moe/cute_dsl/fused_moe.py
```

入口：

```text
cute_dsl_fused_moe_nvfp4(...)
CuteDslMoEWrapper.run(...)
```

核心实现 `_moe_core_impl` 把 MoE 分成 4 步：

```text
1. moe_sort
2. GEMM1 + SwiGLU + FP4 requant
3. zero output
4. GEMM2 + weighted scatter finalize
```

这条 CuteDSL 路径展示的是一种更“kernel 内融合”的 NVFP4 设计。它的输入 `x/x_sf` 已经是 NVFP4 packed activation 和对应 scale：

```text
x    : packed NVFP4 activation
x_sf : activation block scale
```

因此在 CuteDSL `_moe_core_impl` 里，**GEMM1 输入量化已经发生在调用者之前**。但是 GEMM1 后的中间激活 `mid = silu(gate) * up` 会在 GEMM1 epilogue 内再次量化为 FP4，作为 GEMM2 的输入：

```text
bf16/fp32 accumulator
  -> SwiGLU
  -> quantize mid to FP4
  -> intermediate + intermediate_sf
```

所以 CuteDSL 的 NVFP4 量化分成两类：

```text
GEMM1 input quant:
  不在 _moe_core_impl 内展示，caller 传入 x/x_sf 时已经完成。

GEMM2 input quant:
  融合在 GEMM1 + SwiGLU kernel 的 epilogue 中完成。
```

### 6.1 Step 1: `moe_sort`

输入：

```text
token_selected_experts: [T, top_k]
token_final_scales    : [T, top_k]
```

输出：

```text
tile_idx_to_expert_idx
tile_idx_to_mn_limit
expanded_idx_to_permuted_idx
permuted_idx_to_expanded_idx
total_num_padded_tokens
num_non_exiting_tiles
```

作用是把 token-expert pair 按 expert 分组并按 tile 对齐，供 grouped GEMM 使用。

关键点：`moe_sort` 主要生成 mapping，不一定真的搬运 input。后续 GEMM kernel 会根据 `permuted_idx_to_expanded_idx` 从原始 input 中 gather 对应 token。

对 `T=4096, top_k=8`：

```text
expanded rows = 4096 * 8 = 32768
```

这些 rows 会按 expert 分组并 pad 到 tile size。

### 6.2 Step 2: GEMM1 + SwiGLU + FP4 requant

调用：

```text
blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4(...)
```

输入：

```text
a       = x                      # NVFP4 packed activations
a_scale = x_sf                   # activation block scales
b       = w1_weight              # packed W13
b_scale = w1_weight_sf
alpha   = w1_alpha
token_id_mapping = permuted_idx_to_expanded_idx
```

kernel 内部做：

```text
for each expert tile:
  row = gather token by token_id_mapping
  acc = NVFP4_block_scaled_GEMM(x[row], W13[expert])
  gate, up = split(acc)
  mid = silu(gate) * up
  intermediate = quantize_to_fp4(mid, fc2_input_scale)
```

输出：

```text
intermediate    # packed FP4, GEMM2 input
intermediate_sf # intermediate 的 block scale
```

这一层把几个传统 kernel 合并了：

```text
permute/gather + GEMM1 + activation + GEMM2-input quantization
```

### 6.3 Step 3: zero output

GEMM2 finalize 使用 scatter add：

```text
out[token] += topk_weight * expert_output
```

因此 `out` 必须先清零。FlashInfer 的 `_moe_core_impl` 会对 active slice 做 `zero_()`，并可用 aux stream overlap。

### 6.4 Step 4: GEMM2 + weighted scatter finalize

调用：

```text
blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(...)
```

输入：

```text
a       = intermediate
a_scale = intermediate_sf
b       = w2_weight
b_scale = w2_weight_sf
alpha   = w2_alpha
permuted_idx_to_expanded_idx
token_final_scales
out
```

kernel 内部做：

```text
for each permuted expert row:
  expert_output = NVFP4_block_scaled_GEMM(intermediate[row], W2[expert])
  token_id, topk_slot = decode(permuted_idx_to_expanded_idx[row])
  weighted = token_final_scales[token_id, topk_slot] * expert_output
  atomicAdd(out[token_id], weighted)
```

这一层融合了：

```text
GEMM2 + multiply topk weight + unpermute + reduce
```

因此不需要单独的 `moe_unpermute` 或 `moe_fused_mul_sum`。

---

## 7. FlashInfer CUTLASS vs CuteDSL 的差异

| 维度 | FlashInfer CUTLASS path | FlashInfer CuteDSL NVFP4 path |
|---|---|---|
| vLLM 当前接入 | 是，`FlashInferExperts.apply` 调用 | 不是当前这条 vLLM `FlashInferExperts` 主路径 |
| public API | `cutlass_fused_moe` | `cute_dsl_fused_moe_nvfp4` |
| router/top-k | vLLM 外部完成 | caller 外部完成 |
| 输入 | 支持 bf16/fp16/fp8/NVFP4 等 | NVFP4 packed input + scale |
| 核心结构 | CUTLASS/TRT-LLM fused MoE runner | `moe_sort -> GEMM1 fusion -> zero -> GEMM2 finalize` |
| finalize | backend 内部处理 | GEMM2 kernel atomic scatter-add |
| 适用 | 多架构 backend，vLLM 集成路径 | Blackwell CuteDSL 专门路径，结构更显式 |

两条路径语义一致：

```text
out[t] = sum_j topk_weights[t,j] * W2[e_j] @ silu_gate(W13[e_j] @ x[t])
```

差异主要在 kernel 划分、layout、是否显式暴露中间 buffer。

---

## 8. NVFP4 支持点

NVFP4 MoE kernel 需要同时处理几类量化信息：

### 8.1 Packed FP4 weights

两个 FP4 元素 pack 到一个 `uint8`。所以 `K` 维会减半：

```text
logical K = 2048 -> packed K = 1024
logical K = 512  -> packed K = 256
```

### 8.2 Block scale

group size 16 意味着每 16 个 FP4 元素有一个 scale：

```text
scale_count = logical_K / 16
```

FlashInfer/CUTLASS kernel 通常要求 scale 是特定 swizzled/block layout，而不只是自然 `[E, rows, K/16]` layout。

### 8.3 Global scale / alpha

NVFP4 GEMM 的最终数值通常包含：

```text
output ~= MMA(fp4_input, fp4_weight, block_scales) * global_alpha
```

对 MoE 有两层 GEMM，所以会有：

```text
gemm1 activation global scale
gemm1 weight block scales
gemm1 dequant/global alpha
gemm2 activation global scale
gemm2 weight block scales
gemm2 dequant/global alpha
```

### 8.4 Intermediate requantization

GEMM1 后经过 SwiGLU 得到 `[num_pairs, I]` 中间激活。为了让 GEMM2 也走 FP4 tensor core，FlashInfer CuteDSL 会在 GEMM1 epilogue 里把 `mid` 再量化成 FP4，并输出 `intermediate_sf`。

这样 GEMM2 的输入也是 NVFP4 packed。

---

## 9. Prefill 与 Decode 的性能含义

### Prefill

`T=4096` 时：

```text
topk pairs = 32768
```

特点：

- `moe_sort` 开销能被大量 grouped GEMM amortize。
- 每个 expert 通常会收到较多 token。
- GEMM1/GEMM2 计算密度较高。
- grouped GEMM 的 tile 利用率通常较好。

### Decode

`T=1` 时：

```text
topk pairs = 8
```

特点：

- 每步只激活 8 个专家。
- 算术量很小，但需要读 8 个 expert 的 W13/W2。
- 容易 memory bandwidth bound。
- kernel launch、排序、metadata 准备的相对开销明显变大。

因此 decode MoE 的优化通常会倾向于：

- 减少 kernel launch 数。
- 融合 input quant、GEMM1、activation、GEMM2、finalize。
- 使用更适合小 M 的 micro kernel。
- 避免显式 permute/unpermute 的中间写回。

FlashInfer 的 SM120/SM121 `b12x_fused_moe` 就是这一类思路：bf16 input 进入后，在一个 kernel 中融合 quantization、routing、FC1、activation、FC2、scatter。

---

## 10. 一句话总结

Qwen3.5 MoE expert 的数学语义很简单：

```text
shared_out + sum(topk_weight * W2[silu(W13*x)])
```

但高性能实现的关键在于 kernel 分层：

```text
router/top-k
  -> expert grouping / sort
  -> grouped GEMM1
  -> SwiGLU
  -> intermediate FP4 requant
  -> grouped GEMM2
  -> weighted scatter/reduce
```

vLLM 提供 `FusedMoE -> MoERunner -> FusedMoEKernel` 的统一封装；FlashInfer 则在 backend 内把 MoE 的 dispatch、GEMM、activation、quantization、finalize 尽可能融合，NVFP4 支持主要依赖 packed FP4 weight/input、block scale layout、global scale，以及 GEMM1 到 GEMM2 之间的 FP4 requantization。

---

## 11. FlashInfer CUTLASS MoE 后端 Kernel 解析

FlashInfer CUTLASS MoE 后端的 GPU kernel 逐层解析已独立成文，见：

[`flashinfer_cutlass_moe_kernel_breakdown.md`](flashinfer_cutlass_moe_kernel_breakdown.md)

---

## 12. ATREX MoE 后端 Kernel 解析

ATREX MoE 后端的 GPU kernel 逐层解析已独立成文，见：

[`atrex_moe_kernel_breakdown.md`](atrex_moe_kernel_breakdown.md)
