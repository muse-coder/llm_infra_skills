# vLLM CUDA Graph 知识库 — 以 Qwen3.5 为例

## 1. 整体架构：双模式 CUDA Graph

在 v1 引擎、且 piecewise compilation 可用时，vLLM 的默认目标模式通常是 `FULL_AND_PIECEWISE`（`vllm/config/compilation.py:63`，`docs/design/cuda_graphs.md`）。但最终运行模式还会经过 attention backend 能力检查和 downgrade；如果 backend 或编译条件不满足，可能会退化为 `FULL`、`FULL_DECODE_ONLY`、`PIECEWISE` 甚至 `NONE`。

| 批次类型 | CUDAGraphMode | 策略 |
|---|---|---|
| Decode（纯解码，uniform batch） | `FULL` | 整个 forward 捕获为一个 CUDA Graph |
| Prefill / Mixed（预填充或混合批次） | `PIECEWISE` | FX 图在 attention 算子处切断，分片捕获 |

```python
# vllm/config/compilation.py:53-63
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)   # ← 默认

    def decode_mode(self):   # → FULL
    def mixed_mode(self):    # → PIECEWISE
```

Qwen3.5 是混合架构（含 `full_attention` 和 GDN-based `linear_attention` 层）。在当前仓库实现中，Qwen3.5 这条 GDN attention backend 的 CG 支持等级是 `UNIFORM_BATCH`，因此：

- `uniform decode` 可以使用 `FULL` graph
- `prefill / mixed` 不能使用 `FULL` graph（mixed 要求 backend 支持 `ALWAYS`）
- 在默认/常见配置下，prefill / mixed 会落到 `PIECEWISE` 路径

也就是说，**“prefill 走 PIECEWISE”这个结论是对的，但原因不是 `UNIFORM_SINGLE_TOKEN_DECODE`，而是 mixed batch 不满足 `FULL` 所需的 `ALWAYS` 能力。**

---

## 2. Splitting Ops（切割算子）

编译时 FX 图在以下算子处被切断，这些算子 eager 执行，不进入 CUDA Graph：

```python
# vllm/config/compilation.py:717-730
_attention_ops: ClassVar[list[str]] = [
    "vllm::unified_attention_with_output",      # 标准 FlashAttention
    "vllm::unified_mla_attention_with_output",   # MLA attention
    "vllm::mamba_mixer2",                        # Mamba v2
    "vllm::mamba_mixer",                         # Mamba v1
    "vllm::short_conv",                          # 短卷积
    "vllm::linear_attention",                    # 线性注意力
    "vllm::plamo2_mamba_mixer",                  # PLaMo2 Mamba
    "vllm::gdn_attention_core",                  # GDN（Qwen3.5 的 linear attention）
    "vllm::olmo_hybrid_gdn_full_forward",        # OLMo hybrid GDN
    "vllm::kda_attention",                       # KDA attention
    "vllm::sparse_attn_indexer",                 # sparse attention
    "vllm::rocm_aiter_sparse_attn_indexer",      # ROCm sparse attention
]
```

另外，在默认 piecewise 编译路径里、且 `use_inductor_graph_partition=False` 时，`vllm::unified_kv_cache_update` 也会被追加到 `splitting_ops` 中（`compilation.py:1088`）。这不是无条件行为；如果启用了 Inductor graph partition，这个说法未必成立。

**对 Qwen3.5 而言，涉及两个切割算子：**

| 切割算子 | 对应层类型 |
|---|---|
| `vllm::unified_attention_with_output` | `full_attention` 层的 FlashAttention |
| `vllm::gdn_attention_core` | `linear_attention` 层的 GDN 核心计算 |

---

## 3. 共同入口：execute_model

Prefill 和 decode 共享同一个入口，在运行时通过 `cudagraph_mode` 分流：

```
GPUWorker.execute_model(scheduler_output)                    # gpu_worker.py:748
  └─ GPUModelRunner.execute_model(scheduler_output)           # gpu_model_runner.py:3877
       │
       ├─ [1] _determine_batch_execution_and_padding()        # :3548
       │    ├─ _is_uniform_decode()                           # :3528
       │    │    条件: max_num_scheduled_tokens == uniform_decode_query_len
       │    │          且 num_tokens == max_num_scheduled_tokens * num_reqs
       │    │    → True = decode,  False = prefill/mixed
       │    │
       │    └─ cudagraph_dispatcher.dispatch()                 # :3593
       │         根据 uniform_decode 标志匹配已捕获的 graph key
       │         → prefill: CUDAGraphMode.PIECEWISE
       │         → decode:  CUDAGraphMode.FULL
       │
       ├─ [2] preprocess_mamba()                              # :3939
       │    GDN 的 conv_state / ssm_state 拷贝和索引管理
       │    (完全在 graph 生命周期之外)
       │
       ├─ [3] _build_attention_metadata()                     # :3973
       │    构造各 attention backend 的 metadata
       │
       ├─ [4] _preprocess()                                   # :3996
       │    准备 input_ids, positions, inputs_embeds
       │
       └─ [5] set_forward_context(                            # :4020
       │         cudagraph_runtime_mode=mode,
       │         batch_descriptor=batch_desc,
       │         attn_metadata=attn_metadata)
       │    将 mode 和 metadata 写入全局 ForwardContext
       │
       └─ [6] _model_forward(input_ids, positions, ...)       # :4038
              → self.model(...)
              从此处开始，prefill 和 decode 路径分叉
```

---

## 4. Prefill 路径（PIECEWISE 模式）完整调用栈

```
self.model(input_ids, positions, ...)
│
│  self.model 实际是 CUDAGraphWrapper(runtime_mode=FULL)
│  读取 ForwardContext.cudagraph_runtime_mode = PIECEWISE
│  PIECEWISE ≠ FULL → 不匹配，直接透传 self.runnable(...)
│                                                          # cuda_graph.py:244-254
│
└─ Qwen3_5ForCausalLM.forward()                            # qwen3_5.py:521
   └─ Qwen3_5Model.__call__()                              # qwen3_5.py:207
      │
      │  被 @support_torch_compile 装饰                      # decorators.py:462
      │  编译后的 FX 图被切成 N+1 个子图 (N = 层数)
      │
      │  ╔═══════════ 子图 0: PIECEWISE CUDAGraph ═══════════╗
      │  ║ CUDAGraphWrapper(PIECEWISE).__call__()            ║  # cuda_graph.py:233
      │  ║   ├─ 首次: torch.cuda.graph() 捕获                ║  # :308
      │  ║   └─ 后续: entry.cudagraph.replay()               ║  # :355
      │  ║                                                    ║
      │  ║ 包含的算子:                                         ║
      │  ║   embed_tokens(input_ids)             # :511       ║
      │  ║   input_layernorm(hidden_states)      # :407-409   ║
      │  ║   ┌ full_attention 层:                             ║
      │  ║   │  qkv_proj(hidden_states)          # :287       ║
      │  ║   │  q_norm / k_norm                  # :301-306   ║
      │  ║   │  rotary_emb(positions, q, k)      # :308       ║
      │  ║   └ linear_attention(GDN) 层:                      ║
      │  ║      in_proj_qkvz(hidden_states)      # gdn:538    ║
      │  ║      in_proj_ba(hidden_states)         # gdn:539    ║
      │  ╚════════════════════════════════════════════════════╝
      │
      │  ┌─── 切割算子: eager 执行 (layer 0) ─────────────────┐
      │  │                                                      │
      │  │  ★ full_attention 层:                                │
      │  │    Attention.forward()                # attn:440     │
      │  │    ├─ torch.ops.vllm.unified_kv_cache_update()       │
      │  │    │    └─ FlashAttentionImpl.do_kv_cache_update()    │
      │  │    │                                  # attn:686     │
      │  │    └─ torch.ops.vllm.unified_attention_with_output() │
      │  │         └─ FlashAttentionImpl.forward()              │
      │  │              ├─ flash_attn_varlen_func()  (prefill)  │
      │  │              └─ flash_attn_with_kvcache()  (decode)  │
      │  │                                  # flash_attn.py:673 │
      │  │                                                      │
      │  │  ★ linear_attention(GDN) 层:                         │
      │  │    torch.ops.vllm.gdn_attention_core()  # gdn:571   │
      │  │    └─ GatedDeltaNetAttention._forward_core() # :779  │
      │  │         ├─ causal_conv1d_fn()       (prefill)  # :876│
      │  │         │  或 causal_conv1d_update() (decode)  # :889│
      │  │         ├─ fused_post_conv_prep()              # :921│
      │  │         ├─ chunk_gated_delta_rule()  (prefill)  # :978
      │  │         │  └─ ChunkGatedDeltaRule.forward_native()   │
      │  │         │       └─ fla_chunk_gated_delta_rule()      │
      │  │         │     或 ChunkGatedDeltaRule.forward_cuda()  │
      │  │         │       └─ fi_chunk_gated_delta_rule()       │
      │  │         │  或 fused_sigmoid_gating_delta_rule_update()│
      │  │         │                              (decode) # :997
      │  │         └─ ssm_state 写回              # :992        │
      │  └──────────────────────────────────────────────────────┘
      │
      │  ╔═══════════ 子图 1: PIECEWISE CUDAGraph ═══════════╗
      │  ║ CUDAGraphWrapper(PIECEWISE).__call__()            ║
      │  ║                                                    ║
      │  ║ 包含的算子:                                         ║
      │  ║   ┌ full_attention 层:                             ║
      │  ║   │  attn_output_gate (sigmoid * gate)  # :312-314 ║
      │  ║   │  o_proj(attn_output)                # :316     ║
      │  ║   └ linear_attention(GDN) 层:                      ║
      │  ║      norm(core_attn_out, z)             # gdn:586  ║
      │  ║      out_proj(core_attn_out)            # gdn:589  ║
      │  ║                                                    ║
      │  ║   post_attention_layernorm()            # :438     ║
      │  ║   MLP: gate_up_proj → SiLU → down_proj  # :439    ║
      │  ║   layer_scale (如启用)                   # :441-455 ║
      │  ║                                                    ║
      │  ║   (下一层的 input_layernorm 和投影也融入此子图)      ║
      │  ╚════════════════════════════════════════════════════╝
      │
      │  ┌─── 切割算子: eager 执行 (layer 1) ───┐
      │  │  ... 同 layer 0 ...                    │
      │  └────────────────────────────────────────┘
      │
      │  ... 重复至 layer N-1 ...
      │
      │  ╔═══════════ 子图 N: PIECEWISE CUDAGraph ═══════════╗
      │  ║ 包含的算子:                                         ║
      │  ║   最后一层的 o_proj / out_proj                      ║
      │  ║   最后一层的 post_attention_layernorm + MLP          ║
      │  ║   final norm (Qwen3_5RMSNorm)          # :536      ║
      │  ╚════════════════════════════════════════════════════╝
      │
      └─ return hidden_states
```

---

## 5. Decode 路径（FULL 模式）完整调用栈

```
self.model(input_ids, positions, ...)
│
│  self.model 实际是 CUDAGraphWrapper(runtime_mode=FULL)
│  读取 ForwardContext.cudagraph_runtime_mode = FULL
│  FULL == FULL → 匹配!
│                                                          # cuda_graph.py:233
│  ├─ 首次: torch.cuda.graph() 捕获整个 forward             # :308
│  └─ 后续: entry.cudagraph.replay()                       # :355
│       (单次 replay 执行整个 forward，接近零 CPU 开销)
│
│  ┌────────── 以下全部录进一个 FULL CUDA Graph ──────────┐
│  │                                                        │
│  │  Qwen3_5ForCausalLM.forward()           # qwen3_5:521 │
│  │  └─ Qwen3_5Model.__call__()             # qwen3_5:207 │
│  │     │                                                  │
│  │     │  @support_torch_compile 的 __call__               │
│  │     │  内部 PIECEWISE wrapper 看到 runtime=FULL         │
│  │     │  FULL ≠ PIECEWISE → 全部透传 runnable             │
│  │     │                                   # cuda_graph:246│
│  │     │                                                  │
│  │     ├─ embed_tokens(input_ids)          # :511         │
│  │     │                                                  │
│  │     ├─ Layer 0 ~ N-1 (每层结构相同):                    │
│  │     │  │                                               │
│  │     │  ├─ input_layernorm()             # :407-409     │
│  │     │  │                                               │
│  │     │  ├─ ★ full_attention 层:                         │
│  │     │  │  qkv_proj(hidden_states)       # :287         │
│  │     │  │  q_norm / k_norm               # :301-306     │
│  │     │  │  rotary_emb(positions, q, k)   # :308         │
│  │     │  │  Attention.forward()           # attn:440     │
│  │     │  │    unified_kv_cache_update()   # attn:521     │
│  │     │  │    unified_attention_with_output()             │
│  │     │  │      └─ FlashAttentionImpl.forward()          │
│  │     │  │           └─ flash_attn_with_kvcache()        │
│  │     │  │                               # flash_attn:673│
│  │     │  │  attn_output_gate()            # :312-314     │
│  │     │  │  o_proj(attn_output)           # :316         │
│  │     │  │                                               │
│  │     │  ├─ ★ linear_attention(GDN) 层:                  │
│  │     │  │  in_proj_qkvz(hidden_states)   # gdn:538      │
│  │     │  │  in_proj_ba(hidden_states)     # gdn:539      │
│  │     │  │  gdn_attention_core()          # gdn:571      │
│  │     │  │    └─ _forward_core()          # gdn:779      │
│  │     │  │         causal_conv1d_update()  # gdn:889     │
│  │     │  │         fused_sigmoid_gating_delta_rule_update()│
│  │     │  │                                # gdn:997      │
│  │     │  │  norm(core_attn_out, z)        # gdn:586      │
│  │     │  │  out_proj(core_attn_out)       # gdn:589      │
│  │     │  │                                               │
│  │     │  ├─ post_attention_layernorm()     # :438         │
│  │     │  ├─ MLP: gate_up_proj → SiLU → down_proj  # :439│
│  │     │  └─ layer_scale (如启用)           # :441-455     │
│  │     │                                                  │
│  │     └─ final norm                       # :536         │
│  │                                                        │
│  └────────── 全部在一个 CUDA Graph 中 ──────────────────┘
│
└─ return hidden_states
```

---

## 6. 逐算子分类：在 Graph 中 vs 不在 Graph 中

### 6.1 full_attention 层（一个 DecoderLayer）

```
  ┌──────────────────────────────────────────────────────────┐
  │  input_layernorm (RMSNorm)                               │  ← 在 Graph 中
  │  qkv_proj (QKVParallelLinear → Q/K/V)                    │  ← 在 Graph 中
  │  q_norm / k_norm (RMSNorm)                               │  ← 在 Graph 中
  │  rotary_emb (RoPE)                                       │  ← 在 Graph 中
  ├──────────────────── 切割点 ──────────────────────────────┤
  │  ★ unified_kv_cache_update (KV cache 写入)               │  ← 不在 Graph 中 (eager)
  │  ★ unified_attention_with_output (FlashAttention)        │  ← 不在 Graph 中 (eager)
  ├──────────────────── 切割点 ──────────────────────────────┤
  │  attn_output_gate (sigmoid * gate，如果有)                │  ← 在 Graph 中
  │  o_proj (RowParallelLinear)                               │  ← 在 Graph 中
  │  post_attention_layernorm (RMSNorm)                       │  ← 在 Graph 中
  │  MLP (gate_up_proj → SiLU → down_proj)                   │  ← 在 Graph 中
  │  layer_scale (如果启用)                                    │  ← 在 Graph 中
  └──────────────────────────────────────────────────────────┘
```

### 6.2 linear_attention(GDN) 层（一个 DecoderLayer）

```
  ┌──────────────────────────────────────────────────────────┐
  │  input_layernorm (RMSNorm)                               │  ← 在 Graph 中
  │  in_proj_qkvz (MergedColumnParallelLinear → Q/K/V/Z)     │  ← 在 Graph 中
  │  in_proj_ba (MergedColumnParallelLinear → B/A)            │  ← 在 Graph 中
  ├──────────────────── 切割点 ──────────────────────────────┤
  │  ★ gdn_attention_core:                                   │  ← 不在 Graph 中 (eager)
  │    - causal_conv1d_fn / causal_conv1d_update (卷积)       │
  │    - fused_post_conv_prep (后处理)                         │
  │    - chunk_gated_delta_rule (prefill 循环注意力)            │
  │      或 fused_sigmoid_gating_delta_rule_update (decode)   │
  │    - conv_state / ssm_state 读写                          │
  ├──────────────────── 切割点 ──────────────────────────────┤
  │  norm (RMSNormGated, 带 z gate)                          │  ← 在 Graph 中
  │  out_proj (RowParallelLinear)                              │  ← 在 Graph 中
  │  post_attention_layernorm (RMSNorm)                       │  ← 在 Graph 中
  │  MLP (gate_up_proj → SiLU → down_proj)                   │  ← 在 Graph 中
  │  layer_scale (如果启用)                                    │  ← 在 Graph 中
  └──────────────────────────────────────────────────────────┘
```

### 6.3 为什么这么切

**在 Graph 里的（各子图）：**
- 所有线性投影（qkv_proj, o_proj, in_proj_qkvz, out_proj, MLP）
- 所有 LayerNorm / RMSNorm
- RoPE
- 激活函数（SiLU, sigmoid gate）
- 这些都是 shape 固定、无动态分支的纯计算 kernel

**不在 Graph 里的（切割点，针对当前 Qwen3.5 的 piecewise 路径）：**
- **FlashAttention**：prefill 时序列长度变化，内部调度依赖动态 shape
- **GDN 核心**：包含 conv1d 状态更新、循环注意力（chunk_gated_delta_rule）、ssm_state 读写等，涉及动态序列长度和状态管理
- **KV cache 更新**：默认 piecewise 编译路径下会因字符串参数影响 Inductor 复用而被切出去；若启用 Inductor graph partition，行为可能不同

---

## 7. Graph 在整个模型中的全局结构

假设模型有 N 层（full_attention 和 linear_attention 交替），prefill 时的结构：

```
Graph piece 0:  embedding + layer0 前半（layernorm + 投影）
  ↓ eager: attention/gdn op (layer 0)
Graph piece 1:  layer0 后半 + layer1 前半
  ↓ eager: attention/gdn op (layer 1)
Graph piece 2:  layer1 后半 + layer2 前半
  ...
  ↓ eager: attention/gdn op (layer N-1)
Graph piece N:  layer(N-1) 后半 + final_norm
```

一共 **N+1 个 CUDA Graph 片段**，**N 个 eager 执行的 attention/GDN 算子**。

---

## 8. Graph 生命周期之外的操作

以下操作在 `execute_model()` 中 model forward **之前**执行，不参与任何 graph：

| 操作 | 位置 | 说明 |
|---|---|---|
| `preprocess_mamba()` | `gpu_model_runner.py:3939` | GDN 的 conv_state/ssm_state 拷贝、索引管理 |
| `_preprocess()` | `:3996` | input_ids, positions 等输入准备 |
| `_build_attention_metadata()` | `:3973` | 构造各 backend 的 attn metadata |
| `_determine_batch_execution_and_padding()` | `:3548` | batch 类型判断、padding |
| `_get_slot_mappings()` | `:3962` | KV cache slot 映射 |

---

## 9. CUDAGraphWrapper 的捕获/回放机制

```python
# vllm/compilation/cuda_graph.py:233-356
class CUDAGraphWrapper:
    def __call__(self, *args, **kwargs):
        mode = get_forward_context().cudagraph_runtime_mode
        batch_desc = get_forward_context().batch_descriptor

        if mode == NONE or mode != self.runtime_mode:
            # 不匹配 → 直接调用 runnable（透传）
            return self.runnable(*args, **kwargs)

        entry = self.concrete_cudagraph_entries[batch_desc]

        if entry.cudagraph is None:
            # ── 捕获 ──
            cudagraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                output = self.runnable(*args, **kwargs)
            entry.cudagraph = cudagraph
            entry.output = weak_ref_tensors(output)
            return output
        else:
            # ── 回放 ──
            entry.cudagraph.replay()
            return entry.output
```

---

## 10. 从 Qwen3NextModel.forward 到 Graph 触发：完整编译链路

### 10.1 问题：`Qwen3NextModel.forward()` 只是普通 Python 方法，Graph 是怎么被触发的？

关键在于 `@support_torch_compile` 装饰器**替换了类的 `__init__` 和 `__call__`**，使得调用 `model(...)` 时走的不是 `forward()`，而是一套编译 + graph 调度逻辑。

### 10.2 装饰器改造过程（初始化阶段）

```python
# decorators.py:115-246
@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, ...})
class Qwen3_5Model(Qwen3NextModel):
    ...
```

装饰器在**类定义时**做了三件事：

```
                        ┌──────────────────────────────────┐
                        │  @support_torch_compile 装饰器    │
                        └──────────────────────────────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                   ▼
           [1] 注入基类        [2] 替换 __init__     [3] 替换 __call__
```

**[1] 注入基类** (`decorators.py:343`)

```python
cls.__bases__ = cls.__bases__ + (TorchCompileWithNoGuardsWrapper,)
```

让 `Qwen3_5Model` 继承 `TorchCompileWithNoGuardsWrapper`，获得 `torch.compile` 能力。

**[2] 替换 `__init__`** (`decorators.py:349-407`)

新的 `__init__` 在调用原始 `__init__` 之后，额外执行：

```python
def __init__(self, *, vllm_config, prefix, **kwargs):
    old_init(self, ...)                         # 原始初始化（创建 layers 等）

    self.do_not_compile = (mode == NONE or ...)  # 判断是否跳过编译
    if self.do_not_compile:
        return

    # 调用 TorchCompileWithNoGuardsWrapper.__init__()     # wrapper.py:78
    #   → 内部执行:
    #     backend = compilation_config.init_backend(vllm_config)  # 创建 VllmBackend
    #     self._compiled_callable = torch.compile(
    #         self.forward,                    # ← 待编译的函数
    #         fullgraph=True,
    #         dynamic=False,
    #         backend=backend,                 # ← VllmBackend 实例
    #         options={guard_filter_fn: skip_all_guards_unsafe}  # 跳过所有 guard
    #     )
```

此时 `torch.compile()` 只是创建了一个惰性编译对象 `_compiled_callable`，**还没有真正编译**。

**[3] 替换 `__call__`** (`decorators.py:462-632`)

```python
cls.__call__ = __call__   # 新的 __call__ 替代了 nn.Module 默认的 __call__
```

### 10.3 第一次调用：触发编译（Profile / Warmup 阶段）

当 `Qwen3_5ForCausalLM.forward()` 调用 `self.model(input_ids, positions, ...)` 时，
走的是被替换的 `__call__`：

```
Qwen3_5Model.__call__(input_ids, positions, ...)           # decorators.py:462
│
├─ [检查] self.do_not_compile?  → False（需要编译）
├─ [检查] self.compiled?        → False（第一次调用）
│
├─ _mark_dynamic_inputs(self, ...)                          # :535
│    标记 input_ids 的 dim0 为动态, positions 的 dim-1 为动态
│    → torch._dynamo.mark_dynamic(input_ids, [0])
│    → torch._dynamo.mark_dynamic(positions, [-1])
│
└─ TorchCompileWithNoGuardsWrapper.__call__(self, ...)      # wrapper.py:185
   │
   └─ self._compiled_callable(*args, **kwargs)              # :213
      │
      │  这是 torch.compile 包装过的函数
      │  首次调用 → 触发 Dynamo tracing + 后端编译
      │
      ├─ [Dynamo] 追踪 Qwen3_5Model.forward() 的字节码
      │    生成 FX Graph (torch.fx.GraphModule)
      │    遇到 custom_op 节点：
      │      - vllm::unified_attention_with_output
      │      - vllm::unified_kv_cache_update
      │      - vllm::gdn_attention_core
      │    这些 op 有 fake_impl，Dynamo 可以追踪穿过
      │
      └─ [Backend] VllmBackend.__call__(graph, example_inputs)  # backends.py:981
         │
         │  ┌──────────────────────────────────────────────┐
         │  │  VllmBackend 编译流程 (backends.py:981-1182)  │
         │  └──────────────────────────────────────────────┘
         │
         ├─ [1] split_graph(graph, splitting_ops)               # backends.py:1138
         │    │  遍历 FX Graph 的每个节点
         │    │  在 splitting_ops 处切断（split_graph, :532-606）
         │    │    splitting_ops = [
         │    │      "vllm::unified_attention_with_output",
         │    │      "vllm::unified_kv_cache_update",
         │    │      "vllm::gdn_attention_core",
         │    │    ]
         │    │
         │    └─ 产出:
         │         split_gm:        拼接后的 GraphModule（调度器）
         │         piecewise_graphs: [SplitItem(...), ...]
         │           每个 SplitItem 标记 is_splitting_graph=True/False
         │           True  = attention/gdn 算子（不编译，eager 执行）
         │           False = 计算子图（需要 Inductor 编译）
         │
         ├─ [2] PiecewiseCompileInterpreter.run()               # backends.py:706
         │    │  遍历 split_gm 的每个子模块
         │    │
         │    └─ call_module(target, args, kwargs)               # :709
         │         │
         │         │  对于需要编译的子图 (not is_splitting_graph):
         │         │
         │         ├─ PiecewiseBackend(submod, ...)              # :734
         │         │    创建分片后端，内部调用 Inductor 编译子图
         │         │
         │         └─ wrap_with_cudagraph_if_needed(             # :745
         │              piecewise_backend, ...)
         │            │                                          # backends.py:612
         │            │  检查 cudagraph_mode.has_piecewise_cudagraphs()
         │            │  → True（FULL_AND_PIECEWISE 模式下）
         │            │
         │            └─ CUDAGraphWrapper(                       # :654
         │                 runnable=piecewise_backend,
         │                 runtime_mode=CUDAGraphMode.PIECEWISE
         │               )
         │               将 Inductor 编译后的子图包裹进 PIECEWISE wrapper
         │
         │    最终 split_gm 的子模块被原地替换：
         │      submod_0 → CUDAGraphWrapper(PIECEWISE, piece_0)
         │      submod_1 → [原始 splitting op, eager]
         │      submod_2 → CUDAGraphWrapper(PIECEWISE, piece_1)
         │      submod_3 → [原始 splitting op, eager]
         │      ...
         │
         └─ [3] 返回 split_gm 作为 compiled callable
              split_gm 就是"编译后的模型"，被 torch.compile 缓存
              后续调用直接走 split_gm.__call__()
```

### 10.4 后续调用：编译完成后的运行路径

```
Qwen3_5Model.__call__(input_ids, positions, ...)           # decorators.py:462
│
├─ [检查] self.compiled?  → True
│
└─ TorchCompileWithNoGuardsWrapper.__call__(self, ...)      # wrapper.py:185
   │
   └─ self._compiled_callable(*args, **kwargs)              # :213
      │
      │  Dynamo 跳过 guard 检查（skip_all_guards_unsafe）
      │  直接执行缓存的 split_gm
      │
      └─ split_gm(input_ids, positions, ...)
         │
         ├─ submod_0 → CUDAGraphWrapper(PIECEWISE).__call__()
         │    读取 ForwardContext.cudagraph_runtime_mode
         │    ├─ PIECEWISE 匹配 → 捕获或回放此子图
         │    └─ FULL 不匹配 → 透传 runnable (Inductor 编译后的代码)
         │
         ├─ submod_1 → attention/gdn op (直接 eager 执行)
         │
         ├─ submod_2 → CUDAGraphWrapper(PIECEWISE).__call__()
         │    ... 同 submod_0 逻辑 ...
         │
         └─ ... 交替执行至 submod_N ...
```

### 10.5 外层 FULL Wrapper 的包装

以上是 `Qwen3_5Model` 内部的结构。在 `GPUModelRunner` 初始化时，还会在**外层再包一层**：

```python
# gpu_model_runner.py:4892
self.model = CUDAGraphWrapper(
    self.model,                       # ← Qwen3_5ForCausalLM（内含编译后的 Qwen3_5Model）
    runtime_mode=CUDAGraphMode.FULL
)
```

所以最终的嵌套结构是：

```
self.model = CUDAGraphWrapper(FULL)                    ← 外层
  └─ Qwen3_5ForCausalLM
       └─ Qwen3_5Model (编译后 = split_gm)
            ├─ CUDAGraphWrapper(PIECEWISE) piece 0     ← 内层
            ├─ [eager] attention/gdn op
            ├─ CUDAGraphWrapper(PIECEWISE) piece 1
            ├─ [eager] attention/gdn op
            └─ ...
```

### 10.6 运行时调度总结

```
ForwardContext.cudagraph_runtime_mode = ?
                │
        ┌───────┴───────┐
        ▼               ▼
    PIECEWISE           FULL
        │                │
外层 FULL wrapper:   外层 FULL wrapper:
  PIECEWISE≠FULL       FULL==FULL
  → 透传                → 捕获/回放整个 forward
        │                │
        ▼                ▼
内层 PIECEWISE        内层 PIECEWISE wrapper:
  wrappers:            FULL≠PIECEWISE
  PIECEWISE==           → 透传 runnable
  PIECEWISE             (所有计算都被外层
  → 捕获/回放            FULL graph 录制)
  每个子图
        │
        ▼
attention/gdn ops:
  直接 eager 执行
  (不在任何 graph 中)
```

---

## 11. 启动时预捕获

```python
# gpu_model_runner.py:6001
def capture_model():
    for runtime_mode, batch_descs in cudagraph_dispatcher.get_capture_descs():
        # 先捕获 PIECEWISE（prefill/mixed），再捕获 FULL（decode）
        _capture_cudagraphs(batch_descs, runtime_mode)

def _warmup_and_capture(desc, mode):
    # 1. 配置驱动的 warmup（mode=NONE，eager 执行）
    #    默认 cudagraph_num_of_warmups = 0，因此默认不会多跑 warmup
    for _ in range(num_warmups):
        _dummy_run(desc.num_tokens, cudagraph_runtime_mode=NONE)
    # 2. 一次 capture run
    _dummy_run(desc.num_tokens, cudagraph_runtime_mode=mode)
```

日志输出：
```
"Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)"
"Capturing CUDA graphs (decode, FULL)"
```

补充说明：

- capture 顺序确实是 **PIECEWISE 先，FULL 后**
- warmup 是 **GPUModelRunner 控制** 的，不在 `CUDAGraphWrapper` 内部
- 默认 `cudagraph_num_of_warmups = 0`，所以“若干次 warmup”应理解为“可配置的 warmup”，不是默认必然发生

---

## 12. Prefill vs Decode 对比总结

| 维度 | Prefill (PIECEWISE) | Decode (FULL) |
|---|---|---|
| 外层 CUDAGraphWrapper(FULL) | 透传（mode不匹配） | 捕获/回放整个 forward |
| 内层 CUDAGraphWrapper(PIECEWISE) | 每片独立捕获/回放 | 透传（mode不匹配） |
| Attention / GDN 算子 | eager 执行 | 录进 FULL graph |
| 线性层 / Norm / MLP | 在 PIECEWISE graph 中 | 在 FULL graph 中 |
| graph 数量 | N+1 个小 graph（N=层数） | 1 个大 graph |
| CPU overhead | 每层有 eager dispatch 开销 | 接近零（单次 replay） |
| 适用场景 | 变长序列、混合 batch | 固定 batch、单 token decode |

---

## 13. 关键文件索引

| 功能 | 文件 | 行号 |
|---|---|---|
| CUDAGraphMode 定义 | `vllm/config/compilation.py` | 53-103 |
| splitting_ops 列表 | `vllm/config/compilation.py` | 717-730 |
| CUDAGraphWrapper | `vllm/compilation/cuda_graph.py` | 145-357 |
| @support_torch_compile | `vllm/compilation/decorators.py` | 115-246, 462 |
| GPUWorker.execute_model | `vllm/v1/worker/gpu_worker.py` | 748 |
| GPUModelRunner.execute_model | `vllm/v1/worker/gpu_model_runner.py` | 3877 |
| _is_uniform_decode | `vllm/v1/worker/gpu_model_runner.py` | 3528 |
| _determine_batch_execution_and_padding | `vllm/v1/worker/gpu_model_runner.py` | 3548 |
| _model_forward | `vllm/v1/worker/gpu_model_runner.py` | 3495 |
| capture_model | `vllm/v1/worker/gpu_model_runner.py` | 6001 |
| CudagraphDispatcher | `vllm/v1/cudagraph_dispatcher.py` | 234-323 |
| ForwardContext | `vllm/forward_context.py` | 205 |
| Qwen3_5Model | `vllm/model_executor/models/qwen3_5.py` | 207 |
| Qwen3_5DecoderLayer | `vllm/model_executor/models/qwen3_5.py` | 118 |
| Qwen3NextDecoderLayer.forward | `vllm/model_executor/models/qwen3_next.py` | 398 |
| Qwen3NextAttention.forward | `vllm/model_executor/models/qwen3_next.py` | 281 |
| Attention.forward | `vllm/model_executor/layers/attention/attention.py` | 440 |
| unified_attention_with_output | `vllm/model_executor/layers/attention/attention.py` | 729 |
| unified_kv_cache_update | `vllm/model_executor/layers/attention/attention.py` | 686 |
| FlashAttentionImpl.forward | `vllm/v1/attention/backends/flash_attn.py` | 673 |
| GatedDeltaNetAttention | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | 217 |
| GatedDeltaNetAttention.forward_cuda | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | 513 |
| gdn_attention_core (custom op) | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | 1087 |
| _forward_core | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | 779 |
| ChunkGatedDeltaRule | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | 120 |
| **编译链路** | | |
| _support_torch_compile（装饰器实现） | `vllm/compilation/decorators.py` | 325-632 |
| 替换 __init__ | `vllm/compilation/decorators.py` | 349-407 |
| 替换 __call__（首次编译 + 后续调用） | `vllm/compilation/decorators.py` | 462-632 |
| TorchCompileWithNoGuardsWrapper | `vllm/compilation/wrapper.py` | 47-298 |
| TorchCompileWithNoGuardsWrapper.__init__（torch.compile 创建） | `vllm/compilation/wrapper.py` | 78-174 |
| TorchCompileWithNoGuardsWrapper.__call__（运行时入口） | `vllm/compilation/wrapper.py` | 185-215 |
| VllmBackend（编译后端） | `vllm/compilation/backends.py` | 784 |
| VllmBackend.__call__（Dynamo 回调入口） | `vllm/compilation/backends.py` | 981 |
| split_graph（FX 图切割） | `vllm/compilation/backends.py` | 532-606 |
| PiecewiseCompileInterpreter（子图编译 + wrapper 包装） | `vllm/compilation/backends.py` | 666-755 |
| wrap_with_cudagraph_if_needed（子图包装 PIECEWISE wrapper） | `vllm/compilation/backends.py` | 612-663 |
| 外层 FULL wrapper 包装 | `vllm/v1/worker/gpu_model_runner.py` | 4892 |
