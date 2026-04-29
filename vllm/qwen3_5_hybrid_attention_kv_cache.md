# Qwen3.5 Hybrid Attention 架构与 KV Cache 管理知识库

> 基于 vLLM 真实源码分析，覆盖模型架构实现、GatedDeltaNet 状态管理、KV Cache 系统管理、Page Size 对齐、FP8 支持等完整主题。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [Hybrid Attention 实现](#2-hybrid-attention-实现)
3. [GatedDeltaNet 状态管理](#3-gateddeltanet-状态管理)
4. [KV Cache Spec 生成](#4-kv-cache-spec-生成)
5. [KV Cache 分组与 Page Size 对齐](#5-kv-cache-分组与-page-size-对齐)
6. [Hybrid KV Cache 协调器](#6-hybrid-kv-cache-协调器)
7. [Prefix Caching 策略](#7-prefix-caching-策略)
8. [MambaManager 完整行为](#8-mambamanager-完整行为)
9. [Mamba Cache Mode](#9-mamba-cache-mode)
10. [FP8 KV Cache 支持](#10-fp8-kv-cache-支持)
11. [Qwen3.5 vs Qwen3Next 关键差异](#11-qwen35-vs-qwen3next-关键差异)
12. [完整数据流图](#12-完整数据流图)
13. [参考文件索引](#13-参考文件索引)

---

## 1. 整体架构概览

Qwen3.5 采用 **Linear Attention（GatedDeltaNet）+ Full Attention（标准 GQA）** 混合架构：

- **Linear Attention 层**：类 Mamba 的状态空间模型，KV cache 大小固定，不随序列长度增长
- **Full Attention 层**：标准 GQA，保留完整注意力能力，KV cache 随序列长度线性增长

层类型由 `config.layer_types` 决定，默认 `full_attention_interval=4`，即每 4 层一个 `full_attention`，其余 3 层都是 `linear_attention`：

```python
# vllm/transformers_utils/configs/qwen3_5.py
self.layer_types = [
    "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
    for i in range(self.num_hidden_layers)
]
# 32 层示例: [L, L, L, F, L, L, L, F, L, L, L, F, ...]
# → 24 个 linear_attention + 8 个 full_attention
```

**Qwen3.5 Dense 核心配置**（`vllm/transformers_utils/configs/qwen3_5.py`）：

| 参数 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 248320 | |
| `hidden_size` | 4096 | |
| `num_hidden_layers` | 32 | |
| `num_attention_heads` | 16 | Q heads（Full Attention 层） |
| `num_key_value_heads` | 4 | KV heads（GQA 4:1） |
| `head_dim` | 256 | |
| `linear_conv_kernel_dim` | 4 | 短卷积核大小 |
| `linear_key_head_dim` | 128 | Linear Attention K head 维度 |
| `linear_value_head_dim` | 128 | Linear Attention V head 维度 |
| `linear_num_key_heads` | 16 | Linear Attention K heads 数量 |
| `linear_num_value_heads` | 32 | Linear Attention V heads 数量 |

**Qwen3Next 配置对比**（`vllm/transformers_utils/configs/qwen3_next.py`）：

| 参数 | Qwen3.5 | Qwen3Next |
|---|---|---|
| `vocab_size` | 248320 | 151936 |
| `hidden_size` | 4096 | 2048 |
| `num_hidden_layers` | 32 | 48 |
| `num_key_value_heads` | 4 | 2 |
| `num_experts` | — | 512 |
| `num_experts_per_tok` | — | 10 |
| `full_attention_interval` | 可自定义 | 硬编码为 4 |

**KV Cache 内存节省来源（三重优化）**：

| 优化手段 | 节省比例 | 说明 |
|---|---|---|
| GQA（Dense 版本） | 4x | 16 Q heads : 4 KV heads |
| GQA（MoE 版本） | 8x | 16 Q heads : 2 KV heads |
| Linear Attention 层 | ∞ | 不需要传统 KV cache，只保留固定状态矩阵 |

---

## 2. Hybrid Attention 实现

### 2.1 DecoderLayer 的分支逻辑

**文件**：`vllm/model_executor/models/qwen3_5.py`（839 行）

`Qwen3_5DecoderLayer` 继承自 `Qwen3NextDecoderLayer`，**没有自己的 `forward` 方法**，完全使用父类的 forward。它只重写了 `__init__`：

```python
class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(self, vllm_config, layer_type, prefix):
        # 注意：跳过直接父类的 __init__，直接调用 nn.Module.__init__
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        # ① 根据层类型选择注意力机制
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,       # Qwen3.5 使用非交错布局
                create_in_proj_qkvz=vllm_config.lora_config is None,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config, model_config=model_config,
                cache_config=cache_config, quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        # ② 根据模型类型选择 MLP（Dense 或 MoE）
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(vllm_config=vllm_config, ...)
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(hidden_size=config.hidden_size, ...)

        # ③ Layer Norm + 可选 Layer Scale
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.ffn_layer_scale = torch.nn.Parameter(torch.zeros(1, 1, config.hidden_size))
```

**Forward 方法**（定义在父类 `Qwen3NextDecoderLayer`，`vllm/model_executor/models/qwen3_next.py`）：

```python
def forward(self, hidden_states, residual, positions=None, **kwargs):
    # Step 1: Pre-Norm（fused add-norm）
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    # Step 2: Attention（根据层类型分支）
    self_attention_output = torch.empty_like(hidden_states)
    if self.layer_type == "linear_attention":
        self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
    elif self.layer_type == "full_attention":
        self.self_attn(hidden_states=hidden_states, output=self_attention_output, positions=positions)
    hidden_states = self_attention_output

    # Step 3: 可选 Layer Scale
    if self.layer_scale:
        hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)

    # Step 4: Post-Norm + MLP
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)

    # Step 5: 可选 FFN Layer Scale
    if self.layer_scale:
        hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)

    return hidden_states, residual
```

### 2.2 Full Attention 层：Qwen3NextAttention

**文件**：`vllm/model_executor/models/qwen3_next.py`

```python
class Qwen3NextAttention(nn.Module):
    def __init__(self, config, model_config=None, cache_config=None, quant_config=None, prefix=""):
        self.total_num_heads = config.num_attention_heads   # 16
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)  # 4 // tp
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)  # 256
        self.scaling = self.head_dim**-0.5
        self.attn_output_gate = getattr(config, "attn_output_gate", True)  # 默认开启

        # QKV 投影：Q 维度翻倍（后半部分作为 gate）
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size, self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),  # gate 时 Q 维度 ×2
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
        )
        self.o_proj = RowParallelLinear(...)
        self.rotary_emb = get_rope(...)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling,
                              num_kv_heads=self.num_kv_heads, cache_config=cache_config)
        # QK-Norm（Qwen3 系列标志性设计）
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions, output, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        # Attention Gate：Q 维度翻倍，后半部分作为 gate
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-Norm（per-head RMSNorm，防止注意力分数爆炸）
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.num_heads * self.head_dim)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.num_kv_heads * self.head_dim)

        # RoPE 位置编码
        q, k = self.rotary_emb(positions, q, k)

        # 标准注意力计算（同时读写 KV cache）
        attn_output = self.attn(q, k, v)

        # 应用 Attention Gate
        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        # 输出投影
        output[:], _ = self.o_proj(attn_output)
```

### 2.3 Linear Attention 层：GatedDeltaNetAttention

**文件**：`vllm/model_executor/layers/mamba/gdn_linear_attn.py`（1352 行）

```python
@PluggableLayer.register("gated_delta_net_attention")
class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    """
    核心公式: S_t = exp(g_t) · S_{t-1} + β_t · (v_t - exp(g_t) · S_{t-1} · k_t) · k_t^T
    用递归更新状态矩阵 S，不需要存储历史 token 的 KV
    """

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def __init__(self, config, vllm_config, prefix="",
                 create_in_proj_qkvz=True, gqa_interleaved_layout=False):
        self.num_v_heads = config.linear_num_value_heads   # 32
        self.num_k_heads = config.linear_num_key_heads     # 16
        self.head_k_dim = config.linear_key_head_dim       # 128
        self.head_v_dim = config.linear_value_head_dim     # 128
        self.key_dim = self.head_k_dim * self.num_k_heads  # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads # 4096
        self.conv_kernel_size = config.linear_conv_kernel_dim  # 4
        self.conv_dim = self.key_dim * 2 + self.value_dim  # 8192

        # 短卷积层（causal conv1d）
        self.conv1d = ColumnParallelLinear(input_size=self.conv_kernel_size, output_size=self.conv_dim)

        # QKVZ 投影
        if create_in_proj_qkvz:
            # 非 LoRA 路径：融合的 in_proj_qkvz
            self.in_proj_qkvz = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            )
        else:
            # LoRA 路径：分离的 in_proj_qkv 和 in_proj_z
            self.in_proj_qkv = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[self.key_dim, self.key_dim, self.value_dim],
            )
            self.in_proj_z = ColumnParallelLinear(
                input_size=self.hidden_size, output_size=self.value_dim,
            )

        # BA 投影（beta/学习率 + alpha/遗忘因子）
        self.in_proj_ba = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.num_v_heads] * 2,
        )

        # Delta Rule 参数
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads // self.tp_size))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads // self.tp_size, dtype=torch.float32))

        # 门控归一化 + 输出投影
        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size, bias=False)
```

### 2.4 与 KV Cache 系统的对接

**文件**：`vllm/model_executor/models/qwen3_5.py`

```python
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid):
    # IsHybrid 接口告诉 vLLM 这是混合模型

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        """返回 linear_attention 层的固定状态大小"""
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size, hf_config.linear_num_key_heads, hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim, hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim, num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
```

---

## 3. GatedDeltaNet 状态管理

### 3.1 状态定义与真实形状

**`MambaStateShapeCalculator.gated_delta_net_state_shape`**（`vllm/model_executor/layers/mamba/mamba_utils.py`）：

```python
@classmethod
def gated_delta_net_state_shape(cls, tp_world_size, num_k_heads, num_v_heads,
                                 head_k_dim, head_v_dim, conv_kernel_size, num_spec=0):
    # conv_dim 包含 Q、K、V 三部分
    conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads

    conv_state_shape = cls._orient_conv_shape(
        divide(conv_dim, tp_world_size),
        conv_kernel_size - 1 + num_spec,
    )
    temporal_state_shape = (
        divide(num_v_heads, tp_world_size),
        head_v_dim,
        head_k_dim,
    )
    return conv_state_shape, temporal_state_shape
```

**以 Qwen3.5 默认配置（tp=1, num_spec=0）为例**：

```
conv_dim = 128 * 16 * 2 + 128 * 32 = 8192

conv_state_shape  = (8192, 3)        # (conv_dim, conv_kernel_size - 1)
temporal_state_shape = (32, 128, 128)  # (num_v_heads, head_v_dim, head_k_dim)

总元素 = 8192 × 3 + 32 × 128 × 128 = 24,576 + 524,288 = 548,864
bf16 bytes = 548,864 × 2 = 1,097,728 bytes ≈ 1 MB/层
```

| 状态 | 存储位置 | 形状 | 含义 |
|---|---|---|---|
| `conv_state` | `self.kv_cache[0]` | `(8192, 3)` | 短卷积历史缓冲，保存最近 `conv_kernel_size-1` 个 token 的 Q/K/V 拼接 |
| `ssm_state` | `self.kv_cache[1]` | `(32, 128, 128)` | Delta Rule 核心状态矩阵 S，压缩了所有历史 KV 信息 |

### 3.2 Forward 流程（forward_cuda）

```python
def forward_cuda(self, hidden_states, output):
    num_tokens = hidden_states.size(0)

    # Part 1: Input Projection
    if hasattr(self, "in_proj_qkv"):
        # LoRA 路径：分离的 in_proj_qkv 和 in_proj_z
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()
    else:
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        if self.gqa_interleaved_layout:
            # Qwen3-Next：解包交错 GQA 布局
            query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5：直接按大小切分
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()

    # Part 2: Core Attention（自定义 CUDA op）
    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    # 内部完成：causal_conv1d（更新 conv_state）+ gated_delta_rule（更新 ssm_state）
    torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, self.prefix)

    # Part 3: Output Projection
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)  # RMSNorm(x) * sigmoid(z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    output[:num_tokens], _ = self.out_proj(core_attn_out)
```

### 3.3 核心计算：_forward_core

`torch.ops.vllm.gdn_attention_core` 内部调用 `self._forward_core`，分两步处理：

**Step 1: Convolution（短卷积，更新 conv_state）**

| 阶段 | 函数 | 说明 |
|---|---|---|
| Prefill | `causal_conv1d_fn` | 并行处理整个序列 |
| Decode | `causal_conv1d_update` | 逐 token 更新 |

**Step 2: Recurrent Attention（Delta Rule，更新 ssm_state）**

| 阶段 | 函数 | 说明 |
|---|---|---|
| Prefill | `chunk_gated_delta_rule` | 分块并行，支持 FlashInfer（SM90+）和 Triton/FLA 两种后端 |
| Decode | `fused_sigmoid_gating_delta_rule_update` | 逐 token 更新，延迟极低 |
| Packed Decode | `fused_recurrent_gated_delta_rule_packed_decode` | 多请求打包处理，提升吞吐 |

### 3.4 Delta Rule 数学原理

```
Gated Delta Rule（增加遗忘门 α）：
  g_t = -exp(A_log) · softplus(a_t + dt_bias)   # 遗忘因子（log 域）
  β_t = sigmoid(b_t)                              # 学习率
  S_t = exp(g_t) · S_{t-1} + β_t · (v_t - exp(g_t) · S_{t-1} · k_t) · k_t^T

输出：
  output_t = Q_t · S_t   # 线性注意力输出

其中：
  S_t ∈ R^{d_v × d_k}：状态矩阵（ssm_state），压缩了所有历史 KV 信息
  k_t ∈ R^{d_k}：key 向量（经过 L2 归一化）
  v_t ∈ R^{d_v}：value 向量（经过短卷积处理）
  β_t ∈ R：学习率（由 in_proj_ba 的 b 分支生成）
  g_t ∈ R：遗忘因子（由 in_proj_ba 的 a 分支 + A_log + dt_bias 生成）
```

**`fused_gdn_gating` Triton kernel 的真实计算**：

```python
x = a + dt_bias
softplus_x = (1/β) * log(1 + exp(β * x))   # softplus，β=1, threshold=20
g = -exp(A_log) * softplus_x
beta_output = sigmoid(b)
```

---

## 4. KV Cache Spec 生成

不同层类型生成不同的 KV Cache Spec：

### 4.1 Full Attention → FullAttentionSpec

**文件**：`vllm/model_executor/layers/attention/attention.py`

```python
def get_kv_cache_spec(self, vllm_config):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=self.num_kv_heads,
        head_size=self.head_size,
        dtype=self.kv_cache_torch_dtype,     # 由 cache_config.cache_dtype 决定
        kv_quant_mode=quant_mode,            # FP8_PER_TENSOR / NONE
    )
```

### 4.2 Linear Attention → MambaSpec

**文件**：`vllm/model_executor/layers/mamba/abstract.py`

```python
def get_kv_cache_spec(self, vllm_config):
    return MambaSpec(
        shapes=self.get_state_shape(),       # (conv_state_shape, ssm_state_shape)
        dtypes=self.get_state_dtype(),       # 由模型配置决定，通常 bf16
        block_size=mamba_block_size,
        mamba_type=self.mamba_type,           # "gdn_attention"
        mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
        num_speculative_blocks=...,
    )
```

### 4.3 FullAttentionSpec 真实定义

**文件**：`vllm/v1/kv_cache_interface.py`（735 行）

```python
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    head_size_v: int = None
    sliding_window: int | None = None
    attention_chunk_size: int | None = None

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    def max_memory_usage_bytes(self, vllm_config):
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

### 4.4 MambaSpec 真实定义

```python
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_type: str = "mamba2"
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

    def max_memory_usage_bytes(self, vllm_config):
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return cdiv(max_model_len, self.block_size) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)
```

### 4.5 Linear Attention vs Full Attention KV Cache 对比

| 维度 | Full Attention | Linear Attention |
|---|---|---|
| **缓存内容** | 每 token 的 K/V 向量 | 固定大小的内部状态（conv_state + ssm_state） |
| **Spec 类型** | `FullAttentionSpec` | `MambaSpec` |
| **内存增长** | O(序列长度) | O(1)，与序列长度无关 |
| **基类** | `nn.Module` | `MambaBase` |
| **核心操作** | softmax(QK^T/√d) · V | 递归状态更新 S_t = f(S_{t-1}, k_t, v_t) |
| **归一化** | Q/K RMSNorm + RoPE | Q/K L2 Norm（在 GDN kernel 内部） |
| **Spec 字段** | `num_kv_heads`, `head_size`, `block_size` | `shapes`, `dtypes`, `mamba_type`, `mamba_cache_mode` |

---

## 5. KV Cache 分组与 Page Size 对齐

### 5.1 分组策略

**文件**：`vllm/v1/core/kv_cache_utils.py`

`_get_kv_cache_groups_uniform_page_size()` 按 spec 类型分组：

```python
same_type_layers = defaultdict(list)
for layer_name, layer_spec in kv_cache_spec.items():
    same_type_layers[layer_spec].append(layer_name)
```

Qwen3.5 分组结果：

```
Group 0: 24 个 linear_attention 层 → MambaSpec
Group 1:  8 个 full_attention 层   → FullAttentionSpec
```

### 5.2 统一 Page Size 约束

**文件**：`vllm/v1/core/kv_cache_utils.py`

```python
def get_uniform_page_size(kv_cache_specs):
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    assert len(page_sizes) == 1  # 所有 group 的 page_size 必须相同
```

**为什么必须统一？** vLLM 为 hybrid 模型创建 `group_size` 个共享 tensor，每个 tensor 被每个 group 的第 i 层共享。如果 page_size 不同，就无法用同一个 `num_blocks` 来分配这些 tensor。

### 5.3 对齐机制

**文件**：`vllm/platforms/interface.py`

三种对齐方式：

**1. block_size 放大**：
```python
if cache_config.block_size < attn_block_size:
    cache_config.block_size = attn_block_size
```

**2. mamba_block_size = attention block_size**（align 模式）：
```python
if cache_config.mamba_cache_mode == "align":
    cache_config.mamba_block_size = cache_config.block_size
```

**3. mamba page_size 填充到匹配 attention page_size**：
```python
attn_page_size = cache_config.block_size * attn_page_size_1_token
cache_config.mamba_page_size_padded = attn_page_size
```

### 5.4 Page Size 计算公式

| 层类型 | 公式 | 说明 |
|---|---|---|
| **FullAttentionSpec** | `block_size × num_kv_heads × (head_size + head_size_v) × dtype_size` | 随 block_size 线性增长 |
| **MambaSpec** | `Σ(shape_i × dtype_i)`，可被 `page_size_padded` 覆盖 | 固定值 ~1 MB/层（bf16） |

由于 mamba 的 page_size 是固定的（约 1 MB），远小于 attention page_size 乘以 block_size 后的值，vLLM 把 attention page_size 作为基准，给 mamba 的 MambaSpec 加 `page_size_padded` 填充到完全一致。

**对齐代价**：mamba 层实际数据 ~1 MB，填充后 ~1.1 MB，浪费约 10%。但 mamba 层本身内存占比很小（MB 级别 vs attention 的 GB 级别），实际影响可忽略。

### 5.5 MultiGroupBlockTable

**文件**：`vllm/v1/worker/block_table.py`

每个 KV cache group 维护独立的 `BlockTable`：

```python
class MultiGroupBlockTable:
    """不同 group 可能有不同的 block_size, kernel_block_size, max_num_blocks_per_req"""
```

---

## 6. Hybrid KV Cache 协调器

**文件**：`vllm/v1/core/kv_cache_coordinator.py`（570 行）

当 `len(kv_cache_groups) > 1` 时使用 `HybridKVCacheCoordinator`：

- 每个 group 有独立的 `SingleTypeKVCacheManager`
- 用 LCM（最小公倍数）对齐不同 group 的 block size
- `find_longest_cache_hit()` 使用迭代固定点算法

### 6.1 find_longest_cache_hit 完整实现

```python
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    num_groups = len(self.kv_cache_config.kv_cache_groups)
    hit_length = max_cache_hit_length
    hit_blocks_by_group = [None] * num_groups

    # 优化：只有 2 种 attention 类型且第一种是 FullAttention 时，只需迭代一次
    is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
        self.attention_groups[0][0], FullAttentionSpec
    )

    while True:
        curr_hit_length = hit_length

        for spec, group_ids, manager_cls in self.attention_groups:
            is_full_attn = isinstance(spec, FullAttentionSpec)
            cached_blocks = hit_blocks_by_group[group_ids[0]]

            if is_full_attn and cached_blocks is not None:
                # Full Attention 向下封闭属性：直接截断到 block 边界
                num_blocks = curr_hit_length // spec.block_size
                curr_hit_length = num_blocks * spec.block_size
            else:
                # 调用对应类型的 manager 查找
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=..., max_length=curr_hit_length,
                    kv_cache_group_ids=group_ids, block_pool=self.block_pool,
                    kv_cache_spec=spec, use_eagle=self.use_eagle,
                    alignment_tokens=self.lcm_block_size,
                )
                curr_hit_length = len(hit_blocks[0]) * spec.block_size
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

        if curr_hit_length >= hit_length:
            break   # 收敛
        hit_length = curr_hit_length
        if is_simple_hybrid:
            break   # 简单 hybrid 只需一次

    # 最终截断：确保 full attention 块数与 hit_length 一致
    spec, group_ids, _ = self.attention_groups[0]
    if isinstance(spec, FullAttentionSpec):
        num_blocks = hit_length // spec.block_size
        for group_id in group_ids:
            if (blks := hit_blocks_by_group[group_id]) is not None:
                del blks[num_blocks:]

    return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length
```

### 6.2 迭代固定点算法示例

```
初始：hit_length = 1000 tokens

Round 1:
  FullAttention：从左到右扫描，命中到 800 tokens → curr = 800
  MambaManager：在 800 范围内从右到左扫描，命中到 750 tokens → curr = 750

curr(750) < hit_length(1000)，继续
hit_length = 750

Round 2:
  FullAttention：cached_blocks 已存在，截断到 block 边界 → curr = 750
  MambaManager：在 750 范围内，命中到 750 → curr = 750

curr(750) >= hit_length(750)，收敛！最终 hit_length = 750
```

---

## 7. Prefix Caching 策略

### 7.1 FullAttentionManager：从左到右扫描

**文件**：`vllm/v1/core/single_type_kv_cache_manager.py`

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                            block_pool, kv_cache_spec, use_eagle, alignment_tokens,
                            dcp_world_size=1, pcp_world_size=1):
    block_size = kv_cache_spec.block_size
    if dcp_world_size * pcp_world_size > 1:
        block_size *= dcp_world_size * pcp_world_size
    max_num_blocks = max_length // block_size

    # 从左到右扫描，遇到 cache miss 立即停止（必须连续）
    for block_hash in itertools.islice(block_hashes, max_num_blocks):
        if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        else:
            break

    # EAGLE 投机解码：去掉最后一个块
    if use_eagle and computed_blocks[0]:
        for computed in computed_blocks:
            computed.pop()

    # alignment 对齐：确保命中长度是 lcm_block_size 的整数倍
    while (block_size != alignment_tokens
           and len(computed_blocks[0]) * block_size % alignment_tokens != 0):
        for computed in computed_blocks:
            computed.pop()

    return computed_blocks
```

### 7.2 MambaManager：从右到左扫描

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                            block_pool, kv_cache_spec, use_eagle, alignment_tokens, ...):
    block_size = kv_cache_spec.block_size
    max_num_blocks = max_length // block_size

    # 从右到左扫描，找到最后一个匹配即停止
    for i in range(max_num_blocks - 1, -1, -1):
        if cached_block := block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids):
            # alignment 对齐检查
            if (block_size != alignment_tokens
                    and (i + 1) * block_size % alignment_tokens != 0):
                continue
            for computed, cached in zip(computed_blocks, cached_block):
                computed.extend([block_pool.null_block] * i)  # 前面填充 null block
                computed.append(cached)
            break  # 只需最后一个匹配

    return computed_blocks
```

### 7.3 策略差异

| 类型 | 扫描方向 | 命中条件 | 原因 |
|---|---|---|---|
| Full Attention | 从左到右，遇 miss 停止 | 必须从头连续命中 | 注意力需要访问所有历史 token 的 KV |
| Mamba/GDN | 从右到左，找到即停 | 只需最新状态命中 | 状态矩阵已压缩所有历史，只需恢复最新状态 |

---

## 8. MambaManager 完整行为

### 8.1 核心方法

```python
def get_num_skipped_tokens(self, num_computed_tokens):
    """Mamba 只需保留最后一个 token 的状态，前面所有 token 都可以跳过"""
    return num_computed_tokens - 1
```

### 8.2 三种 mamba_cache_mode 的行为

| 模式 | 内存占用 | 行为 | 适用场景 |
|---|---|---|---|
| `none`（默认） | `(1 + num_spec_blocks) × page_size` | 只保留当前状态 | 普通推理 |
| `align` | `(2 + num_spec_blocks) × page_size` | 保留当前 + 上一步状态 | 投机解码（MTP/EAGLE） |
| `all` | `ceil(max_model_len / block_size) × page_size` | 缓存所有 token 的状态 | **Qwen3.5 不支持** |

### 8.3 align 模式的特殊逻辑

```python
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    if self.mamba_cache_mode == "align":
        num_tokens = num_tokens_main_model  # 忽略 lookahead tokens
        num_required_blocks = cdiv(num_tokens, self.block_size) + self.num_speculative_blocks

        if request_id in self._allocated_block_reqs:
            num_new_blocks = 1  # 老请求：复用上一步投机 blocks，最多新分配 1 个
        else:
            num_new_blocks = 1 + self.num_speculative_blocks  # 首次 prefill
```

### 8.4 跨请求缓存隔离

```python
def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks, ...):
    if (len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step):
        # Mamba 不能依赖当前 step 中其他请求生成的 blocks
        # 返回 num_gpu_blocks + 1 让调度器跳过本次调度
        return self.block_pool.num_gpu_blocks + 1
```

---

## 9. Mamba Cache Mode

**文件**：`vllm/model_executor/models/qwen3_5.py`

Qwen3.5 只支持 `none` 和 `align` 模式：

```python
if cache_config.mamba_cache_mode == "all":
    raise NotImplementedError(
        "Qwen3.5 currently does not support 'all' prefix caching, "
        "please use '--mamba-cache-mode=align' instead"
    )
```

各模式下的内存计算（`MambaSpec.max_memory_usage_bytes`）：

```python
"align" → page_size_bytes * (2 + num_speculative_blocks)   # Qwen3.5 使用
"none"  → page_size_bytes * (1 + num_speculative_blocks)
"all"   → ceil(max_model_len / block_size) * page_size_bytes  # 不支持
```

---

## 10. FP8 KV Cache 支持

### 10.1 已有支持

vLLM 有 Qwen3.5 + FP8 KV Cache 的测试配置：

**文件**：`tests/evals/gsm8k/configs/Qwen3.5-35B-A3B-FP8-DEP2.yaml`
```yaml
model_name: "Qwen/Qwen3.5-35B-A3B-FP8"
server_args: >-
  --max-model-len 4096
  --data-parallel-size 2
  --enable-expert-parallel
  --kv-cache-dtype fp8
```

### 10.2 FP8 在 Hybrid 中的行为

FP8 KV Cache **只对 full_attention 层生效**，linear_attention 层的 Mamba 状态仍使用模型原始 dtype：

| 层类型 | Spec | KV Cache dtype |
|---|---|---|
| `full_attention` | `FullAttentionSpec` | `fp8`（由 `--kv-cache-dtype fp8` 控制） |
| `linear_attention` | `MambaSpec` | 模型 dtype（通常 bf16） |

**原因**：
- `MambaSpec` 的 dtype 由模型自身通过 `get_state_dtype()` 直接给出，不经过 `kv_cache_dtype` 配置
- `FullAttentionSpec` 的 dtype 从 `cache_config.cache_dtype` 读取

### 10.3 FP8 不需要换 Attention 算子

FP8 KV Cache 对 Qwen3.5 模型层代码**完全透明**，不需要使用其他 attention 算子。FP8 的量化/反量化完全在 attention backend 内部自动完成：

```
Qwen3NextAttention（模型层，不感知 FP8）
  └── self.attn = Attention(...)              ← 标准 Attention 类
        └── TritonAttentionBackend            ← backend 内部处理 FP8
              ├── reshape_and_cache()         ← 写入时：BF16 → FP8 量化
              └── unified_attention()         ← 计算时：FP8 → BF16 反量化
```

### 10.4 Attention Backend 选择

当 `kv_cache_dtype=fp8` 时，vLLM 自动选择 **TritonAttentionBackend**（FlashAttention 原生不支持 FP8 KV Cache）：

**文件**：`vllm/v1/attention/selector.py`、`vllm/platforms/cuda.py`

```python
def get_attn_backend(head_size, dtype, kv_cache_dtype, ...):
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size, dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,  # "fp8"
    )
    # → 选择 TritonAttentionBackend（支持 FP8 per-tensor 量化）
```

### 10.5 量化模式

**文件**：`vllm/v1/kv_cache_interface.py`

```python
class KVQuantMode(IntEnum):
    NONE = 0                    # 不量化
    FP8_PER_TENSOR = 1          # --kv-cache-dtype fp8 走这条路（单个 tensor-wide scale）
    INT8_PER_TOKEN_HEAD = 2     # INT8 per-token-head
    FP8_PER_TOKEN_HEAD = 3      # FP8 per-token-head（实验性，每个 token 每个 head 独立 scale）
```

映射逻辑：
```python
def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    if kv_cache_dtype.startswith("fp8"):     # "fp8", "fp8_e4m3" 等
        return KVQuantMode.FP8_PER_TENSOR
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    return KVQuantMode.NONE
```

### 10.6 量化/反量化的具体实现

**写入时量化**（`reshape_and_cache`）：

**文件**：`vllm/v1/attention/backends/triton_attn.py`

```python
def reshape_and_cache(self, layer, key, value, kv_cache, slot_mapping):
    if self._is_per_token_head_quant:
        # Per-token-head 量化路径
        triton_reshape_and_cache_flash_per_token_head_quant(
            key, value, key_cache, value_cache,
            self._k_scale_cache, self._v_scale_cache, slot_mapping
        )
    else:
        # FP8 per-tensor 量化路径（--kv-cache-dtype fp8 走这里）
        triton_reshape_and_cache_flash(
            key, value, key_cache, value_cache,
            slot_mapping, self.kv_cache_dtype,
            layer._k_scale, layer._v_scale     # per-tensor scale
        )
```

**计算时反量化**（`unified_attention` kernel 内部）：

**文件**：`vllm/v1/attention/ops/triton_unified_attention.py`

```python
@triton.jit
def _prepare_kv_tile(data, Q, tensor_scale, ..., KV_QUANT_MODE):
    if KV_QUANT_MODE == 1:  # FP8_PER_TENSOR
        if Q.dtype.is_fp8():
            return data.to(Q.dtype), unused_scales
        # 反量化：FP8 → float32 → Q.dtype
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype), unused_scales
    if KV_QUANT_MODE >= 2:  # per-token-head
        token_head_scales = tl.load(scale_cache_ptr + scale_idx, ...)
        return data.to(Q.dtype), token_head_scales
```

### 10.7 完整调用链

```
用户设置 --kv-cache-dtype fp8
    ↓
Attention.get_kv_cache_spec() → quant_mode = FP8_PER_TENSOR
    ↓
get_attn_backend() → 选择 TritonAttentionBackend
    ↓
TritonAttentionImpl.forward():
    ├── reshape_and_cache()
    │     └── triton_reshape_and_cache_flash()    [BF16 → FP8 量化，写入 cache]
    └── unified_attention()
          └── _prepare_kv_tile()                  [FP8 → BF16 反量化，参与计算]
```

### 10.8 关键文件

| 文件 | 作用 |
|---|---|
| `vllm/v1/attention/selector.py` | Attention backend 选择逻辑 |
| `vllm/v1/attention/backends/triton_attn.py` | TritonAttentionBackend 实现（量化写入 + forward） |
| `vllm/v1/attention/ops/triton_unified_attention.py` | Attention Triton kernel（反量化读取） |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | KV cache 写入 Triton kernel（量化） |

---

## 11. Qwen3.5 vs Qwen3Next 关键差异

| 特性 | Qwen3.5 | Qwen3Next |
|---|---|---|
| **GQA 布局** | `gqa_interleaved_layout=False`（非交错） | `gqa_interleaved_layout=True`（交错） |
| **QKVZ 投影** | 4 个独立输出 `[key_dim, key_dim, value_dim, value_dim]` | 单个融合输出 |
| **BA 投影** | 2 个独立输出 `[num_v_heads, num_v_heads]` | 单个融合输出 `[num_v_heads * 2]` |
| **LoRA 支持** | 支持（分离 `in_proj_qkv` + `in_proj_z`） | 不支持 |
| **权重解包** | 直接 `split` | 需要 `fix_query_key_value_ordering` 解交错 |
| **MLP 类型** | 根据 `model_type` 选择 | 根据 `decoder_sparse_step` 选择 |
| **`full_attention_interval`** | 支持自定义 | 硬编码为 4 |
| **`mamba_cache_mode="all"`** | 不支持 | 不支持 |

**GQA 布局差异的影响**（`create_qkvz_proj` 方法）：

```python
output_sizes = (
    # Qwen3-Next（交错布局）：单个融合输出
    [sum((key_dim, key_dim, value_dim, value_dim))]
    if self.gqa_interleaved_layout
    # Qwen3.5（非交错布局）：4 个独立输出
    else [key_dim, key_dim, value_dim, value_dim]
)
```

---

## 12. 完整数据流图

```
输入 Token IDs
      ↓
Embedding Layer（VocabParallelEmbedding）
      ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Qwen3_5Model（32 层，4 层为一个周期）                   │
│                                                                     │
│  Layer 0,1,2（linear_attention）：                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Pre-Norm → GatedDeltaNetAttention.forward_cuda:               │  │
│  │   [in_proj_qkvz] → mixed_qkv, z                              │  │
│  │   [in_proj_ba]   → b (beta), a (alpha)                        │  │
│  │   causal_conv1d → 更新 conv_state (8192, 3)                   │  │
│  │   gated_delta_rule → 更新 ssm_state (32, 128, 128)            │  │
│  │   RMSNormGated(output, z) → [out_proj]                        │  │
│  │ Post-Norm → MLP                                               │  │
│  │ KV Cache：MambaSpec（固定 ~1 MB/层）                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Layer 3（full_attention）：                                        │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Pre-Norm → Qwen3NextAttention.forward:                        │  │
│  │   [qkv_proj] → Q (×2 for gate), K, V                         │  │
│  │   QK-Norm → RoPE → Attention(Q,K,V) → Gate → [o_proj]        │  │
│  │ Post-Norm → MLP                                               │  │
│  │ KV Cache：FullAttentionSpec（O(seq_len)）                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Layer 4,5,6（linear）... Layer 7（full）... 以此类推               │
└─────────────────────────────────────────────────────────────────────┘
      ↓
Final Norm → LM Head → Logits → 下一个 Token


KV Cache 管理层：
┌─────────────────────────────────────────────────────────────────────┐
│  HybridKVCacheCoordinator                                           │
│  ├── FullAttentionManager（8 个 full_attn 层）                      │
│  │     从左到右扫描，必须连续命中                                    │
│  │     支持 EAGLE / DCP / PCP                                       │
│  ├── MambaManager（24 个 linear_attn 层）                           │
│  │     从右到左扫描，只需最新状态命中                                │
│  │     支持 none / align 两种缓存模式                                │
│  └── 迭代固定点算法协调两种类型                                      │
│                                                                     │
│  Page Size 对齐：mamba page_size 填充到与 attention 一致             │
│  FP8：只对 full_attention 层生效，linear_attention 层保持 bf16       │
│  MultiGroupBlockTable：每个 group 独立的 BlockTable                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 13. 参考文件索引

| 组件 | 文件路径 | 行数 |
|---|---|---|
| Qwen3.5 模型实现 | `vllm/model_executor/models/qwen3_5.py` | 839 |
| Qwen3Next Attention | `vllm/model_executor/models/qwen3_next.py` | 810 |
| GatedDeltaNetAttention | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | 1352 |
| Mamba 状态工具 | `vllm/model_executor/layers/mamba/mamba_utils.py` | — |
| MambaBase 抽象类 | `vllm/model_executor/layers/mamba/abstract.py` | — |
| IsHybrid 接口 | `vllm/model_executor/models/interfaces.py` | — |
| Qwen3.5 Dense 配置 | `vllm/transformers_utils/configs/qwen3_5.py` | 175 |
| Qwen3.5 MoE 配置 | `vllm/transformers_utils/configs/qwen3_5_moe.py` | — |
| Qwen3Next 配置 | `vllm/transformers_utils/configs/qwen3_next.py` | 368 |
| KV Cache Spec 定义 | `vllm/v1/kv_cache_interface.py` | 735 |
| KV Cache 分组 | `vllm/v1/core/kv_cache_utils.py` | — |
| Hybrid KV 协调器 | `vllm/v1/core/kv_cache_coordinator.py` | 570 |
| 单类型 KV Cache 管理器 | `vllm/v1/core/single_type_kv_cache_manager.py` | 819 |
| Block Pool | `vllm/v1/core/block_pool.py` | — |
| MultiGroup Block Table | `vllm/v1/worker/block_table.py` | — |
| Page Size 对齐逻辑 | `vllm/platforms/interface.py` | — |
| CacheConfig | `vllm/config/cache.py` | — |
| Hybrid KV Cache 设计文档 | `docs/design/hybrid_kv_cache_manager.md` | — |
