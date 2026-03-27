# ROCm NonAsm Prefill 精度问题 Debug 记录

> **Commit**: `692872a5d40232885206845e9ec228169ca26281`
> **标题**: `refactor: [rocm] replace flash_attn_varlen_func with mha_batch_prefill_func`
> **作者**: moudi.mou
> **日期**: 2026-03-27
> **涉及文件**:
> - `rtp_llm/cpp/kernels/unfused_attention_kernels.cu`
> - `rtp_llm/models_py/bindings/rocm/FusedRopeKVCacheOp.cc`
> - `rtp_llm/models_py/modules/factory/attention/rocm_impl/aiter.py`

---

## 一、问题背景

在 ROCm 平台上，当 `asm_pa=0`（即使用 NonAsm/V1 路径）时，模型推理输出乱码，精度完全不对。ASM 路径（`asm_pa=1`）正常。

**核心目标**：让 NonAsm 版本的 `invokeAddFusedQKVBiasTransposePrefillV1` 调用后，结果写到 `q_output` 中，和 ASM 保持一致，并将 prefill attention 从 `flash_attn_varlen_func` 切换到 `mha_batch_prefill_func`（paged KV cache attention）。

---

## 二、Debug 过程

### 2.1 问题一：RoPE 位置编码错误（padding_offset 缺失）

**现象**：NonAsm 路径的 Q 输出值与 ASM 路径完全不同。

**排查过程**：
1. 对比 ASM 和 V1 两个 kernel 的调用参数，发现 V1 路径传入的 `padding_offset` 为 `nullptr`
2. 在 `FusedRopeKVCacheOp.cc` 中，ASM 路径传了 `params->padding_offset.data_ptr<int>()`，而 V1 路径传了 `nullptr`

**根因**：`padding_offset` 用于将 packed token 索引映射到 padded 序列位置，是 RoPE 位置编码计算的关键输入。缺少它会导致 RoPE 计算出错误的位置编码，进而导致 Q/K 值完全错误。

**修复**：在 V1 路径也传入 `params->padding_offset.data_ptr<int>()`：

```cpp
// 修复前（V1 路径）
nullptr,  // padding_offset - 缺失！

// 修复后
params->padding_offset.data_ptr<int>(),
```

### 2.2 问题二：Q 输出 layout 不一致

**现象**：修复 padding_offset 后，Q 值正确了，但 attention 输出仍然不对。

**排查过程**：
1. 对比 ASM 和 V1 的 `q_output` shape：
   - ASM（`use_paged_fmha=true`）：`[token_num, head, dim]`（packed-token layout）
   - V1（`use_paged_fmha=false`）：`[batch, head, seq, dim]`（padded layout）
2. 下游的 `mha_batch_prefill` 期望的 Q 输入是 packed-token layout `[token_num, head, dim]`

**根因**：V1 路径的 `use_paged_fmha` 参数为 `false`，导致 kernel 将 Q 写入 padded layout，与下游 attention 期望的 packed-token layout 不匹配。

**修复**：V1 路径也强制设置 `use_paged_fmha=true`：

```cpp
// 修复前
use_paged_fmha,   // use_paged_fmha - 可能为 false

// 修复后
true,       // use_paged_fmha=true: V1 writes packed-token layout [token_num, head, dim]
```

同时 `q_output` 的分配也统一为 packed-token layout：

```cpp
// 修复前：根据 use_paged_fmha 分配不同 shape
torch::Tensor q_output = use_paged_fmha ?
    torch::zeros({q_output_token_num, local_head_num, size_per_head}, ...) :
    torch::zeros({batch_size, local_head_num, seq_len, size_per_head}, ...);

// 修复后：统一为 packed-token layout
torch::Tensor q_output = torch::zeros(
    {q_output_token_num, local_head_num, size_per_head}, ...);
```

### 2.3 问题三：KV Cache layout 不匹配（核心问题）

**现象**：Q 输出完全一致后，attention 结果仍然错误。

**排查过程**：
1. 深入分析 CUDA kernel 中 KV cache 的写入方式：
   - **ASM kernel** 使用 `getKVLocalIdx` → 写入**线性 layout** `[head, token, dim]`
   - **V1/NonAsm kernel** 使用 `getKLocalIdx` / `getVLocalIdx` → 写入**vectorized layout**
2. Python 端的 `_reshape_kv_cache_vectorized` 方法假设 KV cache 是线性格式，对其做 reshape 转换为 5D vectorized 格式
3. 对于 NonAsm 路径，KV cache 已经是 vectorized 格式，再做一次 reshape 就变成了**双重 vectorize**，数据完全错乱

**关键代码分析**：

```cpp
// ASM kernel 的 KV cache 写入（线性 layout）
const int inKVBlockIdx = kv_block_array.getKVLocalIdx(
    dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);

// V1/NonAsm kernel 的 KV cache 写入（vectorized layout）
const int inKBlockIdx = kv_block_array.getKLocalIdx<KvCacheDataType::BASE>(
    dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
const int inVBlockIdx = kv_block_array.getVLocalIdx(
    dst_kv_seq_idx, head_idx, size_per_head, tidx * vec_size + vec_i);
```

**Vectorized Layout 详解**：

`mha_batch_prefill` 期望的 KV cache 5D layout：
- **K**: `[num_blocks, num_kv_heads, head_dim/vs, page_size, vs]`
- **V**: `[num_blocks, num_kv_heads, page_size/vs, head_dim, vs]`

其中 `vs = 16 / element_size`（BF16 时 vs=8）。

**根因**：ASM 和 V1 kernel 写入 KV cache 的 layout 不同：
- ASM → 线性 `[head, token, dim]`，需要 Python 端 reshape 成 vectorized 5D
- V1 → 已经是 vectorized，Python 端不应再做 reshape

**修复**：在 `AiterPrefillAttnOp` 中添加 `v1_kv_layout` 标志，根据 kernel 类型走不同的 reshape 分支：

```python
class AiterPrefillAttnOp:
    def __init__(self, attn_configs: AttentionConfigs, v1_kv_layout: bool = False):
        self.v1_kv_layout = v1_kv_layout

    def _reshape_kv_cache_vectorized(self, kv_cache_base):
        flat = kv_cache_base[:, :expected_elems].reshape(block_num, 2, hk, ps * hd)

        # K: V1 kernel 通过 getKLocalIdx<BASE> 写入 vectorized [hd//vs, ps, vs]
        # 直接 view 即可
        k_cache = flat[:, 0, :, :].view(block_num, hk, hd // vs, ps, vs)

        if self.v1_kv_layout:
            # V1 kernel 的 V 通过 getVLocalIdx（非模板版本）写入线性 [hd, ps]
            # 需要 permute: [hd, ps] → [hd, ps//vs, vs] → [ps//vs, hd, vs]
            v_linear = flat[:, 1, :, :].view(block_num, hk, hd, ps)
            v_cache = (
                v_linear.reshape(block_num, hk, hd, ps // vs, vs)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
        else:
            # ASM kernel 的 V 通过 getVLocalIdx<BASE> 写入 vectorized [ps//vs, hd, vs]
            v_cache = flat[:, 1, :, :].view(block_num, hk, ps // vs, hd, vs)

        return k_cache, v_cache
```

**注意**：这里有一个微妙的差异：
- `getKLocalIdx<BASE>` 和 `getVLocalIdx<BASE>`（模板版本）都写入 vectorized layout
- `getVLocalIdx`（非模板版本，V1 使用）写入的是**线性 layout** `[hd, ps]`
- 所以 V1 的 K 是 vectorized 的，但 V 是线性的，需要额外的 permute

---

## 三、C++ 端的简化

### 3.1 删除 prefix prompt 加载逻辑

修复前，C++ 端对非 paged_fmha 路径有一段 `invokeLoadPrefixKVCacheAiter` / `invokeLoadPrefixKVCacheAiterV1` 的调用。由于现在统一走 paged KV cache attention，prefix prompt 的 KV 已经在 cache 中，不需要单独加载，这段逻辑被删除。

### 3.2 统一 store 标志

```cpp
// 修复前：根据 use_paged_fmha 决定
bool store_qkv = !use_paged_fmha;
bool store_kv  = !use_paged_fmha;

// 修复后：统一不存储（KV 直接写入 cache）
bool store_qkv = false;
bool store_kv  = false;
```

### 3.3 删除 GatherSequences 后处理

修复前，非 paged_fmha 路径需要调用 `invokeGatherSequencesCombined` 将 padded 的 Q/K/V 重新排列为 `[head, total_tokens, dim]` 格式。统一走 paged 路径后，这段逻辑不再需要。

---

## 四、Python 端的重构

### 4.1 从 `flash_attn_varlen_func` 切换到 `mha_batch_prefill_func`

修复前，prefill attention 使用 `aiter.flash_attn_varlen_func`，需要完整的 Q/K/V 线性 tensor。修复后，统一使用 `aiter.mha_batch_prefill_func`，直接从 paged KV cache 读取。

```python
# 修复前
res = aiter.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)

# 修复后
k_cache, v_cache = self._reshape_kv_cache_vectorized(kv_cache.kv_cache_base)
res = aiter.mha_batch_prefill_func(
    q_tensor, k_cache, v_cache, ...,
    LAYOUT="VECTORIZED_LAYOUT",
)
```

### 4.2 保留 `flash_attn_varlen_func` 作为 fallback

对于没有 KV cache 的场景（如 encoder-only 模型 BERT），仍然使用 `flash_attn_varlen_func`：

```python
def forward(self, qkv, kv_cache, fmha_params):
    if kv_cache is None:
        return self._forward_varlen(qkv, fmha_params)  # fallback
    # 正常路径：mha_batch_prefill from paged KV cache
    ...
```

---

## 五、关键技术概念

### 5.1 KV Cache Layout 对照表

| 写入函数 | 使用者 | K layout | V layout |
|---------|--------|----------|----------|
| `getKVLocalIdx` | ASM kernel | 线性 `[token, dim]` | 线性 `[token, dim]` |
| `getKLocalIdx<BASE>` | V1 kernel | vectorized `[dim/vs, token, vs]` | - |
| `getVLocalIdx<BASE>` (模板) | ASM kernel | - | vectorized `[token/vs, dim, vs]` |
| `getVLocalIdx` (非模板) | V1 kernel | - | 线性 `[dim, token]` |

### 5.2 mha_batch_prefill 期望的 5D layout

- **K**: `[num_blocks, num_kv_heads, head_dim/vs, page_size, vs]`
- **V**: `[num_blocks, num_kv_heads, page_size/vs, head_dim, vs]`
- `vs = 16 / element_size`（BF16: vs=8, FP16: vs=8）

### 5.3 padding_offset 的作用

`padding_offset` 将 packed token 索引映射到 padded 序列中的实际位置。在 varlen packed-token 模式下，多个不等长序列被紧密拼接（无 padding），但 RoPE 需要知道每个 token 在其所属序列中的真实位置。

```
packed tokens:  [A0, A1, A2, B0, B1]  (seq A len=3, seq B len=2, max_len=3)
padded tokens:  [A0, A1, A2, B0, B1, PAD]
padding_offset: [0,  0,  0,  1,  1]  (每个 packed token 需要加的偏移量)
```

### 5.4 reuse_cache 机制

**匹配逻辑**（`FullKVCacheGroup::match`）：
- 逐 block 匹配 cache_key，遇到第一个不匹配就停止
- `reuse_length = reuse_blocks * seq_size_per_block`
- 匹配时会丢弃最后一个 cache key（partial block），所以最小非零 reuse_length = `seq_size_per_block`

```cpp
MatchResult FullKVCacheGroup::match(const CacheKeysType& cache_keys) {
    MatchResult final_result;
    for (const auto& cache_key : cache_keys) {
        auto result = block_cache_->match(cache_key, group_id_);
        if (isNullBlockIdx(result.matched_index)) break;
        final_result.reuse_blocks++;
        final_result.block_indices.push_back(result.matched_index);
    }
    final_result.reuse_length = final_result.reuse_blocks * seqSizePerBlock();
    return final_result;
}
```

---

## 六、总结

### 修复的三个层次

1. **C++ kernel 调用层** (`FusedRopeKVCacheOp.cc`)：
   - V1 路径补上 `padding_offset`
   - V1 路径强制 `use_paged_fmha=true`
   - 统一 `q_output` 为 packed-token layout
   - 删除不再需要的 prefix loading 和 GatherSequences

2. **Python attention 层** (`aiter.py`)：
   - 从 `flash_attn_varlen_func` 切换到 `mha_batch_prefill_func`
   - 新增 `_reshape_kv_cache_vectorized` 处理 KV cache 5D reshape
   - 根据 `v1_kv_layout` 标志区分 ASM/V1 的 V cache layout 差异

3. **CUDA kernel 层** (`unfused_attention_kernels.cu`)：
   - V1 kernel 的 `getVLocalIdx` 去掉模板参数（使用非模板版本）
   - 代码格式化调整

### 经验教训

- **不同 kernel 可能写入不同的内存 layout**：ASM 和 V1 kernel 虽然功能相同，但 KV cache 的写入 layout 完全不同，必须在 Python 端做对应的 reshape
- **padding_offset 是 packed-token 模式的必要参数**：缺少它会导致 RoPE 位置编码错误，表现为输出乱码
- **use_paged_fmha 影响 Q 的输出 layout**：这个标志不仅控制 KV cache 的写入方式，还决定了 Q 的输出格式（packed vs padded）
- **getVLocalIdx 的模板版本和非模板版本行为不同**：模板版本写入 vectorized layout，非模板版本写入线性 layout，这是一个容易忽略的细节

---

## 七、后续 Bug 修复记录

### 7.1 FP8 prefill 路径报错（flash_attn_varlen_fp8_pertensor_func 参数缺失）

**现象**：FP8 模型（如 `ptpc_qwen3_32b_fp8_amd_pymodel`）在 prefill 阶段 crash：
```
TypeError: flash_attn_varlen_fp8_pertensor_func() missing 3 required positional arguments:
  'cu_seqlens_k', 'max_seqlen_q', and 'max_seqlen_k'
```

**根因**：692872a commit 重构 FP8 路径时，移除了原来的 3 个 `None` descale 参数：

```python
# 修复前（692872a 之前的代码 —— 正确）
res = aiter.flash_attn_varlen_fp8_pertensor_func(
    q_tensor, k_tensor, v_tensor,
    None, None, None,         # ← q_descale, k_descale, v_descale
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    causal=self.is_causal,
)

# 692872a 重构后（错误 —— 缺少 3 个 descale 参数）
res = aiter.flash_attn_varlen_fp8_pertensor_func(
    query, key, value,
    cu_seqlens_q, cu_seqlens_k,       # ← 这些被当成了 descale 参数
    fmha_params.max_seqlen_q,         # ← 被当成 cu_seqlens_q
    fmha_params.max_seqlen_k,         # ← 被当成 cu_seqlens_k
    causal=self.is_causal,            # ← 剩下3个位置参数缺失
)
```

`flash_attn_varlen_fp8_pertensor_func` 的函数签名：
```python
def flash_attn_varlen_fp8_pertensor_func(
    q, k, v,
    q_descale, k_descale, v_descale,  # ← 必填位置参数！
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, ...
)
```

**修复方案**：删除单独的 FP8 `flash_attn_varlen_fp8_pertensor_func` 路径，统一走 `mha_batch_prefill_func`（paged KV cache attention），由 `mha_batch_prefill_func` 内部自动处理 FP8 KV cache。通过检测 `k_cache.dtype` 是否为 FP8 来传递 descale 参数。

### 7.2 KV Cache reshape 方法不一致导致精度问题

**现象**：非 FP8 模型也出现精度不对齐。

**根因**：692872a 的 `_reshape_kv_cache_vectorized` 方法与代码库中其他地方（decode attention、paged prefill、triton PA）使用的标准 `select(1, 0) + view` 模式不一致。其他地方的标准模式：

```python
# 标准模式（AiterPrefillAttnOpPaged、AiterDecodeAttnOpAsm、_run_triton_paged_attention 等统一使用）
key_cache = kv_cache_base.select(1, 0)   # [num_blocks, num_kv_heads, page_size, head_dim]
value_cache = kv_cache_base.select(1, 1)
x = 16 // key_cache.element_size()
kv_sizes = key_cache.shape
key_cache = key_cache.view(kv_sizes[0], kv_sizes[1], kv_sizes[3] // x, kv_sizes[2], x)
value_cache = value_cache.view(kv_sizes[0], kv_sizes[1], kv_sizes[2] // x, kv_sizes[3], x)
```

而 692872a 的方法是自己做 flat slice + reshape：
```python
# 692872a 的非标准方式
flat = kv_cache_base[:, :expected_elems].reshape(block_num, 2, hk, ps * hd)
k_cache = flat[:, 0, :, :].view(block_num, hk, hd // vs, ps, vs)
```

当 `kv_cache_base` 已经是 5D `[num_blocks, 2, hk, ps, hd]` 格式时（通过 `reshape_paged_kv_cache` 或 C++ 端已经 reshape），这个 flat 方式的 `[:, :expected_elems]` 切片会在 dim-1 上操作（dim-1 只有 2 个元素），导致行为不一致。

**修复方案**：将 `_reshape_kv_cache_vectorized` 替换为 `_reshape_kv_cache_to_5d`，使用与代码库一致的 `select + view` 标准模式。

### 7.3 Encoder-only 模型（BERT）kv_cache=None 崩溃

**现象**：BERT/Roberta 等 encoder-only 模型报错 `AttributeError: 'NoneType' object has no attribute 'kv_cache_base'`，后续修复后又报 `'NoneType' object has no attribute 'requires_grad'`。

**根因链条**（两层问题叠加）：

1. **Impl 层**：`AiterPrefillImplAsm.forward` 即使 `kv_cache=None`，仍然调用 `FusedRopeKVCachePrefillOpAsm.forward(qkv, kv_cache=None)`。该 fused rope+KV cache 算子在 `kv_cache=None` 时返回 `(Q_tensor, None, None)` —— K/V 数据丢失。

2. **AttnOp 层**：`AiterPrefillAttnOp.forward` 收到 `(Q, None, None)` 后，试图访问 `kv_cache.kv_cache_base`（第一版错误）或传递 `None` 给 `flash_attn_varlen_func`（第二版错误）。

**关键调试发现**：
- BERT 的 `need_rope_kv_cache` 实际上为 **True**（之前误以为 False）
- 通过添加 debug logging 发现 `qkv` 到达 `AiterPrefillAttnOp.forward` 时已经是 `tuple(Tensor[46, 12, 64], None, None)`

**修复方案**（两层保护）：

```python
# Impl 层：kv_cache=None 时跳过 rope+cache，直接传原始 QKV 给底层
class AiterPrefillImplAsm(FMHAImplBase):
    def forward(self, qkv, kv_cache):
        if kv_cache is None:
            return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)
        # 正常 rope + cache 路径...

# AttnOp 层：kv_cache=None 时走 flash_attn_varlen_func fallback
class AiterPrefillAttnOp:
    def forward(self, qkv, kv_cache, fmha_params):
        if kv_cache is None:
            return self._forward_varlen(qkv, fmha_params)
        # 正常 mha_batch_prefill 路径...
```

`_forward_varlen` 将原始 QKV tensor 拆分为 Q/K/V 后调用 `aiter.flash_attn_varlen_func`（恢复 692872a 之前的行为）。

### 7.4 关键知识点补充

#### FusedRopeKVCache 在 kv_cache=None 时的行为

`FusedRopeKVCachePrefillOpAsm` 是一个 fused 算子，将 RoPE + KV Cache 写入合并为一步：
- 输入：raw QKV tensor
- 输出：tuple `(Q_with_rope, K_in_cache, V_in_cache)` 或 `(Q_with_rope, None, None)`

当 `kv_cache=None` 时，算子只能完成 RoPE 部分，K/V 无处写入，因此返回 `(Q, None, None)`。下游必须感知这一点。

#### kv_cache_base 的两种形态

| 来源 | shape | 何时出现 |
|------|-------|---------|
| C++ 端直接分配 | `[num_blocks, 2, num_kv_heads, page_size, head_dim]` | decode attention、标准 prefill |
| hybrid cache 模式 | `[num_blocks, flat_stride]`（2D） | 需要调用 `reshape_paged_kv_cache` 转换为 5D |

`reshape_paged_kv_cache` 检查 `dim != 2` 则直接返回，所以对已经是 5D 的 cache 无影响。

#### descale 参数规范

`mha_batch_prefill_func` 和 `flash_attn_varlen_fp8_pertensor_func` 都支持 FP8 KV cache，但接口不同：
- `mha_batch_prefill_func`：`q_descale`, `k_descale`, `v_descale` 为 **keyword 可选参数**
- `flash_attn_varlen_fp8_pertensor_func`：`q_descale`, `k_descale`, `v_descale` 为 **positional 必填参数**

使用 `mha_batch_prefill_func` 可以统一 FP8 和非 FP8 路径，只需在检测到 FP8 dtype 时额外传入 descale 参数即可。
