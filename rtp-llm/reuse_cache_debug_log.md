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

在 ROCm 平台上，当 `USE_ASM_PA=0`（即使用 NonAsm/V1 路径）时，模型推理输出乱码，精度完全不对。ASM 路径（`USE_ASM_PA=1`）正常。

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

### 2.3 问题三：KV Cache V layout 不匹配（核心问题）

**现象**：Q 输出完全一致后，attention 结果仍然错误。进一步分析发现，`res` tensor 中 dim=0 的值在 ASM 和 NonAsm 之间完全一致，但 dim>=1 的所有列完全不同。

**排查过程**：
1. 深入分析 `kv_cache_utils.h` 中的索引函数：
   - `getKLocalIdx<BASE>` — 模板版本，写入 vectorized layout `[dim/vs, token, vs]`
   - `getVLocalIdx<BASE>` — 模板版本，写入 vectorized layout `[token/vs, dim, vs]`
   - `getVLocalIdx()` — **非模板版本**，写入线性 layout `[dim, token]`

2. V1 kernel 中 K 使用 `getKLocalIdx<BASE>`（vectorized），V 使用 `getVLocalIdx()`（**非模板，线性 layout**）

3. Python 端的 `_reshape_kv_cache_vectorized` 对 V 做的 `view(block_num, hk, ps // vs, hd, vs)` 假设物理数据是 vectorized 的 `[token/vs, dim, vs]`，但 V1 实际写的是线性 `[dim, token]`

**dim=0 一致、dim>=1 不一致的数学解释**：
- 线性 V layout 的物理位置：`index(h,d,t) = h*D*P + d*P + t`
- Vectorized V layout 的物理位置：`index(h,tg,d,ts) = h*D*P + tg*D*vs + d*vs + ts`
- 当 `d=0` 时：线性偏移 = `t`，vectorized 偏移 = `tg*vs + ts = t`，**两者相同**
- 当 `d>=1` 时：线性偏移 = `d*P + t`，vectorized 偏移 = `tg*D*vs + d*vs + ts`（其中 `D != P`），**两者不同**

这完美解释了为什么只有第一列是正确的。

### 2.4 问题四：方案选择——性能 vs 正确性

**第一次尝试（方案1）**：
- 修改 C++ V1 kernel：把 `getVLocalIdx()` 改为 `getVLocalIdx<BASE>()`，让 V1 也写 vectorized V layout
- **结果**：Prefill 正确了，但 **decode 阶段仍然乱码**

**Decode 乱码的根因**：
- Decode 使用 `paged_attention_atrex` kernel，它期望 V cache 是 **线性 layout** `[head, dim, token]`
- C++ 端的 `aiterPA.cc` 已经有 `v_shuffle = use_asm_pa_` 的分支来处理这个差异（给 `paged_attention_rocm` 用），但 `paged_attention_atrex` 没有 `v_shuffle` 参数，**硬编码期望线性 V layout**
- 把 V1 kernel 改成 vectorized V layout 后，decode 端的 `view(block, hk, hd, ps)` 对 vectorized 物理数据的重解释完全错误

**第二次尝试（方案1 + decode workaround）**：
- 在 `AiterDecodeAttnOpNonAsm.forward()` 中加入 `permute + contiguous` de-vectorize V cache
- **结果**：精度正确了，但 **性能严重下降**
- **原因**：`permute + contiguous` 在每个 decode step 都触发全量 V cache 的 GPU 数据拷贝。Decode 是逐 token 执行的，这个开销不可接受

**最终方案（方案2）——把开销从 decode 移到 prefill**：

核心洞察：Prefill 只执行一次，Decode 每个 token 都执行。Layout 转换应该放在 prefill 端。

1. **C++ 端**：撤回 V1 kernel 的 V 修改，恢复使用非模板 `getVLocalIdx()`（线性 V layout）
2. **Python prefill 端**（`_reshape_kv_cache_vectorized`）：当 `v1_kv_layout=True` 时，对 V cache 做 `permute + contiguous` 将线性 `[hd, ps]` 转为 vectorized `[ps//vs, hd, vs]`（只在 prefill 时执行一次）
3. **Python decode 端**：保持原始的 `view` 操作，不做任何额外转换（零开销）

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
)
```

### 4.2 V cache layout 转换（`_reshape_kv_cache_vectorized`）

根据 `v1_kv_layout` 标志区分 ASM/V1 的 V cache layout 差异：

```python
class AiterPrefillAttnOp:
    def __init__(self, attn_configs: AttentionConfigs, v1_kv_layout: bool = False):
        self.v1_kv_layout = v1_kv_layout

    def _reshape_kv_cache_vectorized(self, kv_cache_base):
        vs = 16 // kv_cache_base.element_size()
        flat = kv_cache_base[:, :expected_elems].reshape(block_num, 2, hk, ps * hd)

        # K: 两种 kernel 都使用 getKLocalIdx<BASE>，layout 一致
        k_cache = flat[:, 0, :, :].view(block_num, hk, hd // vs, ps, vs)

        if self.v1_kv_layout:
            # V1 kernel 用 getVLocalIdx（非模板）写入线性 [hd, ps]
            # 需要 permute 到 vectorized [ps//vs, hd, vs]
            v_linear = flat[:, 1, :, :].view(block_num, hk, hd, ps)
            v_cache = (
                v_linear.reshape(block_num, hk, hd, ps // vs, vs)
                .permute(0, 1, 3, 2, 4)    # [block, hk, ps//vs, hd, vs]
                .contiguous()
            )
        else:
            # ASM kernel 用 getVLocalIdx<BASE> 写入 vectorized [ps//vs, hd, vs]
            v_cache = flat[:, 1, :, :].view(block_num, hk, ps // vs, hd, vs)

        return k_cache, v_cache
```

### 4.3 保留 `flash_attn_varlen_func` 作为 fallback（BERT 类模型）

对于没有 KV cache 的场景（如 encoder-only 模型 BERT），仍然使用 `flash_attn_varlen_func`：

```python
def forward(self, qkv, kv_cache, fmha_params):
    # FP8 模型：走独立的 flash_attn_varlen_fp8_pertensor_func 路径
    if q_tensor.dtype == torch.float8_e4m3fnuz:
        return self._forward_fp8(qkv, fmha_params)

    # BERT 等无 KV cache 模型：走 flash_attn_varlen_func fallback
    if kv_cache is None:
        return self._forward_varlen(qkv, fmha_params)

    # 正常路径：mha_batch_prefill from paged KV cache
    ...
```

### 4.4 Impl 层对 `kv_cache=None` 的保护

```python
class AiterPrefillImplAsm(FMHAImplBase):
    def forward(self, qkv, kv_cache):
        # kv_cache=None 时（BERT 等），跳过 rope+cache，直接传原始 QKV
        if kv_cache is None:
            return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)

        # 正常 rope + cache 路径...
        if self.need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, ...)
        ...
```

---

## 五、关键技术概念

### 5.1 KV Cache Layout 对照表

| 写入函数 | 使用者 | K layout | V layout |
|---------|--------|----------|----------|
| `getKVLocalIdx` | ASM kernel（旧式） | 线性 `[token, dim]` | 线性 `[token, dim]` |
| `getKLocalIdx<BASE>` | ASM + V1 kernel | vectorized `[dim/vs, token, vs]` | - |
| `getVLocalIdx<BASE>` (模板) | ASM kernel | - | vectorized `[token/vs, dim, vs]` |
| `getVLocalIdx` (非模板) | V1 kernel | - | 线性 `[dim, token]` |

> **关键差异**：K 的模板版本 `getKLocalIdx<BASE>` ASM 和 V1 都使用，但 V 的 `getVLocalIdx` 有**模板版本和非模板版本两种**，它们的 layout 完全不同。V1 kernel 使用的是**非模板版本**。

### 5.2 mha_batch_prefill 期望的 5D layout

- **K**: `[num_blocks, num_kv_heads, head_dim/vs, page_size, vs]`
- **V**: `[num_blocks, num_kv_heads, page_size/vs, head_dim, vs]`
- `vs = 16 / element_size`（BF16: vs=8, FP16: vs=8）

### 5.3 paged_attention_atrex 期望的 layout（decode）

- **K**: `[num_blocks, num_kv_heads, head_dim/x, page_size, x]`（vectorized，与 mha_batch_prefill 相同）
- **V**: `[num_blocks, num_kv_heads, head_dim, page_size]`（**线性**，注意和 mha_batch_prefill 不同！）
- `x = 16 / element_size`

这就是为什么 V1 kernel 的 V 不能改成 vectorized 的根本原因：prefill 和 decode 对 V cache layout 的期望不一致。

### 5.4 paged_attention_rocm 的 `v_shuffle` 参数

C++ 端 `aiterPA.cc` 中的 `paged_attention_rocm`（用于 `max_seq_len > 16384` 或 FP8 场景）原生支持两种 V layout，通过 `v_shuffle` 参数控制：

```cpp
bool v_shuffle = use_asm_pa_;  // ASM: true (vectorized V), NonAsm: false (linear V)
if (use_asm_pa_) {
    // v_cache [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[2] / x, kv_sizes[3], x});
} else {
    // v_cache [num_blocks, num_kv_heads, head_size, kv_block_size]
    value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3], kv_sizes[2]});
}
```

而 `paged_attention_atrex` **没有** `v_shuffle` 参数，硬编码期望线性 V layout。

### 5.5 padding_offset 的作用

`padding_offset` 将 packed token 索引映射到 padded 序列中的实际位置。在 varlen packed-token 模式下，多个不等长序列被紧密拼接（无 padding），但 RoPE 需要知道每个 token 在其所属序列中的真实位置。

```
packed tokens:  [A0, A1, A2, B0, B1]  (seq A len=3, seq B len=2, max_len=3)
padded tokens:  [A0, A1, A2, B0, B1, PAD]
padding_offset: [0,  0,  0,  1,  1]  (每个 packed token 需要加的偏移量)
```

### 5.6 reuse_cache 机制

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

## 六、后续 Bug 修复记录

### 6.1 FP8 prefill 路径报错（flash_attn_varlen_fp8_pertensor_func 参数缺失）

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

**最终修复方案**：FP8 prefill 不走 `mha_batch_prefill_func`（因为 FP8 QKV 来自 `qkv_buf_fp8`，需要从中拆分出线性 Q/K/V），保持走 `flash_attn_varlen_fp8_pertensor_func`。在 `forward()` 入口处通过 `dtype` 检查提前分流：

```python
def forward(self, qkv, kv_cache, fmha_params):
    q_tensor = qkv[0] if isinstance(qkv, (tuple, list)) else qkv

    # FP8 路径：提前分流，不走 paged KV cache
    if q_tensor.dtype == torch.float8_e4m3fnuz:
        query, key, value = self._split_qkv_fp8(q_tensor)
        ...
        res = aiter.flash_attn_varlen_fp8_pertensor_func(
            query, key, value,
            cu_seqlens_q, cu_seqlens_k,
            fmha_params.max_seqlen_q, fmha_params.max_seqlen_k,
            causal=self.is_causal,
        )
        return res.reshape(...)

    # 非 FP8 正常路径...
```

> **注意**：这里 `flash_attn_varlen_fp8_pertensor_func` 调用没有传 descale 参数，是因为最新版 aiter 的接口中 descale 已变为可选参数（keyword argument）。如果使用的 aiter 版本中 descale 仍是必填参数，需要加上 `None, None, None`。

### 6.2 Encoder-only 模型（BERT）kv_cache=None 崩溃

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

### 6.3 Decode 阶段乱码（V cache layout 不匹配）

**现象**：Prefill 阶段（通过方案1 C++ 修改）输出正确。但 decode 阶段的输出全是乱码。以 Qwen 模型为例，prompt "人工智能的全称是什么？" 的 prefill 输出 "人工智能" 是正确的（这是 prefill 阶段产出的第一个 token），但后续 decode 阶段产出的 token 全部乱码（如 `green�kdirче至于UFACT.TryParse giác {{`）。

**方案演进**：

| 方案 | C++ V1 kernel V layout | Prefill 读取 | Decode 读取 | 结果 |
|------|----------------------|-------------|------------|------|
| 方案1 | `getVLocalIdx<BASE>` (vectorized) | `view` (零开销) | `view` → 数据错乱 | Prefill 正确，Decode 乱码 |
| 方案1 + decode workaround | vectorized | `view` | `permute+contiguous` (每 token) | 正确但性能差 |
| **方案2（最终）** | `getVLocalIdx()` (线性) | `permute+contiguous` (一次) | `view` (零开销) | **正确且性能好** |

**最终方案的完整改动**：

**C++ `unfused_attention_kernels.cu`**：所有 V1 kernel 中 V 的写入保持使用非模板 `getVLocalIdx()`（共 8 处：prefill FP8/BASE 各 2 处，decode FP8/BASE 各 2 处，prefix load 2 处）。ASM kernel 保持使用模板 `getVLocalIdx<BASE>`。

**Python `aiter.py`**：
- `AiterPrefillAttnOp.__init__` 增加 `v1_kv_layout` 参数
- `_reshape_kv_cache_vectorized` 中 `v1_kv_layout=True` 时做 V 的 `permute+contiguous`
- `AiterPrefillImplNonAsm` 传 `v1_kv_layout=True`
- `AiterDecodeAttnOpNonAsm.forward()` 保持原始 `view` 操作（无额外开销）

---

## 七、关键知识点补充

### 7.1 FusedRopeKVCache 在 kv_cache=None 时的行为

`FusedRopeKVCachePrefillOpAsm` 是一个 fused 算子，将 RoPE + KV Cache 写入合并为一步：
- 输入：raw QKV tensor
- 输出：tuple `(Q_with_rope, K_in_cache, V_in_cache)` 或 `(Q_with_rope, None, None)`

当 `kv_cache=None` 时，算子只能完成 RoPE 部分，K/V 无处写入，因此返回 `(Q, None, None)`。下游必须感知这一点。

### 7.2 kv_cache_base 的两种形态

| 来源 | shape | 何时出现 |
|------|-------|---------|
| C++ 端直接分配 | `[num_blocks, 2, num_kv_heads, page_size, head_dim]` | decode attention、标准 prefill |
| hybrid cache 模式 | `[num_blocks, flat_stride]`（2D） | 需要调用 `reshape_paged_kv_cache` 转换为 5D |

`reshape_paged_kv_cache` 检查 `dim != 2` 则直接返回，所以对已经是 5D 的 cache 无影响。

### 7.3 descale 参数规范

`mha_batch_prefill_func` 和 `flash_attn_varlen_fp8_pertensor_func` 都支持 FP8 KV cache，但接口不同：
- `mha_batch_prefill_func`：`q_descale`, `k_descale`, `v_descale` 为 **keyword 可选参数**
- `flash_attn_varlen_fp8_pertensor_func`：`q_descale`, `k_descale`, `v_descale` 可能是 **positional 必填参数**（取决于 aiter 版本）

使用 `mha_batch_prefill_func` 可以统一 FP8 和非 FP8 路径，只需在检测到 FP8 dtype 时额外传入 descale 参数即可。

---

## 八、开发注意事项

### 8.1 如何绕过 FP8 路径

FP8 模型（如 `ptpc_qwen3_32b_fp8_amd_pymodel`）的 QKV 数据来自 `qkv_buf_fp8`，dtype 为 `torch.float8_e4m3fnuz`。在 `AiterPrefillAttnOp.forward()` 入口处通过 dtype 检查提前分流：

```python
if q_tensor.dtype == torch.float8_e4m3fnuz:
    # FP8 路径：拆分 QKV 后走 flash_attn_varlen_fp8_pertensor_func
    # 不走 paged KV cache + mha_batch_prefill 路径
    query, key, value = self._split_qkv_fp8(q_tensor)
    ...
    return ...
```

**为什么 FP8 不走 `mha_batch_prefill_func`**：FP8 场景下 QKV 是从一个完整的 FP8 buffer 中拆分出来的，不需要从 paged KV cache 读取。直接使用 `flash_attn_varlen_fp8_pertensor_func` 更简单。

### 8.2 如何绕过 BERT 类模型

BERT 等 encoder-only 模型没有 KV cache（`kv_cache=None`），不需要做 paged attention。两层保护确保不会进入 paged KV cache 路径：

**第一层**（Impl 层）：`kv_cache is None` 时跳过 `FusedRopeKVCache` 算子（否则 K/V 会丢失），直接传原始 QKV 给 AttnOp。

**第二层**（AttnOp 层）：`kv_cache is None` 时走 `_forward_varlen` fallback，使用 `flash_attn_varlen_func` 处理原始 Q/K/V。

**注意**：BERT 的 `need_rope_kv_cache` 可能为 True（不能假设为 False），但即使 True，当 `kv_cache=None` 时也不应该调用 fused rope+KV cache 算子。

### 8.3 修改 `unfused_attention_kernels.cu` 的注意事项

1. **只改 V1 kernel**：函数名含 `_v1` 后缀（如 `add_fusedQKV_bias_transpose_prefill_kernel_v1`、`load_prefix_KVCache_kernel_aiter_v1`），不要改非 V1 的 ASM kernel
2. **V 和 K 要区分对待**：K 的 `getKLocalIdx<BASE>` 两种 kernel 都用，不需要改；V 的 `getVLocalIdx` 有模板/非模板两种版本，V1 必须用非模板版本
3. **FP8 和 BASE 都要改**：每个 kernel 中有 `if constexpr (std::is_same<Tcache, __nv_fp8_e4m3>::value)` 分支，FP8 和 BASE 分支的 V 都需要改
4. **prefix load kernel 也要改**：`load_prefix_KVCache_kernel_aiter_v1` 是**读取** V cache 的（不是写入），但它需要使用与写入 kernel 相同的索引函数才能正确读取
5. **必须重新编译**：`.cu` 文件的修改需要重新编译 HIP kernel，Python 端的修改不需要

### 8.4 调试技巧

1. **对比 ASM 和 NonAsm 的 tensor 输出**：在关键位置（如 `forward()` 入口、KV cache reshape 后、attention 输出后）打印 tensor 的前几个值，用 `USE_ASM_PA=0` 和 `USE_ASM_PA=1` 分别跑，对比差异
2. **dim=0 一致而 dim>=1 不一致**的 pattern 强烈暗示 V cache layout 不匹配（线性 vs vectorized 的偏移在 d=0 时恰好重合）
3. **Prefill 正确但 Decode 乱码**的 pattern 说明 KV cache 的写入/读取在不同阶段的 kernel 间不一致
4. **C++ 改了但效果没变**：检查是否重新编译了（`.cu` 文件需要编译才能生效）

---

## 九、总结

### 修改的三个层次

1. **C++ kernel 调用层** (`FusedRopeKVCacheOp.cc`)：
   - V1 路径补上 `padding_offset`
   - V1 路径强制 `use_paged_fmha=true`
   - 统一 `q_output` 为 packed-token layout
   - 删除不再需要的 prefix loading 和 GatherSequences

2. **Python attention 层** (`aiter.py`)：
   - 从 `flash_attn_varlen_func` 切换到 `mha_batch_prefill_func`
   - 新增 `_reshape_kv_cache_vectorized` 处理 KV cache 5D reshape
   - 根据 `v1_kv_layout` 标志区分 ASM/V1 的 V cache layout 差异
   - 保留 FP8 独立路径（`flash_attn_varlen_fp8_pertensor_func`）
   - 保留 BERT fallback 路径（`flash_attn_varlen_func`）

3. **CUDA kernel 层** (`unfused_attention_kernels.cu`)：
   - V1 kernel 的 `getVLocalIdx` 保持使用非模板版本（线性 V layout）
   - ASM kernel 的 `getVLocalIdx<BASE>` 保持使用模板版本（vectorized V layout）

### 方案选择的核心原则

**同一块 KV cache 被 prefill 和 decode 两个阶段共享读取**，但两个阶段的 attention kernel 对 V cache layout 的期望不同：
- `mha_batch_prefill`：期望 vectorized V `[token/vs, dim, vs]`
- `paged_attention_atrex`：期望线性 V `[dim, token]`

因此不管 V1 kernel 以哪种 layout 写入 V，总有一端需要做 layout 转换。**应该把转换开销放在 prefill 端**（只执行一次），而不是 decode 端（每个 token 都执行）。

### 经验教训

- **不同 kernel 可能写入不同的内存 layout**：ASM 和 V1 kernel 虽然功能相同，但 KV cache 的写入 layout 完全不同，必须在 Python 端做对应的 reshape
- **padding_offset 是 packed-token 模式的必要参数**：缺少它会导致 RoPE 位置编码错误，表现为输出乱码
- **use_paged_fmha 影响 Q 的输出 layout**：这个标志不仅控制 KV cache 的写入方式，还决定了 Q 的输出格式（packed vs padded）
- **getVLocalIdx 的模板版本和非模板版本行为不同**：模板版本写入 vectorized layout，非模板版本写入线性 layout，这是一个容易忽略的细节
- **性能 vs 正确性的权衡**：layout 转换一定要放在执行次数少的路径上（prefill 而非 decode）
- **FP8 和 BERT 是容易被忽略的边界场景**：任何 attention 路径重构都必须验证这两类模型不受影响
