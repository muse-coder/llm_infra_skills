# vLLM Qwen3.5：Paged KV、NHD/HND、Prefill、Decode 与 Prefix Cache

> 范围：当前本地 vLLM 与 FlashInfer 源码，重点讨论 Qwen3.5 hybrid
> attention、NVIDIA GPU、FlashAttention 与 FlashInfer/TRTLLM-Gen。这里的
> `prefill` 指 causal context attention；`prefix cache` 指跨请求复用已经计算好的
> full-attention KV page。

## 目录

1. [核心结论](#1-核心结论)
2. [需要分开的四个概念](#2-需要分开的四个概念)
3. [Paged KV 是怎样存储和寻址的](#3-paged-kv-是怎样存储和寻址的)
4. [NHD 与 HND 的意义](#4-nhd-与-hnd-的意义)
5. [vLLM 中 K/V 的实际打包方式](#5-vllm-中-kv-的实际打包方式)
6. [Prefix cache 下的 causal prefill](#6-prefix-cache-下的-causal-prefill)
7. [Decode 如何读取同一份 cache](#7-decode-如何读取同一份-cache)
8. [Continuous batching 如何混合 prefill 与 decode](#8-continuous-batching-如何混合-prefill-与-decode)
9. [FlashAttention 路径](#9-flashattention-路径)
10. [FlashInfer/TRTLLM-Gen 路径](#10-flashinfertrtllm-gen-路径)
11. [Qwen3.5 hybrid 的 block/page 对齐](#11-qwen35-hybrid-的-blockpage-对齐)
12. [旧 PagedAttention V1/V2 与当前实现的关系](#12-旧-pagedattention-v1v2-与当前实现的关系)
13. [源码索引](#13-源码索引)

---

## 1. 核心结论

1. **Prefix cache 中 full-attention KV 仍然是 Paged KV。** Prefix cache 不创建
   另一份连续 KV，只让新请求的 block table 引用已有的物理 KV pages。
2. **Prefill 和 decode 共享同一份 KV cache，也共享同一种物理 layout。** 不会
   prefill 用 NHD、decode 再把整份 cache 转成 HND。
3. **NHD/HND 只决定 page 内 token 与 head 的物理顺序。** 它们不改变 attention
   数学、prefix-cache hash、block table 语义或 cache 容量。
4. **FlashAttention 通常使用 NHD；SM100 的 FlashInfer/TRTLLM-Gen 路径强制 HND。**
5. **FA 的 prefill/decode 共用一个 varlen API，但内部会按 shape 选择 kernel
   specialization。TRTLLM-Gen 明确使用 context 与 decode 两套 kernel API。**
6. **Qwen3.5 的 full-attention 使用 Paged KV；GDN/linear-attention 保存 recurrent
   state，不是传统 K/V。Hybrid cache manager 负责协调两类 page 的边界。**

---

## 2. 需要分开的四个概念

以下概念相互关联，但不是同一件事：

```text
阶段                 KV 寻址方式             page 内 layout       kernel
prefill/decode   ×   contiguous/paged   ×   NHD/HND         ×   FA/TRTLLM/Triton
```

- `prefill/decode`：描述当前 Q 的工作负载。
- `Paged KV`：描述逻辑 token 如何映射到物理 KV page。
- `NHD/HND`：描述单个物理 page 内的维度顺序。
- `FA/TRTLLM-Gen`：描述执行 attention 的 kernel family。

因此，“decode 使用 PagedAttention”更准确的说法是：decode kernel 直接支持
page table 并读取 paged KV；它不一定调用一个名字恰好叫 `PagedAttention` 的 kernel。

---

## 3. Paged KV 是怎样存储和寻址的

设：

```text
P = num_pages
N = page_size / block_size（一个 page 包含的 token 数）
H = num_kv_heads_local
D = head_dim
```

每个请求保存一张逻辑 page 到物理 page 的映射：

```text
block_tables[request, logical_page] = physical_page
```

逻辑 token `t` 的寻址过程：

```text
logical_page = t // N
page_offset  = t % N
physical_page = block_tables[request, logical_page]
```

然后 kernel 从：

```text
K[physical_page, ..., page_offset, ...]
V[physical_page, ..., page_offset, ...]
```

读取对应数据。`seq_lens`/`seqused_k` 给出每个请求的有效 KV token 数，最后一个
page 中未使用的位置不会参与 attention。

### Prefix cache 如何复用 page

```text
request A block table: [P3, P8, P12, A0, A1]
request B block table: [P3, P8, P12, B0]
                         └─ shared prefix ─┘
```

两个请求只共享物理 page ID；不会复制、拼接或 gather 成连续 KV tensor。Runtime 负责
hash、引用计数、释放和必要的 copy-on-write，attention kernel 只消费 block table。

---

## 4. NHD 与 HND 的意义

### 4.1 NHD：token-major

```text
[P, N, H, D]
page → token → head → dim
```

`N=3, H=2` 时的内存顺序：

```text
t0h0, t0h1, t1h0, t1h1, t2h0, t2h1
```

特点：

- 同一个 token 的所有 KV heads 相邻；
- 与 `[num_tokens, heads, D]` 的 token-major Q/K/V activation 一致；
- 适合按 token 组织的 prefill、varlen batching 和 cache 写入；
- FlashAttention 的 paged KV 路径通常使用这种布局。

### 4.2 HND：head-major

```text
[P, H, N, D]
page → head → token → dim
```

`N=3, H=2` 时的内存顺序：

```text
h0t0, h0t1, h0t2, h1t0, h1t1, h1t2
```

特点：

- 同一个 KV head 的历史 tokens 相邻；
- decode 对固定 KV head 扫描长历史时更自然；
- GQA/MQA 中多个 query heads 可复用同一个 KV-head tile；
- 适合 Blackwell TRTLLM-Gen 的 TMA 搬运方式。

### 4.3 为什么不能每个阶段各用一种

Continuous batching 中，同一步经常同时存在 prefill 与 decode。如果阶段之间切换
layout，需要反复执行：

```text
[P, N, H, D] → transpose + copy → [P, H, N, D]
```

长上下文下复制整份 KV 的成本很高。因此 vLLM 在 backend 初始化时确定全局 layout，
之后 prefill、decode、prefix-cache hit 都使用它。

---

## 5. vLLM 中 K/V 的实际打包方式

普通 BF16/FP16 cache 中，vLLM 将 K/V 打包在最后一维，而不是维护两块完全独立的
连续 allocation。

### NHD physical storage

```text
[P, N, H, 2D]
           K | V
```

### HND physical storage

```text
[P, H, N, 2D]
           K | V
```

每个 `(page, token, head)` 或 `(page, head, token)` 的内容是：

```text
[K0 ... K(D-1), V0 ... V(D-1)]
```

进入 backend 后通过 `split(D, dim=-1)` 得到两个零拷贝 view：

```text
K view: [..., D]
V view: [..., D]
```

以 HND contiguous combined cache 为例，K/V view 的 stride 是：

```text
stride(page)  = H * N * 2D
stride(head)  = N * 2D
stride(token) = 2D
stride(dim)   = 1
```

所以它在语义上是 `[P,H,N,D]`，但固定 head 下相邻 K token 的 stride 是 `2D`，
因为中间穿插了对应 token 的 V。TRTLLM-Gen 从 tensor 读取真实 stride，只强制最后
的 `D` stride 为 1，因此可以直接消费该 view。

---

## 6. Prefix cache 下的 causal prefill

命中 prefix cache 后，本轮 Q 只包含未命中的 suffix，而 K/V 逻辑序列包含：

```text
cached prefix + current suffix
```

例如：

```text
cached prefix = 512 tokens
current chunk = 256 tokens
page_size     = 128 tokens

q_len   = 256
seq_len = 768
block_table = [cached P0, P1, P2, P3, new P4, P5]
```

本轮流程：

```text
1. Runtime 找到 prefix-cache hit pages
2. 为 suffix 分配新的物理 pages
3. 计算当前 chunk 的 Q/K/V
4. 根据 slot_mapping 把新 K/V scatter 写入同一份 paged cache
5. causal prefill kernel 读取 cached pages + new pages
6. 只输出 suffix Q 对应的 O
```

Causal mask 使用右下角对齐。第 `i` 个 suffix query 可以看到：

```text
全部 cached prefix
+
当前 suffix 中位置 <= i 的 K/V
```

Prefix cache 不改变 layout：

```text
FLASH_ATTN backend  → 命中 page 与新 page 都是 NHD
TRTLLM-Gen backend  → 命中 page 与新 page 都是 HND
```

---

## 7. Decode 如何读取同一份 cache

Prefill 完成后不做 layout conversion。下一步 decode 直接复用原 block table，并为新
token 分配 slot：

```text
prefill/context kernel
       │ 写入/读取同一份 paged KV
       ▼
decode kernel
```

Decode 通常满足：

```text
q_len = 1
kv_len = 完整历史长度 + 当前 token
```

当前 token 的 K/V 在 attention 前已经通过 `slot_mapping` 写入 cache，因此 decode
kernel 只需使用 `block_tables + seq_lens` 扫描逻辑历史。

Speculative decode/MTP 可令每请求 `q_len > 1`。TRTLLM-Gen 通过
`q_len_per_req` 或 `cum_seq_lens_q` 表示；某些 DCP 条件下会转入 context/prefill
路径以保证 causal position 正确。

---

## 8. Continuous batching 如何混合 prefill 与 decode

一个 scheduler step 可以同时包含：

```text
[decode tokens | prefill/chunked-prefill tokens]
```

公共部分：

```text
统一分配 blocks/slots
      ↓
统一计算 Q/K/V
      ↓
统一将新 K/V scatter 到 paged cache
      ↓
backend attention dispatch
```

### FlashAttention

通常使用一次 mixed varlen 调用。`query_start_loc` 表达每个请求不同的 `q_len`，
`seq_lens` 表达各自完整 KV 长度。

```text
mixed batch → flash_attn_varlen_func → shape-dependent kernel specialization
```

### FlashInfer/TRTLLM-Gen

vLLM 在同一 flattened batch 中切出两个 slice：

```text
decode slice  → trtllm_batch_decode_with_kv_cache
prefill slice → trtllm_batch_context_with_kv_cache
```

两个 kernel 写回同一个 output tensor 的不同区域，读取同一份 HND KV cache。

---

## 9. FlashAttention 路径

FlashAttention backend 的 page/block token 数必须是 16 的倍数。逻辑 cache shape 为：

```text
[num_pages, num_kv_heads, page_size, 2 * head_dim]
```

默认 NHD physical stride 对应：

```text
[num_pages, page_size, num_kv_heads, 2 * head_dim]
```

Prefill 与 decode 都进入：

```python
flash_attn_varlen_func(
    q=query,
    k=key_cache,
    v=value_cache,
    seqused_k=seq_lens,
    block_table=block_table,
    max_seqlen_q=max_query_len,
    max_seqlen_k=max_seq_len,
    fa_version=...,
)
```

所以二者共享 Python/C++ API 与 cache layout，但 `q_len=1` 的 decode 和长 Q prefill
可能选择不同 CuTeDSL schedule、tile 或 split-K specialization。

### Qwen3.5 的 FA4 注意点

当前本地 vLLM 的标准 Qwen3.5 默认 `head_dim=256`。在 SM100 上，当前 selector 对
FA4 的 `head_size > 128`（192 除外）组合回退到 FA2。因此：

```text
SM100 + Qwen3.5 head_dim=256 + FLASH_ATTN
→ 当前 vLLM 通常解析为 FA2，而不是 FA4
```

该结论属于当前源码版本约束，后续 FA4 kernel 能力变化后需要重新核对 selector。

---

## 10. FlashInfer/TRTLLM-Gen 路径

SM100 上 FlashInfer backend 要求 HND。vLLM 的 combined storage 是：

```text
[P, H, N, 2D]
```

拆成传给 FlashInfer 的两个 view：

```text
K: [P, H, N, D]
V: [P, H, N, D]
```

### Prefill/context API

```python
trtllm_batch_context_with_kv_cache(
    query=prefill_query,
    kv_cache=(k_cache, v_cache),
    block_tables=block_tables,
    seq_lens=seq_lens,
    cum_seq_lens_q=cum_seq_lens_q,
    ...,
)
```

### Decode API

```python
trtllm_batch_decode_with_kv_cache(
    query=decode_query,
    kv_cache=(k_cache, v_cache),
    block_tables=block_tables,
    seq_lens=seq_lens,
    kv_layout="HND",
    backend="trtllm-gen",
    ...,
)
```

因此 TRTLLM-Gen 的 context 与 decode 是不同 kernel family，但不使用不同 cache。

### TRTLLM-Gen 的真实 stride 支持

FlashInfer Python API 将输入规范化为 HND-shaped view。底层 launcher 读取：

```cpp
page_size = key_cache.size(-2);
num_kv_heads = key_cache.size(-3);
kv_stride_keys_values = key_cache.stride(-2);
kv_stride_heads = key_cache.stride(-3);
kv_stride_batch = key_cache.stride(0);
```

随后把 stride 写入 TRTLLM-Gen runner/TMA descriptor。最后的 `head_dim` 必须 stride
为 1；head 与 page stride 可以来自实际 tensor，因此 vLLM 的 `[K|V]` packed view
无需重新复制成独立 contiguous K/V。

### Block table 格式

vLLM 默认：

```text
uses_shared_paged_kv_idx=True
block_tables: [batch, max_pages_per_seq]
```

K/V 共用一个 physical page ID。FlashInfer 也支持原生 TRT-LLM 风格：

```text
uses_shared_paged_kv_idx=False
block_tables: [batch, 2, max_pages_per_seq]
```

此时 K/V 可以分别使用 page index；vLLM 常规路径不使用这种形式。

### NHD 输入的例外

FlashInfer API 允许传 NHD。普通 BF16/FP16 可通过 transpose view 形成 HND-shaped
view，并把实际 stride 交给 kernel。NVFP4 的 NHD 数据及 scale tensor 会执行
`.contiguous()` 转为 HND，产生真实 copy，因此 vLLM 在 SM100 直接选择 HND。

---

## 11. Qwen3.5 hybrid 的 block/page 对齐

Qwen3.5 默认按 `layer_types` 交替使用：

```text
linear_attention → Gated DeltaNet recurrent state
full_attention   → Paged K/V
```

Full-attention 每 token 的 cache 字节数为：

```text
attention_bytes_per_token
  = (K head_dim + V head_dim) * num_kv_heads_local * dtype_bytes
  = 2 * num_kv_heads_local * head_dim * dtype_bytes
```

一页的实际字节数：

```text
attention_page_bytes = block_size * attention_bytes_per_token
```

GDN 需要保存 conv state 与 temporal state。vLLM 会把 attention block size 放大到
其 page 至少能容纳一份 GDN state，并将 GDN page padding 到同样字节数：

```text
attention_page_bytes >= gdn_state_bytes
mamba_page_size_padded = attention_page_bytes
```

因此 Qwen3.5 的实际 block size 可能明显大于默认 16，且随模型、TP、dtype、GDN
heads 与 speculative tokens 改变。`block_size` 是 cache page 的 token 粒度，不是
attention kernel 的 MMA tile size。

Prefix caching 下 Qwen3.5 当前使用 `mamba_cache_mode=align`，让 full-attention KV
与 GDN state 的可复用边界一致。最终 prefix hit 是两类 cache 都支持的最长公共前缀。

---

## 12. 旧 PagedAttention V1/V2 与当前实现的关系

vLLM 早期 NVIDIA CUDA decode 使用独立的：

```text
paged_attention_v1
paged_attention_v2
```

- V1：一个 CTA 处理一个 `(sequence, query_head)` 并扫描全部历史 KV。
- V2：将长 KV 分成 partitions，先生成 partial O/max/sum，再用 reduction kernel 合并。
- Prefill 当时也不是 V1/V2，而通常走单独的 Triton `context_attention_fwd`。

当前 NVIDIA 主路径通常使用 FA、FlashInfer/TRTLLM-Gen 或 Triton unified attention。
它们将 page-table 支持集成进各自 kernel，因此不再需要调用旧名字的 CUDA
PagedAttention V1/V2，但 Paged KV 的核心设计没有改变。

历史实现可通过 vLLM git 查看：

```bash
git show 928de46888:csrc/attention/attention_kernels.cu
git show 6f9d81d03b^:vllm/attention/ops/paged_attn.py
```

---

## 13. 源码索引

### vLLM

- Qwen3.5 full/linear layer 构造：
  [`qwen3_5.py`](../../vllm/vllm/model_executor/models/qwen3_5.py)
- Qwen3.5 默认 head/GDN 配置：
  [`qwen3_5.py`](../../vllm/vllm/transformers_utils/configs/qwen3_5.py)
- FlashAttention KV shape、stride 与 forward：
  [`flash_attn.py`](../../vllm/vllm/v1/attention/backends/flash_attn.py)
- FlashInfer/TRTLLM metadata、layout 与 forward：
  [`flashinfer.py`](../../vllm/vllm/v1/attention/backends/flashinfer.py)
- Attention cache spec：
  [`attention.py`](../../vllm/vllm/model_executor/layers/attention/attention.py)
- Hybrid block/page 对齐：
  [`interface.py`](../../vllm/vllm/platforms/interface.py)
- Hybrid prefix-cache hit 协调：
  [`kv_cache_coordinator.py`](../../vllm/vllm/v1/core/kv_cache_coordinator.py)
- GDN state shape：
  [`mamba_utils.py`](../../vllm/vllm/model_executor/layers/mamba/mamba_utils.py)
- 当前 Triton paged attention：
  [`chunked_prefill_paged_decode.py`](../../vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py)

### FlashInfer

- TRTLLM decode API 与 layout 文档：
  [`decode.py`](../../kernels/flashinfer/flashinfer/decode.py)
- TRTLLM runner 参数与实际 stride 提取：
  [`trtllm_fmha_kernel_launcher.cu`](../../kernels/flashinfer/csrc/trtllm_fmha_kernel_launcher.cu)
- HND TMA shape/stride：
  [`kernelParams.h`](../../kernels/flashinfer/include/flashinfer/trtllm/fmha/kernelParams.h)

---

## 最终心智模型

```text
                    一个 backend，全生命周期一种 KV layout

FLASH_ATTN                         FlashInfer/TRTLLM-Gen on SM100
通常 NHD [P,N,H,2D]               HND [P,H,N,2D]
       │                                  │
       ├─ prefix-cache causal prefill      ├─ context kernel
       ├─ chunked prefill                  ├─ decode kernel
       └─ decode                           └─ spec decode
       │                                  │
       └────── 都通过 block table 读取 Paged KV ──────┘

Prefix cache：决定复用哪些 physical pages
NHD/HND：决定 page 内 token/head 的物理顺序
Prefill/decode：决定 Q workload 与 kernel 调度
```
