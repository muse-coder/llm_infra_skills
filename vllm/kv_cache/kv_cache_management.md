# vLLM KV Cache 管理机制详解

> 本文档系统性地介绍 vLLM 的 KV Cache 管理架构、核心流程，以及 FP8 量化与 KV Cache 的结合原理。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [核心文件索引](#2-核心文件索引)
3. [核心数据结构](#3-核心数据结构)
4. [KV Cache 管理全流程](#4-kv-cache-管理全流程)
   - 4.1 [初始化阶段](#41-初始化阶段)
   - 4.2 [请求调度阶段](#42-请求调度阶段)
   - 4.3 [模型执行阶段](#43-模型执行阶段)
   - 4.4 [Block 释放阶段](#44-block-释放阶段)
5. [Prefix Caching 详解](#5-prefix-caching-详解)
6. [FP8 量化与 KV Cache 的结合](#6-fp8-量化与-kv-cache-的结合)
   - 6.1 [为什么要量化 KV Cache](#61-为什么要量化-kv-cache)
   - 6.2 [量化的核心挑战](#62-量化的核心挑战)
   - 6.3 [支持的数据类型](#63-支持的数据类型)
   - 6.4 [量化策略详解](#64-量化策略详解)
   - 6.5 [Scale 因子的管理](#65-scale-因子的管理)
   - 6.6 [写入与读取的量化/反量化流程](#66-写入与读取的量化反量化流程)
   - 6.7 [CUDA Kernel 实现](#67-cuda-kernel-实现)
   - 6.8 [Scale 校准方式](#68-scale-校准方式)
7. [关键设计亮点](#7-关键设计亮点)

---

## 1. 整体架构概览

vLLM 的 KV Cache 管理基于 **PagedAttention** 思想，将 GPU 显存切分成固定大小的 **Block（页）**，通过类似操作系统虚拟内存的方式管理，彻底消除内存碎片和预分配浪费。

```
┌─────────────────────────────────────────────────────────────────┐
│                      vLLM KV Cache 架构                          │
│                                                                 │
│  用户请求                                                        │
│     ↓                                                           │
│  Scheduler ──────→ KVCacheManager ──────→ KVCacheCoordinator   │
│  (调度决策)         (分配/释放接口)        (多 group 协调)        │
│                           │                      │              │
│                       BlockPool              BlockPool          │
│                      (GPU blocks)           (per group)         │
│                           │                                     │
│                  ┌────────┴────────┐                            │
│             FreeBlockQueue    PrefixCache                       │
│             (双向链表 LRU)   (hash → block)                     │
│                           │                                     │
│                  GPU KV Cache Tensor                            │
│         [num_blocks, 2, block_size, num_kv_heads, head_size]    │
│                           │                                     │
│                  ┌────────┴────────┐                            │
│              Key Cache         Value Cache                      │
│           (FP16/BF16/FP8)   (FP16/BF16/FP8)                    │
└─────────────────────────────────────────────────────────────────┘
```

**核心设计思想：**

- **Block 化管理**：KV Cache 被切分成固定大小的 block（默认 16 tokens/block），每个 block 是最小分配单元
- **非连续内存**：不同请求的 block 在物理上不连续，通过 `block_table` 做地址翻译
- **引用计数**：每个 block 维护 `ref_cnt`，支持多请求共享同一 prefix block
- **LRU 驱逐**：`FreeBlockQueue` 双向链表，按最近最少使用顺序管理空闲 block

---

## 2. 核心文件索引

| 文件路径 | 职责 |
|---------|------|
| `vllm/v1/core/block_pool.py` | Block 分配、释放、LRU 驱逐、Prefix Cache 哈希表 |
| `vllm/v1/core/kv_cache_manager.py` | KV Cache 核心管理接口，协调 block 生命周期 |
| `vllm/v1/core/kv_cache_coordinator.py` | 多 KV cache group 协调（full/sliding window/mamba） |
| `vllm/v1/core/sched/scheduler.py` | 调度器，决定请求何时分配/释放 block |
| `vllm/v1/worker/block_table.py` | 维护 req→block_id 映射，计算 slot_mapping |
| `vllm/v1/kv_cache_interface.py` | KVCacheSpec/KVCacheConfig 数据结构定义 |
| `vllm/v1/attention/backends/flashinfer.py` | PagedAttention 前向计算（FlashInfer backend） |
| `vllm/model_executor/layers/quantization/kv_cache.py` | FP8 scale 管理（k_scale/v_scale 的创建与加载） |
| `vllm/config/cache.py` | CacheConfig、CacheDType 定义 |
| `csrc/cache_kernels.cu` | `reshape_and_cache_flash` CUDA kernel（量化写入） |
| `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh` | FP8 量化 dispatch 宏，CUDA 模板特化 |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | Per-token-head 动态量化 Triton kernel |

---

## 3. 核心数据结构

### KVCacheBlock

```python
class KVCacheBlock:
    block_id: int                          # GPU 内存中的 block 索引
    ref_cnt: int                           # 引用计数（>0 时不会被驱逐）
    block_hash: BlockHashWithGroupId | None  # prefix cache 的哈希键
    is_null: bool                          # 是否为占位符（sliding window 用）
```

### KV Cache Tensor 内存布局

```
# NHD layout（默认）
kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_size]
           ─────────  ─  ──────────  ────────────  ─────────
           block 数量  K/V  每块token数   头数        头维度

# 其中 dim=1：0 = Key cache，1 = Value cache
# FP8 时 dtype=uint8，FP16/BF16 时 dtype 对应精度
```

### BlockTable

```
block_table: [max_num_reqs, max_num_blocks_per_req]  # int32
             每个请求的 block ID 列表

slot_mapping: [max_num_batched_tokens]               # int64
              每个 token 对应的 KV cache slot 位置
              slot = block_id * block_size + offset_in_block
```

---

## 4. KV Cache 管理全流程

### 4.1 初始化阶段

```
Engine 启动
  │
  ├── 计算 num_gpu_blocks
  │     = (GPU 总显存 - 模型权重占用 - 预留) / 单个 block 大小
  │     单个 block 大小 = 2 × block_size × num_kv_heads × head_size × dtype_bytes × num_layers
  │
  ├── GPUModelRunner.initialize_kv_cache()
  │     ├── _allocate_kv_cache_tensors()
  │     │     └── 分配 int8 原始内存 tensor（兼容量化，FP8 用 uint8 视图）
  │     ├── _reshape_kv_cache_tensors()
  │     │     └── reshape 为 [num_blocks, 2, block_size, num_kv_heads, head_size]
  │     └── 初始化 attention backend（FlashInfer / Flash-Attn 等）
  │
  └── BlockPool 初始化
        ├── 创建 num_gpu_blocks 个 KVCacheBlock 对象
        ├── 初始化 FreeKVCacheBlockQueue（双向链表，所有 block 入队）
        └── 初始化 cached_block_hash_to_block（prefix cache 哈希表，初始为空）
```

### 4.2 请求调度阶段

```
新请求到达 → waiting queue
  │
  └── Scheduler.schedule()
        │
        ├── Step 1: get_computed_blocks()  ← Prefix Cache 命中检测
        │     ├── 计算请求 token 序列的滚动 block_hash
        │     ├── 在 cached_block_hash_to_block 中逐 block 查找
        │     ├── 命中：touch(block)，ref_cnt++，从 free_block_queue 移除
        │     └── 返回：命中的 blocks 列表 + 命中的 token 数量
        │
        ├── Step 2: allocate_slots()  ← 分配新 block
        │     ├── 计算需要分配的 block 数量
        │     │     = ceil((total_tokens - cached_tokens) / block_size)
        │     ├── 检查 free_block_queue 是否有足够的空闲 block
        │     │     不足 → 返回 None → 请求进入等待或抢占
        │     ├── BlockPool.get_new_blocks(num_blocks)
        │     │     ├── 从 free_block_queue 头部弹出 block（LRU 最旧的）
        │     │     └── 若 block 在 prefix cache 中，先驱逐（清除 block_hash）
        │     └── 缓存新的 full blocks（cache_full_blocks，写入 prefix cache）
        │
        └── Step 3: 构建 SchedulerOutput
              └── 包含 block_ids, num_computed_tokens, slot_mapping 等
```

### 4.3 模型执行阶段

```
GPUModelRunner.execute_model()
  │
  ├── 准备输入
  │     ├── BlockTable.commit_block_table()
  │     │     └── 更新 [req_idx, block_idx] → block_id 的 GPU tensor
  │     └── BlockTable.compute_slot_mapping()
  │           └── 用 Triton kernel 计算每个 token 的精确 slot 位置
  │               slot = block_table[req_idx, token_pos // block_size]
  │                       * block_size + token_pos % block_size
  │
  └── 模型前向传播（每层 Attention）
        │
        ├── 计算 Q, K, V（线性变换）
        │
        ├── do_kv_cache_update()  ← 写入 KV Cache
        │     └── reshape_and_cache_flash(K, V, kv_cache, slot_mapping,
        │                                 kv_cache_dtype, k_scale, v_scale)
        │           按 slot_mapping 把 K/V 写到对应 block 的对应位置
        │           （FP8 时同步做量化，见第 6 节）
        │
        └── PagedAttention forward()  ← 读取 KV Cache 计算 attention
              使用 block_table 非连续地访问 KV cache
              （FP8 时同步做反量化，见第 6 节）
```

### 4.4 Block 释放阶段

```
请求完成（EOS 或 max_tokens 达到）
  │
  └── Scheduler._free_request()
        │
        └── KVCacheManager.free(request)
              │
              └── BlockPool.free_blocks(blocks)
                    ├── block.ref_cnt -= 1（每个 block）
                    └── ref_cnt == 0 的 block → 追加回 free_block_queue 尾部
                          注意：block_hash 保留！
                          → 后续请求仍可通过 prefix cache 命中此 block
                          → 只有当 block 被重新分配时才清除 block_hash
```

---

## 5. Prefix Caching 详解

### Block Hash 计算

```
每个 block 的 hash 依赖其内容 + 前序所有 block 的 hash（滚动哈希）：

block_0_hash = hash(token_ids[0:block_size])
block_1_hash = hash(block_0_hash || token_ids[block_size:2*block_size])
block_2_hash = hash(block_1_hash || token_ids[2*block_size:3*block_size])
...

这保证了：相同前缀的请求，对应位置的 block_hash 完全相同
```

### 命中与驱逐流程

```
┌─────────────────────────────────────────────────────────────┐
│                   Prefix Cache 状态机                        │
└─────────────────────────────────────────────────────────────┘

Block 状态：
  [已分配，ref_cnt > 0]  ←→  [已分配，ref_cnt = 0，在 free_queue]
         ↑                              ↓
      touch()                    被新请求重新分配
         │                    _maybe_evict_cached_block()
         │                    清除 block_hash，block 重新使用
         │
  cache_full_blocks()
  写入 cached_block_hash_to_block

命中时：
  1. 在 cached_block_hash_to_block 找到 block
  2. block.ref_cnt += 1（touch）
  3. 若 block 在 free_queue 中，从 free_queue 移除
  4. 该 block 不会被驱逐，直到请求完成后 ref_cnt 归零

驱逐时（LRU）：
  1. 需要新 block 但 free_queue 头部的 block 有 block_hash
  2. 从 cached_block_hash_to_block 删除该 hash 记录
  3. block.block_hash = None
  4. block 被重新分配给新请求
```

---

## 6. FP8 量化与 KV Cache 的结合

### 6.1 为什么要量化 KV Cache

KV Cache 是 LLM 推理中显存占用的主要来源之一。以 LLaMA-70B 为例：

```
KV Cache 显存占用（每个 token，单层）：
  = 2（K+V）× num_kv_heads × head_size × dtype_bytes
  = 2 × 8 × 128 × 2（FP16）= 4096 bytes = 4 KB

总占用（32 层，batch=32，seq_len=4096）：
  = 4 KB × 32 layers × 32 batch × 4096 tokens ≈ 16 GB

FP8 量化后：
  = 16 GB × (1 byte / 2 bytes) = 8 GB  ← 节省 50%
```

**量化 KV Cache 的收益：**
- 显存减少 ~50%（FP16→FP8），可支持更大 batch 或更长序列
- 内存带宽减少，decode 阶段（memory-bound）速度提升
- 代价：引入量化误差，需要 scale 因子管理

### 6.2 量化的核心挑战

KV Cache 量化与权重量化有本质区别：

| 对比维度 | 权重量化 | KV Cache 量化 |
|---------|---------|--------------|
| **量化时机** | 离线（加载时一次性） | 在线（每次 forward 实时量化） |
| **数据分布** | 静态，训练后固定 | 动态，每个 token 不同 |
| **Scale 来源** | 校准数据集统计 | 静态预设 or 动态计算 |
| **精度影响** | 主要影响模型权重精度 | 影响历史 token 的 K/V 精度 |
| **累积误差** | 无 | 有（早期 token 的量化误差会影响后续 attention） |

### 6.3 支持的数据类型

```python
# vllm/config/cache.py
CacheDType = Literal[
    "auto",               # 与模型计算精度一致（fp16/bf16），不量化
    "fp8",                # = fp8_e4m3（默认 FP8，推荐）
    "fp8_e4m3",           # 4位指数 + 3位尾数，精度更高，范围较小
    "fp8_e5m2",           # 5位指数 + 2位尾数，范围更大，精度略低
    "fp8_inc",            # Intel Gaudi 平台专用 FP8
    "fp8_ds_mla",         # DeepSeek MLA 架构专用 FP8
    "fp8_per_token_head", # 动态 per-(token, head) 量化，精度最高
    "int8_per_token_head",# INT8 动态 per-(token, head) 量化
]

# 内存存储类型映射（vllm/utils/torch_utils.py）
STR_DTYPE_TO_TORCH_DTYPE = {
    "fp8":                torch.uint8,         # 用 uint8 存储 FP8 数据
    "fp8_e4m3":           torch.uint8,
    "fp8_e5m2":           torch.uint8,
    "fp8_inc":            torch.float8_e4m3fn, # 原生 FP8 类型
    "fp8_per_token_head": torch.uint8,
}
```

### 6.4 量化策略详解

vLLM 支持三种粒度的 KV Cache 量化策略：

#### 策略一：Per-Tensor 量化

```
K/V tensor 整体使用一个 scale：

K: [num_tokens, num_heads, head_size]  →  scale: [1]

量化：quantized_K = round(K / scale).clamp(FP8_MIN, FP8_MAX)
反量化：K ≈ quantized_K * scale

优点：实现简单，overhead 最小
缺点：不同 head 的数值范围差异大时，精度损失明显
```

#### 策略二：Per-Head 量化

```
每个 attention head 独立使用一个 scale：

K: [num_tokens, num_heads, head_size]  →  scale: [num_heads]

量化：quantized_K[token, head, :] = round(K[token, head, :] / scale[head])
反量化：K[token, head, :] ≈ quantized_K[token, head, :] * scale[head]

优点：不同 head 的数值范围差异被 scale 吸收，精度更好
缺点：需要存储 num_heads 个 scale（开销极小）
推荐：生产环境推荐使用
```

#### 策略三：Per-Token-Head 动态量化

```
每个 (token, head) 独立动态计算 scale：

K: [num_tokens, num_heads, head_size]  →  scale: [num_tokens, num_heads]（动态）

量化（Triton kernel 实时计算）：
  absmax = max(|K[token, head, :]|)
  scale[token, head] = absmax / FP8_MAX
  quantized_K[token, head, :] = round(K[token, head, :] / scale[token, head])

scale 单独存储在 k_scale_cache: [num_blocks, block_size, num_kv_heads]

反量化：
  K[token, head, :] ≈ quantized_K[token, head, :] * scale_cache[block, offset, head]

优点：精度最高，完全适应动态数值范围
缺点：需要额外存储 scale cache，计算 overhead 略高
```

**三种策略对比：**

| 策略 | Scale 形状 | 精度 | 额外存储 | 适用场景 |
|------|-----------|------|---------|---------|
| Per-Tensor | `[1]` | ★★☆ | 极小 | 快速验证、精度要求低 |
| Per-Head | `[num_heads]` | ★★★ | 极小 | **生产环境推荐** |
| Per-Token-Head | `[tokens, heads]`（动态） | ★★★★ | 中等 | 精度敏感场景 |

### 6.5 Scale 因子的管理

Scale 因子由 `BaseKVCacheMethod` 管理（`vllm/model_executor/layers/quantization/kv_cache.py`）：

```python
class BaseKVCacheMethod(QuantizeMethodBase):

    def create_weights(self, layer: torch.nn.Module):
        """在 Attention layer 上创建 scale 参数"""
        # 初始值 -1.0 表示"未设置"
        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        # 内部使用的 float 版本（避免 tensor 操作 overhead）
        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """加载权重后处理 scale"""

        # Per-token-head：动态计算，不需要静态 scale
        if kv_cache_uses_per_token_head_scales(layer.kv_cache_dtype):
            layer._k_scale.copy_(1.0)
            layer._v_scale.copy_(1.0)
            del layer.k_scale  # 删除静态 scale，节省显存
            del layer.v_scale
            return

        if is_quantized_kv_cache(layer.kv_cache_dtype):
            if layer.k_scale > 0.0 and layer.v_scale > 0.0:
                # checkpoint 中有 scale，直接加载
                k_scale = layer.k_scale.item()
                v_scale = layer.v_scale.item()
            elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
                # checkpoint 中没有 scale，使用默认值 1.0（精度较差）
                k_scale = 1.0
                v_scale = 1.0
            # AMD FNUZ FP8 需要额外 ×2 修正
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

            layer._k_scale.copy_(k_scale)
            layer._v_scale.copy_(v_scale)
            layer._k_scale_float = k_scale  # 缓存 float 值，kernel 调用更高效
            layer._v_scale_float = v_scale
```

### 6.6 写入与读取的量化/反量化流程

#### 写入 KV Cache（量化）

```
Attention.forward()
  │
  ├── 计算 K, V（FP16/BF16 精度）
  │
  └── do_kv_cache_update()
        │
        └── reshape_and_cache_flash(
                key,           # [num_tokens, num_heads, head_size]  FP16
                value,         # [num_tokens, num_heads, head_size]  FP16
                key_cache,     # [num_blocks, block_size, num_heads, head_size]  uint8（FP8）
                value_cache,   # [num_blocks, block_size, num_heads, head_size]  uint8（FP8）
                slot_mapping,  # [num_tokens]  每个 token 的目标 slot
                kv_cache_dtype="fp8_e4m3",
                k_scale,       # [1] 或 [num_heads]
                v_scale,       # [1] 或 [num_heads]
            )
            │
            └── CUDA Kernel 内部（per token, per head）：
                  quantized_k = float_to_fp8(K[token, head, :] / k_scale[head])
                  key_cache[slot_mapping[token], :] = quantized_k
```

#### 读取 KV Cache（反量化）

```
PagedAttention Kernel（CUDA）
  │
  ├── 根据 block_table 定位 K/V 数据（非连续内存访问）
  │
  ├── 反量化：
  │     dequant_K = fp8_to_float(key_cache[block, offset, head, :]) * k_scale[head]
  │     dequant_V = fp8_to_float(value_cache[block, offset, head, :]) * v_scale[head]
  │
  └── 计算 Attention：
        scores = Q @ dequant_K.T / sqrt(head_size)
        output = softmax(scores) @ dequant_V
```

#### Per-Token-Head 动态量化（Triton 实现）

```python
# vllm/v1/attention/ops/triton_reshape_and_cache_flash.py

def triton_reshape_and_cache_flash_per_token_head_quant(
    key, value,
    key_cache, value_cache,
    k_scale_cache,  # [num_blocks, block_size, num_kv_heads]  存储动态 scale
    v_scale_cache,
    slot_mapping,
):
    # Triton kernel 内部（每个 (token, head) 独立处理）：
    #
    # 1. 计算动态 scale
    absmax = max(|K[token, head, :]|)
    scale = absmax / FP8_MAX  # FP8_MAX = 448.0 for e4m3
    #
    # 2. 量化并写入 KV cache
    quantized = round(K[token, head, :] / scale).clamp(FP8_MIN, FP8_MAX)
    key_cache[block_id, offset, head, :] = quantized
    #
    # 3. 存储 scale（供 attention 计算时反量化使用）
    k_scale_cache[block_id, offset, head] = scale
```

### 6.7 CUDA Kernel 实现

FP8 量化的 dispatch 通过宏实现，根据 `kv_cache_dtype` 字符串选择对应的 CUDA 模板特化：

```cpp
// csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh

#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)
  if (KV_DTYPE == "auto") {
    // 不量化，直接存储原始精度
    if (SRC_DTYPE == at::ScalarType::Half)
      FN(uint16_t, uint16_t, Fp8KVCacheDataType::kAuto);
    else if (SRC_DTYPE == at::ScalarType::BFloat16)
      FN(__nv_bfloat16, __nv_bfloat16, Fp8KVCacheDataType::kAuto);
  } else if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {
    // 量化为 FP8 E4M3
    if (SRC_DTYPE == at::ScalarType::Half)
      FN(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E4M3);
    else if (SRC_DTYPE == at::ScalarType::BFloat16)
      FN(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3);
  } else if (KV_DTYPE == "fp8_e5m2") {
    // 量化为 FP8 E5M2
    if (SRC_DTYPE == at::ScalarType::Half)
      FN(uint16_t, uint8_t, Fp8KVCacheDataType::kFp8E5M2);
    ...
  }

// reshape_and_cache_flash kernel 调用示例
void reshape_and_cache_flash(
    torch::Tensor& key,         // [num_tokens, num_heads, head_size]
    torch::Tensor& value,
    torch::Tensor& key_cache,   // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,     // [1] 或 [num_heads]
    torch::Tensor& v_scale) {

  int kv_scale_stride = (k_scale.numel() > 1) ? 1 : 0;  // per-head vs per-tensor

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}
```

### 6.8 Scale 校准方式

Scale 的质量直接决定量化精度，vLLM 支持三种校准方式：

#### 方式一：无校准（scale = 1.0）

```python
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False,  # 默认
)
# scale 固定为 1.0
# 优点：零开销，开箱即用
# 缺点：精度损失较大，K/V 数值范围可能超出 FP8 表示范围
```

#### 方式二：随机 Token 动态校准

```python
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    kv_cache_dtype="fp8_e4m3",
    calculate_kv_scales=True,  # warmup 时用随机 token 估算 scale
)
# warmup 阶段：用一批随机 token 跑前向，统计 K/V 的数值范围
# 计算：scale = max(|K|) / FP8_MAX
# 优点：比 scale=1.0 精度好，无需额外数据集
# 缺点：随机 token 不代表真实分布，scale 估算可能不准
```

#### 方式三：数据集离线校准（推荐）

```bash
# 使用 llm-compressor 在代表性数据集上校准
python -m llmcompressor.transformers.calibrate \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset "ultrachat" \
    --kv_cache_scheme fp8 \
    --output_dir ./calibrated_model

# 加载时自动读取 checkpoint 中的 k_scale/v_scale
llm = LLM(
    model="./calibrated_model",
    kv_cache_dtype="fp8",
)
```

```
校准流程：
  代表性数据集（1000~2000 条样本）
    ↓
  前向传播，收集每层每 head 的 K/V 激活值
    ↓
  统计 max(|K|) 和 max(|V|)（per-tensor 或 per-head）
    ↓
  scale = max_abs / FP8_MAX
    ↓
  保存到 checkpoint 的 k_scale / v_scale 参数
    ↓
  vLLM 加载时通过 process_weights_after_loading() 读取
```

**三种校准方式对比：**

| 方式 | 精度 | 便利性 | 适用场景 |
|------|------|-------|---------|
| 无校准（scale=1.0） | ★★☆ | ★★★★★ | 快速实验、精度要求低 |
| 随机 Token 校准 | ★★★☆ | ★★★★☆ | 无校准数据集时的折中 |
| 数据集离线校准 | ★★★★★ | ★★★☆☆ | **生产环境强烈推荐** |

---

## 7. 关键设计亮点

### PagedAttention：非连续内存的高效访问

```
传统方案（连续内存）：
  请求 A: [K0, K1, K2, K3, K4, K5, K6, K7]  ← 必须预分配最大长度
  请求 B: [K0, K1, K2]                        ← 剩余空间浪费

PagedAttention（分页内存）：
  Block 0: [K0, K1]  ← 请求 A 的前 2 个 token
  Block 1: [K2, K3]  ← 请求 A 的 3-4 token
  Block 2: [K0, K1]  ← 请求 B 的前 2 个 token（可能与 A 共享！）
  Block 3: [K4, K5]  ← 请求 A 的 5-6 token
  ...
  block_table[A] = [0, 1, 3, ...]  ← 地址翻译表
  block_table[B] = [2, ...]
```

### Prefix Caching：跨请求复用 KV Cache

```
请求 A: "你好，请介绍一下 Python" → 计算并缓存 block_0, block_1
请求 B: "你好，请介绍一下 Java"   → block_0 命中！直接复用，只需计算 block_1'

节省：block_0 对应的所有层的 K/V 计算（num_layers × 2 × block_size 次矩阵乘法）
```

### FP8 量化：显存与精度的平衡

```
FP8 E4M3 数值范围：[-448, 448]，精度约 3 位有效数字
FP16 数值范围：[-65504, 65504]，精度约 3-4 位有效数字

量化误差来源：
  1. 截断误差：K/V 值超出 FP8 范围时被 clamp
  2. 舍入误差：FP8 精度不足导致的舍入
  3. Scale 误差：静态 scale 无法完美适配动态数值范围

缓解方案：
  - Per-head scale：每个 head 独立 scale，减少截断误差
  - Per-token-head scale：动态适应每个 token 的数值范围
  - 数据集校准：用真实数据统计准确的 scale
```

### 引用计数：安全的 Block 共享

```
场景：请求 A 和 B 共享 prefix block_0

block_0.ref_cnt = 2  ← A 和 B 都在使用

请求 A 完成：
  block_0.ref_cnt = 1  ← B 仍在使用，block 不释放

请求 B 完成：
  block_0.ref_cnt = 0  ← 加入 free_queue 尾部（LRU）
                          block_hash 保留，可被后续请求命中
```

---

## 参考文档

- [PagedAttention 设计文档](paged_attention.md)
- [Prefix Caching 设计文档](prefix_caching.md)
- [Hybrid KV Cache Manager](hybrid_kv_cache_manager.md)
- [量化 KV Cache 用户文档](../features/quantization/quantized_kvcache.md)
- [Attention Backends](attention_backends.md)
