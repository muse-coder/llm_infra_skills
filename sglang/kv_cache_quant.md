# SGLang KV Cache 量化详解

## 一、概述

KV Cache 量化通过使用低精度数据类型（FP8 / FP4）替代默认的 BF16 来减少 KV Cache 的显存占用。在自回归生成过程中，LLM 会缓存已计算的 Key-Value 对以避免重复计算，KV Cache 通常占据 GPU 显存的很大比例，尤其是长序列场景。

量化 KV Cache 是一种**显存优化技术**，主要通过允许缓存更多 token 来提升吞吐量，但可能引入轻微的精度损失。

## 二、支持的量化格式

### 2.1 FP8 格式

OCP (Open Compute Project) 定义了两种 8-bit 浮点格式：

| 格式 | 指数位 | 尾数位 | 最大值 | 特点 |
|------|--------|--------|--------|------|
| **E4M3** (`fp8_e4m3`) | 4 | 3 | ±240.0 | 精度高，动态范围小，**推荐** |
| **E5M2** (`fp8_e5m2`) | 5 | 2 | ±57344.0 | 精度低，动态范围大 |

### 2.2 FP4 格式（实验性）

OCP 定义的 MXFP4 (Microscaling FP4) 格式：

| 格式 | 符号位 | 指数位 | 尾数位 | 可表示值 |
|------|--------|--------|--------|----------|
| **E2M1** (`fp4_e2m1`) | 1 | 2 | 1 | {0, 0.5, 1, 1.5, 2, 3, 4, 6} 共 8 个值 |

FP4 使用 **block-wise microscaling**：每 16 个连续元素共享一个 8-bit 指数 scale factor。

### 2.3 格式对比

| 特性 | FP8 (E4M3) | FP4 (E2M1 / MXFP4) |
|------|------------|---------------------|
| **位宽** | 8 bits | 4 bits |
| **Scale 粒度** | Per-tensor（整个 tensor 一个 scale） | Per-block（每 16 个元素一个 scale） |
| **Scale 来源** | 从 checkpoint 加载或默认 1.0 | 动态计算（无需预量化模型） |
| **Scale 存储** | 标量 float32 | uint8 指数形式，每 16 元素一个 |
| **内存节省** | 相比 BF16 节省 **2×** | 相比 BF16 节省 **~3.56×**（含 scale 开销） |
| **精度** | 简单/复杂任务均较好 | 简单任务好，复杂推理任务下降明显 |
| **硬件要求** | CUDA 11.8+ | CUDA 12.8+ / PyTorch 2.8+ |
| **成熟度** | 生产就绪 | 实验性 |
| **融合算子** | ✅ 有 Triton 融合 kernel | ❌ 目前无融合 kernel，使用 `@torch.compile` |

## 三、整体架构

SGLang 的 KV Cache 量化采用三层架构：

```
配置层 (ServerArgs / ModelRunner)
    ↓  --kv-cache-dtype 参数解析，确定量化类型
内存池层 (KVCache / TokenToKVPool)
    ↓  管理量化数据的存储、读写、scale buffer
Kernel 层 (Triton / CUDA / torch.compile)
    ↓  执行实际的量化/反量化计算
```

### 3.1 关键文件索引

| 文件 | 作用 |
|------|------|
| `python/sglang/srt/server_args.py` | `--kv-cache-dtype` 命令行参数定义与兼容性检查 |
| `python/sglang/srt/model_executor/model_runner.py` | KV cache dtype 解析和配置 (`configure_kv_cache_dtype`) |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | 内存池初始化、FP4/FP8 池选择逻辑 |
| `python/sglang/srt/layers/quantization/kv_cache.py` | FP8 scale 创建/加载/处理 (`BaseKVCacheMethod`) |
| `python/sglang/srt/layers/quantization/kvfp4_tensor.py` | FP4 量化/反量化核心算法 (`KVFP4QuantizeUtil`) |
| `python/sglang/srt/mem_cache/memory_pool.py` | KV Cache 内存池（`MHATokenToKVPool`, `MLATokenToKVPool`, `*FP4` 变体） |
| `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py` | FP8 融合 Triton kernel (`fused_fp8_set_kv_buffer`) |
| `python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant.cuh` | NVFP4 硬件加速 CUDA kernel（SM100+ Blackwell） |
| `python/sglang/jit_kernel/nvfp4.py` | NVFP4 JIT 编译封装 |
| `sgl-kernel/python/sgl_kernel/flash_mla.py` | FlashMLA FP8 attention kernel 接口 |

## 四、配置入口

### 4.1 命令行参数

```bash
python3 -m sglang.launch_server \
    --model-path <model> \
    --kv-cache-dtype <dtype>   # auto | fp8_e4m3 | fp8_e5m2 | bf16 | bfloat16 | fp4_e2m1
```

**📄 `python/sglang/srt/server_args.py` (行 3783-3788)**

```python
parser.add_argument(
    "--kv-cache-dtype",
    type=str,
    default=ServerArgs.kv_cache_dtype,
    choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],
)
```

### 4.2 dtype 解析逻辑

**📄 `python/sglang/srt/model_executor/model_runner.py` (行 1809-1860)**

```python
def configure_kv_cache_dtype(self):
    if self.server_args.kv_cache_dtype == "auto":
        # 从模型的 quant_config 中读取 kv_cache_quant_algo
        quant_config = getattr(self.model, "quant_config", None)
        kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if kv_cache_quant_algo == "FP8":
            self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            self.kv_cache_dtype = self.dtype  # 不量化，使用模型原始 dtype
    elif self.server_args.kv_cache_dtype == "fp8_e4m3":
        self.kv_cache_dtype = torch.float8_e4m3fn
    elif self.server_args.kv_cache_dtype == "fp8_e5m2":
        self.kv_cache_dtype = torch.float8_e5m2
    elif self.server_args.kv_cache_dtype == "fp4_e2m1":
        if hasattr(torch, "float4_e2m1fn_x2"):
            self.kv_cache_dtype = torch.float4_e2m1fn_x2
        else:
            self.kv_cache_dtype = self.dtype  # PyTorch 版本不支持，回退
    elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
        self.kv_cache_dtype = torch.bfloat16
```

### 4.3 内存池选择

**📄 `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` (行 554-685)**

根据 `kv_cache_dtype` 和模型架构（MHA/MLA）选择对应的内存池：

- **MLA + FP4** → `MLATokenToKVPoolFP4`
- **MLA + FP8/BF16** → `MLATokenToKVPool`
- **MHA + FP4** → `MHATokenToKVPoolFP4`
- **MHA + FP8/BF16** → `MHATokenToKVPool`

## 五、FP8 KV Cache 量化实现

### 5.1 量化原理

FP8 使用 **per-tensor scaling**：整个 K/V tensor 共享一个 scale factor。

```
量化:   FP8_value = BF16_value / scale
反量化: BF16_value = FP8_value × scale
```

### 5.2 Scale 的创建与加载

**📄 `python/sglang/srt/layers/quantization/kv_cache.py`**

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    def create_weights(self, layer):
        # 初始化为 -1.0（无效值，表示尚未加载）
        layer.k_scale = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )
        layer.v_scale = torch.nn.Parameter(
            torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )

    def process_weights_after_loading(self, layer):
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            # ✅ 从 checkpoint 成功加载了 scale
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            # ❌ 没有加载到 scale，使用默认值 1.0（即不缩放，直接截断）
            k_scale = 1.0
            v_scale = 1.0
        else:
            # 只有一个 kv_scale，复制给 k 和 v
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = v_scale = scale_to_duplicate.to("cpu").tolist()

        layer.k_scale_float = k_scale
        layer.v_scale_float = v_scale
```

**Scale 来源**：
1. **静态 scale**：从预量化模型 checkpoint 中加载（如 NVIDIA ModelOpt 量化的模型）
2. **JSON 文件**：通过 `--quantization-param-path` 提供
3. **默认 scale = 1.0**：如果 checkpoint 中没有 scale，直接用 1.0

### 5.3 内存池存储

**📄 `python/sglang/srt/mem_cache/memory_pool.py` (行 632-670, 850-870)**

```python
class KVCache(abc.ABC):
    def __init__(self, ...):
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # FP8 存储为 uint8，因为 PyTorch 的 index_put 不支持 float8 类型
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype

class MHATokenToKVPool(KVCache):
    def _create_buffers(self):
        # K/V buffer 形状: [size, head_num, head_dim]，dtype 为 uint8
        self.k_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,  # torch.uint8 (FP8 的存储形式)
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
```

### 5.4 量化写入 KV Cache

FP8 有两条路径：

#### 路径 A：通用路径（MHATokenToKVPool.set_kv_buffer）

**📄 `python/sglang/srt/mem_cache/memory_pool.py`**

```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale, v_scale, ...):
    if cache_k.dtype != self.dtype:
        # 1. 除以 scale 进行缩放
        if k_scale is not None:
            cache_k.div_(k_scale)
        if v_scale is not None:
            cache_v.div_(v_scale)
        # 2. 类型转换到 FP8
        cache_k = cache_k.to(self.dtype)
        cache_v = cache_v.to(self.dtype)

    if self.store_dtype != self.dtype:
        # 3. view 为 uint8 以便 index_put
        cache_k = cache_k.view(self.store_dtype)
        cache_v = cache_v.view(self.store_dtype)

    # 4. 写入 cache
    self.k_buffer[layer_id][loc] = cache_k
    self.v_buffer[layer_id][loc] = cache_v
```

#### 路径 B：融合 Triton Kernel（TRTLLM MHA Backend 专用）

**📄 `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`**

这是一个**融合 kernel**，将量化 + 写入 cache 合并为一个 kernel，避免中间 FP8 tensor 的内存分配：

```python
@triton.jit
def _process_kv_tensor(...):
    # 从输入加载 BF16/FP16 数据
    block = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

    # 量化到 FP8（乘以 inverse scale）
    if use_provided_scale:
        block_fp8 = (block * inv_scale).to(tl.float8e4nv)
    else:
        block_fp8 = block.to(tl.float8e4nv)

    # 直接写入 paged cache
    tl.store(cache_ptr + cache_offsets, block_fp8, mask=mask)
```

**Grid 设计**：`(num_tokens, num_head_blocks, 2)`
- dim 0: 每个 token 一个 program
- dim 1: head 分块处理（BLOCK_HEAD = min(num_kv_heads, 8)）
- dim 2: K=0, V=1

**性能优势**：
- 消除中间 FP8 tensor 的内存分配
- 减少 kernel launch 开销
- 更好的内存带宽利用率
- 支持 CUDA Graph capture（inverse scale 在 GPU 上计算，避免 GPU→CPU 同步）

### 5.5 反量化读取

FP8 的反量化通常**融合在 attention kernel 内部**，不需要显式反量化。FlashInfer / TRTLLM / FlashMLA 等 attention backend 直接接受 FP8 格式的 KV cache，并在计算 attention score 时内部处理 scale。

例如 FlashMLA 中：

**📄 `sgl-kernel/python/sgl_kernel/flash_mla.py`**

```python
# FP8 KV cache 直接传入 attention kernel，配合 descale 参数
out, softmax_lse = torch.ops.sgl_kernel.fwd_kvcache_mla_fp8.default(
    q, k_cache, head_dim_v, cache_seqlens, block_table,
    softmax_scale, causal, tile_scheduler_metadata, num_splits,
    descale_q, descale_k,  # 反量化 scale 在 kernel 内部应用
)
```

## 六、FP4 KV Cache 量化实现

### 6.1 量化原理

FP4 使用 **MXFP4 (Microscaling FP4)** 格式，采用 **block-wise quantization**：

```
每 16 个连续元素共享一个 8-bit 指数 scale factor
```

FP4 E2M1 格式只能表示 8 个值：`{0, 0.5, 1, 1.5, 2, 3, 4, 6}`

### 6.2 量化核心算法

**📄 `python/sglang/srt/layers/quantization/kvfp4_tensor.py`**

#### 常量定义

```python
E2M1_MAX = 6.0  # FP4 E2M1 能表示的最大值
E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])  # 所有可表示的值
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])  # 量化边界
```

#### 量化过程 (`batched_quantize`)

```python
@staticmethod
@torch.compile
def batched_quantize(tensor):  # tensor: [B, M, N]
    b, m, n = tensor.shape

    # Step 1: 按 block=16 重塑
    reshaped = tensor.view(b, m * n // 16, 16)

    # Step 2: 计算每个 block 的 scale（指数形式）
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))
    scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)  # 偏移存储

    # Step 3: 缩放到 FP4 范围
    scaled = reshaped / torch.exp2(scale_exp)

    # Step 4: 量化到 FP4（查找最近的 E2M1 值）
    sign_bits = (scaled < 0).to(torch.uint8) << 3
    abs_vals = scaled.abs()
    magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)
    fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)

    # Step 5: 打包（两个 FP4 值 → 一个 uint8）
    fp4_reshaped = fp4_vals.view(b, m, n)
    packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]

    return packed, scale_factors  # packed: [B,M,N/2], scale: [B, M*N/16]
```

#### 反量化过程 (`batched_dequantize`)

```python
@staticmethod
@torch.compile
def batched_dequantize(quant_tensor, scale_factors, dtype=torch.bfloat16):
    b, m, n_half = quant_tensor.shape
    n = n_half * 2

    # Step 1: 解包 uint8 → 两个 FP4 值
    fp4_vals[..., 0::2] = quant_tensor & 0x0F
    fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F

    # Step 2: 提取符号和幅度，查表得到浮点值
    magnitude_idx = fp4_vals & 0x07
    float_vals = E2M1_VALUES[magnitude_idx.long()]
    float_vals = torch.where(sign_mask, -float_vals, float_vals)

    # Step 3: 应用 scale 恢复原始值
    reshaped = float_vals.view(b, m * n // 16, 16)
    scale_exp = scale_factors.float() - 127  # 恢复原始指数
    scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))

    return scaled.view(b, m, n).to(dtype)
```

### 6.3 FP4 内存池

#### MHA 架构

**📄 `python/sglang/srt/mem_cache/memory_pool.py` (行 1081-1230)**

```python
class MHATokenToKVPoolFP4(MHATokenToKVPool):
    def _create_buffers(self):
        m = self.size + self.page_size
        n = self.head_num
        k = self.head_dim
        scale_block_size = 16
        self.store_dtype = torch.uint8

        # K/V buffer: 压缩到一半大小（两个 FP4 打包为一个 uint8）
        self.k_buffer = [torch.zeros((m, n, k // 2), dtype=torch.uint8, ...)]
        self.v_buffer = [torch.zeros((m, n, k // 2), dtype=torch.uint8, ...)]

        # Scale buffer: 每 16 个元素一个 scale
        self.k_scale_buffer = [torch.zeros((m, (n*k) // 16), dtype=torch.uint8, ...)]
        self.v_scale_buffer = [torch.zeros((m, (n*k) // 16), dtype=torch.uint8, ...)]
```

#### MLA 架构（DeepSeek 系列）

**📄 `python/sglang/srt/mem_cache/memory_pool.py` (行 1658-1780)**

```python
class MLATokenToKVPoolFP4(MLATokenToKVPool):
    def _create_buffers(self):
        m = self.size + self.page_size
        k = self.kv_cache_dim  # kv_lora_rank + qk_rope_head_dim

        # KV buffer（MLA 合并存储 K 和 V）
        self.kv_buffer = [torch.zeros((m, 1, k // 2), dtype=torch.uint8, ...)]
        # Scale buffer
        self.kv_scale_buffer = [torch.zeros((m, k // 16), dtype=torch.uint8, ...)]
```

### 6.4 FP4 的写入和读取

**📄 `python/sglang/srt/mem_cache/memory_pool.py` (行 1175-1230)**

```python
# 写入（量化）— 无融合 kernel，分步执行
def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale, v_scale, ...):
    if cache_k.dtype != self.dtype:
        if k_scale is not None:
            cache_k.div_(k_scale)  # 先应用 FP8 的 scale（如果有）
        # 调用 FP4 量化（@torch.compile 优化，非手写融合 kernel）
        cache_k, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
        cache_v, cache_v_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_v)

    # 分别写入数据和 scale
    self.k_buffer[layer_id][loc] = cache_k
    self.k_scale_buffer[layer_id][loc] = cache_k_fp4_sf

# 读取（反量化）
def _get_key_buffer(self, layer_id):
    if self.store_dtype != self.dtype:
        cache_k_fp4 = self.k_buffer[layer_id].view(torch.uint8)
        cache_k_sf = self.k_scale_buffer[layer_id]
        return KVFP4QuantizeUtil.batched_dequantize(cache_k_fp4, cache_k_sf)
    return self.k_buffer[layer_id]
```

### 6.5 NVFP4 硬件加速 Kernel（权重量化用，非 KV Cache）

**📄 `python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant.cuh`**

在 SM100+ (Blackwell) 架构上，使用 PTX 指令直接进行 FP4 转换：

```cpp
// 使用 PTX 指令将 8 个 float32 转为 8 个 e2m1 值
SGL_DEVICE uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
    uint32_t val;
    asm volatile(
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
        // ...
    );
    return val;
}
```

> **注意**：这些 NVFP4 kernel 目前用于**权重量化和 MoE**，不用于 KV Cache 量化。KV Cache 的 FP4 量化使用的是纯 PyTorch 的 `KVFP4QuantizeUtil`（`@torch.compile` 优化）。

## 七、融合算子现状

### 7.1 FP8 — 有融合算子 ✅

**📄 `python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py`**

`fused_fp8_set_kv_buffer` 将以下操作融合为一个 Triton kernel：
1. BF16/FP16 → FP8 量化（乘以 inverse scale）
2. 写入 paged KV cache

### 7.2 FP4 — 无融合算子 ❌

FP4 KV Cache 的写入是**分步执行**的：
1. `KVFP4QuantizeUtil.batched_quantize` — `@torch.compile` 优化的纯 PyTorch 实现
2. 分别写入 data buffer 和 scale buffer

**潜在优化方向**：参考 FP8 的 `trtllm_fp8_kv_kernel.py`，可以写一个 `fused_fp4_set_kv_buffer` Triton kernel，将 block-wise scale 计算 + FP4 量化 + paged cache 写入三步融合。

## 八、Attention Backend 支持矩阵

### 8.1 MHA Backends

| Backend | FP8 KV Cache | FP4 KV Cache |
|---------|:---:|:---:|
| **FlashInfer** | ✅ | ❌ |
| **FA3 (FlashAttention 3)** | ✅ | ❌ |
| **FA4 (FlashAttention 4)** | ❌ | ✅ |
| **Triton** | ✅ | ✅ |
| **Torch Native (SDPA)** | ✅ | ✅ |
| **FlexAttention** | ❌ | ✅ |
| **TRTLLM MHA** | ✅ | ✅ |
| AITER (ROCm) | ✅ | ❌ |

### 8.2 MLA Backends

| Backend | FP8 KV Cache | FP4 KV Cache |
|---------|:---:|:---:|
| **FlashInfer MLA** | ❌ | ✅ |
| **FlashMLA** | ✅ | ✅ |
| **Cutlass MLA** | ✅ | ✅ |
| **TRTLLM MLA (Blackwell)** | ✅ | ✅ |
| **FA4** | ❌ | ✅ |
| FA3 | ❌ | ❌ |
| Triton | ❌ | ❌ |

## 九、模型支持

### 9.1 FP4 KV Cache 不限定特定模型

FP4 KV Cache 量化**不限定特定模型架构**，理论上所有 MHA 和 MLA 架构的模型都支持，关键约束是：
1. 选择支持 FP4 的 **attention backend**（见上表）
2. 满足 **CUDA 12.8+ / PyTorch 2.8+** 的环境要求

### 9.2 经过精度验证的模型

| 模型 | 数据集 | KV16 | KV8 (FP8) | KV4 (FP4) |
|------|--------|------|-----------|-----------|
| Qwen3-235B-A22B | gsm8k | 0.9168 | 0.9181 | 0.9186 |
| Qwen3-235B-A22B | aime25 | 0.7733 | 0.7333 | 0.6000 |
| DeepSeek-R1-0528 | gsm8k | 0.9157 | 0.9154 | 0.9124 |
| DeepSeek-R1-0528 | aime25 | 0.5067 | 0.4934 | 0.4000 |
| GPT-OSS-120B | gsm8k | 0.9161 | 0.9163 | 0.9152 |
| GPT-OSS-120B | aime25 | 0.7533 | 0.7667 | 0.3533 |

**关键结论**：
- **简单数据集**（如 gsm8k）：FP4 精度接近 FP8/BF16
- **大模型**（200B+）：比小模型更能容忍 FP4 量化
- **复杂推理任务**（如 aime25）：FP4 精度下降明显，建议使用 FP8

### 9.3 CI 测试中涉及的模型

- **Kimi-K2.5-MXFP4** — `test/registered/amd/test_kimi_k25_mxfp4.py`
- **DeepSeek-R1-MXFP4** — `test/registered/amd/accuracy/mi35x/`
- **DeepSeek-V3.2-NVFP4** — `docs/basic_usage/deepseek_v32.md`

## 十、使用示例

### 10.1 FP8 KV Cache

```bash
# FP8 E4M3（推荐）
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --kv-cache-dtype fp8_e4m3

# FP8 E4M3 + 自定义 scale 文件
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --kv-cache-dtype fp8_e4m3 \
    --quantization-param-path quant_params.json
```

Scale JSON 格式：
```json
{
  "kv_cache": {
    "dtype": "float8_e4m3fn",
    "scaling_factor": {
      "0": { "0": 1.0, "1": 1.0 }
    }
  }
}
```

### 10.2 FP4 KV Cache

```bash
# FP4 E2M1（实验性，无需 scale 文件）
python3 -m sglang.launch_server \
    --model-path nvidia/DeepSeek-R1-0528-NVFP4 \
    --kv-cache-dtype fp4_e2m1
```

## 十一、最佳实践

1. **优先使用 FP8 E4M3**：精度损失最小，生产就绪
2. **FP4 适合大模型 + 简单任务**：200B+ 参数的模型在 gsm8k 等简单任务上精度无损
3. **复杂推理任务避免 FP4**：aime25、gpqa_diamond 等复杂任务精度下降明显
4. **检查 backend 兼容性**：不是所有 attention backend 都支持量化 KV cache
5. **使用预量化模型**：优先使用 checkpoint 中包含 scale 的模型，避免默认 scale=1.0 带来的精度问题
6. **注意 FP4 的环境要求**：需要 CUDA 12.8+ 和 PyTorch 2.8.0+
