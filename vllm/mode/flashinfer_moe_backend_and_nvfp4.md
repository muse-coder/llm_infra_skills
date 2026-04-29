# `VLLM_FLASHINFER_MOE_BACKEND` 和 NVFP4 kernel 区别

## 先说结论

这几个名字很像，但它们控制的层级不同：

| 开关 | 控制范围 | 典型值 | 作用 |
|---|---|---|---|
| `--performance-mode` | vLLM 顶层运行策略 | `balanced` / `interactivity` / `throughput` | 决定整体更偏低延迟还是高吞吐 |
| `VLLM_FLASHINFER_MOE_BACKEND` | **MoE 层**里 FlashInfer 后端选择 | `latency` / `throughput` / `masked_gemm` | 决定 MoE 用 TRTLLM、CUTLASS 还是 CuteDSL |
| `VLLM_NVFP4_GEMM_BACKEND` | **普通 Linear 层** 的 NVFP4 GEMM kernel | `flashinfer-trtllm` / `flashinfer-cutlass` / `cutlass` / `marlin` / ... | 决定 NVFP4 线性层用哪个 GEMM 内核 |

最容易搞混的一点是：

- `VLLM_FLASHINFER_MOE_BACKEND` 只管 **MoE**
- `VLLM_NVFP4_GEMM_BACKEND` 只管 **普通 Linear**
- 两者都和 `--performance-mode` 不是一回事

---

## 1. `VLLM_FLASHINFER_MOE_BACKEND` 到底控制什么

它控制的是 **FlashInfer MoE backend 的映射**。

当前代码里的映射关系是：

```python
backend_map = {
    "throughput": FlashinferMoeBackend.CUTLASS,
    "latency": FlashinferMoeBackend.TENSORRT_LLM,
    "masked_gemm": FlashinferMoeBackend.CUTEDSL,
}
```

也就是：

| 环境变量值 | 实际后端 | 适用直觉 |
|---|---|---|
| `latency` | `TensorRT-LLM` | 小 batch / 低延迟倾向 |
| `throughput` | `CUTLASS` | 大 batch / 高吞吐倾向 |
| `masked_gemm` | `CuteDSL` | 更特化的 MoE kernel 路径 |

### 一个很关键的真实行为

虽然名字叫 `latency`，但它**不是在所有 GPU 上都真的走 TRTLLM**。

代码里有显式降级逻辑：

- 如果你设置了 `VLLM_FLASHINFER_MOE_BACKEND=latency`
- 但设备不是 `SM100 family`（不是 Blackwell）
- 会自动回退到 `CUTLASS`

也就是说：

- 在 **B100/B200** 这类 SM100+ 上，`latency` 才真正对应 `TRTLLM`
- 在 **H100/H200** 这类非 SM100 上，`latency` 实际上会退回 `CUTLASS`

所以不能简单理解成：

- `latency` = 一定用 TRTLLM

更准确的说法是：

- `latency` = **优先请求 TRTLLM；不支持时退回 CUTLASS**

---

## 2. 它对哪些 MoE 类型生效

`VLLM_FLASHINFER_MOE_BACKEND` 不是“所有 MoE 一律生效”，而是会进入不同量化路径的 oracle。

### 2.1 Unquantized MoE

无量化 MoE 里：

- `throughput` -> `FlashInfer CUTLASS`
- `latency` -> `FlashInfer TRTLLM`
- `masked_gemm` -> **不支持**

也就是说，`masked_gemm` 不是通用 MoE 模式，它不是无量化路径的合法选择。

### 2.2 FP8 MoE

FP8 MoE 里也是类似：

- `throughput` -> `FLASHINFER_CUTLASS`
- `latency` -> `FLASHINFER_TRTLLM`
- `masked_gemm` -> **不支持 FP8 MoE**

所以如果你是 FP8 MoE，主要就是在：

- `TRTLLM`
- `CUTLASS`

这两条路之间切换。

### 2.3 NVFP4 MoE

NVFP4 MoE 是这几个选项最“完整支持”的地方：

- `throughput` -> `FLASHINFER_CUTLASS`
- `latency` -> `FLASHINFER_TRTLLM`
- `masked_gemm` -> `FLASHINFER_CUTEDSL`

而且 NVFP4 MoE 还有自己更完整的一层 backend 枚举：

- `FLASHINFER_TRTLLM`
- `FLASHINFER_CUTLASS`
- `FLASHINFER_CUTEDSL`
- `FLASHINFER_CUTEDSL_BATCHED`
- `VLLM_CUTLASS`
- `MARLIN`

其中 `VLLM_FLASHINFER_MOE_BACKEND` 只负责把 FlashInfer 那几条路径映射进去。

---

## 3. `latency` / `throughput` / `masked_gemm` 三者的本质区别

## 3.1 `latency` -> TRTLLM

这条路的特点是：

- 更偏融合式 kernel
- 更偏小 batch / 低 launch overhead
- 更依赖新硬件能力

在 NVFP4 MoE 的 TRTLLM 实现里，代码明确显示：

- 只支持 **Blackwell / SM100 family**
- `supports_chunking() -> False`
- `supports_expert_map() -> False`
- `hidden_dim % 512 == 0`
- 激活只支持 `SiLU` 和 `RELU2_NO_MUL`

所以 TRTLLM 路线的典型画像是：

- 优点：小 batch 时更容易压低 kernel launch 和中间搬运开销
- 缺点：限制更多，硬件门槛更高，灵活性更差

## 3.2 `throughput` -> CUTLASS

这条路的特点是：

- 更偏通用、高吞吐
- 支持范围更广
- 对大 batch 更友好

在 FlashInfer CUTLASS MoE 里：

- 支持 `SM90+`、`SM100+`、`SM110+`、`SM120+`
- 支持 unquantized / FP8 / NVFP4 等多种量化路径
- 更适合大 batch 和高并发

可以把它理解成：

- TRTLLM 更像“小 batch 冲延迟的专用快车”
- CUTLASS 更像“大 batch 吃满吞吐的主力卡车”

## 3.3 `masked_gemm` -> CuteDSL

这条路更特化，不应该把它简单看成“第三种通用 latency/throughput 模式”。

它的特点是：

- 对应 `FlashInfer CuteDSL`
- 在当前代码里，最主要是 **NVFP4 MoE** 路线会用到
- 设备要求高，通常要求 `SM100 family`
- 限制更多，例如当前实现里只支持 `SiLU`

因此：

- `masked_gemm` 更像特定内核路线
- 不是普适意义上的“比 throughput 更快”或“比 latency 更慢”

---

## 4. NVFP4 要分成两件事看

如果模型用了 NVFP4，最容易犯的错误是把所有 kernel 都归到一个开关上。

实际上要分成两类：

### 4.1 NVFP4 MoE kernel

这部分由：

- `VLLM_FLASHINFER_MOE_BACKEND`
- `--moe-backend`
- `VLLM_USE_FLASHINFER_MOE_FP4`

这些 MoE 相关配置影响。

它控制的是：

- expert GEMM
- routing + expert 计算的融合方式
- MoE prepare/finalize 路径

### 4.2 NVFP4 Linear kernel

这部分由：

- `VLLM_NVFP4_GEMM_BACKEND`

控制。

它影响的是模型里的普通线性层，例如：

- attention 里的 QKV / out projection
- MLP 的 up / down / gate projection

**它不管 MoE expert kernel。**

所以 NVFP4 下经常会出现这样的组合：

- MoE 层：`FLASHINFER_TRTLLM` 或 `FLASHINFER_CUTLASS`
- 普通 Linear 层：`FlashInferCutlassNvFp4LinearKernel` 或 `MarlinNvFp4LinearKernel`

这两块可以不是同一个 backend 家族。

---

## 5. NVFP4 MoE backend 的区别

## 5.1 自动选择顺序

NVFP4 MoE 的自动优先级是：

1. `FLASHINFER_TRTLLM`
2. `FLASHINFER_CUTEDSL`
3. `FLASHINFER_CUTEDSL_BATCHED`
4. `FLASHINFER_CUTLASS`
5. `VLLM_CUTLASS`
6. `MARLIN`

这说明一件事：

- 在能满足条件的情况下，系统会优先尝试更偏 FlashInfer / 原生 FP4 的实现
- 不满足时再回退到更通用或更慢的路径

## 5.2 `FLASHINFER_TRTLLM`

特点：

- 偏低延迟
- `SM100+` 才能走
- 更融合
- 不支持 chunking
- 不支持 expert_map
- hidden dim 要满足更严格的 shape 约束

适合：

- Blackwell
- 小 batch
- 想优先压 MoE expert 计算延迟

## 5.3 `FLASHINFER_CUTLASS`

特点：

- 偏高吞吐
- 支持面更广
- 大 batch 更友好
- 对高并发或大 token 数更稳

适合：

- H100 / H200 / Blackwell
- 高并发
- 批量生成

## 5.4 `FLASHINFER_CUTEDSL` / `FLASHINFER_CUTEDSL_BATCHED`

特点：

- 更特定的 NVFP4 MoE 路线
- 常和 `masked_gemm` 对应
- 对硬件和激活函数限制更强

什么时候会看到 batched 版本：

- 当部署配置使用 batched expert activation format 时
- 代码会把 `FLASHINFER_CUTEDSL` 转成 `FLASHINFER_CUTEDSL_BATCHED`

## 5.5 `VLLM_CUTLASS`

这是 vLLM 自带的 CUTLASS MoE 路线，不依赖 FlashInfer 的那条包装接口。

特点：

- 更通用
- 通常作为 FlashInfer 路线不满足条件时的备选

## 5.6 `MARLIN`

这是更保底的回退方案。

特点：

- 兼容性强
- 但不是原生 NVFP4 W4A4 高性能路径
- 更像 “能跑、但别期待最优吞吐”

---

## 6. NVFP4 Linear kernel 的区别

普通 NVFP4 Linear 的自动优先级和 MoE 不一样。当前 CUDA 平台的顺序是：

1. `FlashInferCutlassNvFp4LinearKernel`
2. `CutlassNvFp4LinearKernel`
3. `MarlinNvFp4LinearKernel`
4. `FlashInferTrtllmNvFp4LinearKernel`
5. `FlashInferCudnnNvFp4LinearKernel`
6. `FbgemmNvFp4LinearKernel`
7. `EmulationNvFp4LinearKernel`

这说明：

- 对普通 Linear 层，默认并**不是**先选 TRTLLM
- 优先级首先偏向 CUTLASS / 通用高性能实现

这和 MoE 路线不一样，别混了。

## 6.1 `flashinfer-cutlass`

特点：

- 要求 `FlashInfer + SM100+`
- 权重会做 `swizzle + pad`
- 激活量化也走 CUTLASS 对应布局
- 更偏大 batch 吞吐

这是普通 NVFP4 Linear 里最像“主力默认高性能路径”的实现。

## 6.2 `flashinfer-trtllm`

特点：

- 权重做 `shuffle_matrix_a` / `shuffle_matrix_sf_a`
- 调用 `backend="trtllm"` 的 FlashInfer FP4 GEMM
- 更偏小 batch / 低延迟

但注意：

- 在普通 Linear 自动选择顺序里，它不是排第一
- 所以如果你想强制试它，需要显式设置：

```bash
VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm
```

## 6.3 `cutlass`

这是 vLLM 自带 CUTLASS NVFP4 Linear，不依赖 FlashInfer 包装层。

适合：

- 想避开 FlashInfer 依赖
- 仍然希望保留 CUTLASS 路线

## 6.4 `flashinfer-cudnn`

特点：

- 也是 FlashInfer 封装
- 权重布局和 CUTLASS 路线相近
- 在部分 shape 上可能有自己的 autotuning 优势

## 6.5 `marlin`

这个非常重要，因为很多人会误以为 “NVFP4 = 一定是原生 FP4 GEMM”。

其实不是。

`MarlinNvFp4LinearKernel` 的注释写得很明确：

- 它是 **weight-only FP4 compression via Marlin (W4A16)**

也就是说：

- 权重是 FP4 压缩格式
- 计算不是原生 W4A4 NVFP4 tensor core 路径
- 更像兼容性 fallback

文案里甚至还会警告：

- 你的 GPU 没有原生 FP4 计算支持
- 会退回到 Marlin
- 对 compute-heavy workload 可能有性能损失

所以：

- **H100 上如果最终走 Marlin，不代表你“在跑原生 NVFP4”**
- 它更接近 “FP4 weight-only fallback”

## 6.6 `emulation`

这是最后兜底：

- 软件模拟
- 本质是 dequant 到 BF16 再 matmul
- 主要用于调试、验证和没有优化 kernel 可用时保底运行

如果自动选择最后落到它，性能通常会很差。

---

## 7. 两套 NVFP4 选择逻辑，为什么经常看起来矛盾

很多人会问：

- 为什么 MoE 里 `latency` 往往联想到 TRTLLM
- 但普通 Linear 的 NVFP4 自动选择顺序里，TRTLLM 又不靠前

原因是：

- **MoE expert kernel** 和 **普通 Linear kernel** 是两套独立体系
- 它们的 shape、融合方式、瓶颈位置都不一样
- 所以最优 backend 也不一定一样

简单讲：

- MoE 更在意 routing + experts 的融合
- 普通 Linear 更像通用 GEMM 选型问题

所以完全可能出现：

- MoE 层：TRTLLM 最优
- 普通 Linear：CUTLASS 最优

这不是矛盾，是两个子问题。

---

## 8. 实际怎么选

## 8.1 如果你关心的是 MoE

优先看：

- `VLLM_FLASHINFER_MOE_BACKEND`
- `--moe-backend`
- 当前量化类型是不是 NVFP4 / FP8 / unquantized
- GPU 是不是 SM100+

推荐经验：

- **Blackwell + 小 batch 低延迟**：先试 `VLLM_FLASHINFER_MOE_BACKEND=latency`
- **H100/H200 或大 batch 高吞吐**：先试 `VLLM_FLASHINFER_MOE_BACKEND=throughput`
- **你明确在试 CuteDSL 路线**：再试 `masked_gemm`

## 8.2 如果你关心的是 NVFP4 普通 Linear

优先看：

- `VLLM_NVFP4_GEMM_BACKEND`
- 自动选择最终落在哪个 kernel

推荐经验：

- **Blackwell + 高吞吐**：先试 `flashinfer-cutlass`
- **Blackwell + 小 batch 低延迟**：可以对比 `flashinfer-trtllm`
- **没有原生 FP4 能力的 GPU**：大概率会落到 `marlin`
- **调试 / 验证**：才用 `emulation`

---

## 9. 最后给一个最实用的判断法

如果你只想快速判断，不想把所有代码都看一遍，可以记这三句：

1. `VLLM_FLASHINFER_MOE_BACKEND` 只管 **MoE**
2. `VLLM_NVFP4_GEMM_BACKEND` 只管 **普通 NVFP4 Linear**
3. `latency` 这个词在不同层里不一定表示同一个实现，尤其在非 SM100 上，`MoE latency` 可能实际回退成 `CUTLASS`

---

## 代码定位

- `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`
  - `VLLM_FLASHINFER_MOE_BACKEND` -> `TRTLLM/CUTLASS/CUTEDSL` 的映射
- `vllm/model_executor/layers/fused_moe/oracle/unquantized.py`
  - 无量化 MoE 下如何使用 `VLLM_FLASHINFER_MOE_BACKEND`
- `vllm/model_executor/layers/fused_moe/oracle/fp8.py`
  - FP8 MoE 下如何使用 `VLLM_FLASHINFER_MOE_BACKEND`
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`
  - NVFP4 MoE backend 总选择逻辑
- `vllm/model_executor/kernels/linear/__init__.py`
  - `VLLM_NVFP4_GEMM_BACKEND` 和普通 NVFP4 Linear 自动选择逻辑
- `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py`
  - `flashinfer-cutlass` / `flashinfer-trtllm` / `flashinfer-cudnn`
- `vllm/model_executor/kernels/linear/nvfp4/marlin.py`
  - `marlin` 是 W4A16 fallback
- `vllm/model_executor/kernels/linear/nvfp4/emulation.py`
  - `emulation` 是软件模拟兜底
