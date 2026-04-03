# LLM Compressor 中的 GPTQ 与 NVFP4 知识库

> 这份文档偏知识点、工程判断和常见疑问，目标是帮助快速理解 `GPTQ`、`NVFP4`、`NVFP4A16`、`actorder`、`weight_g_idx`、导出格式以及 vLLM 执行路径。  
> 如果你想看 **GPTQ 算法本身如何实现、如何迁移到其他仓库**，请看配套文档：`llm_compressor_gptq_implementation.md`。

---

## 目录

1. [先看结论](#1-先看结论)
2. [GPTQ 是什么，NVFP4 又是什么](#2-gptq-是什么nvfp4-又是什么)
3. [GPTQ 和普通 NVFP4 的关系](#3-gptq-和普通-nvfp4-的关系)
4. [GPTQ + NVFP4 在 LLM Compressor 中怎么配置](#4-gptq--nvfp4-在-llm-compressor-中怎么配置)
5. [NVFP4 和 NVFP4A16 的区别](#5-nvfp4-和-nvfp4a16-的区别)
6. [Activation Ordering 对量化的影响](#6-activation-ordering-对量化的影响)
7. [`weight_g_idx` 是什么，有什么意义](#7-weight_g_idx-是什么有什么意义)
8. [导出格式与保存字段](#8-导出格式与保存字段)
9. [vLLM 加载和执行路径是否一致](#9-vllm-加载和执行路径是否一致)
10. [常见误区与结论](#10-常见误区与结论)

---

## 1. 先看结论

### 1.1 最重要的三句话

1. **GPTQ 是算法，NVFP4 是量化格式。**
2. **GPTQ + NVFP4 和普通 NVFP4 的主导出格式通常一样，差别主要在量化结果数值，而不是 checkpoint 容器本身。**
3. **`actorder=static` 或 `weight` 时，GPTQ+NVFP4 通常和普通 NVFP4 走相同的 vLLM runtime 路径；`actorder=group` 时可能因为 `weight_g_idx` 引入额外映射语义。**

### 1.2 工程上怎么记

- `NVFP4` 决定：权重/激活如何表示成 FP4，scale 怎么存，checkpoint 怎么导出
- `GPTQ` 决定：在这个 FP4 表示空间里，权重值怎么选才更好
- `actorder` 决定：GPTQ 按什么顺序处理列，以及 group qparams 是否跟着重排后的列重新绑定

---

## 2. GPTQ 是什么，NVFP4 又是什么

### 2.1 GPTQ

GPTQ 是一种**基于校准数据和二阶近似的逐层权重量化算法**。  
它和 RTN 的最大不同不是 bit-width，而是：

- RTN：每个值单独 round
- GPTQ：逐列量化，并把当前列误差补偿到后续列

在 `llm-compressor` 里，GPTQ 的实现主入口是：

```python
# src/llmcompressor/modifiers/gptq/base.py
class GPTQModifier(Modifier, QuantizationMixin):
    ...
```

真正执行 GPTQ 数学过程的是：

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py
def quantize_weight(...):
    ...
```

### 2.2 NVFP4

NVFP4 是一种 **4-bit floating-point** 量化格式，在这个仓库的语义下，它不是简单的“裸 FP4 权重”，而是带有：

- `weight_global_scale`
- `weight_scale`
- `weight_zero_point`
- `group_size=16`

也就是说它是一个 **两级尺度体系**：

```text
global_scale + local/group scale
```

NVFP4 在这个仓库里对应的压缩导出格式是：

```text
nvfp4_pack_quantized
```

文档依据：

```77:78:docs/steps/choosing-scheme.md
| NVFP4A16 - float | nvfp4_pack_quantized |
| NVFP4 - float  | nvfp4_pack_quantized   |
```

---

## 3. GPTQ 和普通 NVFP4 的关系

### 3.1 相同点

如果目标 scheme 都是 `NVFP4`，那么两者：

- 最终目标格式一样
- scale 体系一样
- 导出 compressor 一样
- runtime 看到的主量化格式一样

### 3.2 不同点

差异在于“**值怎么选**”：

- 普通 NVFP4：更接近 observer + fake quant / RTN 的方式
- GPTQ + NVFP4：使用 Hessian、逐列量化、误差补偿来决定最终落到哪个 NVFP4 可表示值

所以更准确地说：

> **GPTQ 不是另一种 NVFP4 格式，而是另一种“求 NVFP4 量化结果”的算法。**

### 3.3 一个非常重要的区分

不要把下面两层混在一起：

#### 格式层
- 是不是 `NVFP4`
- 用什么 compressor
- runtime 是不是把它当作 NVFP4 加载

#### 算法层
- 是 RTN、GPTQ、AWQ 还是其他算法得到的这个 NVFP4 模型

格式层一样，不代表数值一样。  
算法层不同，也不一定意味着 runtime 路径不同。

---

## 4. GPTQ + NVFP4 在 LLM Compressor 中怎么配置

最明确的手动配置示例在：

```python
# examples/quantization_w4a4_fp4/llama3_gptq_example.py
NVFP4 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        actorder="static",
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=False,
        group_size=16,
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
        observer="memoryless_minmax",
    ),
    input_activations=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=DynamicType.LOCAL,
        group_size=16,
        observer="static_minmax",
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    ),
    targets=["Linear"],
)
recipe = GPTQModifier(config_groups={"group_0": NVFP4}, ignore=["lm_head"])
```

### 4.1 这段配置的含义

#### weights
- `num_bits=4`：权重 4 bit
- `type=FLOAT`：不是 INT4，而是 FP4
- `strategy=TENSOR_GROUP`：说明既有 tensor 级全局尺度，也有 group 级局部尺度
- `group_size=16`：每 16 个元素一组
- `symmetric=True`：对称量化配置

#### input_activations
- 激活也量化成 FP4
- `dynamic=DynamicType.LOCAL`：局部动态量化
- 离线校准主要为了得到 `input_global_scale`
- 推理时还会动态计算 local activation scale

### 4.2 这段配置为什么代表“GPTQ + NVFP4”

因为：

- `GPTQModifier` 决定算法是 GPTQ
- `weights.type=FLOAT + num_bits=4 + strategy=TENSOR_GROUP + group_size=16` 决定权重格式是 NVFP4 风格
- `input_activations` 决定激活也走 NVFP4 风格的动态 FP4 路径

---

## 5. NVFP4 和 NVFP4A16 的区别

这是最容易被混淆的一组。

### 5.1 一句话结论

- `NVFP4A16`：**权重 FP4，激活 FP16**
- `NVFP4`：**权重 FP4，激活也 FP4**

### 5.2 工程影响

| 项目 | NVFP4A16 | NVFP4 |
|---|---|---|
| 权重量化 | 是 | 是 |
| 激活量化 | 否 | 是 |
| 是否需要校准数据 | 通常不需要 | 需要 |
| 是否需要 `input_global_scale` | 否 | 是 |
| 推理 kernel | W4A16 / mixed precision | W4A4 / 全 FP4 |

### 5.3 代码依据

#### NVFP4A16 示例

```python
# examples/quantization_w4a16_fp4/nvfp4/llama3_example.py
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)
```

#### NVFP4 示例

```python
# examples/quantization_w4a4_fp4/llama3_example.py
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

你会看到：

- `NVFP4A16` 不传 dataset
- `NVFP4` 要传 dataset

根本原因不是权重，而是激活量化需要校准 `input_global_scale`。

---

## 6. Activation Ordering 对量化的影响

这是 GPTQ 里最容易让人困惑的部分。

### 6.1 它到底控制什么

`actorder` 控制的是：

> **GPTQ 在逐列量化时，列是按什么顺序被处理的。**

因为 GPTQ 的误差会传播到后续列，所以顺序一变，最终结果就会变。

### 6.2 三种模式

| actorder | 含义 | 是否重排列 | 是否重算 scale/zp | 是否保存 `weight_g_idx` |
|---|---|---|---|---|
| `static` | 原始列顺序 | 否 | 否 | 否 |
| `weight` | 重要列优先量化 | 是 | 否 | 通常否 |
| `group` | 重要列优先，且 group qparams 跟随重排 | 是 | 是 | 是 |

### 6.3 为什么顺序会影响量化

GPTQ 的误差补偿逻辑决定了：

- 第 i 列量化完后，其误差会传播到后面的列
- 所以后面列进入量化时，值已经被改了

因此：

- 先量化谁
- 谁更早把误差传出去

都会改变最终结果。

### 6.4 `static`

最简单：

- 不重排
- 按原始列顺序量化
- 默认 group 绑定不变

代码上表现为：没有进入 `GROUP` / `WEIGHT` 特殊分支。

### 6.5 `weight`

先按 Hessian 对角线大小排序，把更重要的列放前面：

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py
elif actorder == ActivationOrdering.WEIGHT:
    W, H, perm = _apply_activation_ordering(W, H)
    g_idx = g_idx[perm]
```

量化完成后恢复列顺序：

```python
if actorder == ActivationOrdering.WEIGHT:
    invperm = torch.argsort(perm)
    W = W[:, invperm]
```

它的影响主要是：

- 改变误差传播路径
- 通常不改变最终 checkpoint 的字段结构

### 6.6 `group`

这时不仅重排，还会在重排后的 `W` 上重新计算 `scale/zp`：

```python
if actorder == ActivationOrdering.GROUP:
    W, H, perm = _apply_activation_ordering(W, H)
    scale, zero_point = observer(W)
```

量化完成后恢复列顺序时，还要保存 `weight_g_idx`：

```python
elif actorder == ActivationOrdering.GROUP:
    invperm = torch.argsort(perm)
    W = W[:, invperm]
    g_idx = g_idx[invperm]
    has_gidx = True
```

这意味着：

- 不只是顺序变了
- “每列对应哪组 qparams” 也变了

### 6.7 工程上怎么选

- 想简单、稳妥、运行路径统一：优先 `static`
- 想尝试更强恢复能力，但不想引入额外导出字段：试 `weight`
- 只有在你明确需要“重排后的 group 绑定语义”时，才用 `group`

---

## 7. `weight_g_idx` 是什么，有什么意义

### 7.1 一句话定义

`weight_g_idx` 是：

> **列到 weight group 的映射表。**

它回答的问题不是“这列排第几”，而是：

> **这列应该去查哪一组 `weight_scale` / `weight_zero_point`。**

### 7.2 它影响什么

它影响的是：

- weight-side qparams 绑定
- dequant 解释逻辑
- fused low-bit GEMM 里 weight qparams 的查表

### 7.3 它是不是影响反量化

是的。

即使 runtime 没有显式地把 weight 全部 dequant 成 FP16/FP32，它在 fused kernel 里也必须“按正确的 scale 解释每列权重码值”。

如果 `weight_g_idx` 错了，本质上就是：

- 用错了 group scale
- 数值解释错了
- 输出会偏

### 7.4 如果 weight 和 activation 都是 NVFP4，它还有意义吗

有意义，但只在 `actorder=group` 这类需要它的配置下有意义。

很多人容易误以为：

> “既然 W 和 A 都是 NVFP4，kernel 不就直接乘了吗？”

但 NVFP4 不是“裸 FP4 张量直接乘”，它还依赖：

- weight 的 global scale
- weight 的 local/group scale
- activation 的 global scale
- activation 的动态 local scale

所以 `weight_g_idx` 解决的是：

> **weight 列该绑定哪组 weight qparams**

这和 activation 是否也是 NVFP4 是两个不同问题。

### 7.5 什么时候你通常不用关心它

如果是：

- `actorder=static`
- `actorder=weight`

通常不会持久化 `weight_g_idx`，这时你大多数情况下不用专门考虑它。

---

## 8. 导出格式与保存字段

### 8.1 GPTQ+NVFP4 和普通 NVFP4 的主导出格式

在导出格式层面，两者通常是一样的：

- 都是 `nvfp4_pack_quantized`
- `config.json` 的主量化格式都是 NVFP4

### 8.2 差别主要不在格式，而在数值

也就是说：

- **容器**通常一样
- **导出字段集合**通常也很接近
- **里面的值**不一样

GPTQ 主要会影响：

- `weight`
- `weight_scale`
- `weight_global_scale`
- 以及少数配置下的 `weight_g_idx`

### 8.3 GPTQ 回写的参数

代码里 GPTQ 显式列出的参数是：

```python
_GPTQ_Q_PARAMS = ["weight", "weight_scale", "weight_zero_point", "weight_g_idx"]
```

最终写回模块：

```python
q_param_dict = {
    "weight": W,
    "weight_scale": scale.to(dtype=final_dtype),
    "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
}
if g_idx is not None:
    q_param_dict["weight_g_idx"] = g_idx
```

### 8.4 最靠谱的区分方式

如果你拿到两个 NVFP4 checkpoint，想知道哪个是 GPTQ 压的：

1. 先看 recipe
2. 再看有没有 `weight_g_idx`
3. 不要只看 `quantization_config.format`

因为主格式可能完全一样。

---

## 9. vLLM 加载和执行路径是否一致

这是工程上最关键的判断之一。

### 9.1 在什么条件下一致

如果比较的是：

- 普通 `NVFP4`
- `GPTQ + NVFP4`，且 `actorder=static`
- `GPTQ + NVFP4`，且 `actorder=weight`

并且：

- scheme 都是 `NVFP4`
- 导出格式都是 `nvfp4_pack_quantized`
- 没有额外持久化 `weight_g_idx`
- 使用同一 vLLM 版本和同一硬件环境

那么通常可以理解为：

> **vLLM 的 load 和 execute 路径是一致的。**

### 9.2 这句话更精确的含义

它不是说：

- 权重值一样
- 精度表现一样

而是说：

- vLLM 看到的是同一类 checkpoint 格式
- 会进入同一类 NVFP4 loader
- 会走同一类 NVFP4 kernel 路径

差异主要体现在：

- checkpoint 里参数数值不同

### 9.3 什么时候可能不一致

这些情况可能导致路径不同：

- 一个是 `NVFP4`，另一个是 `NVFP4A16`
- 硬件不支持 activation quantization，runtime 自动退化成 weight-only
- `actorder=group` 且导出含 `weight_g_idx`
- quantization config 细节不同

### 9.4 为什么这个结论重要

如果你把 GPTQ+NVFP4 移植到别的仓库，一个很理想的设计就是：

> **让 GPTQ+NVFP4 和普通 NVFP4 共享同一 runtime 路径。**

这样 runtime 不需要知道模型是不是 GPTQ 压的，只需要知道它是 NVFP4。

---

## 10. 常见误区与结论

### 误区 1：GPTQ 就等于对称量化

不对。

- GPTQ 是算法
- 对称/非对称是 quantization args 的属性

### 误区 2：保存了 `zero_point` 就一定是非对称量化

不对。

在这个仓库里，`zero_point` 是统一量化接口的通用字段。  
即使 `symmetric=True`，它也可能以固定零值或中性占位的形式存在。

### 误区 3：`weight_g_idx` 表示列坐标变了

不对。

最终保存回去的权重列顺序会恢复原样。  
`weight_g_idx` 记录的是：

> **列恢复原位后，该去查哪组 qparams。**

### 误区 4：W/A 都是 NVFP4 时，`weight_g_idx` 就没意义了

不对。

只要 `actorder=group` 让 weight-side 的 group 绑定发生变化，`weight_g_idx` 就仍然有意义。

### 误区 5：GPTQ+NVFP4 一定会走不同的 runtime 路径

不一定。

在 `static / weight` 这些常见设置下，它通常和普通 NVFP4 共享同一路径。  
差异主要是数值，不是执行分支。

---

## 推荐阅读顺序

如果你当前最关心的是：

- **理解 GPTQ + NVFP4 的工程语义**  
  先看：`2 -> 3 -> 5 -> 6 -> 7 -> 8 -> 9`

- **理解 `actorder` 和 `weight_g_idx`**  
  先看：`6 -> 7 -> 9`

- **准备自己实现 GPTQ**  
  继续看：`llm_compressor_gptq_implementation.md`
# LLM Compressor GPTQ 深度解析

> 基于 `vllm-project/llm-compressor` 仓库源码整理，涵盖 GPTQ 算法原理、核心代码、NVFP4 集成方式、配置影响及完整数据流。

---

## 目录

1. [GPTQ 算法概述](#1-gptq-算法概述)
2. [核心代码架构](#2-核心代码架构)
3. [Hessian 统计与积累](#3-hessian-统计与积累)
4. [权重量化与误差补偿](#4-权重量化与误差补偿)
5. [校准阶段精度分析](#5-校准阶段精度分析)
6. [GPTQ + NVFP4 集成](#6-gptq--nvfp4-集成)
7. [NVFP4 vs NVFP4A16](#7-nvfp4-vs-nvfp4a16)
8. [Activation Ordering 配置详解](#8-activation-ordering-配置详解)
9. [模型保存与导出格式](#9-模型保存与导出格式)
10. [GPTQ 支持的精度总览](#10-gptq-支持的精度总览)
11. [完整数据流示例：GPTQ + NVFP4](#11-完整数据流示例gptq--nvfp4)
12. [GPTQ 最小可移植实现](#12-gptq-最小可移植实现)
13. [把 GPTQ 移植到其他仓库时的检查清单](#13-把-gptq-移植到其他仓库时的检查清单)

---

## 1. GPTQ 算法概述

### 1.1 它在优化什么

GPTQ（论文：https://arxiv.org/abs/2210.17323）不是简单最小化 `||W - Q(W)||`，而是试图最小化量化对**层输出**造成的损失。用二阶近似表示：

```
ΔL ≈ 1/2 (w - q)^T H (w - q)
```

其中：
- `w` 是原始权重
- `q` 是量化后权重
- `H` 是和层输入相关的 Hessian 近似

### 1.2 逐列量化 + 误差补偿

GPTQ 的核心是：
- 当前列量化后得到误差
- 利用 `H^{-1}` 的局部信息，把这个误差传播到后续未量化列
- 后面列在量化前已经被"预补偿"过了

这就是 GPTQ 通常比 RTN 更准（尤其在 4-bit 场景）的根本原因。

### 1.3 和 RTN 的本质区别

- **RTN**：每个权重独立 round 到最近可表示值
- **GPTQ**：利用二阶信息，逐列量化并向后传播误差，使整个层输出误差更小

---

## 2. 核心代码架构

### 2.1 文件布局

```
src/llmcompressor/modifiers/gptq/
├── __init__.py
├── base.py              # GPTQModifier 生命周期管理
└── gptq_quantize.py     # GPTQ 数学核心（Hessian、量化、误差补偿）

src/llmcompressor/modifiers/quantization/
├── calibration.py       # observer 初始化、校准 hook、global scale 计算
└── quantization/
    ├── base.py          # QuantizationModifier
    └── mixin.py         # QuantizationMixin（量化配置解析、observer、hook 管理）

src/llmcompressor/pipelines/sequential/
└── pipeline.py          # SequentialPipeline（逐子图校准 + 压缩）
```

### 2.2 GPTQModifier 生命周期

```python
# src/llmcompressor/modifiers/gptq/base.py

class GPTQModifier(Modifier, QuantizationMixin):
    """
    Lifecycle:
    - on_initialize
        - apply config to model
    - on_start
        - add activation calibration hooks
        - add gptq weight calibration hooks
    - on_sequential_epoch_end
        - quantize_weight
    - on_finalize
        - remove_hooks()
        - model.apply(freeze_module_quantization)
    """
```

关键参数：
- `sequential_targets`：逐层压缩的目标层名
- `block_size`：GPTQ 列块大小（默认 128）
- `dampening_frac`：Hessian 对角线阻尼系数（默认 0.01）
- `actorder`：列量化顺序（默认 "static"）
- `offload_hessians`：是否将 Hessian 卸载到 CPU 以减少显存

---

## 3. Hessian 统计与积累

### 3.1 创建空 Hessian

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py

GPTQ_PRECISION = torch.float32

def make_empty_hessian(
    module: torch.nn.Module, device: torch.device | None = None
) -> torch.Tensor:
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)
```

Hessian 的维度是 `[in_features, in_features]`，刻画输入维度上的二阶相关性。

### 3.2 积累 Hessian

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py

def accumulate_hessian(
    inp: torch.Tensor,
    module: torch.nn.Module,
    H: torch.Tensor | None,
    num_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)

    num_added = inp.shape[0]

    match module:
        case torch.nn.Linear() | transformers.Conv1D():
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        case torch.nn.Conv2d():
            unfold = torch.nn.Unfold(...)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

    num_samples += num_added

    # 关键：转成 FP32 做统计
    inp = inp.to(dtype=GPTQ_PRECISION)
    inp = math.sqrt(2) * inp
    H += inp.matmul(inp.t())

    return H, num_samples
```

本质是在累计 `X X^T` 二阶统计量，最后除以样本数得到 Hessian 近似。

### 3.3 校准 hook

```python
# src/llmcompressor/modifiers/gptq/base.py

def calibrate_module(self, module, args, _output):
    inp = args[0]

    if module not in self._num_samples:
        init_device = "cpu" if self.offload_hessians else get_execution_device(module)
        self._hessians[module] = make_empty_hessian(module, device=init_device)
        self._num_samples[module] = torch.zeros(tuple(), device=get_execution_device(module))

    with self._maybe_onload_hessian(module):
        self._hessians[module], self._num_samples[module] = accumulate_hessian(
            inp, module, self._hessians[module], self._num_samples[module],
        )
```

每个 batch 的输入激活都会被 hook 捕获，送入 `accumulate_hessian()` 积累统计量。

---

## 4. 权重量化与误差补偿

### 4.1 quantize_weight() 主流程

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py

def quantize_weight(module, quant_args, hessian, blocksize=128, percdamp=0.01):
    W = module.weight.clone()
    H = hessian

    # 1. 创建 observer 计算量化参数
    observer = Observer.load_from_registry(...)
    W = W.to(dtype=GPTQ_PRECISION)  # 转 FP32
    scale, zero_point = observer(W)

    # 2. 处理 activation ordering（见第 8 节）

    # 3. 计算逆 Hessian
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    # 4. 逐列块量化 + 误差补偿（见 4.2）

    # 5. 返回量化结果
    return (loss, q_param_dict)
```

### 4.2 逐列量化 + 误差补偿（核心中的核心）

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py  行 167-254

for i1 in range(0, num_columns, blocksize):
    i2 = min(i1 + blocksize, num_columns)
    count = i2 - i1

    W1 = W[:, i1:i2].clone()
    Q1 = torch.zeros_like(W1)
    Err1 = torch.zeros_like(W1)
    Hinv1 = Hinv[i1:i2, i1:i2]

    for i in range(count):
        w = W1[:, i]           # 当前列原始值
        d = Hinv1[i, i]        # 当前列在逆 Hessian 上的对角项
        q = w.clone()

        # 按目标格式量化当前列
        q = fake_quantize(q, scale, zero_point, quant_args, global_scale=global_scale)

        # 记录量化结果
        Q1[:, i] = q
        losses1[:, i] = (w - q) ** 2 / d**2

        # === 误差补偿（GPTQ 的核心） ===
        err1 = (w - q) / d
        w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        W1[:, i:] -= w1_err    # 修改后续列的浮点值
        Err1[:, i] = err1

    # 块级误差传播
    W[:, i1:i2] = Q1
    losses += torch.sum(losses1, 1) / 2
    w_err = Err1.matmul(Hinv[i1:i2, i2:])
    W[:, i2:] -= w_err         # 将误差传播到后续块
```

**误差补偿的含义：**
- `err1 = (w - q) / d`：归一化误差
- `Hinv[i, i:]`：当前列误差和后续列之间的耦合强弱
- `W1[:, i:] -= w1_err`：修改**工作副本中未量化列的浮点值**
- 目的：让后续列在量化时，能部分抵消当前列造成的输出误差

**GPTQ 的补偿不是修改已量化列或 scale，而是修改后续未量化列的浮点值。**

### 4.2.1 把这段代码翻译成算法步骤

如果不看工程细节，只看算法本体，`quantize_weight()` 的真正流程可以翻译成下面 10 步：

1. 取出目标层权重 `W`
2. 把 `W` 规范成二维矩阵 `[out_features, in_features]`
3. 用 observer 或量化器预先求出目标格式需要的 `scale / zero_point / global_scale`
4. 如果启用了 `actorder`，先对列和 Hessian 做同样的重排
5. 对 Hessian 做阻尼、Cholesky、逆，得到一个可用于误差传播的 `Hinv`
6. 按列块遍历输入维度
7. 在每个块内逐列量化当前列 `w -> q`
8. 计算当前列误差 `(w - q)`，并用 `Hinv` 投影到后续列
9. 把补偿后的后续列继续送入下一轮量化
10. 恢复原始列顺序和原始张量形状，返回量化参数

从可移植性角度看，真正不可缺的只有四个对象：

- `W`: 当前层权重矩阵
- `H`: 输入激活导出的二阶统计
- `quantizer`: 给定 `w` 返回量化后的 `q`
- `Hinv`: 控制误差如何传播到后续列

只要别的仓库里能构造这四样，GPTQ 就能移植。

### 4.2.2 最核心的数学对象分别在干什么

#### `W`
- 代表当前层待量化的权重副本
- 在 GPTQ 里不是只读的
- 会在量化过程中被持续修改

#### `Q1`
- 代表当前块里已经决定下来的量化结果
- 一旦一列被写进 `Q1`，这一列就不会再被改

#### `Err1`
- 代表当前块内每一列量化后产生的误差向量
- 这些误差稍后会继续传播到块外后续列

#### `Hinv`
- 不是普通“矩阵求逆结果”的装饰品
- 它是 GPTQ 里决定“当前列误差应该沿什么方向分摊到后续列”的核心结构

#### `d = Hinv[i, i]`
- 当前列自己的敏感度归一化项
- 用于把误差做尺度标准化

#### `g_idx`
- 列到 group 的映射
- 只在 group / tensor_group 策略下需要
- 如果列顺序被重排但 group 绑定也发生了变化，就可能需要保存

### 4.2.3 逐列补偿的直观解释

假设当前块只有三列：`c0, c1, c2`。

#### 第 1 步：量化第一列

```text
w0 -> q0
e0 = w0 - q0
```

#### 第 2 步：不立刻去量化 `c1`

GPTQ 不会直接拿原始 `c1` 去量化，而是先做：

```text
c1' = c1 - compensate(e0)
c2' = c2 - compensate(e0)
```

#### 第 3 步：量化第二列

```text
w1 = c1'
q1 = Quantize(w1)
e1 = w1 - q1
```

#### 第 4 步：再把第二列误差继续传给后面的列

```text
c2'' = c2' - compensate(e1)
```

#### 第 5 步：量化第三列

```text
w2 = c2''
q2 = Quantize(w2)
```

最终得到的是：

```text
[q0, q1, q2]
```

但注意：

- `q1` 不是原始 `c1` 的直接量化结果
- `q2` 也不是原始 `c2` 的直接量化结果
- 它们都是在“前面列误差已经注入后”的结果

这就是 GPTQ 和 RTN 的本质不同。

### 4.2.4 代码中“修改权重”的位置到底在哪里

GPTQ 里真正改浮点权重副本的就是这两处：

```python
# 块内传播
W1[:, i:] -= w1_err

# 块间传播
W[:, i2:] -= w_err
```

这两行意味着：

- GPTQ 不是“只算一个更好的 rounding”
- 它是真的把未量化列的浮点值改了
- 最后输出的量化结果是沿着这条补偿路径得到的

### 4.2.5 把 llm-compressor 里的实现抽象成可移植伪代码

下面这段伪代码可以直接当作迁移到其他仓库时的骨架：

```python
def gptq_quantize_weight(W, H, quantizer, block_size=128, damp=0.01):
    # W: [out_features, in_features], float32 working copy
    # H: [in_features, in_features], Hessian approximation
    # quantizer: object/function that maps float column -> quantized float column

    W = W.clone().float()
    H = H.clone().float()

    # damped inverse Hessian
    diag_idx = torch.arange(H.shape[0], device=H.device)
    H[diag_idx, diag_idx] += damp * torch.mean(torch.diag(H))
    U = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(U)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)

    for i1 in range(0, W.shape[1], block_size):
        i2 = min(i1 + block_size, W.shape[1])
        W_blk = W[:, i1:i2].clone()
        Hinv_blk = Hinv[i1:i2, i1:i2]
        Err_blk = torch.zeros_like(W_blk)

        for i in range(i2 - i1):
            w = W_blk[:, i]
            d = Hinv_blk[i, i]

            q = quantizer(w)
            Q[:, i1 + i] = q

            err = (w - q) / d
            Err_blk[:, i] = err

            # propagate to columns inside current block
            W_blk[:, i:] -= err.unsqueeze(1) @ Hinv_blk[i, i:].unsqueeze(0)

        # write quantized block back
        W[:, i1:i2] = Q[:, i1:i2]

        # propagate to columns outside current block
        if i2 < W.shape[1]:
            W[:, i2:] -= Err_blk @ Hinv[i1:i2, i2:]

    return Q
```

如果你要迁移到别的仓库，这段骨架加上一个适配本仓库量化格式的 `quantizer(w)`，就已经很接近可用实现了。

### 4.2.6 在其他仓库里如何实现 `quantizer(w)`

把 GPTQ 挪到别的仓库时，最容易耦合的是“目标格式量化器”。推荐把它拆成独立函数：

```python
def quantizer(w, qparams):
    # 1. 根据列查到 scale / zp / global_scale
    # 2. 把 w 映射到目标低比特格式可表示集合
    # 3. 返回 dequant 后的浮点近似值 q
    return q
```

对于不同格式：

- INT4/INT8：`q = dequant(quant(w))`
- FP4/NVFP4：`q = map_to_fp4_grid(w, scale, global_scale)`
- block/group 策略：先查 group/block 对应的 qparams，再量化

这样 GPTQ 主体根本不用知道目标格式是 int 还是 float。

### 4.3 数值稳定性

```python
try:
    damp = percdamp * torch.mean(torch.diag(H))
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
except torch._C._LinAlgError:
    # 不稳定时退化为单位阵，相当于回退到更接近 RTN 的行为
    Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)
```

- 先对对角线加阻尼
- 用 Cholesky 做稳定求逆
- 如果还是不稳定，回退为单位阵

### 4.4 压缩模块调用

```python
# src/llmcompressor/modifiers/gptq/base.py

def compress_module_list(self, module_list):
    for module in module_list:
        quant_args = getattr_chain(module, "quantization_scheme.weights")
        with torch.no_grad(), align_module_device(module), ...:
            loss, q_param_dict = quantize_weight(
                module=module,
                quant_args=quant_args,
                hessian=self._hessians.pop(module) / self._num_samples.pop(module),
                blocksize=self.block_size,
                percdamp=self.dampening_frac,
            )
        for attr, val in q_param_dict.items():
            update_offload_parameter(module, attr, val)
```

注意 `hessian / num_samples`：前面累计的是总量，这里才做平均。

---

## 5. 校准阶段精度分析

### 5.1 前向 GEMM 精度

在 GPTQ 校准期间，pipeline 会**禁用量化前向**：

```python
# src/llmcompressor/utils/helpers.py
DISABLE_QAC_MODIFIERS = ["GPTQModifier", "AWQModifier", "AutoRoundModifier"]

# src/llmcompressor/pipelines/sequential/pipeline.py
disable_qac = any(
    type(mod).__name__ in DISABLE_QAC_MODIFIERS
    for mod in session.lifecycle.recipe.modifiers
)
if not dataset_args.quantization_aware_calibration or disable_qac:
    stack.enter_context(DisableQuantization(model))
```

所以校准阶段的 `Linear(x, W)` 还是**原始权重 dtype 的 GEMM**（通常 bf16/fp16），不是低比特 GEMM。

### 5.2 GPTQ 内部计算精度

```python
GPTQ_PRECISION = torch.float32
```

- Hessian 统计：`FP32`
- 权重工作副本：`FP32`
- Cholesky/逆 Hessian：`FP32`
- 误差传播：`FP32`

### 5.3 精度总结

| 阶段 | dtype |
|------|-------|
| 校准前向 GEMM | 原模型 dtype（bf16/fp16） |
| Hessian 统计 matmul | FP32 |
| 逆 Hessian 求解 | FP32 |
| GPTQ 逐列优化 | FP32 |
| fake_quantize 映射 | FP32 -> 目标格式可表示值 |
| 最终回写模块 | cast 回原始 dtype |

---

## 6. GPTQ + NVFP4 集成

### 6.1 NVFP4 量化配置

```python
# examples/quantization_w4a4_fp4/llama3_gptq_example.py

NVFP4 = dict(
    weights=QuantizationArgs(
        num_bits=4,
        actorder="static",
        type=QuantizationType.FLOAT,          # 浮点量化，不是整数
        strategy=QuantizationStrategy.TENSOR_GROUP,  # 两级尺度体系
        symmetric=True,
        dynamic=False,
        group_size=16,                        # 每 16 个元素一组
        scale_dtype=FP8_E4M3_DATA.dtype,      # scale 用 FP8 保存
        zp_dtype=FP8_E4M3_DATA.dtype,
        observer="memoryless_minmax",
    ),
    input_activations=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=DynamicType.LOCAL,            # 局部动态量化
        group_size=16,
        observer="static_minmax",
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    ),
    targets=["Linear"],
)
recipe = GPTQModifier(config_groups={"group_0": NVFP4}, ignore=["lm_head"])
```

### 6.2 NVFP4 的两级尺度体系

NVFP4 不是简单的整数 pack，依赖**全局 scale + 局部分组 scale**：

```
原始权重 W
  -> 先确定一个 tensor 级别的 global_scale
  -> 再按 group_size=16 划分小组
  -> 每组求本地 scale / zp
  -> 每个值映射到 NVFP4 的 4-bit float 可表示集合
```

### 6.3 global_scale 的计算

对于 `TENSOR_GROUP` 策略，必须先有 `global_scale`：

```python
# src/llmcompressor/observers/base.py

def _check_has_global_scale(self, global_scale):
    if self.args.strategy == QuantizationStrategy.TENSOR_GROUP and global_scale is None:
        raise ValueError("Cannot compute scale and zero points without first computing global scale")

def _get_global_scale_with_minmax(self, observed):
    observed = observed.reshape((1, 1, -1))  # per tensor reshape
    global_min_vals, global_max_vals = self.get_global_min_max(observed)
    global_scale = generate_gparam(global_min_vals, global_max_vals)
    return global_scale, global_min_vals, global_max_vals
```

触发时机：

```python
# src/llmcompressor/modifiers/quantization/calibration.py

def update_weight_global_scale(module):
    if getattr_chain(module, "quantization_scheme.weights.strategy", None) != QuantizationStrategy.TENSOR_GROUP:
        return
    call_observer(module, base_name="weight", should_calculate_gparam=True, should_calculate_qparams=False)
```

### 6.4 fused global scale（vLLM 兼容要求）

某些 fused 权重组要共享同一个 global scale：

```python
# src/llmcompressor/modifiers/utils/helpers.py

def update_fused_layer_weight_global_scales(submodule):
    """
    When running NVFP4 quantization, update the global scale such that
    q,k,v layers are treated as one tensor with the same global_scale
    and gate_proj/up_proj layers are treated as one tensor with the
    same global scale.
    """
    # attention: q/k/v 取最小值共享
    if is_attention_module(submodule):
        global_scale = torch.min(torch.cat((
            submodule.q_proj.weight_global_scale.data,
            submodule.k_proj.weight_global_scale.data,
            submodule.v_proj.weight_global_scale.data,
        ))).reshape([1])
        # 写回 q/k/v 共享同一个 global_scale

    # MLP: gate/up 取最小值共享
    if _is_mlp_module(submodule):
        global_scale = torch.min(torch.cat((
            submodule.gate_proj.weight_global_scale.data,
            submodule.up_proj.weight_global_scale.data,
        ))).reshape([1])
        # 写回 gate/up 共享同一个 global_scale
```

### 6.5 GPTQ 在 NVFP4 里怎么"选值"

GPTQ 在量化 NVFP4 权重时，对每一列调用 `fake_quantize`，并传入 `global_scale`：

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py

global_scale = getattr(module, "weight_global_scale", None)

# 对 TENSOR_GROUP 策略
elif strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
    group_index = g_idx[column_idx]
    altered_qargs = copy(quant_args)
    altered_qargs.strategy = QuantizationStrategy.CHANNEL

    q = fake_quantize(
        q,
        scale[:, group_index],
        zero_point[:, group_index],
        altered_qargs,
        global_scale=global_scale,
    )
```

`fake_quantize` 内部会把值映射到 NVFP4 的 4-bit float 可表示集合，受 `global_scale` 和 `local group scale` 共同约束。

### 6.6 GPTQ 和普通 NVFP4 的核心区别

**两者目标格式相同，优化过程不同：**

| | 普通 NVFP4 (RTN) | GPTQ + NVFP4 |
|---|---|---|
| 量化格式 | NVFP4 | NVFP4 |
| scale 体系 | global_scale + group scale | global_scale + group scale |
| 是否需要校准数据 | 不需要（NVFP4A16）或少量（NVFP4） | 需要（构建 Hessian） |
| 值的选择方式 | 独立 round 到最近可表示值 | Hessian 引导逐列优化 + 误差补偿 |
| 是否修改后续列 | 否 | 是 |

**GPTQ 不是换了格子，而是更聪明地选格点。**

### 6.7 从“算法层”和“格式层”分开理解 GPTQ+NVFP4

这是最容易混的地方，建议强行拆成两层：

#### 格式层
NVFP4 定义的是：
- 权重如何用 4-bit float 表示
- scale 怎么存
- `global_scale + local scale` 怎么配合
- 导出时怎么 pack 成 `nvfp4_pack_quantized`

#### 算法层
GPTQ 定义的是：
- 列按什么顺序量化
- Hessian 怎么统计
- 当前列误差怎么传给后续列
- 在固定 NVFP4 网格下，选哪个可表示值更合适

所以：

- `NVFP4` 决定“你能站在哪些格点上”
- `GPTQ` 决定“你应该站到哪个格点上”

### 6.8 如果把 GPTQ+NVFP4 移植到别的仓库，最少要带走什么

最少要带走 6 件事：

1. **权重 qparams 生成器**
   - 要支持 `global_scale`
   - 要支持 `group_size=16`
   - 要支持 float4/NVFP4 可表示集合

2. **激活 qparams 生成器**
   - 对 `NVFP4` 而言，要有 `input_global_scale`
   - 动态 local scale 可以在 runtime 做

3. **Hessian 统计器**
   - 能从 layer input 生成 `H`

4. **GPTQ 主循环**
   - 就是前一节的伪代码

5. **导出格式适配**
   - 如果目标 runtime 是 vLLM/Blackwell，需要和 `nvfp4_pack_quantized` 等价的布局

6. **runtime 侧解释逻辑**
   - 至少要保证 weight qparams、activation qparams、可能的 `weight_g_idx` 都能被正确使用

---

## 7. NVFP4 vs NVFP4A16

### 7.1 核心区别：激活是否参与量化

| | **NVFP4A16** | **NVFP4** |
|---|---|---|
| 权重 | FP4，group_size=16，带 global scale | FP4，group_size=16，带 global scale |
| 激活 | **保持 FP16，不量化** | **也量化成 FP4，动态 per-group** |
| `input_activations` 配置 | `None` | 有，4-bit float，dynamic=LOCAL |
| 需要 dataset | 不需要 | 需要（校准激活 global scale） |
| 压缩格式 | `nvfp4_pack_quantized` | `nvfp4_pack_quantized` |
| 推理 GEMM | FP4 weight × FP16 activation | FP4 weight × FP4 activation |
| 能否走全 FP4 kernel | 不能 | 能（仅 Blackwell SM100+） |

名字里的 `A16` 就是 "Activation 16-bit" 的意思。

### 7.2 NVFP4A16 示例

```python
# examples/quantization_w4a16_fp4/nvfp4/llama3_example.py

recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)  # 不需要 dataset
```

### 7.3 NVFP4 示例

```python
# examples/quantization_w4a4_fp4/llama3_example.py

recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
oneshot(
    model=model, dataset=ds, recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
```

### 7.4 激活校准的代码逻辑

当 `input_activations` 存在且 `dynamic=DynamicType.LOCAL` 时，系统会给模块创建 input observer 和注册校准 hook：

```python
# src/llmcompressor/modifiers/quantization/quantization/mixin.py

def _initialize_observers(self, module):
    scheme = module.quantization_scheme
    input = scheme.input_activations and scheme.input_activations.dynamic in (False, DynamicType.LOCAL)
    if input:
        initialize_observer(module, base_name="input")

def _initialize_hooks(self, module):
    if input:
        hooks.add(self.register_hook(module, calibrate_input_hook, "forward_pre"))
```

校准时只算 `input_global_scale`，不算局部 scale/zp：

```python
# src/llmcompressor/modifiers/quantization/calibration.py

def calibrate_activations(module, value, base_name):
    quantization_args = getattr_chain(module, args_attr, None)

    calculate_qparams = True
    calculate_gparam = False

    if quantization_args is not None:
        if quantization_args.dynamic in (True, DynamicType.LOCAL):
            calculate_qparams = False     # 局部 scale 推理时动态算
        if quantization_args.strategy == QuantizationStrategy.TENSOR_GROUP:
            calculate_gparam = True       # global_scale 需要离线校准

    call_observer(module, base_name, value,
                  should_calculate_gparam=calculate_gparam,
                  should_calculate_qparams=calculate_qparams)
```

### 7.5 推理时的行为

- **NVFP4A16**：runtime 读取 FP4 权重，激活保持 FP16，走 mixed-precision kernel
- **NVFP4**：runtime 读取 FP4 权重 + 预校准的 `input_global_scale`，每个 batch 动态算激活的 local scale 后做 FP4 量化，走 **W4A4 FP4 GEMM kernel**
- 在非 SM100 硬件上，vLLM 会自动退化为只做 weight-only 量化

### 7.6 这部分对移植的意义

如果你想把 GPTQ+NVFP4 挪到别的仓库，需要先决定你要支持哪一种：

#### 只支持 `NVFP4A16`
优点：
- 实现简单
- 不需要激活校准
- 不需要 runtime 动态 FP4 activation quantization

缺点：
- 推理时不是全 FP4 路径

#### 支持 `NVFP4`
优点：
- 可以对接真正的 W4A4 路径
- 更贴近 Blackwell 上的高吞吐目标

缺点：
- 需要离线校准 `input_global_scale`
- 需要 runtime 支持动态 local activation scale

因此从移植复杂度上：

> `NVFP4A16` 更像第一阶段实现目标，`NVFP4` 更像完整实现目标。

---

## 8. Activation Ordering 配置详解

### 8.1 三种模式

`actorder` 控制 GPTQ 量化时列的处理顺序。只在 `GROUP / TENSOR_GROUP` 策略下生效。

| actorder | 列是否重排 | scale/zp 是否在重排后重算 | 是否保存 `weight_g_idx` |
|---|---|---|---|
| `static` | 否 | 否 | 否 |
| `weight` | 是 | 否 | 否 |
| `group` | 是 | 是 | 是 |

### 8.2 static（默认）

不做列重排，用原始列顺序做 GPTQ：

```python
# GPTQModifier 默认
actorder: Optional[Union[ActivationOrdering, Sentinel]] = Sentinel("static")
```

代码中没有 `STATIC` 的显式分支，不进入 `GROUP` / `WEIGHT` 分支即为 static 行为。

### 8.3 weight

先按 Hessian 对角线大小对列做重排（优先处理更重要的列），再做 GPTQ：

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py

elif actorder == ActivationOrdering.WEIGHT:
    W, H, perm = _apply_activation_ordering(W, H)
    g_idx = g_idx[perm]
```

量化完成后恢复原始列顺序：

```python
if actorder == ActivationOrdering.WEIGHT:
    invperm = torch.argsort(perm)
    W = W[:, invperm]
```

**通常不保存 `weight_g_idx`**。更准确地说，`weight` 模式里：
- `scale/zp` 是在**重排前**的 `W` 上计算出来的
- 量化过程中会临时把 `g_idx` 跟着列一起重排，用于在 permuted 视角下给当前列取对的 group qparams
- 量化完成后只恢复 `W` 的原始列顺序，不持久化 `g_idx`

因此从最终 checkpoint 的视角看，不需要额外保存一个非恒等的列到 group 的映射表。

### 8.4 group

也会按重要性重排列，但关键不同是**对重排后的权重重新计算 scale/zp**：

```python
if actorder == ActivationOrdering.GROUP:
    W, H, perm = _apply_activation_ordering(W, H)
    # 对重排后的 W 重新计算 scale/zp
    scale, zero_point = observer(W)
```

量化完成后恢复列顺序，但此时 group 映射不再是默认顺序：

```python
elif actorder == ActivationOrdering.GROUP:
    invperm = torch.argsort(perm)
    W = W[:, invperm]
    g_idx = g_idx[invperm]
    has_gidx = True  # 必须保存
```

### 8.5 `_apply_activation_ordering` 实现

```python
def _apply_activation_ordering(W, H):
    perm = torch.argsort(torch.diag(H), descending=True)
    return W[:, perm], H[perm][:, perm], perm
```

按 Hessian 对角线（反映每个输入维度的"重要性"）降序排列。

### 8.6 `weight_g_idx` 是什么

`weight_g_idx` 是"列到量化 group 的映射表"。

正常情况（static/weight）下，映射是天然顺序的：
```
列:        c0  c1  c2  c3  c4  c5  c6  c7
group:      0   0   0   0   1   1   1   1
```

但 `group` 模式下，因为 scale/zp 是基于重排后的列计算的，恢复列顺序后映射变成非顺序：
```
列:        c0  c1  c2  c3  c4  c5  c6  c7
g_idx:      0   1   0   1   1   0   1   0
```

**重要：`weight_g_idx` 不是记录"列被挪到了哪里"，而是记录"恢复到原位后，每列该去查哪个 group 的 scale"。**

### 8.7 测试验证

```python
# tests/llmcompressor/transformers/gptq/test_gptq_oneshot.py

if weight_args.actorder == ActivationOrdering.GROUP:
    assert hasattr(targetted_linear_layer, "weight_g_idx"), "GROUP actorder should have g_idx"
elif weight_args.actorder == ActivationOrdering.WEIGHT:
    assert not hasattr(targetted_linear_layer, "weight_g_idx"), "WEIGHT actorder should not have g_idx"
```

### 8.8 actorder 对量化结果到底影响了什么

这里尽量说得更工程化一些。

#### `static` 影响的是：
- 误差传播路径按原始列顺序展开
- qparams 和原始 group 布局绑定
- checkpoint 最简单

#### `weight` 影响的是：
- 更重要的列会更早被量化
- 它们的误差会更早传播到不重要列
- 最终数值可能不同，但导出结构通常和 `static` 一样简单

#### `group` 影响的是：
- 不只是“谁先量化”
- 而是“重排后的列集合对应哪一组 scale”
- 因此不仅量化路径变了，qparams 绑定语义也变了

从移植视角看：

- `static` 最容易实现
- `weight` 次之
- `group` 除了 GPTQ 主体，还需要保证 runtime/导出逻辑能理解 `weight_g_idx`

---

## 9. 模型保存与导出格式

### 9.1 保存流程

```python
# src/llmcompressor/transformers/compression/compressed_tensors_utils.py

def save_pretrained_wrapper(save_directory, quantization_format=None, save_compressed=True, **kwargs):
    compressor = ModelCompressor.from_pretrained_model(model, quantization_format=quantization_format)
    if save_compressed:
        compressor.compress_model(model)

    original_save_fn.__get__(model, model_class)(save_directory, **kwargs)
    compressor.update_config(save_directory)
    update_and_save_recipe(model.name_or_path, save_directory)
```

### 9.2 NVFP4 对应的压缩器

不管是普通 NVFP4 还是 GPTQ+NVFP4，只要 scheme 是 NVFP4，导出都用 `nvfp4_pack_quantized`：

| Quantization | Quant Compressor |
|---|---|
| NVFP4A16 - float | nvfp4_pack_quantized |
| NVFP4 - float | nvfp4_pack_quantized |

### 9.3 GPTQ 产出的模块参数

```python
_GPTQ_Q_PARAMS = ["weight", "weight_scale", "weight_zero_point", "weight_g_idx"]
```

最终写回模块：

```python
q_param_dict = {
    "weight": W,
    "weight_scale": scale.to(dtype=final_dtype),
    "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
}
if g_idx is not None:
    q_param_dict["weight_g_idx"] = g_idx
```

### 9.4 GPTQ 和普通 NVFP4 保存格式对比

| | 普通 NVFP4 | GPTQ + NVFP4 |
|---|---|---|
| 压缩器 | nvfp4_pack_quantized | nvfp4_pack_quantized |
| config.json 主格式 | 相同 | 相同 |
| weight | 有 | 有（数值不同） |
| weight_scale | 有 | 有（数值可能不同） |
| weight_zero_point | 有 | 有 |
| weight_global_scale | 有 | 有 |
| weight_g_idx | 无 | 可能有（actorder=group 时） |
| recipe | 记录 QuantizationModifier | 记录 GPTQModifier |

### 9.5 static / weight 下和普通 NVFP4 的运行路径关系

如果比较的是：
- 普通 `NVFP4`
- `GPTQ + NVFP4`，且 `actorder=static`
- `GPTQ + NVFP4`，且 `actorder=weight`

那么在以下前提成立时：
- scheme 都是 `NVFP4`，不是 `NVFP4A16`
- 导出格式都是 `nvfp4_pack_quantized`
- 没有额外持久化 `weight_g_idx`
- 使用同一 vLLM 版本和同一硬件环境

可以把它们理解为：

> **vLLM 的 load 和执行路径通常是一致的。**

更具体地说：
- 加载器看到的是同一类 NVFP4 压缩格式
- 运行时看到的是同一类权重字段结构
- 如果是 Blackwell / SM100+，都可以走同一类 `W4A4` NVFP4 kernel 路径
- 差异主要来自 checkpoint 中保存的**数值**不同，而不是 runtime 分支不同

真正可能让执行路径不同的因素通常是：
- 一个 checkpoint 是 `NVFP4`，另一个是 `NVFP4A16`
- 硬件不支持 activation quantization，runtime 自动退化为 weight-only
- `actorder=group` 并保存了 `weight_g_idx`
- 两边的 quantization config 实际并不相同

### 9.5.1 为什么这个结论对移植很重要

如果你把 GPTQ+NVFP4 移植到别的仓库，一个很关键的设计目标是：

> **尽量让“普通 NVFP4”和“GPTQ+NVFP4”共享同一套 runtime 路径。**

这样做的好处是：
- runtime 不需要知道模型是否由 GPTQ 生成
- runtime 只需要知道 checkpoint 是 `NVFP4`
- GPTQ 的差异仅体现在 checkpoint 里的数值内容，而不是执行分支

`llm-compressor` 当前在 `actorder=static/weight` 下基本就是这个思路：
- 导出格式一致
- vLLM load 路径通常一致
- 差异主要体现在数值，不在于调用不同 kernel

### 9.6 如何判断 checkpoint 来源

1. **最靠谱**：看 recipe 文件，有 `GPTQModifier` 就是 GPTQ
2. **辅助证据**：safetensors 里有 `weight_g_idx` 强烈暗示 GPTQ
3. 仅靠 `config.json` 的 `quantization_config` 通常无法区分

---

## 10. GPTQ 支持的精度总览

### 10.1 算法内部计算精度

固定 `FP32`（`GPTQ_PRECISION = torch.float32`）。

### 10.2 量化后目标精度

| 方案 | 权重精度 | 激活精度 | 数值类型 | 示例 scheme |
|---|---|---|---|---|
| W4A16 | 4-bit int | 16-bit float | INT | `"W4A16"` |
| W8A16 | 8-bit int | 16-bit float | INT | `"W8A16"` |
| W8A8-INT8 | 8-bit int | 8-bit int | INT | `"W8A8"` |
| W4AFP8 | 4-bit int | FP8 | INT+FLOAT | `"W4AFP8"` |
| MXFP4A16 | 4-bit float | 16-bit float | FLOAT | `"MXFP4A16"` |
| NVFP4A16 | 4-bit float | 16-bit float | FLOAT | `"NVFP4A16"` |
| NVFP4 | 4-bit float | 4-bit float | FLOAT | 自定义 config_groups |

### 10.3 GPTQ 支持的量化策略

代码中 `quantize_weight()` 的 `fake_quantize` 分支覆盖：

```python
if strategy == QuantizationStrategy.TENSOR: ...
elif strategy == QuantizationStrategy.CHANNEL: ...
elif strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP): ...
elif strategy == QuantizationStrategy.BLOCK: ...
else:
    raise ValueError(f"Quantization strategy is not supported for GPTQ: {strategy}")
```

---

## 11. 完整数据流示例：GPTQ + NVFP4

以 `GPTQ + NVFP4`（权重和激活都量化到 FP4）为例：

```
[1] 加载原始模型
    model.weight: bf16/fp16
    |
    v
[2] 挂载 GPTQ + NVFP4 量化配置
    weights: num_bits=4, type=float, strategy=tensor_group, group_size=16
    input_activations: num_bits=4, type=float, dynamic=local, group_size=16
    |
    v
[3] 校准前向（calibration forward）
    Linear GEMM 还是原始高精度 GEMM（bf16/fp16），不是 NVFP4 GEMM
    |
    +--> [3a] GPTQ hook 抓输入 x
    |         x -> reshape/transpose -> 转 FP32
    |         累计 H += x x^T （FP32 matmul）
    |
    +--> [3b] 激活 observer 统计范围
    |         计算 input_global_scale
    |
    +--> [3c] 权重 observer 统计范围
              计算 weight_global_scale
    |
    v
[4] 子图校准结束
    每层得到：Hessian 近似 H、weight_global_scale、input_global_scale
    |
    v
[5] fused global scale 对齐
    q/k/v 共享一个 global_scale（取最小值）
    gate/up 共享一个 global_scale（取最小值）
    |
    v
[6] GPTQ quantize_weight()
    W -> clone -> 转 FP32
    H -> 加 damp -> Cholesky -> 近似 H^{-1}
    逐列/逐块做 GPTQ：
      当前列 w
        -> 用 NVFP4 对应 qparams + global_scale 做 fake_quantize
        -> 得到 q
        -> err = (w - q) / d
        -> 用 H^{-1} 把误差传播到后续列
    输出：weight, weight_scale, weight_zero_point, (weight_g_idx)
    |
    v
[7] 回写到模块
    module.weight <- quantized weight
    module.weight_scale <- scale
    module.weight_zero_point <- zp
    module.weight_global_scale <- global scale
    module.input_global_scale <- activation global scale
    |
    v
[8] 保存 compressed model
    compressor = nvfp4_pack_quantized
    save_pretrained(save_compressed=True)
    -> 按 NVFP4 packed layout 落盘
    -> 更新 config.json quantization_config
    -> 保存 recipe（包含 GPTQModifier 信息）
    |
    v
[9] 部署推理（vLLM / Blackwell SM100+）
    读取 NVFP4 packed weights + scale 参数
    读取预校准的 input_global_scale
    每个 batch：用 input_global_scale + 动态 local scale 量化激活
    执行 W4A4 FP4 GEMM kernel
    （非 SM100 硬件自动退化为 weight-only FP4）
```

### 11.1 这个数据流里最值得移植的三个接口

如果你准备把 GPTQ 搬到别的项目，建议优先把这三个接口抽象出来：

#### 接口 A：`collect_hessian(layer, inputs)`

输入：
- 某一层
- 某批 calibration inputs

输出：
- 该层的 Hessian 近似累积结果

#### 接口 B：`build_qparams(layer_or_tensor, scheme)`

输入：
- 权重或激活张量
- 量化 scheme（INT4 / FP4 / NVFP4）

输出：
- `scale`
- `zero_point`
- `global_scale`
- 可能的 `g_idx`

#### 接口 C：`gptq_quantize(layer_weight, hessian, qparams, actorder)`

输入：
- 权重矩阵
- Hessian
- qparams 构造器或查表器
- `actorder`

输出：
- 量化后的权重参数集合

只要把这三层接口理顺，GPTQ 在不同仓库之间迁移会容易很多。

---

## 12. GPTQ 最小可移植实现

这一节专门从“我要在别的仓库里重写 GPTQ”角度写。

### 12.1 最小依赖版本

如果你完全不想依赖 `llm-compressor`，最少需要这些能力：

1. 能拿到某层的输入激活
2. 能把该层权重 reshape 成 `[out_features, in_features]`
3. 能根据目标格式实现一个 `quantizer(w)`
4. 能在 GPU 或 CPU 上做 Cholesky / inverse

### 12.2 最小实现骨架

```python
class PortableGPTQ:
    def __init__(self, quantizer, block_size=128, damp=0.01):
        self.quantizer = quantizer
        self.block_size = block_size
        self.damp = damp

    def accumulate_hessian(self, x, H):
        # x: [batch, seq, in_features] or [N, in_features]
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        x = x.t().float()
        H += (math.sqrt(2) * x) @ (math.sqrt(2) * x).t()
        return H

    def quantize(self, W, H):
        W = W.float().clone()
        H = H.float().clone()

        diag_idx = torch.arange(H.shape[0], device=H.device)
        H[diag_idx, diag_idx] += self.damp * torch.mean(torch.diag(H))

        U = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(U)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        Q = torch.zeros_like(W)

        for i1 in range(0, W.shape[1], self.block_size):
            i2 = min(i1 + self.block_size, W.shape[1])
            Wblk = W[:, i1:i2].clone()
            Hblk = Hinv[i1:i2, i1:i2]
            Err = torch.zeros_like(Wblk)

            for i in range(i2 - i1):
                w = Wblk[:, i]
                d = Hblk[i, i]
                q = self.quantizer(w)
                Q[:, i1 + i] = q

                err = (w - q) / d
                Err[:, i] = err
                Wblk[:, i:] -= err.unsqueeze(1) @ Hblk[i, i:].unsqueeze(0)

            W[:, i1:i2] = Q[:, i1:i2]
            if i2 < W.shape[1]:
                W[:, i2:] -= Err @ Hinv[i1:i2, i2:]

        return Q
```

### 12.3 把这段骨架接到任意模型上的方法

#### 方案 1：hook 采集输入
- 给每个 Linear 注册 forward hook
- 记录输入
- 调 `accumulate_hessian`

#### 方案 2：离线逐层重放
- 先缓存每层输入
- 离线重放这些输入做 Hessian 累积

#### 方案 3：顺序子图压缩
- 模仿 `SequentialPipeline`
- 一层层校准、量化、再传播输出

### 12.4 移植时最容易踩坑的地方

#### 坑 1：Hessian 维度搞错
必须按输入维度建 `[in_features, in_features]`，不是按输出维度建。

#### 坑 2：没有统一 Conv1D / Linear 方向
有些仓库的 `Conv1D` 权重转置方向和 `Linear` 不一样，必须先规范化。

#### 坑 3：量化器返回的是低比特码，而不是 dequant 后浮点
GPTQ 主循环里通常需要的是“dequant 后的浮点近似值 `q`”，因为误差传播是在浮点域完成的。

#### 坑 4：先算 `scale` 再做补偿，还是边补偿边重算 `scale`
这决定了你实现的是：
- 普通 `static/weight` 风格
- 还是 `group` 风格

#### 坑 5：对 `TENSOR_GROUP` 忽略了 `global_scale`
对于 NVFP4 这种格式，没 `global_scale` 就不是等价实现。

### 12.5 最推荐的移植顺序

1. 先实现 `W4A16 + static`
2. 再实现 `W4A16 + weight`
3. 再实现 `NVFP4A16`
4. 再实现 `NVFP4`
5. 最后再补 `group + weight_g_idx`

这是从实现复杂度最低到最高的合理路径。

---

## 13. 把 GPTQ 移植到其他仓库时的检查清单

### 13.1 算法正确性检查

- Hessian 是否按输入维度建立
- Hessian 是否除以样本数
- Cholesky 失败时是否有回退路径
- 量化误差是否真的传播到了后续列
- 最终 `Q` 是否覆盖了所有列

### 13.2 量化器正确性检查

- `quantizer(w)` 返回的是 dequant 后浮点近似值吗
- group / block / tensor 策略是否都能正确查 qparams
- NVFP4 是否真的同时使用了 `global_scale` 和 local scale

### 13.3 导出格式检查

- checkpoint 是否保存了 runtime 所需全部字段
- `weight_g_idx` 是否只在确实需要时保存
- 不同 scheme 是否映射到正确的 compressor / format

### 13.4 runtime 一致性检查

- 普通量化和 GPTQ 量化是否能共用同一 runtime 路径
- runtime 是否只依赖 checkpoint 格式而不依赖“生成算法”
- 对于 `actorder=group`，runtime 是否真的读取了 `weight_g_idx`

### 13.5 评测检查

- MSE / cosine / perplexity
- 第一层、最后一层、attention qkv、MLP up/gate 的误差分布
- 是否存在某些层 Hessian 不稳定、退化成 identity 的情况

### 按张量 dtype 再画一遍

```
[原始输入 token ids]
    |
    v
embedding / transformer hidden states
    dtype: bf16/fp16
    |
    v
[校准前向中的 Linear GEMM]
    x: bf16/fp16,  W: bf16/fp16
    y: bf16/fp16
    |
    +--> GPTQ hook 抓 x -> cast to FP32 -> H += x x^T (FP32)
    |
    v
[GPTQ quantize_weight]
    W_original -> cast to FP32
    H -> FP32 inverse/cholesky
    per-column fake_quantize to NVFP4 representable values (FP32 中操作)
    error compensation in FP32
    |
    v
[量化结果]
    weight, weight_scale, weight_zero_point, weight_global_scale
    target format: NVFP4
    |
    v
[保存 compressed model -> nvfp4_pack_quantized]
    |
    v
[部署推理]
    runtime 读取 NVFP4 packed weights
    低比特 NVFP4 GEMM 在这里真正发生
```

---

## 附录：关键 symmetric 和 zero_point 说明

**"保存了 `weight_zero_point`" 不等于 "用了非对称量化"。**

在 `symmetric=True` 的 NVFP4 配置下：
- `zero_point` 是统一量化框架的通用字段
- 对称量化时通常为固定零值或中性占位
- 真正决定量化效果的是 `scale` 和 `global_scale`
- 这是工程统一接口的设计，不代表所有量化都真的需要非零 zero-point

---

## 附录：常见疑问（FAQ）

### Q1. `weight_g_idx` 到底影响什么？

`weight_g_idx` 影响的是：

> **每一列 weight 在解释 / 使用时，应该绑定哪一组 `weight_scale` / `weight_zero_point`。**

它不是：
- 修改权重矩阵坐标
- 修改 activation 的 group 映射
- 修改 activation quantization 流程

而是一个纯粹的 **weight-side 列到 group 的查表**。

### Q2. `weight_g_idx` 是不是影响反量化？

是的。从数值解释角度看，它直接影响“这列应该用哪组 scale”。

不管 runtime 是：
- 显式先 dequant 再 GEMM
- 还是 fused low-bit GEMM 内部边解释边算

本质都需要把权重码值和**正确的 group scale**对应起来。

如果 `g_idx` 错了，会发生：
- 相同的低比特码值配错 scale
- 数值等效上就是“反量化解释错了”
- 输出会偏

所以在 `actorder=group` 下，`weight_g_idx` 是 correctness 所必需的元数据。

### Q3. 如果权重和 activation 都是 NVFP4，`weight_g_idx` 还有意义吗？

有意义，但只在特定配置下有意义。

如果是：
- `actorder=static`
- 或 `actorder=weight`

通常不会持久化 `weight_g_idx`，也就不需要关心它。

如果是：
- `actorder=group`

那么即使：
- weight 是 NVFP4
- input activation 也是 NVFP4

`weight_g_idx` 依然有意义，因为它解决的是：

> **weight 列如何绑定到 weight group qparams**

而不是：

> activation 是否也是 NVFP4

换句话说，W/A 都是 NVFP4 并不自动消除 weight-side 的 group 映射问题。

### Q4. 用 NVFP4 GEMM 的时候，不是直接输入 NVFP4 的 input 和 weight 就行了吗？

不完全是。NVFP4 不是“裸 4-bit 浮点数组直接相乘”这么简单，它依赖：
- weight 的 `global_scale`
- weight 的 `local/group scale`
- activation 的 `global_scale`
- activation 的动态 local scale

因此 kernel 实际上需要知道：
- weight 低比特码值
- 当前列该查哪组 weight scale
- activation 低比特码值
- activation 当前 token / 当前 group 的 local scale

在 `actorder=group` 时，`weight_g_idx` 正是 kernel 或解释逻辑正确取 weight qparams 的一部分。

### Q5. `static` / `weight` 下，GPTQ+NVFP4 导出权重和普通 NVFP4 是否可以看作同一路径？

可以，前提是讨论的是：
- 同样的 `NVFP4` scheme
- 不是 `NVFP4A16`
- 没有 `weight_g_idx`

这时可以分两层理解：

1. **导出格式层面**
   - 都是 `nvfp4_pack_quantized`
   - 结构上基本一致

2. **vLLM load / execute 层面**
   - 通常走同一类加载路径
   - 通常走同一类 NVFP4 kernel 路径
   - 差异主要来自参数数值不同，不是执行分支不同

### Q6. 那 `group` 模式是不是一定更好？

从这份仓库代码本身，能确定的是：
- `group` 的机制更复杂
- 它会对重排后的列重新计算 qparams
- 它需要额外保存 `weight_g_idx`

但仅凭当前仓库代码，**不能直接证明**：
- `group` 一定比 `weight` 更准
- 或 `weight` 一定比 `static` 更准

这些属于实验 / benchmark 结论，不是单靠源码分支就能严格推出的结论。
