# vLLM throughput vs latency（单卡 GPU 场景）

## 概述

这篇文档讨论的是 vLLM 顶层的 `--performance-mode`。当前代码里它有三个值：

- `balanced`：默认值，做折中
- `interactivity`：偏低延迟
- `throughput`：偏高吞吐

但这里先强调一个最容易混淆的点：

**`throughput / interactivity` 是 vLLM 的顶层运行模式；某些子模块里也会出现名字相似的 `latency / throughput` backend 选项，但它们不是同一个开关。**

例如：

- `--performance-mode` 影响的是运行时的整体策略
- `VLLM_FLASHINFER_MOE_BACKEND=latency|throughput|masked_gemm` 影响的是 FlashInfer MoE kernel 选择
- `--all2all-backend deepep_low_latency|deepep_high_throughput` 影响的是多卡 EP 通信后端

单卡 GPU 场景下，最核心的矛盾仍然是：

- `throughput`：更在意整张卡每秒吐多少 token
- `interactivity`：更在意单个请求多久拿到首 token / 下一个 token

---

## 1. 单卡下这两个模式到底在争什么

放在单卡上，二者本质上是在做下面这个 trade-off：

- **吞吐优先**：更愿意把请求攒起来一起算，让 GPU 更满
- **延迟优先**：更愿意更早发车，减少单个请求等待别人

直观理解：

- `throughput` 像拼车，尽量多凑几个人一起走
- `interactivity` 像专车，人到了就尽快发车

因此它们关注的指标天然不同：

| 关注点 | 吞吐优先 | 延迟优先 |
|---|---|---|
| 主要指标 | `tokens/s`、`req/s` | `TTFT`、`ITL`、`E2E latency` |
| 典型场景 | 离线批量推理、评测、大并发压测 | 在线聊天、低并发交互、流式生成 |
| 典型取舍 | 愿意增加排队时间 | 愿意接受 GPU 没吃满 |

---

## 2. 当前 vLLM 里，`--performance-mode` 真正改了什么

先说结论：

1. `throughput` 会把默认的 `max_num_batched_tokens` 和 `max_num_seqs` **翻倍**
2. `interactivity` **不会自动把这两个值减半**
3. `interactivity` 会把小 batch 的 `CUDA Graph` 捕获做得更细
4. 文档字符串里也明确写了：`interactivity` 倾向小 batch 低延迟，`throughput` 倾向高并发下更高 `tokens/sec`

### 2.1 调度预算：`throughput` 会放大默认值

`vllm/engine/arg_utils.py` 里有明确逻辑：

```python
# If throughput mode is set, double max_num_batched_tokens and max_num_seqs.
if self.performance_mode == "throughput":
    if orig_max_num_batched_tokens is None:
        self.max_num_batched_tokens *= 2
    if orig_max_num_seqs is None:
        self.max_num_seqs *= 2
```

这里的关键限定词是：**仅当你没有手动指定这两个参数时，才会翻倍默认值。**

也就是说：

- 你手动传了 `--max-num-batched-tokens` / `--max-num-seqs`，`throughput` 不会再替你改
- `interactivity` 不是“自动缩小 token budget”，它更多是在 baseline 上偏向低延迟

这也是原来很多表述里最容易说过头的地方。

### 2.2 `balanced` 和 `interactivity` 的默认调度预算并不天然不同

默认值来自 `vllm/engine/arg_utils.py`，会按硬件和使用场景区分：

| 硬件/场景 | `max_num_batched_tokens` | `max_num_seqs` |
|---|---:|---:|
| H100/H200/MI300X 这类大显存非 A100，`LLM_CLASS` | 16384 | 1024 |
| H100/H200/MI300X 这类大显存非 A100，`OPENAI_API_SERVER` | 8192 | 1024 |
| A100 或其他较小默认档位，`LLM_CLASS` | 8192 | 256 |
| A100 或其他较小默认档位，`OPENAI_API_SERVER` | 2048 | 256 |

在这些 baseline 上：

- `balanced`：用 baseline
- `interactivity`：仍然用 baseline，除非你手动覆盖
- `throughput`：在 baseline 基础上翻倍，前提是你没手动设置

所以更准确的说法不是“latency 模式预算更小”，而是：

**在当前 vLLM 实现里，`throughput` 会主动把默认预算放大；`interactivity` 主要不是靠缩小预算来实现低延迟。**

---

## 3. 单卡里最重要的第二个差异：CUDA Graph 捕获粒度

这部分是 `interactivity` 和其他模式真正拉开差距的地方之一。

`vllm/config/vllm.py` 的逻辑是：

- `interactivity`：对小 batch 做更细粒度的 capture
- `balanced` / `throughput`：使用更稀疏的 capture 点

大致是：

```python
if self.performance_mode == "interactivity":
    interactivity_max = min(max_cudagraph_capture_size, 32)
    cudagraph_capture_sizes = list(range(1, interactivity_max + 1))
else:
    cudagraph_capture_sizes = [1, 2, 4]
    cudagraph_capture_sizes += range(8, ..., 8)
    cudagraph_capture_sizes += range(256, ..., 16)
```

这意味着在小 batch decode 下：

- `interactivity` 更容易正好命中 batch=3、5、7 这种小尺寸
- `balanced/throughput` 更可能命中邻近但更大的 capture size，从而需要 padding

### 为什么这对单卡特别重要

单卡在线服务常见的情况是：

- 并发不高
- decode batch 很小
- 每步就几个请求各出 1 个 token

这时如果 graph capture 粒度不够细，就会有额外 padding 和调度开销。  
因此 `interactivity` 在**小并发、小 batch、decode 主导**的场景下，往往更有机会改善：

- `TTFT`
- `ITL`
- 小 batch 下的 step latency

但也要注意：

**这不是无条件收益。**  
最终是否命中 FULL CUDA Graph，还取决于：

- 当前 batch 形状
- attention backend 是否支持
- 当前运行模式是否退化为 `FULL` / `PIECEWISE` / `NONE`

所以更稳妥的表述是“更有利于低延迟”，而不是“必然显著更快”。

---

## 4. 单卡调度视角：throughput 为什么通常更高，latency 为什么通常更快

调度器每一步都有一个 token budget，本质上是“这一轮最多给 GPU 安排多少 token 计算”。

在单卡下可以这样理解：

- `throughput`：更大的默认预算，更愿意把更多请求/更多 token 合并到同一步
- `interactivity`：默认预算不一定更小，但配合更细的 graph capture，更倾向优化小 batch 的单步开销

结果通常是：

| 维度 | `throughput` | `interactivity` |
|---|---|---|
| GPU 利用率 | 更容易打满 | 可能偏低 |
| 单步计算量 | 更大 | 往往更小或更规整 |
| 单请求排队时间 | 通常更长 | 通常更短 |
| 总 `tokens/s` | 通常更高 | 通常更低 |
| `TTFT` / `ITL` | 通常更差 | 通常更好 |

注意这里我用了“通常”而不是“必然”。  
因为真实结果还会受这些因素影响：

- prompt 长度分布
- decode 占比
- 是否开启 chunked prefill
- 模型结构（Dense / MoE）
- attention / quantization backend
- 用户是否手动覆盖调度参数

---

## 5. prefill 和 decode 两阶段，要分开看

### Prefill

Prefill 的特点是：

- 输入 token 多
- 计算更重
- 更容易从大 batch 中获益

因此在单卡上：

- 长 prompt、离线批处理、更高并发时，`throughput` 往往更占优
- 它的优势主要来自更大的 batch 和更高的 GEMM 利用率

### Decode

Decode 的特点是：

- 每步新增 token 很少
- 每一步都很短
- 对 launch 开销、graph 命中、调度等待更敏感

因此在单卡上：

- 在线聊天、低并发、流式输出时，`interactivity` 往往更占优
- 它的优势主要来自更细粒度的小 batch 优化，而不是“强制减少所有预算”

---

## 6. 一个更准确的单卡心智模型

原来常见但不够准确的说法是：

- “latency 模式 = 小 batch”
- “throughput 模式 = 大 batch”

更准确的说法应该是：

- `throughput` **会主动把默认调度上限放大**
- `interactivity` **主要优化小 batch 的执行效率，而不是简单地把 batch 上限缩小**

所以对单卡来说，更推荐这样理解：

1. 如果你什么都不手调：
   - `balanced` = baseline
   - `interactivity` = baseline 调度预算 + 更细粒度的小 batch graph capture
   - `throughput` = 更大的默认调度预算 + 更偏吞吐的整体策略

2. 如果你已经手动设置了 `--max-num-batched-tokens` 和 `--max-num-seqs`：
   - `throughput` 的“预算翻倍”这部分效果就没有了
   - 这时三者差别更多体现在 graph capture 等其他 runtime 行为

---

## 7. 容易混淆但必须分开的概念

这是原文里最需要纠正的地方。

### 7.1 `--performance-mode` 不是 MoE backend 开关

FlashInfer MoE 的环境变量：

```bash
VLLM_FLASHINFER_MOE_BACKEND=latency
VLLM_FLASHINFER_MOE_BACKEND=throughput
VLLM_FLASHINFER_MOE_BACKEND=masked_gemm
```

它控制的是 **MoE kernel 选择**，例如：

- `latency` -> `TensorRT-LLM`
- `throughput` -> `CUTLASS`

这和 `--performance-mode` 不是同一个层级。

两者关系更像：

- `--performance-mode`：决定整机运行倾向
- `VLLM_FLASHINFER_MOE_BACKEND`：决定某一类 MoE kernel 用哪个实现

它们都带有 `latency/throughput` 字样，但**不能直接等同**。

### 7.2 `deepep_low_latency / deepep_high_throughput` 是多卡通信后端，不适用于单卡主线讲解

这组配置属于 DP/EP / all-to-all 通信层面，更适合多卡 MoE 部署。  
单卡 GPU 讲 `throughput vs latency` 时，可以顺带提到，但不应该作为主线。

---

## 8. 单卡选型建议

### 适合 `interactivity`

- 单用户或低并发聊天
- 对 `TTFT` 和 `ITL` 很敏感
- 输出是流式的，用户在前台“盯着看”
- decode 占主要成本

### 适合 `throughput`

- 离线批量生成
- 跑 benchmark / eval / dataset 批处理
- 高并发压测
- 更关心总 `tokens/s`，不太关心单请求是不是多等几十到几百毫秒

### 什么时候用 `balanced`

- 业务既不是极端低延迟，也不是极端离线吞吐
- 你还没有足够 benchmark 数据支撑更激进选择
- 你把它当默认起点，然后再做 sweep

---

## 9. 建议怎样验证，而不是只看名字

单卡上不要只看模式名，最好直接测三组数据：

1. `TTFT`
2. `ITL` / `TPOT`
3. `output tokens/s`

最常见的现象是：

- `throughput` 赢 `tokens/s`
- `interactivity` 赢 `TTFT` / `ITL`
- `balanced` 落在中间

但如果你已经手动设了很大的 `--max-num-batched-tokens`，或者 workload 本身几乎全是长 prefill，那么 `interactivity` 可能并不会带来想象中的收益。

---

## 10. 最终结论

在当前 vLLM 代码里，单卡场景下更准确的总结是：

- `throughput` 的主要区别是：**把默认调度预算放大**，从而更容易提高整卡吞吐
- `interactivity` 的主要区别是：**把小 batch 的 CUDA Graph 捕获做细**，从而更有利于低并发 decode 场景的低延迟
- `balanced` 是两者之间的 baseline

因此：

- 如果你在讲“单卡 GPU 上 throughput 和 latency 的区别”，主线应放在 **调度预算 + 小 batch CUDA Graph 行为 + workload 类型**
- 不应把 MoE/NVFP4/DeepEP 这些子系统里的 `latency/throughput` backend 直接当作 `--performance-mode` 的一部分

---

## 代码定位

- `vllm/config/vllm.py`
  - `PerformanceMode` 定义
  - `performance_mode` 字段说明
  - `interactivity` 的 `cudagraph_capture_sizes` 逻辑
- `vllm/engine/arg_utils.py`
  - 默认 `max_num_batched_tokens` / `max_num_seqs`
  - `throughput` 对默认值翻倍的逻辑
- `vllm/v1/core/sched/scheduler.py`
  - 调度循环和 token budget 消耗方式
- `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`
  - `VLLM_FLASHINFER_MOE_BACKEND` 与 MoE kernel 的映射

如果后面要单独写一篇 MoE / NVFP4 里的 `latency` vs `throughput` backend，对应另起文档会更清楚。
