# Qwen3.5 案例：`qwen3_next.py:forward` start 到 `caching.py:__call__` start 之间发生了什么

## 1. 问题边界

本文只解释下面这个很窄的时间窗：

- 起点：`vllm/model_executor/models/qwen3_next.py(500): forward` 的 `start`
- 终点：`vllm/compilation/caching.py(210): __call__` 的 `start`

案例使用：

- 模型：Qwen3.5 / `Qwen3_5Model`
- trace：`compare_single/single_logs_opensource/graph_vllm_opensource.pt.trace.json`
- 场景：prefill 相关执行路径

要先强调一件事：

- `Qwen3_5Model` 继承自 `Qwen3NextModel`
- `Qwen3_5Model` 没有重写 `forward`
- 所以 trace 里看到 `qwen3_next.py(500): forward`，本质上就是 Qwen3.5 在跑

```python
# vllm/model_executor/models/qwen3_5.py
@support_torch_compile(...)
class Qwen3_5Model(Qwen3NextModel):
    ...
```

## 2. 这段时间窗的结论

在这份新版 trace 里，这个窗口大约只有 `832 us`，主要分成三段：

1. 一串 `vllm/utils/torch_utils.py(737): __init__`
2. 若干 `torch/_compile.py(42): inner` 和 `torch/_dynamo/eval_frame.py(1240): _fn`
3. 然后才进入 `vllm/compilation/caching.py(210): __call__`

它不是模型层计算主路径，也不是 prefill 的 piecewise graph 本体。

更准确地说，它是：

- vLLM 为后续 custom op 准备 `layer_name`
- Torch runtime 把执行从普通 Python `forward` 过渡到 compiled callable
- 然后才真正进入 cached / compiled graph 入口

## 3. 为什么是 `qwen3_next.py:forward`

Qwen3.5 的外层 `forward` 最终会调用 `self.model(...)`：

```python
# vllm/model_executor/models/qwen3_5.py
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
    hidden_states = self.model(
        input_ids, positions, intermediate_tensors, inputs_embeds
    )
    return hidden_states
```

而 `self.model` 是 `Qwen3_5Model`，它继承 `Qwen3NextModel`，因此真正的骨架是：

```python
# vllm/model_executor/models/qwen3_next.py
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        ...

    for layer_idx, layer in enumerate(...):
        hidden_states, residual = layer(...)
    ...
```

所以 trace 里用 `qwen3_next.py(500): forward` 标记 Qwen3.5，是符合源码结构的。

## 4. 第一段：大量 `torch_utils.py(737): __init__` 是什么

这不是算子执行，而是在构造 `LayerName`：

```python
# vllm/utils/torch_utils.py
class LayerName(OpaqueBase):
    def __init__(self, value: str):
        self.value = value

def _encode_layer_name(layer_name: str) -> str | LayerName:
    return LayerName(layer_name) if _USE_LAYERNAME else layer_name
```

它的目的不是业务逻辑，而是编译友好性：

- 把层名字从 `str` 包成 opaque object
- 让 `torch.compile` 把它当作 hoisted input，而不是 baked constant
- 避免 custom op 因为层名不同而触发 per-layer recompilation

这个 `layer_name` 后面会传给 attention / KV cache / GDN 等路径，例如：

```python
# vllm/model_executor/layers/attention/attention.py
torch.ops.vllm.maybe_calc_kv_scales(
    query, key, value, _encode_layer_name(self.layer_name)
)
```

```python
# vllm/model_executor/layers/attention/attention.py
encoded = _encode_layer_name(self.layer_name)
```

因此，trace 里那一串 `__init__` 可以理解成：

- `forward` 一开始先为很多层 / many custom-op call site 准备 `LayerName`
- 它们是极轻的 Python 对象构造
- 不是 attention / GDN / KV cache 真正在算

## 5. 第二段：`torch/_compile.py(42): inner` 是什么

这来自 PyTorch 的 `_disable_dynamo()` 内部 wrapper：

```python
# torch/_compile.py
def _disable_dynamo(fn=None, recursive=True):
    if fn is not None:
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            disable_fn = getattr(fn, "__dynamo_disable", None)
            if disable_fn is None:
                import torch._dynamo
                disable_fn = torch._dynamo.disable(fn, recursive, wrapping=False)
                fn.__dynamo_disable = disable_fn
            return disable_fn(*args, **kwargs)
        return inner
```

它的职责是：

- 找出某个函数对应的 `disable_fn`
- 第一次调用时懒生成这个 `disable_fn`
- 然后把调用转发给它

所以 trace 里看到 `torch/_compile.py(42): inner`，本质上不是模型计算，而是：

- Torch runtime 在调用一个“不要继续被 Dynamo 追踪”的 helper wrapper

## 6. 第三段：`torch/_dynamo/eval_frame.py(1240): _fn` 是什么

这个 `_fn` 是 `torch._dynamo.disable(...)` 返回出来的真正 wrapper：

```python
# torch/_dynamo/eval_frame.py
def _fn(*args, **kwargs):
    prior = set_eval_frame(None)
    try:
        _maybe_set_eval_frame(_callback_from_stance(self.callback))
        try:
            fn_name = getattr(fn, "__name__", type(fn).__name__)
            return fn(*args, **kwargs)
        finally:
            set_eval_frame(None)
    finally:
        _maybe_set_eval_frame(prior)
```

它做的事情是：

1. 暂时关掉当前 frame 的 Dynamo hook
2. 切换 eval-frame 状态
3. 执行原始函数
4. 恢复之前的状态

所以 trace 里伴随它出现的这些小事件：

- `set_eval_frame`
- `getattr`
- `is_exporting`
- `call_size`
- `call_stride`

都属于“进入 compiled callable 之前的 Torch runtime 控制逻辑”。

## 7. 那个“bubble”到底是什么

如果只盯 timeline，很容易看到：

- 一串 `LayerName.__init__`
- 然后中间像空了一段
- 接着才到 `caching.py:__call__`

这段“空白”不能简单理解成“vLLM 在做大量未命名业务逻辑”。

对这份新版 trace，更贴近事实的解释是：

### 7.1 它不是模型层计算

在 `caching.py:__call__` 开始之前，还没有真正进入 compiled callable：

```python
# vllm/compilation/caching.py
def __call__(self, *args, **kwargs):
    return self.optimized_call(*args, **kwargs)
```

也就是说：

- piecewise graph 本体还没开始
- eager `gdn_attention_core` / `unified_attention_with_output` 也还没开始
- 这段不属于 prefill 主计算本体

### 7.2 它更像“compiled callable entry overhead”

这段 bubble 更合适的名字是：

- compiled-callable entry overhead
- 或“进入编译后 callable 前的 runtime 过渡开销”

组成大致包括：

- `LayerName` 构造结束后的 Torch wrapper 调度
- eval-frame 切换
- 少量 shape / stride helper
- profiler bookkeeping
- 以及一部分 trace 上不明显的 C++ / runtime 过渡时间

### 7.3 为什么看起来像“大 bubble”

因为 profiler 的可见性不均匀：

- `LayerName.__init__` 被明确标成 `python_function`
- `caching.__call__` 也有清晰名字
- 中间很多 runtime 过渡很短、很碎，或者根本没有按同样粒度被标出来

于是视觉上会变成：

```text
forward start
  -> 一堆 LayerName.__init__
  -> 中间像空了一块
  -> caching.__call__ start
```

这个“空了一块”不等于 vLLM 有一段单独的大 Python 业务函数，只是 trace 标注粒度造成的观感。

## 8. 它和 prefill graph 的关系

对 Qwen3.5 的 prefill 来说，真正的 graph / eager 交替结构是在 `caching.__call__` 之后才开始明显出现：

- graph piece
- `gdn_attention_core` eager
- graph piece
- `gdn_attention_core` eager
- graph piece
- `unified_kv_cache_update + unified_attention_with_output` eager
- ...

所以：

- `forward start -> caching.__call__ start` 这段，是 prefill 主执行之前的入口准备
- `caching.__call__` 之后，才是 piecewise graph 真正开始接管执行

## 9. 和旧版大 bubble 的区别

如果你在旧版 vLLM 或另一份 trace 里看到的是**更长**的 bubble，比如毫秒级甚至 10ms+，那就不一定只是这里说的这 `832 us` 入口过渡了。

旧版 vLLM 一个常见问题是：

- Dynamo 生成很长的 compiled bytecode preamble
- 在进入 `__compiled_fn_*` 之前，先做大量参数提取
- 里面会出现大量 `dict.__getitem__` / `self._modules` / `self._parameters` 访问

这类开销的详细分析见：

- `compile_cpu_overhead_analysis.md`

也就是说，要区分两类现象：

### A. 新版 trace 里这段短窗口

- 主要是 `LayerName` 构造 + Torch runtime wrapper
- 量级约 `sub-ms`

### B. 旧版 trace 里那种长前导

- 主要是 Dynamo 生成的参数提取 preamble
- 量级可能是 `ms` 到 `10+ms`

两者位置相似，但根因不完全相同。

## 10. 一句话总结

以 Qwen3.5 为例，`qwen3_next.py:forward` start 到 `caching.py:__call__` start 之间，CPU 主要在做的不是模型计算，而是：

1. 用 `LayerName.__init__` 为后续 custom op 准备 `layer_name`
2. 通过 `torch/_compile.py:inner` 和 `torch/_dynamo/eval_frame.py:_fn` 做 Torch runtime 的 disable / eval-frame 过渡
3. 最后把执行切换到 `VllmSerializableFunction.optimized_call`

所以，这段“很多 `__init__` + 一个 bubble”本质上是**compiled graph 入口前的准备与切换阶段**，不是 prefill 主体计算本身。
