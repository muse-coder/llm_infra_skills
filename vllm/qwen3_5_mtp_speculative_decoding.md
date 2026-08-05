# Qwen3.5 MTP 推理逻辑

## 1. MTP 是什么

Qwen3.5 的 MTP 用一个轻量 draft 模块预测后续 token，再由完整 Qwen3.5
verify 模型验证这些候选。

它的目标不是让 draft 每次都猜对，而是用较小的计算量提前猜多个 token，争取让
完整模型一次 forward 推进多个位置。

Qwen3.5 的 MTP draft 不是另一套完整语言模型。它主要使用：

```text
当前 token embedding
        +
完整 Qwen3.5 的 hidden state
        |
        v
轻量 MTP decoder
        |
        v
下一个 draft token
```

## 2. Draft 模型与 Verify 模型的区别

| Draft / MTP | Verify / Target |
|---|---|
| 轻量 MTP 模块 | 完整 Qwen3.5 模型 |
| 负责猜测候选 token | 负责验证候选并决定最终输出 |
| 可以猜错 | 是最终正确性的来源 |
| 为生成多个候选而逐 token 执行 | 一次并行验证多个候选位置 |

假设配置生成 3 个 draft token：

```text
MTP 串行猜测：
MTP -> A -> MTP -> B -> MTP -> C

完整模型并行验证：
Qwen3.5([A, B, C]) -> 一次 forward
```

性能收益来自：用便宜的 MTP 多运行几次，换取昂贵的完整 Qwen3.5 一次验证并
尽可能接受多个 token。

## 3. 猜错后怎么办

Verify 模型从左到右验证 draft。它会保留第一个错误位置之前连续猜对的 token。
第一个 token 被拒绝后，后续所有依赖它的 draft 都会被丢弃。

例如 MTP 猜测：

```text
draft:  A  B  C
```

Verify 认为 A 正确，但 B 应该是 X：

```text
verify: A  X  ...
```

最终处理为：

```text
保留 A
拒绝 B
丢弃 B、C
输出 verify 给出的修正 token X
当前确认序列变成：... A X
```

不需要重新验证 A，也不是从错误的 B 继续。下一次 MTP 从修正后的 X 后面重新
预测：

```text
已确认序列：... A X
                 |
                 v
下一轮 MTP：预测 X 后面的 token
```

### 第一个 token 就猜错

```text
draft:  A B C
verify: X ...
```

A、B、C 全部丢弃，只保留 verify 产生的 X，然后从 X 后重新做 MTP。

### 全部猜对

```text
draft:  A B C
verify: A B C
```

A、B、C 全部保留。完整模型还可以利用最后一组 logits 再产生一个 bonus token D，
因此一次验证最多得到：

```text
A B C D
```

## 4. 为什么猜错不需要重新跑完整前缀

完整模型验证 `[A, B, C]` 时，已经同时计算了每个位置的 logits。

如果 A 被接受、B 被拒绝，那么完整模型已经得到：

```text
P(next token | 原上下文, A)
```

所以可以直接从该位置得到修正 token X。B、C 后面的结果因为依赖错误前缀而被
丢弃，下一轮再从正确的 `... A X` 继续。

相应缓存也只保留正确前缀：A 的状态保留，B、C 的状态失效或被后续 token
覆盖，不需要重算 A 之前的历史。

## 5. MTP KV 与 Qwen3.5 GDN State

### 5.1 MTP 自己的 KV Cache

Qwen3.5 的 MTP decoder 固定使用 full attention，因此有一套独立于 target
backbone 的 KV cache。

第一次 draft forward 会尽量复用 target 相同的 batch shape、position 和 attention
metadata。继续生成第二、第三个 draft 时，每一步更新：

```text
input_id = 上一步 draft token
hidden_state = 上一步 MTP 输出
position += 1
seq_len += 1
```

并为新位置重新计算 slot mapping 和 attention metadata。

发生拒绝后，代码使用 `num_rejected` 缩短有效 query length 和 sequence length，
使 MTP cache 回到正确前缀；错误分支占用的位置随后会被覆盖。

主要代码入口：

- `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`
- `prepare_prefill_inputs()`：根据拒绝数量恢复有效输入长度
- `prepare_decode_inputs()` / `update_draft_inputs()`：推进 draft token、hidden 和位置

### 5.2 Target 的 GDN Recurrent State

Qwen3.5 target 是 GDN linear attention 与 full attention 混合模型。这使它比纯
Transformer 的 speculative decoding 多一个问题：

- Full attention 可以按 token 保存 KV。
- GDN 主要维护递归状态，不能简单保留一串普通 KV。
- Target 一次验证多个候选时，要等 rejection sampling 完成后，才能确定最终应
  保留哪个位置的 GDN state。

vLLM 会为 speculative positions 预留额外状态空间。Qwen3.5 将
`num_speculative_tokens` 传给 GDN state shape 计算，使 convolution state 的历史
长度变为：

```text
conv_kernel_size - 1 + num_speculative_tokens
```

验证完成后，再根据实际接受/输出的 token 数，把 recurrent state 对齐到最终提交
的位置。例如 A 接受、B 拒绝时，保留“处理完 A”对应的状态，而不是错误 B/C 后的
状态。

主要代码入口：

- `vllm/model_executor/models/qwen3_5.py`：
  `get_mamba_state_shape_from_config()`
- `vllm/model_executor/layers/mamba/mamba_utils.py`：
  `gated_delta_net_state_shape()`
- `vllm/v1/worker/mamba_utils.py`：验证后的 state 后处理

当前 Qwen3.5 target 和 MTP 都不支持 `mamba_cache_mode="all"`。启用 prefix
caching 时应使用 `align`；不启用 prefix caching 时通常使用 `none`。

## 6. 两个容易混淆的 Depth 参数

| 参数 | 含义 |
|---|---|
| `mtp_num_hidden_layers` | checkpoint 中包含多少个 MTP decoder layer |
| `num_speculative_tokens` | 推理时一轮最多提出多少个 draft token |

例如 checkpoint 只有一个 MTP layer，但配置：

```text
num_speculative_tokens = 3
```

那么同一个轻量 MTP layer 会自回归运行多次，依次生成 d1、d2、d3。深度越大不
代表一定更快，因为越靠后的候选通常接受率越低。

内部 predictor 原本支持按 step 选择 MTP layer：

```python
current_step_idx = spec_step_idx % self.num_mtp_layers
mtp_layer = self.layers[current_step_idx]
```

但在当前分析的 vLLM 版本中，`Qwen3_5MTP.forward()` 没有继续向内部传递
`spec_step_idx`，通用 autoregressive speculator 也没有传这个参数，所以当前路径
实际一直使用 `layers[0]`。这对常见的 `mtp_num_hidden_layers=1` checkpoint 没有
影响；若使用多 MTP layer checkpoint，需要重新检查这条调用链。

对应代码：

- `vllm/model_executor/models/qwen3_5_mtp.py`：
  `Qwen3_5MultiTokenPredictor.forward()`、`Qwen3_5MTP.forward()`
- `vllm/config/speculative.py`：MTP config 转换与 runtime depth 校验

## 7. Dense、MoE 与多模态的粗略区别

三者的 draft/verify、接受和拒绝流程完全相同，差异主要发生在 MTP decoder
内部。

### Dense Qwen3.5

MTP decoder 的 FFN 使用普通 `Qwen3NextMLP`：

```text
full-attention MTP layer + dense MLP
```

### Qwen3.5-MoE

MTP decoder 的 FFN 使用 `Qwen3NextSparseMoeBlock`：

```text
full-attention MTP layer + routed experts/shared expert
```

当 MoE 使用 sequence parallel 时，token 维可能经过 reduce-scatter，因此 final
norm 前需要把 hidden state 和 residual all-gather 回完整 token 数。

### 多模态 Qwen3.5

`Qwen3_5MTP` 实现了 `SupportsMultiModal`。Draft prefill 可以接收已经计算好的
`inputs_embeds`，将 image/video embedding 合并到对应位置，而不是把多模态占位
token 当作普通词表 ID。

这些分支集中在：

- `vllm/model_executor/models/qwen3_5_mtp.py`
- `vllm/model_executor/models/qwen3_5.py`

## 8. 一句话总结

```text
连续猜对的 token 保留；从第一个猜错位置开始截断；使用 verify 的修正 token；
然后从修正 token 后重新执行 MTP。
```
