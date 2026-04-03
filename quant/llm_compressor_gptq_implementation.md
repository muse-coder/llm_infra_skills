# GPTQ 算法实现细节与可移植说明

> 这份文档专门讲 `GPTQ` 算法本身如何实现，包括 Hessian 统计、逐列量化、误差补偿、`actorder` 处理、和 NVFP4 的接合点。  
> 目标不是只解释 `llm-compressor`，而是把 GPTQ 提炼成可以迁移到其他仓库的实现手册。

---

## 目录

1. [实现目标](#1-实现目标)
2. [最小实现所需对象](#2-最小实现所需对象)
3. [Hessian 统计实现](#3-hessian-统计实现)
4. [GPTQ 主循环实现](#4-gptq-主循环实现)
5. [误差补偿到底做了什么](#5-误差补偿到底做了什么)
6. [actorder 如何影响实现](#6-actorder-如何影响实现)
7. [GPTQ 如何接入 NVFP4](#7-gptq-如何接入-nvfp4)
8. [最小可移植伪代码](#8-最小可移植伪代码)
9. [移植到其他仓库的工程建议](#9-移植到其他仓库的工程建议)
10. [检查清单与常见坑](#10-检查清单与常见坑)

---

## 1. 实现目标

如果把 GPTQ 从任何工程框架里剥离出来，它的目标很简单：

> 给定一个线性层权重 `W` 和该层输入的二阶统计 `H`，在目标量化格式约束下，找到一个更优的量化结果 `Q`，使层输出误差尽可能小。

这件事可以拆成四个子问题：

1. 如何收集该层输入统计，构造 Hessian 近似
2. 如何根据目标格式构造量化器
3. 如何做逐列量化
4. 如何把当前列误差传播到后续列

---

## 2. 最小实现所需对象

从实现角度，GPTQ 最少只依赖这 4 个核心对象：

### 2.1 `W`

- 当前层权重矩阵
- 统一视角下应规范成二维：`[out_features, in_features]`
- GPTQ 会修改它的工作副本

### 2.2 `H`

- 输入维度上的 Hessian 近似
- 维度为：`[in_features, in_features]`
- 来源于校准数据的输入激活二阶统计

### 2.3 `quantizer`

- 输入一个浮点列向量 `w`
- 输出量化后、再解释回浮点近似值的 `q`

注意：  
在 GPTQ 主循环里，通常更方便让量化器直接返回“**dequant 后的浮点近似值**”，因为误差传播在浮点域进行。

### 2.4 `Hinv`

- Hessian 处理后的逆近似
- 控制误差如何传播到后续列
- 是 GPTQ 能优于 RTN 的关键

---

## 3. Hessian 统计实现

### 3.1 `llm-compressor` 的实现入口

```python
# src/llmcompressor/modifiers/gptq/gptq_quantize.py

GPTQ_PRECISION = torch.float32

def make_empty_hessian(module, device=None):
    weight = module.weight
    num_columns = weight.shape[1]
    device = device if device is not None else weight.device
    return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)
```

### 3.2 Hessian 为什么按输入维度建

以线性层 `y = xW^T` 为例：

- 权重矩阵形状通常是 `[out_features, in_features]`
- GPTQ 是按“列”量化，也就是按输入维度方向量化
- 所以 Hessian 要刻画输入维度之间的相关性

因此：

```text
H.shape = [in_features, in_features]
```

### 3.3 输入张量如何规范化

```python
def accumulate_hessian(inp, module, H, num_samples):
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
```

对 `Linear` 而言，本质是：

1. `[batch, seq, hidden]`
2. reshape 成 `[N, hidden]`
3. transpose 成 `[hidden, N]`

这样后面做：

```python
H += inp.matmul(inp.t())
```

就是标准的 `X X^T` 统计。

### 3.4 为什么转成 FP32

```python
inp = inp.to(dtype=GPTQ_PRECISION)
inp = math.sqrt(2) * inp
H += inp.matmul(inp.t())
```

这里的 FP32 非常重要：

- 避免 bf16/fp16 统计误差过大
- 后面 Cholesky / inverse 更稳定

因此要区分：

- **校准前向 GEMM**：通常还是原模型 bf16/fp16
- **Hessian 统计 matmul**：FP32

### 3.5 可迁移版本的 Hessian 收集器

```python
def accumulate_hessian_linear(x, H):
    # x: [batch, seq, hidden] or [N, hidden]
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    x = x.t().float()                # [hidden, N]
    x = math.sqrt(2) * x
    H += x @ x.t()
    return H
```

这段已经足够迁移到大多数纯 PyTorch 框架。

---

## 4. GPTQ 主循环实现

### 4.1 主入口

```python
def quantize_weight(module, quant_args, hessian, blocksize=128, percdamp=0.01):
    strategy = quant_args.strategy
    actorder = quant_args.actorder
    global_scale = getattr(module, "weight_global_scale", None)
    final_shape = module.weight.shape
    final_dtype = module.weight.dtype
    W = module.weight.clone()
    H = hessian
```

### 4.2 先把权重规范化成可量化矩阵

```python
match module:
    case torch.nn.Conv2d():
        W = W.flatten(1)
    case transformers.Conv1D():
        W.transpose_(0, 1)
W = W.to(dtype=GPTQ_PRECISION)
```

迁移时一定要注意：

- 不同框架里的 `Linear` / `Conv1D` 权重布局可能不同
- GPTQ 主循环最好只处理统一好的 `[out_features, in_features]`

### 4.3 先求 qparams

```python
observer = Observer.load_from_registry(...)
scale, zero_point = observer(W)
```

这一步和格式强绑定：

- INT4/INT8：普通 affine / symmetric qparams
- NVFP4：需要 `global_scale + group scale`

所以把 `quantizer` 和 `qparams builder` 独立抽象出来，会大大降低移植难度。

### 4.4 处理 `actorder`

```python
if strategy in (QuantizationStrategy.GROUP, QuantizationStrategy.TENSOR_GROUP):
    g_idx = torch.arange(num_columns, device=W.device, dtype=torch.int) // quant_args.group_size

    if actorder == ActivationOrdering.GROUP:
        W, H, perm = _apply_activation_ordering(W, H)
        scale, zero_point = observer(W)

    elif actorder == ActivationOrdering.WEIGHT:
        W, H, perm = _apply_activation_ordering(W, H)
        g_idx = g_idx[perm]
```

实现层面记住三件事：

1. `static`：不动
2. `weight`：重排列，但 qparams 不在重排后重算
3. `group`：重排列，而且 qparams 在重排后重算

### 4.5 求 `Hinv`

```python
try:
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H
except torch._C._LinAlgError:
    Hinv = H = torch.eye(num_columns, dtype=H.dtype, device=H.device)
```

这里的工程含义是：

- 不直接裸 inverse
- 先加阻尼
- 用 Cholesky 做稳定求解
- 失败时回退 identity

迁移时这个分支非常建议保留。

### 4.6 逐块遍历

```python
for i1 in range(0, num_columns, blocksize):
    i2 = min(i1 + blocksize, num_columns)
    count = i2 - i1

    W1 = W[:, i1:i2].clone()
    Q1 = torch.zeros_like(W1)
    Err1 = torch.zeros_like(W1)
    losses1 = torch.zeros_like(W1)
    Hinv1 = Hinv[i1:i2, i1:i2]
```

为什么要 block-wise：

- 节省显存
- 避免一次性操作整个大矩阵
- 保持较好的并行性和稳定性

### 4.7 逐列量化

```python
for i in range(count):
    w = W1[:, i]
    d = Hinv1[i, i]
    q = w.clone()

    # quantize column
    ...

    Q1[:, i] = q
    losses1[:, i] = (w - q) ** 2 / d**2

    err1 = (w - q) / d
    w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    W1[:, i:] -= w1_err
    Err1[:, i] = err1
```

这就是 GPTQ 的核心。

### 4.8 块内和块外传播

```python
# 当前块写回
W[:, i1:i2] = Q1
losses += torch.sum(losses1, 1) / 2

# 向块外传播
w_err = Err1.matmul(Hinv[i1:i2, i2:])
W[:, i2:] -= w_err
```

所以 GPTQ 的补偿有两层：

- 块内传播
- 块外传播

这也是为什么它最终结果比“只在块内修正”的简化版更好。

---

## 5. 误差补偿到底做了什么

### 5.1 不是改已量化列

GPTQ 不会回头改已经写进 `Q1` 的列。  
它改的是：

> **后面还没量化的浮点列。**

### 5.2 为什么这样做

因为目标不是让某一列更接近原值，而是让整个层输出误差更小。

当前列一旦量化产生误差：

```text
e = w - q
```

GPTQ 会问：

> 这个误差能不能通过调整后面列来部分抵消？

而 `Hinv` 就是在回答这个问题。

### 5.3 直观例子

假设有三列：

```text
c0, c1, c2
```

普通量化：

```text
q0 = Q(c0)
q1 = Q(c1)
q2 = Q(c2)
```

GPTQ：

```text
q0 = Q(c0)
e0 = c0 - q0

c1' = c1 - compensate(e0)
c2' = c2 - compensate(e0)

q1 = Q(c1')
e1 = c1' - q1

c2'' = c2' - compensate(e1)
q2 = Q(c2'')
```

最终：

- `q1` 不是原始 `c1` 的直接量化
- `q2` 不是原始 `c2` 的直接量化

这正是 GPTQ 的精华。

### 5.4 代码里真正改权重的地方

```python
W1[:, i:] -= w1_err
W[:, i2:] -= w_err
```

如果你在别的仓库重写 GPTQ，最关键的不是“做了 quantize”，而是**必须真的把这两类传播做出来**。

---

## 6. actorder 如何影响实现

### 6.1 `static`

最简单：

- 不重排
- qparams 按原始列布局求
- 不需要 `weight_g_idx`

迁移复杂度最低。

### 6.2 `weight`

重排列优先量化重要列：

```python
elif actorder == ActivationOrdering.WEIGHT:
    W, H, perm = _apply_activation_ordering(W, H)
    g_idx = g_idx[perm]
```

量化完成后恢复：

```python
if actorder == ActivationOrdering.WEIGHT:
    invperm = torch.argsort(perm)
    W = W[:, invperm]
```

特点：

- 误差传播路径变了
- 导出结构通常不需要新增字段

### 6.3 `group`

它是最复杂的，因为会重算 qparams：

```python
if actorder == ActivationOrdering.GROUP:
    W, H, perm = _apply_activation_ordering(W, H)
    scale, zero_point = observer(W)
```

并在最后持久化：

```python
elif actorder == ActivationOrdering.GROUP:
    invperm = torch.argsort(perm)
    W = W[:, invperm]
    g_idx = g_idx[invperm]
    has_gidx = True
```

迁移含义：

- 不仅主循环要支持
- 导出格式也要支持
- runtime 也要支持

如果你只是 first-pass 实现 GPTQ，建议先不要上 `group`。

---

## 7. GPTQ 如何接入 NVFP4

### 7.1 先分清两层

#### GPTQ 是算法层
- 负责 Hessian
- 负责逐列量化
- 负责误差补偿

#### NVFP4 是格式层
- 负责定义可表示值集合
- 负责定义 `global_scale + local scale`
- 负责定义导出格式

### 7.2 在 `llm-compressor` 里的接合点

GPTQ 主循环内部，真正和 NVFP4 发生接合的是这里：

```python
global_scale = getattr(module, "weight_global_scale", None)

q = fake_quantize(
    q,
    scale[:, group_index],
    zero_point[:, group_index],
    altered_qargs,
    global_scale=global_scale,
)
```

也就是说：

- GPTQ 并不知道“什么叫 NVFP4”
- 它只知道“这里有一个 quantizer，它需要 `scale / zp / global_scale`”

所以迁移时最推荐的做法是：

> **把 GPTQ 主循环和具体格式量化器彻底解耦。**

### 7.3 可迁移的 `quantizer` 抽象

```python
class Quantizer:
    def quantize_column(self, w, column_idx):
        # 查当前列该用的 qparams
        # 把 w 投影到目标格式可表示集合
        # 返回 dequant 后的浮点近似值 q
        return q
```

然后 GPTQ 主循环只依赖：

```python
q = quantizer.quantize_column(w, column_idx)
```

这样：

- INT4 可以用
- FP8 可以用
- NVFP4 也可以用

### 7.4 NVFP4 特有的移植点

如果目标是 NVFP4，你至少还要补这三件事：

1. **weight global scale 生成**
2. **group qparams 生成**
3. **可能的 fused global scale 对齐**

其中 fused global scale 对齐在这个仓库里是为了 vLLM 要求，不一定是 GPTQ 数学本体必需，但如果目标 runtime 是 vLLM/Blackwell，这个行为需要保留。

---

## 8. 最小可移植伪代码

下面是一段比框架更抽象的 GPTQ 伪代码，可以直接作为新仓库实现骨架。

```python
import math
import torch


def accumulate_hessian_for_linear(x, H):
    if x.ndim == 3:
        x = x.reshape(-1, x.shape[-1])
    x = x.t().float()
    x = math.sqrt(2) * x
    H += x @ x.t()
    return H


def gptq_quantize_weight(W, H, quantizer, block_size=128, damp_ratio=0.01):
    # W: [out_features, in_features]
    W = W.clone().float()
    H = H.clone().float()

    # mask dead diagonal
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # damped inverse Hessian
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp_ratio * torch.mean(torch.diag(H))

    try:
        U = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(U)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except RuntimeError:
        Hinv = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

    Q = torch.zeros_like(W)

    for i1 in range(0, W.shape[1], block_size):
        i2 = min(i1 + block_size, W.shape[1])
        Wblk = W[:, i1:i2].clone()
        Hblk = Hinv[i1:i2, i1:i2]
        Err = torch.zeros_like(Wblk)

        for i in range(i2 - i1):
            w = Wblk[:, i]
            d = Hblk[i, i]

            q = quantizer(w, i1 + i)
            Q[:, i1 + i] = q

            err = (w - q) / d
            Err[:, i] = err

            # propagate inside current block
            Wblk[:, i:] -= err.unsqueeze(1) @ Hblk[i, i:].unsqueeze(0)

        # commit current block
        W[:, i1:i2] = Q[:, i1:i2]

        # propagate to remaining columns
        if i2 < W.shape[1]:
            W[:, i2:] -= Err @ Hinv[i1:i2, i2:]

    return Q
```

### 8.1 如果你要支持 actorder

可以在主函数前后加两段：

```python
perm = None

if actorder == "weight":
    perm = torch.argsort(torch.diag(H), descending=True)
    W = W[:, perm]
    H = H[perm][:, perm]
    g_idx = g_idx[perm]

elif actorder == "group":
    perm = torch.argsort(torch.diag(H), descending=True)
    W = W[:, perm]
    H = H[perm][:, perm]
    scale, zp = recompute_qparams_on_permuted_weight(W)

# ... GPTQ main loop ...

if perm is not None:
    invperm = torch.argsort(perm)
    W = W[:, invperm]
```

如果是 `group`，还要额外保留 `g_idx = g_idx[invperm]`。

---

## 9. 移植到其他仓库的工程建议

### 9.1 推荐的模块拆分

如果你要在新仓库里实现 GPTQ，建议拆成这几层：

- `hessian.py`
  - 收集输入统计
  - 构建 `H`

- `qparams.py`
  - 生成 `scale / zero_point / global_scale`

- `formats/`
  - `int4.py`
  - `int8.py`
  - `nvfp4.py`

- `gptq.py`
  - 纯 GPTQ 主循环

- `export.py`
  - 导出 checkpoint

- `runtime_adapter.py`
  - 对接 vLLM / 其他 runtime

### 9.2 最推荐的实现顺序

不要一开始就做全套。建议按下面顺序：

1. `W4A16 + static`
2. `W4A16 + weight`
3. `NVFP4A16`
4. `NVFP4`
5. `group + weight_g_idx`

理由：

- 先验证 GPTQ 主循环本身
- 再逐步增加格式复杂度
- 最后再处理 runtime/导出复杂语义

### 9.3 迁移时最推荐的验证方式

每加一层能力都做三种验证：

1. 数值层
   - MSE
   - cosine similarity

2. 端到端层
   - perplexity
   - generation 质量

3. 结构层
   - checkpoint 字段是否齐全
   - runtime 是否能正常加载

---

## 10. 检查清单与常见坑

### 10.1 算法正确性检查

- Hessian 是否按输入维度建立
- Hessian 是否按样本数归一化
- 是否真的做了块内和块外两级传播
- Cholesky 失败时是否有 fallback
- 最终是否恢复原始权重形状

### 10.2 量化器检查

- `quantizer` 返回的是浮点近似值而不是纯 low-bit code 吗
- group / tensor / block 逻辑是否查对了 qparams
- NVFP4 是否真的同时用了 `global_scale` 和 group scale

### 10.3 actorder 检查

- `weight` 是否只是重排列而不重算 qparams
- `group` 是否真的重算了 qparams
- `group` 是否正确导出 `weight_g_idx`

### 10.4 导出与 runtime 检查

- checkpoint 是否和 runtime 预期格式一致
- 普通 NVFP4 和 GPTQ+NVFP4 是否能共享 runtime 路径
- 对于 `group`，runtime 是否真的会读取 `weight_g_idx`

### 10.5 最容易踩坑的 5 件事

1. **把 Hessian 建成 `[out_features, out_features]`**
   - 这是错的，应该是输入维度

2. **量化器返回 low-bit code 而不是浮点近似值**
   - 会导致误差传播逻辑不成立

3. **忘了对 `TENSOR_GROUP` 处理 `global_scale`**
   - 这样做出来的就不是等价的 NVFP4

4. **只做块内传播，不做块外传播**
   - 会损失 GPTQ 的关键效果

5. **在 `group` 模式下不保存 `weight_g_idx`**
   - 导出后 runtime 可能用错 weight qparams

---

## 最后一句

如果你要把 GPTQ 从 `llm-compressor` 迁移到别的仓库，真正要带走的不是某个类，而是这条主线：

> **输入激活 -> Hessian 近似 -> qparams -> 逐列量化 -> 误差补偿 -> 恢复布局 -> 导出格式适配**

只要这条主线保持不变，GPTQ 就可以独立于具体仓库存在。  
而 `NVFP4`、`INT4`、`FP8` 这些，只是接到这条主线上的不同量化器。
