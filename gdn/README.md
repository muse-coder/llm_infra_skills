# GDN（Gated DeltaNet）Prefill 知识库

> 对象：Qwen3-Next / Qwen3.5 的线性注意力层，**推理 prefill 阶段**（chunked 形式）
> 硬件范围：数据中心 Blackwell（**SM100 / SM103**）为主，Hopper 作对照
> 三个算子库：`flash-linear-attention`（Triton）、`FlashInfer`（CuTe DSL）、`FlashQLA`（Qwen 官方，TileLang）

---

## 文档索引

**先读算法，再读实现。** 三篇实现文档都直接引用算法文档的符号、依赖图编号（环节 ①–⑤）和"四个数学杠杆"编号，跳读会读不通。

| 文档 | 内容 |
|---|---|
| [`GDN_Algorithm.md`](GDN_Algorithm.md) | **必读前置。** 线性注意力 → delta rule → GDN 的演化；Qwen3.5 的参数化与 GVA 配置；chunk 形式的完整三步推导（换元消门 → UT transform → 相似变换）；chunk size = 64 的定量分析；**五环节依赖图与"环节 ④ 是唯一串行瓶颈"这个核心结论**；算术强度与并行度饥饿的量化；**四个数学杠杆**（门与求逆可交换、分块 Schur 补、仿射结合律、指数遗忘）及各自证明；误差界与精度分配原则；decode 作为 $C=1$ 的退化；符号表与三处跨库 API 陷阱 |
| [`FLA_Triton_Baseline.md`](FLA_Triton_Baseline.md) | **flash-linear-attention 的优化方案。** 8 kernel 流水线；唯一的重量级融合（$C=64$ 下 kkt + solve_tril 寄存器内融合）；两级三角求逆与 tf32 精度取舍；全局 `exp2` 化；编译期特化作为主要抽象手段；$w/u$ 为什么保留并物化（训练定位的必然）；**默认递推路径为什么完全不解决串行问题**；intracard CP 的门槛；HBM 流量完整核算（213 → 33 FLOP/Byte）；刻意不做的 11 件事；backend 分派机制 |
| [`FlashInfer_GDN_Blackwell.md`](FlashInfer_GDN_Blackwell.md) | **FlashInfer 的优化方案（SM100/SM103）。** 非 CP 巨核：384 线程 12 warp 的角色划分、**双独立 MMA 发射流**、SMEM 226 KB 与 **TMEM 恰好 512 列**的分配表、两 chunk 成对（真正理由是求逆并行度）、8×8 起步 3 层合并的求逆（走 warp 级 MMA 而非 tcgen05）、零状态编译期剥离、有界 TMA descriptor 让 varlen 尾块全免费；CP 路径：**四阶段精确并行扫描**及其 workspace 定价、gate-free 求逆 + 门夹心（与非 CP 相反的选择）；分页状态池与 fp8 状态量化；SM90 对照 |
| [`FlashQLA_GDN_Blackwell.md`](FlashQLA_GDN_Blackwell.md) | **FlashQLA 的优化方案（SM100/SM103）。** 三段式（CP 开启时六段）调度及其融合边界的理由；gate-free 三角求逆与 Λ 后置；$w/u$ 塌缩；4 warpgroup + producer 拆 4 warp 的结构、TMEM 448/512 列与状态拆半；**核心：门驱动的近似上下文并行**——逐 (段, head) 的 −10 阈值、预热语义、$M$ 矩阵精确兜底、$\sqrt{}$ 段长模型、SM100 上的 256-chunk 阈值坑；标量衰减在 log 域提取；TileLang 到底改变了什么（诚实评估）；GB200 实测数据 |

---

## 一页速览

### 问题

GDN 的 chunk 形式把七个矩阵乘打包起来喂 Tensor Core，理论算术强度约 213 FLOP/Byte（**算力受限**）。但状态递推（环节 ④）严格串行，把并行度锁死在

$$
P = B \times H_v
$$

Qwen3.5-397B 在 TP8 下每卡只有 8 个 v-head，单请求 prefill 就是 **8 个并行任务对上百个 SM**。序列越长、TP 越高越糟，而这恰恰是最常见的配置。

> **头号问题是并行度饥饿（数量级），不是算力也不是带宽；融合问题是次要的（常数因子）。**

### 三家的答案

| | FLA | FlashInfer | FlashQLA |
|---|---|---|---|
| 语言 | Triton | CuTe DSL | TileLang |
| kernel 数（前向） | 8（核心 4） | 1（CP 时 4） | 3（CP 时 6） |
| 破解环节 ④ | ❌ 不破解（CP 需显式开启 + ≥24k + 仅 varlen） | ✅ **杠杆三**，代数精确 | ✅ **杠杆四**，近似 + 杠杆三兜底 |
| 门在求逆内/外 | 内 | 非 CP 内 / **CP 外** | **外** |
| $w/u$ | 保留并物化（为反向） | 合并 | 合并（塌缩） |
| 反向传播 | ✅ | ❌ | ✅ |
| 分页状态池 | ❌ | ✅ **唯一** | ❌ |
| fp8 状态 I/O | ❌ | ✅ **唯一** | ❌ |
| 任意 head dim / 非 NVIDIA | ✅ **唯一** | ❌ | ❌ |

### 读文档时请随时回看的那张表

算法文档 §5 的五环节依赖表是全部实现的蓝图：

| 环节 | 内容 | 依赖 $S_{\mathrm{prev}}$ |
|---|---|---|
| ① 门前处理 | $g=\mathrm{cumsum}(\log\alpha)$、$\Lambda$ | ❌ |
| ② UT transform | $KK^{\top}\to A_{\mathrm{raw}}$（三角求逆） | ❌ |
| ③ value 修正 | $V_{\mathrm{new}}$ | ✅ |
| ④ **状态递推** | $S_{\mathrm{next}}$ | ✅ **严格串行** |
| ⑤ 输出组装 | $O$ | ✅ |

三个库的 kernel 边界、warp 分组、缓冲级数几乎都能追回这张表。两个例子：

- **FlashInfer** 按"是否依赖状态"划分 warp 组（CG0 管 ①②，CG1 管 ③④⑤），于是自然产生了"两条独立 MMA 发射流"——不依赖状态的 GEMM 可以任意超前。
- **FlashQLA** 按"谁拥有哪个量"划分（状态 / $V_{\mathrm{new}}$ / 输出各一个 warpgroup）。

**同一张依赖图，两种切法，两种结构。**

---

## 阅读注意

1. **区分"算法层"与"实现层"的优化。** 算法层的重排（gate-free 求逆、$w/u$ 塌缩、标量衰减提取、仿射结合律、遗忘截断）是可移植的洞察；实现层的调度（TMEM 分配、barrier arrive count、流水级数）绑死在具体架构上。前者值得记住，后者主要是"为什么这么分配"的推理过程值得记住。

2. **源码注释与 docstring 会过期。** `flashinfer` 的 `gated_delta_net_chunked.py` 文件头 docstring 几乎每行表格都与代码不符（详见其文档 §1）；`flashinfer` 的 `benchmarks/README.md` 说 SM90 是 C++ 实现（实际是 CuTe DSL）。**三篇实现文档里的所有结论都以代码为准，并标注了 `file:line`。**

3. **目录结构会误导。** `FlashQLA/blackwell/` 下的 `kkt_solve.py`、`cp_fwd.py`、`cp_bwd.py` 与 `hopper/` 逐字节相同；`flashinfer` 的 `collective_inverse_hmma.py` 只被 CP 的 T 预计算用，非 CP 主 kernel 有自己的 fork。

4. **SM103 基本等于 SM100。** 两个库都没有实质的 SM103 分支，唯一差异是 FlashInfer 的 CP 路径会编译成 `sm_103a`。

5. **性能数字的可信度。** `flashinfer` 仓库里没有任何性能数据，而它自带 benchmark 的 FLOP 模型只数了 7 个 GEMM 里的 2 个（**TFLOPS 低估约 3–4 倍**）。`FlashQLA` 有 GB200/H200 的实测 ms 数据，但用的是 flashinfer 0.6.14。**任何跨库结论都需要在目标硬件上重测，并确认 CP 是否实际启用。**
