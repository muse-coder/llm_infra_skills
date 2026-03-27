# Warp 0: TMA Load Warp 详解

> 源码位置: [`deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh`](../deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh)
> TMA 工具函数: [`deep_gemm/include/deep_gemm/common/tma_utils.cuh`](../deep_gemm/include/deep_gemm/common/tma_utils.cuh)
> Scheduler: [`deep_gemm/include/deep_gemm/common/scheduler.cuh`](../deep_gemm/include/deep_gemm/common/scheduler.cuh)

---

## 概览

Warp 0 是 **TMA (Tensor Memory Accelerator) 数据搬运 warp**，负责将矩阵 A、B 及其 Scale Factor 从 Global Memory 异步搬运到 Shared Memory。它是整个 GEMM 流水线的数据供给端。

**核心特征：**
- 只有 **1 个 thread** 发射 TMA 指令（`warp_idx == 0 and cute::elect_one_sync()`）
- 采用 **persistent kernel** 模式，一个 block 常驻 SM，处理多个 tile
- 使用 **多 stage 流水线**，与 Warp 1 (MMA) 和 Warp 2 (SF Transpose) 异步协作
- 在 2-CTA cluster 模式下，**两个 CTA 各自独立发射 TMA**

---

## Step 0: TMA Descriptor Prefetch

在进入主循环之前，Warp 0 会预取所有 TMA descriptor 到 cache，减少后续 TMA 发射时的延迟：

```cpp
// sm100_fp8_gemm_1d1d.cuh
if (warp_idx == 0 and cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&tensor_map_a);
    cute::prefetch_tma_descriptor(&tensor_map_b);
    cute::prefetch_tma_descriptor(&tensor_map_sfa);
    cute::prefetch_tma_descriptor(&tensor_map_sfb);
    cute::prefetch_tma_descriptor(&tensor_map_cd);
}
```

TMA descriptor 是 host 端创建的 128 字节结构体，描述了全局内存的 layout、swizzle 模式等信息。预取到 cache 后，后续每次 TMA 发射都能快速访问。

> **底层 PTX 指令**：`prefetch.tensormap [addr];`
>
> **PTX ISA 官方描述**（8.5 / 9.2）：
> *"If the `.tensormap` qualifier is specified then the prefetch instruction brings the cache line containing the specified address in the `.const` or `.param` memory state space **into the cache** for subsequent use by TMA instructions."*
>
> 注意：与普通 `prefetch` 指令（可指定 `.L1` / `.L2` 级别）不同，`prefetch.tensormap` **没有 cache level 修饰符**，PTX 规范只保证"预取到 cache"，具体预取到哪一级 cache（L1 / L2）是硬件实现细节，官方未明确指定。TMA descriptor 通过 kernel 参数传入，属于 `.param` 地址空间。

> **原始知识库遗漏**：`MoE grouped gemm B200.md` 中完全没有提到这个 prefetch 步骤。

---

## Step 1: Persistent Kernel 与 Scheduler

### 背景知识：Block、SM 和 Tile 的关系

#### Block 和 SM

在 CUDA 编程模型中，**block（线程块）** 是 GPU 调度的基本单位，每个 block 被分配到一个 **SM（Streaming Multiprocessor）** 上执行。

在 **persistent kernel** 模式下，kernel launch 时恰好 launch **`kNumSMs` 个 block**（B200 有 132 个 SM，所以 launch 132 个 block），**每个 SM 恰好分配一个 block**，block 常驻在 SM 上不退出，通过内部循环依次处理多个 tile，直到所有 tile 处理完毕才退出。

这与普通 kernel 不同——普通 kernel 通常 launch 大量 block（比如几千个），由 GPU 硬件调度器动态分配到 SM 上。

#### Tile 是什么

Tile 是针对**输出矩阵 C** 定义的，位于 **(M, N) 二维平面**上。GEMM 计算 `C[M, N] = A[M, K] × B[K, N]`，一个 tile `(m_block_idx, n_block_idx)` 代表 C 矩阵中一个 `BLOCK_M × BLOCK_N` 大小的输出块。

```
C[M, N] = A[M, K] × B[K, N]

                    B 矩阵 [K, N]
                    ┌──────────────────┐
                    │ n=0  n=1  n=2 ...│
                    │                  │
                    │  K 行 × N 列     │
                    └──────────────────┘

A 矩阵 [M, K]       C 矩阵 [M, N]（输出）
┌──────────┐        ┌──────────────────┐
│ M 行     │        │ tile   tile  tile│
│ × K 列   │   →    │ (0,0)  (0,1)(0,2)│
│          │        │ tile   tile  tile│
│          │        │ (1,0)  (1,1)(1,2)│
└──────────┘        │ ...              │
                    └──────────────────┘
                    每个 tile = BLOCK_M × BLOCK_N
```

**K 维度不体现在 tile 编号中**——为了计算一个 C tile，需要沿 K 维度循环累加，这个循环在 block **内部**完成：

```
计算 tile(2, 1) 即 C 矩阵的第 2 个 M-block、第 1 个 N-block：

for k_block in range(num_k_blocks):
    C[2,1] += A[2, k_block] × B[k_block, 1]
              ↑ BLOCK_M×BLOCK_K   ↑ BLOCK_K×BLOCK_N

K 维度的循环是 Warp 0 (TMA) 和 Warp 1 (MMA) 流水线协作完成的
```

#### 完整的调度图景

```
GPU Launch: 只 launch kNumSMs 个 blocks，每个 block 常驻 SM

blockIdx.x=0 (SM#0)   → tile(0,0) → tile(0,2) → tile(1,1) → ... → 退出
blockIdx.x=1 (SM#1)   → tile(0,1) → tile(1,0) → tile(1,2) → ... → 退出
...
blockIdx.x=131 (SM#131) → tile(...) → tile(...) → ... → 退出

每个 block 处理一个 tile 时：
  Warp 0: 沿 K 维度循环发射 TMA，加载 A 和 B 的数据块到 SMEM
  Warp 1: 沿 K 维度循环做 MMA，累加到 C 的 tile 中
  Warp 2: 转置 Scale Factor
  处理完一个 tile 的所有 K-block 后 → epilogue 写回 C → 领取下一个 tile
```

### Persistent Kernel vs 非 Persistent Kernel

> 参考资料：
> - [CUTLASS Grouped Kernel Schedulers](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/grouped_scheduler.html)
> - [NVIDIA Forum: Question about persistent kernel concept](https://forums.developer.nvidia.com/t/question-about-persistent-kernel-concept/320600)

#### 非 Persistent Kernel（传统方式）

传统 GEMM kernel 为**每个 tile launch 一个 block**。如果 C 矩阵有 1024 个 tile，就 launch 1024 个 block，由 GPU 硬件调度器动态将这些 block 分配到 SM 上执行：

```
传统方式: launch num_tiles 个 blocks

block 0  → tile(0,0) → 退出
block 1  → tile(0,1) → 退出
block 2  → tile(0,2) → 退出
...
block 1023 → tile(31,31) → 退出

GPU 硬件调度器负责将 1024 个 block 动态分配到 132 个 SM 上
每个 block 只处理一个 tile，处理完就退出，SM 再接收下一个 block
```

**优点：**
- **编程简单**：一个 block 对应一个 tile，逻辑清晰
- **硬件调度器自动负载均衡**：SM 空闲时自动领取下一个 block
- **适合 tile 数量少的场景**：当 tile 数 ≤ SM 数时，所有 tile 可以一波并行完成，没有额外开销

**缺点：**
- **block 调度开销**：每个 block 的创建、分配、销毁都有硬件开销（虽然单次很小，但 1024 个 block 累积起来不可忽略）
- **无法跨 tile 复用状态**：每个 block 退出后，Shared Memory 和寄存器中的数据全部丢失，下一个 block 需要重新初始化
- **不适合 Grouped GEMM**：MoE 场景中有多个 expert（多个小 GEMM），如果每个 GEMM 单独 launch 一个 kernel，kernel launch 开销会非常大

#### Persistent Kernel（DeepGEMM 采用的方式）

Persistent kernel 只 launch **`kNumSMs` 个 block**（等于 SM 数量），每个 block 常驻在 SM 上，通过内部循环依次领取并处理多个 tile：

```
Persistent 方式: launch kNumSMs 个 blocks，每个 block 常驻 SM

block 0 (SM#0)   → tile(0,0) → tile(1,2) → tile(3,1) → ... → 退出
block 1 (SM#1)   → tile(0,1) → tile(2,0) → tile(3,2) → ... → 退出
...
block 131 (SM#131) → tile(0,3) → tile(2,1) → ... → 退出

每个 block 通过 Scheduler 的 get_next_block() 循环领取下一个 tile
所有 tile 处理完毕后才退出
```

**优点：**
- **消除 block 调度开销**：只有 `kNumSMs` 个 block，一次性全部常驻，没有反复创建/销毁的开销
- **数据持久化**：block 常驻 SM 意味着 Shared Memory、寄存器中的数据可以跨 tile 复用（如 TMA descriptor prefetch 只需做一次）
- **天然适合 Grouped GEMM**：多个 expert 的 tile 可以在同一个 kernel 内由 Scheduler 统一分配，无需多次 kernel launch。CUTLASS 文档中明确指出："grouped kernel launches multiple problems within a single CUDA kernel launch"
- **更好的 L2 Cache 控制**：通过 Scheduler 的 tile swizzle，可以精确控制 tile 遍历顺序来优化 L2 cache 命中率（传统方式中 block 的执行顺序由硬件调度器决定，不可控）
- **减少 tail effect**：传统方式中最后一波 block 可能只占用部分 SM（如 1024 个 tile 在 132 个 SM 上，最后一波只有 1024 % 132 = 100 个 block 活跃，32 个 SM 空闲）。Persistent kernel 中每个 block 处理的 tile 数量接近均等，负载更均衡

**缺点：**
- **编程复杂度高**：需要自己实现 Scheduler 来分配 tile，处理 group 边界、swizzle 等逻辑
- **需要精确的 occupancy 控制**：必须确保 launch 的 block 数等于 SM 数（或 SM 数 × occupancy），否则可能导致 SM 空闲或 block 无法全部常驻
- **调试困难**：所有 tile 在同一个 kernel 中处理，出错时难以定位是哪个 tile 的问题

#### 什么时候用 Persistent Kernel

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| **大矩阵 GEMM**（tile 数 >> SM 数） | ✅ Persistent | 消除 block 调度开销，控制 L2 cache |
| **Grouped GEMM / MoE** | ✅ Persistent | 多个 problem 在一个 kernel 内处理，避免多次 launch |
| **小矩阵 GEMM**（tile 数 ≤ SM 数） | ❌ 传统 | tile 一波就能并行完成，persistent 的 Scheduler 反而是额外开销 |
| **需要与其他 kernel 并发** | ❌ 传统 | persistent kernel 会占满所有 SM，阻塞其他 kernel |
| **简单原型开发** | ❌ 传统 | 编程简单，快速验证 |

#### DeepGEMM 中的 Persistent Kernel

DeepGEMM 的 persistent kernel 实现在 `scheduler.cuh` 中，核心循环如下：

```cpp
// 伪代码
Scheduler scheduler(shape_m, shape_n, shape_k, grouped_layout);

while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // Warp 0: 沿 K 维度循环发射 TMA
    // Warp 1: 沿 K 维度循环做 MMA
    // Warp 2: 转置 Scale Factor
    // Epilogue: 写回 C
    // → 回到 get_next_block() 领取下一个 tile
}
```

`get_next_block` 内部通过 `current_iter * kNumSMs + blockIdx.x` 计算全局 tile 索引，实现 round-robin 分配：block 0 处理 tile 0, 132, 264, ...；block 1 处理 tile 1, 133, 265, ...；以此类推。

### Scheduler 的 Group 分配（M-Grouped GEMM）

参考 `scheduler.cuh` 中的 `get_next_block`：

```cpp
if constexpr (kGemmType == GemmType::MGroupedMasked) {
    while (true) {
        if (current_group_idx == kNumGroups)
            return false;
        // 从 GPU 全局内存读取当前 expert 的有效 M 行数
        num_m_blocks = ceil_div(
            static_cast<uint32_t>(__ldg(grouped_layout + current_group_idx)), BLOCK_M);
        const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
        if (next_block_idx < current_m_block_cumsum * num_n_blocks)
            break;
        current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
    }
    // Swizzle tile 顺序以提高 L2 cache 命中率
    get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks,
                           m_block_idx, n_block_idx);
}
```

**关键点：**
- 每个 expert 为一个 group，`kNumGroups` 是 expert 数量
- MoE 中每个 expert 获取的 token 数是动态的，`grouped_layout` 记录每个 group 的有效 M 行数
- `__ldg` 原语走 read-only L1/Tex cache，快速读取
- `get_swizzled_block_idx` 对 tile 顺序做 swizzle，让相邻 SM 复用 L2 cache 数据（详见下方深入章节）

### 深入：Tile Swizzle 与 L2 Cache 优化

> 源码位置：[`scheduler.cuh`](../deep_gemm/include/deep_gemm/common/scheduler.cuh) 第 94 行 `get_swizzled_block_idx`，第 14 行 `get_num_1d_blocks_per_group`

#### `kNum1DBlocksPerGroup` 是什么

`kNum1DBlocksPerGroup` 是一个**编译期常量**，表示在 **tile 级别**（即 CTA/block 级别）的分组大小——**一个 swizzle group 在 primary 维度上包含多少个 tile block**。它不是 warp 级别或 thread 级别的概念，而是 **SM 调度层面的 tile 遍历分组**。

具体来说：
- 在 persistent kernel 中，所有 tile 被编号为一个线性序列 `0, 1, 2, ...`
- `kNum1DBlocksPerGroup` 将这些 tile 按 primary 维度分成若干 group，每个 group 包含 `kNum1DBlocksPerGroup` 个连续的 primary 维度 block × 全部 secondary 维度 block
- 候选值为 **8 或 16**，通过 `get_num_1d_blocks_per_group` 在编译期选择最优值

选择逻辑（以 `kIsMulticastOnA = false`，即 grouping on M 为例）：

```cpp
// scheduler.cuh
for (const auto& candidate: {8u, 16u}) {
    const auto& usage = candidate * BLOCK_M +                          // group 内 M 维度覆盖的数据量
                        constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // 同时活跃的 N 维度数据量
    if (usage < min_usage)
        min_usage = usage, num_best_blocks = candidate;
}
```

优化目标是**最小化 L2 cache 的工作集大小**：
- **`candidate * BLOCK_M`**：一个 group 内 A 矩阵覆盖的行数（需要驻留在 L2 中）
- **`ceil_div(kNumSMs, candidate) * BLOCK_N`**：所有 SM 同时活跃时，B 矩阵覆盖的列数（同时有 `kNumSMs / candidate` 个 group 在不同 N 列上工作）
- 两者之和就是 L2 中同时需要驻留的数据量，选择使这个值最小的 candidate

#### Swizzle 的具体映射过程

以 `kIsMulticastOnA = false`（grouping on M）为例，假设 `num_m_blocks = 32`, `num_n_blocks = 4`, `kNum1DBlocksPerGroup = 8`：

```cpp
// get_swizzled_block_idx 核心逻辑
primary_num_blocks   = num_m_blocks;   // = 32
secondary_num_blocks = num_n_blocks;   // = 4
num_blocks_per_group = 4 * 8 = 32;    // 每个 group 包含 32 个 tile

group_idx      = block_idx / 32;       // 第几个 group
first_block_idx = group_idx * 8;       // group 在 M 维的起始 block
in_group_idx   = block_idx % 32;       // group 内的局部索引

m_block_idx = first_block_idx + (in_group_idx % 8);  // M 维：在 group 内循环
n_block_idx = in_group_idx / 8;                       // N 维：每 8 个 tile 换一列
```

用具体例子展示 group 0（block_idx 0~31）的映射：

```
block_idx:  0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | 16 ... | 24 ...
            ─────────────────────── ─────────────────────────
m_block:    0  1  2  3  4  5  6  7 | 0  1  2  3  4  5  6  7 | 0  ... | 0  ...
n_block:    0  0  0  0  0  0  0  0 | 1  1  1  1  1  1  1  1 | 2  ... | 3  ...
```

**关键观察**：在一个 group 内，**先遍历 M 维度的 8 个 block，再移动到下一个 N 列**。

#### 为什么 Swizzle 能提高 L2 Cache 命中率

**没有 Swizzle 的朴素遍历（row-major）：**

```
遍历顺序: (M0,N0) → (M0,N1) → (M0,N2) → (M0,N3) → (M1,N0) → (M1,N1) → ...

SM#0 加载: A[M0], B[N0]
SM#1 加载: A[M0], B[N1]   ← A[M0] 可复用，但 B 不断换列
SM#2 加载: A[M0], B[N2]
SM#3 加载: A[M0], B[N3]
SM#4 加载: A[M1], B[N0]   ← 回到 N0 时，B[N0] 可能已被 L2 驱逐
```

问题：当 N 很大时，遍历完一整行 N 后回到下一行 M 时，之前 B 的数据早已被 L2 驱逐。**B 矩阵的 L2 复用率很低**。

**Swizzle 后的遍历（grouped ordering）：**

```
遍历顺序: (M0,N0) → (M1,N0) → ... → (M7,N0) → (M0,N1) → (M1,N1) → ... → (M7,N1) → ...

SM#0 加载: A[M0], B[N0]
SM#1 加载: A[M1], B[N0]   ← B[N0] 被连续 8 个 SM 复用！
SM#2 加载: A[M2], B[N0]
...
SM#7 加载: A[M7], B[N0]   ← B[N0] 在 L2 中被命中 7 次
SM#8 加载: A[M0], B[N1]   ← 才切换到 B[N1]
```

**本质**：Swizzle 把 tile 遍历从"一行一行扫"变成"一小块一小块扫"：

```
朴素遍历（row-major）:          Swizzle 后（grouped ordering）:

N →                              N →
┌──┬──┬──┬──┐                    ┌──┬──┬──┬──┐
│1 │2 │3 │4 │ M=0                │1 │9 │17│25│ M=0
├──┼──┼──┼──┤                    ├──┼──┼──┼──┤
│5 │6 │7 │8 │ M=1                │2 │10│18│26│ M=1
├──┼──┼──┼──┤                    ├──┼──┼──┼──┤
│9 │10│11│12│ M=2                │3 │11│19│27│ M=2
├──┼──┼──┼──┤                    ├──┼──┼──┼──┤
│..│..│..│..│                    │..│..│..│..│
├──┼──┼──┼──┤                    ├──┼──┼──┼──┤
│29│30│31│32│ M=7                │8 │16│24│32│ M=7
└──┴──┴──┴──┘                    └──┴──┴──┴──┘

B 的每一列被间隔 4 次才复用        B 的每一列被连续 8 次复用 ✅
```

**效果**：
- **B 矩阵**的同一列数据在 L2 中被 `kNum1DBlocksPerGroup`（8 或 16）个连续 tile 共享命中
- **A 矩阵**的不同行在 group 内被快速遍历，每个只用一次，不占用太多 L2 空间
- 整体 L2 工作集 = `kNum1DBlocksPerGroup * BLOCK_M`（A 的行）+ `ceil_div(kNumSMs, kNum1DBlocksPerGroup) * BLOCK_N`（B 的列），远小于朴素遍历的工作集

这与 [Triton 的 L2 Cache Optimization](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations) 中描述的 "grouped ordering" 是同一个思想。

#### M-Grouped GEMM 中的 idx 处理

在 `MGroupedMasked` 模式下，swizzle 前需要先减去当前 expert 之前的累计 tile 数：

```cpp
get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks,
                       m_block_idx, n_block_idx);
```

- **`next_block_idx`**：全局线性 tile 索引（`current_iter * kNumSMs + blockIdx.x`）
- **`current_m_cumsum * num_n_blocks`**：当前 expert 之前所有 expert 累计的 tile 数量
- **相减**：得到**当前 expert 内部的局部 tile 索引**
- 然后 `get_swizzled_block_idx` 将这个局部索引 swizzle 成 `(m_block_idx, n_block_idx)`

这样每个 expert 内部都独立做 swizzle，不会跨 expert 边界混乱。

---

## Step 2: 流水线同步机制

### Pipeline 状态机

Warp 0 使用 `kNumStages` 个 stage 的循环流水线，通过 phase bit 翻转来区分新旧数据：

```cpp
uint32_t stage_idx = 0, phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++ k_block_idx;
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;  // 回到 stage 0 时翻转 phase
};
```

### Barrier 角色

| Barrier | 生产者 | 消费者 | 含义 |
|---------|--------|--------|------|
| `empty_barriers` | Warp 1 (MMA) | Warp 0 (TMA) | "SMEM buffer 已被消费，可以覆写" |
| `full_barriers` | Warp 0 (TMA) | Warp 2 (SF Transpose) | "TMA 数据已到达 SMEM" |
| `with_sf_full_barriers` | Warp 2 (SF Transpose) | Warp 1 (MMA) | "SF 转置完成，可以做 MMA" |

Warp 0 在每次发射 TMA 前，先等待 `empty_barriers` 确认该 stage 的 buffer 已被消费：

```cpp
empty_barriers[stage_idx]->wait(phase ^ 1);
```

注意 `phase ^ 1`：等待的是**上一轮**的 phase，因为当前 phase 的数据还没写入。

---

## Step 3: 坐标计算

### M/N 维度全局坐标

```cpp
uint32_t m_idx = scheduler.template get_global_idx<
    (kGemmType == GemmType::MGroupedMasked), IndexType::MN>(
    shape_m, BLOCK_M, m_block_idx);
uint32_t n_idx = scheduler.template get_global_idx<
    (kMajorB == cute::UMMA::Major::K), IndexType::MN>(
    shape_n, BLOCK_N, n_block_idx, m_block_idx);
```

- `m_idx`：对于 `MGroupedMasked`，需要加上 group 偏移（因为多个 expert 的 M 维度是拼接的）
- `n_idx`：当 B 是 K-major 时，N 是外维，需要根据 group 做偏移

### K 维度坐标

```cpp
uint32_t k_idx = k_block_idx * BLOCK_K;  // 裸偏移，不含 group offset
uint32_t k_a_idx = scheduler.template get_global_idx<
    (kMajorA == cute::UMMA::Major::MN), IndexType::K>(
    shape_k, BLOCK_K, k_block_idx, m_block_idx);
uint32_t k_b_idx = scheduler.template get_global_idx<
    (kMajorB == cute::UMMA::Major::MN), IndexType::K>(
    shape_k, BLOCK_K, k_block_idx, m_block_idx);
```

- `k_idx` 是不含 group 偏移的裸坐标，用于计算 SF 的 K 维坐标
- `k_a_idx` / `k_b_idx` 是考虑了 major 和 group 偏移后的实际全局坐标

**编译期约束**：所有 M-grouped GEMM 中，A 必须是 K-major（行主序），因为 M 维度是 expert token 拼接维度，不能做 M-major。

### 2-CTA Multicast 偏移

```cpp
if constexpr (kNumMulticast > 1) {
    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
}
```

`block_rank_in_cluster()` 返回 0 或 1，两个 CTA 各自加载自己负责的那半块数据。

---

## Step 4: TMA 发射

### A/B 矩阵 TMA

根据 major 选择维度顺序：

```cpp
// A 是 K-major: TMA 按 (K, M) 布局搬运
if constexpr (kMajorA == cute::UMMA::Major::K)
    tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx],
        k_a_idx, m_idx, 1, batch_idx);
// A 是 MN-major: TMA 按 (M, K) 布局搬运
if constexpr (kMajorA == cute::UMMA::Major::MN)
    tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx],
        m_idx, k_a_idx, 1, batch_idx);
```

### tma_copy 内部实现

参考 `tma_utils.cuh`，`tma_copy` 会根据 swizzle mode 将一个大 tile 拆分成多个 atom 发射，并根据 multicast 模式和架构选择不同的 TMA 指令：

```cpp
// tma_utils.cuh
constexpr uint32_t BLOCK_INNER_ATOM = kSwizzleMode == 0 ?
    BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);

if (num_tma_multicast == 1) {
    // 单 CTA 模式：SM90/SM100 共用 SM90_TMA_LOAD_2D
    // （SM100 兼容 SM90 的 TMA 指令，cache hint 枚举值相同）
    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr,
            CacheHintSm100::EVICT_NORMAL,
            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
            inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
    }
} else {
    // 2-CTA 模式
    #if __CUDA_ARCH__ >= 1000
        // SM100: 两个 CTA 各自独立发射，barrier 信号只发到 leader CTA
        for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
            cute::SM100_TMA_2SM_LOAD_2D::copy(desc_ptr, barrier_ptr,
                CacheHintSm100::EVICT_NORMAL,
                smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
        }
    #elif __CUDA_ARCH__ >= 900
        // SM90: 只有 leader CTA 发射 multicast，一次写入两个 CTA 的 SMEM
        if (cute::block_rank_in_cluster() == 0) {
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, barrier_ptr,
                    (1 << num_tma_multicast) - 1, CacheHintSm90::EVICT_NORMAL,
                    smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                    inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
            }
        }
    #endif
}
```

- 当 `kSwizzleMode = 128`（128B swizzle），`BLOCK_INNER_ATOM = 128 / sizeof(fp8) = 128`
- 对于 `BLOCK_K = 128`，只需 1 次 TMA 发射
- 对于更大的 inner 维度，需要多次 TMA 发射
- **SM100 上即使是单 CTA 模式也用 `SM90_TMA_LOAD_2D`**，因为 SM100 向后兼容 SM90 的 TMA 指令

### SM100 2-CTA TMA 模式

> **⚠️ 原始知识库错误更正**：`MoE grouped gemm B200.md` 中描述 "只让 leader CTA load 一次，然后利用 multicast 将数据直接写入两个 CTA 的 shared memory"，这是 **SM90 的行为**，不是 SM100 的行为。

在 SM100 上，2-CTA 模式使用 `SM100_TMA_2SM_LOAD_2D`：

```cpp
// tma_utils.cuh 中的 SM100 分支
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
    // 2-CTA function will send signals to the leader CTA only
    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
        cute::SM100_TMA_2SM_LOAD_2D::copy(desc_ptr, barrier_ptr,
            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
            inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
    }
```

**SM100 vs SM90 的关键区别：**

| 特性 | SM90 (Hopper) | SM100 (Blackwell) |
|------|--------------|-------------------|
| TMA 发射 | **只有 leader CTA** 发射一次 multicast | **两个 CTA 各自独立**发射 TMA |
| 数据写入 | 一次 TMA 写入多个 CTA 的 SMEM | 每个 CTA 的 TMA 写入自己的 SMEM |
| Barrier 信号 | Multicast 到所有 CTA | **只发到 leader CTA** |
| 指令 | `SM90_TMA_LOAD_MULTICAST_2D` | `SM100_TMA_2SM_LOAD_2D` |

```
SM90 Multicast:
Global Memory ──TMA──→ CTA 0 SMEM  (leader CTA 发射一次)
                  └──→ CTA 1 SMEM  (multicast 写入)

SM100 2-CTA:
Global Memory ──TMA──→ CTA 0 SMEM  (CTA 0 自己发射)
Global Memory ──TMA──→ CTA 1 SMEM  (CTA 1 自己发射)
                       barrier 信号只发到 leader CTA
```

在 Warp 0 的代码中，`warp_idx == 0 and cute::elect_one_sync()` 条件下，**两个 CTA 的 warp 0 都会进入这个分支**。`elect_one_sync` 只是在 warp 内选一个 thread，不是在 cluster 内选一个 CTA。所以两个 CTA 各自独立发射 TMA，只是各自加载自己那半块数据（通过 `block_rank_in_cluster()` 偏移坐标）。

### num_arrival_bytes 计算

```cpp
auto num_arrival_bytes =
    SMEM_A_SIZE_PER_STAGE / (std::is_same_v<a_dtype_t, cutlass::float_e4m3_t> ? 1 : 2) +
    SMEM_B_SIZE_PER_STAGE / (std::is_same_v<b_dtype_t, cutlass::float_e4m3_t> ? 1 : 2);
```

> **⚠️ 原始知识库注释更正**：原文注释 "FP4 类型只有 FP8 一半大小，所以要除以 2" 方向正确但不够精确。

准确解释：
- `SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t)`
- 对于 FP8（`float_e4m3_t`），`sizeof = 1`，SMEM 分配大小 = 实际传输字节数，除以 1
- 对于 FP4（subbyte 类型），CUTLASS 中 `sizeof` 仍然是 1 字节（pack 后），但实际每个元素只占 4 bit，TMA 实际传输的字节数是 SMEM 分配大小的 **一半**，所以除以 2
- `num_arrival_bytes` 必须精确等于 TMA 实际传输的字节数，否则 barrier 永远不会 ready

---

## Step 5: Scale Factor TMA

### 条件性加载

SF 的加载频率取决于 K 维度粒度 `kGranKA` / `kGranKB`：

| 粒度 | `kNumSFStagesPerLoad` | 含义 |
|------|----------------------|------|
| `gran_k = 32` | 1 | 每个 K-block (128 元素) 都加载 SF |
| `gran_k = 128` | 4 | 每 4 个 K-block 才加载一次 SF |

```cpp
// 每 kNumSFAStagesPerLoad 个 K-block 加载一次 SFA
if (k_block_idx % kNumSFAStagesPerLoad == 0) {
    tma_copy<BLOCK_M, 1, 0>(  // swizzle mode = 0，不做 swizzle
        &tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx],
        m_block_idx * BLOCK_M,
        scheduler.template get_global_idx<...>(shape_sfa_k, 1,
            ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad)));
    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);  // 每个 SF 是 UE8M0 = 4 bytes
}
```

**为什么 SF 不用 swizzle**：SF 数据量很小（`BLOCK_M` 或 `BLOCK_N` 个 `uint32_t`），不会产生严重的 bank conflict，所以 swizzle mode 设为 0。

### SF 的 K 维坐标

SF 的 K 维坐标计算比较特殊：
- `gran_k = 32` 时，一个 SF 覆盖 32 个 FP8 元素，`BLOCK_K = 128` 包含 4 个 SF
- `gran_k = 128` 时，一个 SF 覆盖 128 个 FP8 元素（即 512 个 FP4），4 个 K-block 共享一个 SF

---

## Step 6: Barrier 通知

```cpp
full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
```

`arrive_and_expect_tx` 告诉 barrier：
1. **arrive**：Warp 0 已经发射了所有 TMA 指令
2. **expect_tx**：预期 `num_arrival_bytes` 字节会异步到达

Barrier 会在 TMA 硬件真正搬完这些字节后自动变为 ready，唤醒等待的消费者：
- **Warp 2** (SF Transpose) 等待 `full_barriers` → 开始转置 SF
- **Warp 1** (MMA) 等待 `with_sf_full_barriers`（需要 Warp 2 转置完成后才 arrive）

---

## 深入：为什么 M-Grouped GEMM 不能 Multicast on A

> **⚠️ 原始知识库错误更正**：`MoE grouped gemm B200.md` 中将原因归结为 "TMEM atom 固定 128 行，你不能只用其中 64 行做 UMMA"，这是**错误的**。UMMA 支持 64/128/256 行的 M 维度。

**真正原因是 M-grouped GEMM 中 expert 边界不对齐。**

在 M-grouped GEMM 中，A 矩阵的 M 维度是多个 expert 的 token 拼接而成的：

```
A 矩阵 M 维度（多 expert 拼接）:
┌─────────────────────────────────────────┐
│ Expert 0: 120 tokens │ Expert 1: 80 tokens │ Expert 2: ...
└─────────────────────────────────────────┘
```

如果对 A 做 multicast（切 M 维），两个 CTA 分别加载 M 维的前半和后半：

```
Multicast on A (切 M 维) — 问题场景:

假设 BLOCK_M = 128, 当前 tile 跨越 expert 边界:
M[0..63]   → Expert 0 的 token     → CTA 0 加载
M[64..127] → Expert 0 + Expert 1   → CTA 1 加载
                                       ↑ 跨越了 expert 边界！

两个 CTA 拿到的数据属于不同 expert，
但它们要共享同一个 B 矩阵做 GEMM — 这在语义上是错误的。
不同 expert 对应不同的权重矩阵 B。
```

而 Multicast on B（切 N 维）没有这个问题，因为 B 矩阵的 N 维度不涉及 expert 拼接：

```
Multicast on B (切 N 维) — 安全:

CTA 0 加载: B[0 ~ N/2-1, :]      N 维度与 expert 无关
CTA 1 加载: B[N/2 ~ N-1, :]      两个 CTA 处理同一个 expert 的不同 N 列
→ 语义正确 ✅
```

所以代码中有断言：
```cpp
DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
```

对于普通 GEMM（`GemmType::Normal`），M 维度没有 expert 拼接，multicast on A 是合法的。

---

## 完整数据流总结

```
Warp 0 (TMA Load) 的一次迭代:

1. wait(empty_barriers)     ← 等 Warp 1 消费完旧数据
2. 计算 m_idx, n_idx, k_idx  ← scheduler 提供 tile 坐标
3. 发射 TMA: A → smem_a      ← 异步，立即返回
4. 发射 TMA: B → smem_b      ← 异步，立即返回
5. 条件发射 TMA: SFA → smem_sfa  ← 每 kNumSFAStagesPerLoad 个 K-block 一次
6. 条件发射 TMA: SFB → smem_sfb  ← 每 kNumSFBStagesPerLoad 个 K-block 一次
7. arrive_and_expect_tx(full_barriers, num_bytes)  ← 通知 Warp 2
8. advance_pipeline          ← 切换到下一个 stage

                    ┌──────────────────────────────────────┐
                    │         Warp 0 (TMA Load)            │
                    │                                      │
                    │  Global Memory ──TMA──→ SMEM         │
                    │  (A, B, SFA, SFB)                    │
                    └──────────┬───────────────────────────┘
                               │ full_barriers
                               ▼
                    ┌──────────────────────────────────────┐
                    │       Warp 2 (SF Transpose)          │
                    │                                      │
                    │  SMEM 上 SF 数据做 warp transpose    │
                    └──────────┬───────────────────────────┘
                               │ with_sf_full_barriers
                               ▼
                    ┌──────────────────────────────────────┐
                    │         Warp 1 (MMA)                 │
                    │                                      │
                    │  UTCCP: SMEM SF → TMEM               │
                    │  UMMA: SMEM A,B → TMEM accum         │
                    └──────────┬───────────────────────────┘
                               │ empty_barriers (释放 SMEM)
                               ▼
                          回到 Warp 0 Step 1
```
