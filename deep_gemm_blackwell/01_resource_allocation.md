# DeepGEMM SM100 FP8 GEMM 资源分配决策

> **核心代码：**
> - 通用启发式逻辑：[`csrc/jit_kernels/heuristics/common.hpp`](../csrc/jit_kernels/heuristics/common.hpp)
> - SM100 架构特化：[`csrc/jit_kernels/heuristics/sm100.hpp`](../csrc/jit_kernels/heuristics/sm100.hpp)
> - Kernel 模板参数消费端：[`deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh`](../deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh)
>
> 本章解析 DeepGEMM 在 JIT 编译阶段如何根据 GEMM shape 和硬件约束，自动选择最优的 tile 尺寸、流水线深度、multicast 策略和 shared memory 布局。

---

## 整体决策流程

资源分配在 host 端的 `get_best_config()` 函数中完成（[`common.hpp` L143](../csrc/jit_kernels/heuristics/common.hpp)），按以下顺序依次决策：

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 确定 Tile 尺寸 (BLOCK_M, BLOCK_N, BLOCK_K)     │
│          ↓                                               │
│  Step 2: 确定 Multicast 策略 (1-CTA vs 2-CTA)           │
│          ↓                                               │
│  Step 3: 确定流水线深度 (num_stages)                      │
│          ↓                                               │
│  Step 4: 计算 Shared Memory 布局                         │
│          ↓                                               │
│  Step 5: 确定线程配置                                     │
│          ↓                                               │
│  输出: GemmConfig（传入 JIT 编译器生成 kernel）            │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Tile 尺寸选择

> **源码：** [`common.hpp` — `get_best_config()`](../csrc/jit_kernels/heuristics/common.hpp) 中的 tile 搜索循环
> **源码：** [`sm100.hpp` — `get_block_m_candidates()` / `get_block_n_candidates()`](../csrc/jit_kernels/heuristics/sm100.hpp)

### 候选值生成

BLOCK_K 固定，BLOCK_M 和 BLOCK_N 从候选列表中选择：

```cpp
// common.hpp — get_best_config()
const auto& block_k = (mma_kind == MmaKind::BF16 ? 64 : 128);
```

```cpp
// sm100.hpp — get_block_m_candidates()
static std::vector<int> get_block_m_candidates(const KernelType& kernel_type,
                                                const cute::UMMA::Major& major_a, const int& m) {
    std::vector<int> candidates{128, 256};
    if ((kernel_type == KernelType::Kernel1D1D or kernel_type == KernelType::KernelNoSF)
        and major_a == cute::UMMA::Major::K) {
        if (m <= 32) candidates.push_back(32);
        if (m <= 64) candidates.push_back(64);
    }
    return candidates;
}

// sm100.hpp — get_block_n_candidates()
static std::vector<int> get_block_n_candidates(const KernelType& kernel_type,
                                                const at::ScalarType& cd_dtype) {
    std::vector<int> candidates = {16};
    for (int i = 32; i <= 256; i += 32)
        candidates.push_back(i);
    return candidates;
}
```

| 维度 | 候选值 | 备注 |
|------|--------|------|
| **BLOCK_K** | FP8: `128`，BF16: `64` | 固定，不参与搜索 |
| **BLOCK_M** | `{128, 256}` 为基础 | 当 `m ≤ 64` 时追加 `64`；当 `m ≤ 32` 时追加 `32` |
| **BLOCK_N** | `{16, 32, 64, 96, 128, 160, 192, 224, 256}` | 以 32 为步长，额外包含 16 |

**特殊场景的候选值约束：**

```cpp
// common.hpp — get_best_config() 中的特殊场景处理
if (gemm_type == GemmType::MGroupedContiguous)
    block_ms = std::vector{get_mk_alignment_for_contiguous_layout()};
if (gemm_type == GemmType::MGroupedMasked or gemm_type == GemmType::MGroupedContiguousWithPsumLayout)
    block_ms = std::vector{64, 128};    // Exclude 256 for performance

// FP4 + MN-major 的约束
if (a_dtype == kPackedFP4 and major_a == cute::UMMA::Major::MN)
    block_ms = std::vector{128};
if (b_dtype == kPackedFP4 and major_b == cute::UMMA::Major::MN)
    block_ns = std::vector{128};
```

| 场景 | BLOCK_M 候选 | 原因 |
|------|-------------|------|
| MGroupedMasked / MGroupedContiguousWithPsumLayout | `{64, 128}` | 排除 256，因为 MoE 场景下每个 expert 的 token 数通常较小，256 会导致大量 padding 浪费 |
| MGroupedContiguous | `{128}` 固定 | 必须与 contiguous layout 的对齐要求一致 |
| FP4 + MN-major | `{128}` 固定 | TMA 的 `.b4x16_p64` 指令仅支持 Swizzle-128B |

**合法性检查（`is_block_size_legal`）：**

> **源码：** [`sm100.hpp` — `is_block_size_legal()`](../csrc/jit_kernels/heuristics/sm100.hpp)

在搜索前，每个 (BLOCK_M, BLOCK_N) 组合需要通过以下检查：

```cpp
// sm100.hpp — is_block_size_legal() 核心逻辑
static bool is_block_size_legal(...) {
    if (block_n % 16 != 0)                                          // ① TMEM Layout A/D 要求
        return false;
    if (kernel_type == KernelType::Kernel1D1D
        and major_b == cute::UMMA::Major::K and block_m > 128)      // ② 性能约束
        return false;
    if (k <= 256 and (block_n > 128 or block_m > 128))              // ③ 小 K 减少 epilogue 瓶颈
        return false;
    // ④ TMEM 列数硬件限制
    if (((2 * block_n) + (sf_block_m / 32) + (sf_block_n / 32)) > 512)
        return false;
    // ⑤ B 为 MN-major 时的 TMA 性能约束
    return major_b == cute::UMMA::Major::K or (block_n * get_element_size(mma_kind)) % 64 == 0;
}
```

其中第 ④ 条的 TMEM 列数计算为：

```
tmem_cols = 2 × BLOCK_N                          // 累加器（2 个 epilogue stage）
          + ceil_align(BLOCK_M, 128) / 32         // SFA
          + ceil_align(BLOCK_N, 128) / 32         // SFB
```

这个约束来自 Blackwell TMEM 的硬件限制：**最多 128 行 × 512 列**。

### 选择策略

> **源码：** [`common.hpp` — `get_best_config()` 中的双层 for 循环](../csrc/jit_kernels/heuristics/common.hpp)

对所有合法的 (BLOCK_M, BLOCK_N) 组合，按以下优先级排序选择最优配置：

```cpp
// common.hpp — 核心指标计算
const auto& get_num_blocks = [=](const int& block_m, const int& block_n) {
    return ceil_div(m, block_m) * ceil_div(n, block_n) * num_groups;
};
const auto& get_num_waves = [=](const int& block_m, const int& block_n) {
    return ceil_div(get_num_blocks(block_m, block_n), num_sms);
};
const auto& get_last_wave_util = [=](const int& block_m, const int& block_n) {
    const auto& num_last_blocks = get_num_blocks(block_m, block_n) % num_sms;
    return num_last_blocks == 0 ? num_sms : num_last_blocks;
};
```

```cpp
// common.hpp — 选择策略（简化后的核心逻辑）
bool success = false;
if (best_block_m == 0 or num_waves < best_num_waves) {
    success = true;                                                    // P0: 更少的 wave
} else if (num_waves == best_num_waves) {
    success = last_util > best_last_util;                              // P1: 更高的尾部利用率
    if (last_util == best_last_util) {
        success |= block_m == best_block_m and block_n < best_block_n; // P2: 同 M，更小 N
        success |= block_n == best_block_n and block_m < best_block_m; // P3: 同 N，更小 M
        success |= block_m != best_block_m and block_n > best_block_n  // P4: 更大 N
                   and block_n <= n and block_m <= m;
    }
}
```

| 优先级 | 规则 | 直觉 |
|--------|------|------|
| **P0** | `num_waves` 更小 | 减少总的 wave 轮次，降低尾部浪费 |
| **P1** | `last_util` 更大 | 最后一个 wave 尽可能多地利用 SM |
| **P2** | 相同 BLOCK_M，选更小的 BLOCK_N | 减少 N 维度的 padding 浪费 |
| **P3** | 相同 BLOCK_N，选更小的 BLOCK_M | 减少 M 维度的 padding 浪费 |
| **P4** | 不同 BLOCK_M 和 BLOCK_N 时，选更大的 BLOCK_N | 更大的 BLOCK_N 通常有更高的计算吞吐（前提：`BLOCK_N ≤ N` 且 `BLOCK_M ≤ M`） |

**设计理念：** 核心目标是 **最小化 wave 数量**，因为每个 wave 都需要所有 SM 同步完成后才能开始下一个 wave。在 wave 数量相同时，优先让最后一个 wave 的 SM 利用率更高，避免大量 SM 空转。

---

## Step 2: Multicast 策略

> **源码：** [`common.hpp` — `get_best_config()` 中的 multicast 决策](../csrc/jit_kernels/heuristics/common.hpp)
> **源码：** [`sm100.hpp` — `get_multicast_legality()`](../csrc/jit_kernels/heuristics/sm100.hpp)
> **源码：** [`common.hpp` — `is_multicast_legal()`](../csrc/jit_kernels/heuristics/common.hpp)

Multicast 是 Blackwell 的 2-CTA cluster 特性：两个相邻 CTA 共享同一个 SM，TMA 只需从 global memory 读取一次数据，就能同时写入两个 CTA 的 shared memory。

### 开启条件

必须**同时满足**以下所有条件：

```cpp
// common.hpp — is_multicast_legal()
static bool is_multicast_legal(const int& shape_dim, const int& block_dim,
                               const int& num_multicast, const int& num_sms,
                               const bool& require_divisible) {
    const bool& divisible = ceil_div(shape_dim, block_dim) % num_multicast == 0
                            or not require_divisible;
    return divisible and num_sms % num_multicast == 0;
}
```

```
1. M ≥ 512                                    ← 问题规模足够大
2. ceil_div(shape_dim, block_dim) % 2 == 0     ← 对应维度的 block 数量为偶数
3. num_SMs % 2 == 0                            ← SM 数量为偶数
```

### Multicast 方向选择

DeepGEMM 的 multicast 只能在 M 维度或 N 维度之一上进行：

| 方向 | 含义 | 效果 |
|------|------|------|
| **Multicast on B（切 N 维）** | 两个 CTA 各加载 B 矩阵的一半 N 列 | `LOAD_BLOCK_N = BLOCK_N / 2` |
| **Multicast on A（切 M 维）** | 两个 CTA 各加载 A 矩阵的一半 M 行 | `LOAD_BLOCK_M = BLOCK_M / 2` |

**方向优先级：** 优先在 **较大的维度** 上做 multicast（`BLOCK_M > BLOCK_N` 时优先 A，反之优先 B）。

```cpp
// common.hpp — get_best_config() 中的方向选择逻辑
bool order[2] = {false, true};           // 默认先尝试 B，再尝试 A
if (best_block_m > best_block_n)
    std::swap(order[0], order[1]);       // BLOCK_M 更大时，优先尝试 A
for (const bool& is_multicast_on_a: order) {
    if (m >= 512 and is_legal[static_cast<int>(is_multicast_on_a)]) {
        best_multicast_config = {2, is_multicast_on_a};
        break;
    }
}
```

**SM100 的特殊约束：** 在当前 DeepGEMM 实现中，SM100 架构 **不支持 Multicast on A**：

```cpp
// sm100.hpp — get_multicast_legality()
static std::pair<bool, bool> get_multicast_legality(...) {
    return {
        false,   // ← Multicast on A: 始终禁止
        is_multicast_legal(m, block_m, 2, num_sms, true) and
            (gemm_type == GemmType::Normal or gemm_type == GemmType::KGroupedContiguous
             or (gemm_type == GemmType::Batched and num_groups <= 32)),
    };
}
```

原因：在 M-grouped GEMM 中，A 矩阵的 M 维度是多个 expert 的 token 拼接，expert 边界不一定与 BLOCK_M 对齐，无法简单地将 M 维一分为二。此外，kernel 中有断言进一步确认了这一点：

```cpp
// sm100_fp8_gemm_1d1d.cuh — kernel 内的静态断言
DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M, "Only support tensor memory layout A/D");
```

```
Multicast on B（切 N 维）的数据流：

全局内存                    Shared Memory
┌──────────┐               ┌──────────────┐
│  B 数据   │──── TMA ────→│  CTA 0 SMEM  │ ← 一次读取
│ (半块 N)  │       │       └──────────────┘
└──────────┘       │       ┌──────────────┐
                   └──────→│  CTA 1 SMEM  │ ← 同时写入（multicast）
                           └──────────────┘

两个 CTA 各自加载 B 的不同半块，但通过 multicast 共享对方的数据
→ 全局内存带宽减半，UMMA 指令以 cta_group::2 模式执行
```

---

## Step 3: 流水线深度（num_stages）

> **源码：** [`common.hpp` — `get_best_config()` 中的 stage 搜索循环](../csrc/jit_kernels/heuristics/common.hpp)
> **源码：** [`sm100.hpp` — `smem_capacity` 常量定义](../csrc/jit_kernels/heuristics/sm100.hpp)

### 选择策略

从最大值 32 开始递减，找到第一个满足 SMEM 容量限制的 stage 数：

```cpp
// common.hpp — get_best_config() 中的 stage 选择
constexpr int smem_capacity = ArchSpec::smem_capacity;  // SM100: 232448 bytes ≈ 227 KB

int best_num_stages = 0;
SharedMemoryConfig best_smem_config;
for (int num_stages = 32; num_stages > 0; -- num_stages) {
    if (not ArchSpec::is_num_stages_legal(mma_kind, cd_dtype, num_stages,
                                          best_block_m, best_block_n, block_k))
        continue;

    best_smem_config = get_smem_config<ArchSpec>(gemm_type, kernel_type,
                                                 m, n, k,
                                                 best_block_m, best_block_n, block_k,
                                                 major_a, major_b,
                                                 mma_kind, cd_dtype,
                                                 num_stages, best_multicast_config);
    if (best_smem_config.smem_size <= smem_capacity) {
        best_num_stages = num_stages;
        break;
    }
}
DG_HOST_ASSERT(best_num_stages != 0);
```

```cpp
// sm100.hpp — SM100 的 SMEM 容量和 stage 合法性（SM100 无额外约束）
static constexpr int smem_capacity = 232448;

static bool is_num_stages_legal(const MmaKind& mma_kind, const at::ScalarType& cd_dtype,
                                const int& num_stages,
                                const int& block_m, const int& block_n, const int& block_k) {
    return true;  // SM100 对 stage 数没有额外限制
}
```

**设计理念：** 更深的流水线可以更好地隐藏 TMA 访存延迟。当 MMA 计算 stage i 的数据时，TMA 可以同时预取 stage i+1, i+2, ... 的数据。但 stage 越多，SMEM 占用越大，受限于 SM100 的 232KB SMEM 容量。

```
Stage = 2（浅流水线）：
  TMA:  [Load S0]  [等待]  [Load S1]  [等待]  ...
  MMA:  [等待]  [MMA S0]  [等待]  [MMA S1]  ...
  ⚠️ MMA 经常需要等待 TMA 完成

Stage = 8（深流水线）：
  TMA:  [S0][S1][S2][S3][S4][S5][S6][S7][S0]...
  MMA:       [S0][S1][S2][S3][S4][S5][S6][S7]...
  ✅ TMA 和 MMA 几乎完全重叠
```

> **注意：** 流水线深度并非越大越好。实际的 overlap 效果取决于具体的 GEMM shape 和 tile 配置，需要通过 Nsight Compute 分析确认。

---

## Step 4: Shared Memory 布局

> **源码（host 端计算）：** [`common.hpp` — `get_smem_config()`](../csrc/jit_kernels/heuristics/common.hpp)
> **源码（SM100 特化）：** [`sm100.hpp` — `get_smem_cd_size()` / `get_sf_smem_size_per_stage()` / `get_barrier_smem_size()`](../csrc/jit_kernels/heuristics/sm100.hpp)
> **源码（kernel 端消费）：** [`sm100_fp8_gemm_1d1d.cuh` — SMEM 指针计算](../deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh)

### SMEM 总体布局

host 端通过 `get_smem_config()` 计算总大小：

```cpp
// common.hpp — get_smem_config() 核心逻辑
int smem_size = 0;
smem_size += smem_tensor_map;                      // ⓪ SM100 上为 0（用 __grid_constant__）
smem_size += smem_cd;                              // ① C/D 输出 buffer
smem_size += num_stages * smem_a_per_stage;        // ② A 矩阵 buffer
smem_size += num_stages * smem_b_per_stage;        // ③ B 矩阵 buffer
smem_size += num_stages * smem_sfa_per_stage;      // ④ Scale Factor A buffer
smem_size += num_stages * smem_sfb_per_stage;      // ⑤ Scale Factor B buffer
smem_size += smem_extra_sfb;                       // SM100 上为 0
smem_size += smem_barrier;                         // ⑥ Barrier 对象
smem_size += smem_tmem_ptr;                        // ⑦ TMEM 指针
```

kernel 端通过指针偏移访问各区域：

```cpp
// sm100_fp8_gemm_1d1d.cuh — kernel 内的 SMEM 指针布局
extern __shared__ __align__(1024) uint8_t smem_buffer[];

// ① C/D buffer（地址 0 开始）
auto smem_cd = PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
});
// ② A buffer（紧接 CD 之后）
auto smem_a = PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
});
// ③ B buffer（紧接 A 之后）
auto smem_b = PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE
        + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
});
// ④⑤ SF buffer（紧接 B 之后）
auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE
    + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
auto smem_sfa = PatternVisitor([=](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
});
auto smem_sfb = PatternVisitor([=](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(sf_start_ptr
        + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
});
// ⑥ Barriers（紧接 SF 之后）
auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
    SMEM_CD_SIZE +
    kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
    kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
// ⑦ TMEM 指针（紧接 Barriers 之后）
auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
    barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);
```

完整布局图：

```
┌─────────────────────────────────────────────────────────────┐  ← 地址 0
│ ① C/D 输出 Buffer（Double Buffer）                           │
│    大小: STORE_BLOCK_M × swizzle_cd_mode × 2                │
│    STORE_BLOCK_M = min(BLOCK_M, 128)                        │
├─────────────────────────────────────────────────────────────┤
│ ② A 矩阵 Buffer（× num_stages）                             │
│    每 stage: LOAD_BLOCK_M × BLOCK_K × sizeof(FP8)           │
│    = BLOCK_M × 128 × 1 字节                                 │
├─────────────────────────────────────────────────────────────┤
│ ③ B 矩阵 Buffer（× num_stages）                             │
│    每 stage: LOAD_BLOCK_N × BLOCK_K × sizeof(FP8)           │
│    = LOAD_BLOCK_N × 128 × 1 字节                            │
├─────────────────────────────────────────────────────────────┤
│ ④ Scale Factor A Buffer（× num_stages）                     │
│    每 stage: SF_BLOCK_M × sizeof(uint32_t)                  │
│    SF_BLOCK_M = ceil_align(BLOCK_M, 128)                    │
├─────────────────────────────────────────────────────────────┤
│ ⑤ Scale Factor B Buffer（× num_stages）                     │
│    每 stage: SF_BLOCK_N × sizeof(uint32_t)                  │
│    SF_BLOCK_N = ceil_align(BLOCK_N, 128)                    │
├─────────────────────────────────────────────────────────────┤
│ ⑥ Barrier 对象                                              │
│    ├─ full_barriers        (num_stages × 8B)                │
│    ├─ empty_barriers       (num_stages × 8B)                │
│    ├─ with_sf_full_barriers(num_stages × 8B)                │
│    ├─ tmem_full_barriers   (kNumEpilogueStages × 8B)        │
│    ├─ tmem_empty_barriers  (kNumEpilogueStages × 8B)        │
│    └─ TC util control      (8B)                             │
│    总计: num_stages × 24 + kNumEpilogueStages × 16 + 8      │
├─────────────────────────────────────────────────────────────┤
│ ⑦ TMEM 指针 (4 bytes)                                       │
└─────────────────────────────────────────────────────────────┘
```

### 各部分详解

#### ① C/D 输出 Buffer

```cpp
// sm100.hpp — get_smem_cd_size()
static int get_smem_cd_size(const KernelType& kernel_type,
                            const int& block_m, const int& block_n,
                            const int& swizzle_cd_mode,
                            const at::ScalarType& cd_dtype) {
    constexpr static int layout_ad_m = 128;
    return std::min(block_m, layout_ad_m) * swizzle_cd_mode * 2;
}
```

```cpp
// sm100_fp8_gemm_1d1d.cuh — kernel 内对应的常量定义
constexpr uint32_t LAYOUT_AD_M = 128;
constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
constexpr uint32_t kNumTMAStoreStages = 2;
constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
```

- `STORE_BLOCK_M` 受限于 TMEM 的 128 行布局，每次最多从 TMEM 读出 128 行
- `× 2` 是 **double buffer**（`kNumTMAStoreStages = 2`）：一个 stage 正在被 TMA store 写回 global memory，另一个 stage 可以同时被 epilogue 填充
- `swizzle_cd_mode` 的选择：找到最大的 mode ∈ {128, 64, 32, 16}，使得 `(BLOCK_N × cd_elem_size) % mode == 0`

#### ② ③ A/B 矩阵 Buffer

```cpp
// sm100_fp8_gemm_1d1d.cuh — kernel 内的常量定义
constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);

// 对齐检查
DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0
    and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
    "Shared memory of A/B must be aligned to 1024 bytes");
```

- 在 2-CTA multicast on B 模式下，`LOAD_BLOCK_N = BLOCK_N / 2`，B 矩阵的 SMEM 占用减半
- 每个 stage 的 buffer 必须 **1024 字节对齐**（swizzle-128B 的要求）

#### ④ ⑤ Scale Factor Buffer

```cpp
// sm100_fp8_gemm_1d1d.cuh — kernel 内的常量定义
constexpr uint32_t kNumUTCCPAlignedElems = 128;
constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
```

```cpp
// sm100.hpp — host 端对应的计算
static std::pair<int, int> get_sf_uttcp_aligned_block_sizes(
    const int& block_m, const int& block_n, const MmaKind& mma_kind) {
    constexpr int num_utccp_aligned_elems = 128;
    switch (mma_kind) {
        case MmaKind::BF16:     return {0, 0};
        case MmaKind::MXFP8FP4: return {align(block_m, num_utccp_aligned_elems),
                                        align(block_n, num_utccp_aligned_elems)};
        default: DG_HOST_UNREACHABLE("Unknown dtype");
    }
}
```

- 对齐到 128 是因为 UTCCP 指令（`tcgen05.cp`）的最小传输粒度为 128 个元素
- Scale factor 的数据类型是 `float_ue8m0_t`（E8M0，1 字节），但在 SMEM 中以 `uint32_t`（4 字节）存储，这是因为 UTCCP 指令要求 32-bit 对齐
- Scale factor 的 **K 维度** 由 `kGranKA` / `kGranKB` 决定：

```cpp
// sm100_fp8_gemm_1d1d.cuh — K 维度的 SF 加载频率
constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");
```

  - `kGranK = 32`：每 32 个 K 元素一个 SF → 每个 BLOCK_K=128 包含 4 个 SF → 每个 K-block 都需要加载新的 SF（`kNumSFStagesPerLoad = 1`）
  - `kGranK = 128`：每 128 个 K 元素一个 SF → 4 个 BLOCK_K 共享一组 SF → 每 4 个 K-block 加载一次（`kNumSFStagesPerLoad = 4`）

#### ⑥ Barrier 对象

```cpp
// sm100.hpp — barrier SMEM 大小计算
static int get_barrier_smem_size(const int& num_stages) {
    // TMA full/empty barriers, with-SF full barriers, tensor memory full/empty barriers
    // NOTES: the last barrier is for tensor core utilization control
    return num_stages * 8 * 3 + 2 * 8 * 2 + 8;
}
```

```cpp
// sm100_fp8_gemm_1d1d.cuh — kernel 内的 barrier 指针分配
auto full_barriers         = PatternVisitor([=](const uint32_t& i) {
    return barrier_start_ptr + (i); });
auto empty_barriers        = PatternVisitor([=](const uint32_t& i) {
    return barrier_start_ptr + (kNumStages + i); });
auto with_sf_full_barriers = PatternVisitor([=](const uint32_t& i) {
    return barrier_start_ptr + (kNumStages * 2 + i); });
auto tmem_full_barriers    = PatternVisitor([=](const uint32_t& i) {
    return barrier_start_ptr + (kNumStages * 3 + i); });
auto tmem_empty_barriers   = PatternVisitor([=](const uint32_t& i) {
    return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });
```

5 类 barrier 的用途：

| Barrier | 数量 | 生产者 | 消费者 | 语义 |
|---------|------|--------|--------|------|
| `full_barriers` | num_stages | Warp 0 (TMA) | Warp 2 (SF transpose) | "SMEM 中的 A/B/SF 数据已就绪" |
| `empty_barriers` | num_stages | Warp 1 (MMA) | Warp 0 (TMA) | "MMA 已消费完该 stage，可以覆盖" |
| `with_sf_full_barriers` | num_stages | Warp 2 (SF transpose) | Warp 1 (MMA) | "SF 转置完成，可以执行 UTCCP + MMA" |
| `tmem_full_barriers` | kNumEpilogueStages | Warp 1 (MMA) | Epilogue warps | "TMEM 中的累加器已就绪" |
| `tmem_empty_barriers` | kNumEpilogueStages | Epilogue warps | Warp 1 (MMA) | "Epilogue 已读完 TMEM，可以覆盖" |

> **Blackwell 特性：** SM100 使用 `__grid_constant__` 将 TMA Tensor Map 存储在常量内存中，不占用 SMEM。
> ```cpp
> // sm100.hpp
> static int get_tensormap_smem_size(const GemmType& gemm_type) { return 0; }
> ```

### Swizzle 模式

> **源码：** [`common.hpp` — `get_swizzle_mode()`](../csrc/jit_kernels/heuristics/common.hpp)

Swizzle 用于避免 shared memory 的 bank conflict。选择逻辑：

```cpp
// common.hpp — get_swizzle_mode()
template <typename size_type_t>
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    // `> 0` means interleaving
    // 16B actually means non-swizzling (but interleaving)
    for (const int& mode: {128, 64, 32, 16}) {
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
    DG_HOST_UNREACHABLE("Unreachable");
}
```

```cpp
// common.hpp — get_smem_config() 中的 swizzle 模式计算
const int& swizzle_a_mode = get_swizzle_mode(
    major_a == cute::UMMA::Major::K ? block_k : load_block_m, ab_elem_size);
const int& swizzle_b_mode = get_swizzle_mode(
    major_b == cute::UMMA::Major::K ? block_k : load_block_n, ab_elem_size);
const int& swizzle_cd_mode = ArchSpec::enable_cd_swizzle(cd_dtype)
    ? get_swizzle_mode(block_n, cd_elem_size) : 0;
```

- A/B 矩阵的 swizzle 基于 **连续维度的大小**：K-major 时基于 BLOCK_K，MN-major 时基于 LOAD_BLOCK_M/N
- C/D 矩阵的 swizzle 基于 BLOCK_N
- Scale factor **不使用 swizzle**（数据量太小，不会产生严重的 bank conflict）

---

## Step 5: 线程配置

> **源码（host 端）：** [`sm100.hpp` — `get_thread_config()`](../csrc/jit_kernels/heuristics/sm100.hpp)
> **源码（kernel 端 warp 分派）：** [`sm100_fp8_gemm_1d1d.cuh` — `if (warp_idx == 0/1/2/...)` 分支](../deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh)

SM100 的线程配置固定为：

```cpp
// sm100.hpp — get_thread_config()
static ThreadConfig get_thread_config(const KernelType& kernel_type,
                                      const int& block_m, const int& block_n) {
    return ThreadConfig::sm100(128, 128);
    //                         ^^^  ^^^
    //          non_epilogue_threads  epilogue_threads
}
```

kernel 入口通过 `__launch_bounds__` 声明总线程数：

```cpp
// sm100_fp8_gemm_1d1d.cuh — kernel 签名
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_gemm_1d1d_impl(...) {
```

总共 256 个线程，分为两组：

| 线程组 | 线程数 | 包含的 Warp | 职责 |
|--------|--------|------------|------|
| Non-epilogue | 128 (4 warps) | Warp 0, 1, 2, 3 | TMA load / MMA / SF transpose / 空闲 |
| Epilogue | 128 (4 warps) | Warp 4, 5, 6, 7 | TMEM → SMEM → Global Memory |

kernel 内通过 `warp_idx` 分派各 warp 的角色：

```cpp
// sm100_fp8_gemm_1d1d.cuh — warp 角色分派
const auto warp_idx = cutlass::canonical_warp_idx_sync();

if (warp_idx == 0 and cute::elect_one_sync()) {
    // Warp 0: TMA Load — 只用 1 个 thread 发射 TMA 异步拷贝指令
    ...
} else if (warp_idx == 1 and is_leader_cta) {
    // Warp 1: MMA Issue — 只用 1 个 thread 发射 UMMA 指令（仅 leader CTA）
    ...
} else if (warp_idx == 2) {
    // Warp 2: SF Transpose — 32 个 thread 协作完成 scale factor 的 SMEM 转置
    ...
} else if (warp_idx >= kNumNonEpilogueThreads / 32
           and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
    // Warp 4-7: Epilogue — 128 个 thread 协作完成 TMEM→SMEM 搬运 + TMA store
    ...
}
```

各 warp 的具体角色：

```
Warp 0: TMA Load        — 只用 1 个 thread 发射 TMA 异步拷贝指令
Warp 1: MMA Issue        — 只用 1 个 thread 发射 UMMA 指令（仅 leader CTA）
Warp 2: SF Transpose     — 32 个 thread 协作完成 scale factor 的 SMEM 转置
Warp 3: 空闲（预留）
Warp 4-7: Epilogue       — 128 个 thread 协作完成 TMEM→SMEM 搬运 + TMA store
```

> **为什么 Warp 0 和 Warp 1 只用 1 个 thread？** TMA 和 UMMA 都是硬件指令，由专用硬件单元执行，不需要多线程协作。一个 thread 发射指令后，硬件异步完成数据搬运/计算。`cute::elect_one_sync()` 确保 warp 内只有一个 thread 执行。

---

## TMEM 布局

> **源码：** [`sm100_fp8_gemm_1d1d.cuh` — TMEM 常量定义和分配](../deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh)
> **源码：** [`sm100.hpp` — `is_block_size_legal()` 中的 TMEM 列数校验](../csrc/jit_kernels/heuristics/sm100.hpp)
> **参考：** [PTX ISA — Tensor Memory Layout](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-layout)

Tensor Memory 是 Blackwell 引入的新型片上存储，固定 **128 行 × 最多 512 列**，用于存放 MMA 的累加器和 scale factor。

```cpp
// sm100_fp8_gemm_1d1d.cuh — TMEM 布局的完整常量定义
constexpr uint32_t LAYOUT_AD_M = 128;
constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;

// SF 在 TMEM 中占用的列数
constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;

// 自动推导 epilogue stage 数：总列数 ≤ 512 时用 double buffer
constexpr uint32_t kNumEpilogueStages =
    (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;

// 累加器占用的列数
constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;

// TMEM 总列数（需要对齐）
constexpr uint32_t kNumTmemCols =
    get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();

// SFA / SFB 在 TMEM 中的起始列偏移
constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

// 合法性检查
DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");
```

```
TMEM 列布局:
┌──────────────────────────────────────────────────────────────────┐
│  累加器区域                              │  SFA  │  SFB          │
│  kNumEpilogueStages × kNumMWaves × BLOCK_N 列  │       │               │
├──────────────────────────────────────────┼───────┼───────────────┤
│  列 0 ────────────────── kNumAccumTmemCols │ SFA列 │ SFB列         │
└──────────────────────────────────────────────────────────────────┘
```

- **累加器**：`kNumEpilogueStages × kNumMWaves × BLOCK_N` 列，每列 128 行（对应 LAYOUT_AD_M=128）
- **SFA**：`SF_BLOCK_M / 32` 列
- **SFB**：`SF_BLOCK_N / 32` 列
- **kNumEpilogueStages** 的推导：当总列数 ≤ 512 时为 2（double buffer），否则为 1
- **kNumMWaves**：`BLOCK_M / min(BLOCK_M, 128)`，当 BLOCK_M > 128 时需要分多个 wave 处理 M 维度（**注意：这不是处理多个 tile，而是同一个 tile 内 M 维度超过 TMEM 128 行限制时的分波处理**）

TMEM 的分配和释放由不同的 warp 负责：

```cpp
// sm100_fp8_gemm_1d1d.cuh — Warp 2 负责分配 TMEM
} else if (warp_idx == 2) {
    Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
}

// sm100_fp8_gemm_1d1d.cuh — Epilogue 最后一个 warp 负责释放 TMEM
if (epilogue_warp_idx == kNumUMMAStoreThreads / 32 - 1)
    Allocator().free(0, kNumTmemCols);
```

---

## 数值示例

以 MoE Grouped GEMM 典型配置为例：`M=4096, N=7168, K=2048, num_groups=8, num_SMs=132`

**Step 1 — Tile 选择：**
```
候选: BLOCK_M ∈ {64, 128}, BLOCK_N ∈ {16, 32, ..., 256}
num_blocks(128, 128) = 32 × 56 × 8 = 14336
num_waves(128, 128)  = ceil(14336 / 132) = 109
last_util(128, 128)  = 14336 % 132 = 68

→ 最终选择: BLOCK_M=128, BLOCK_N=128, BLOCK_K=128
```

**Step 2 — Multicast：**
```
M=4096 ≥ 512 ✓
ceil_div(4096, 128) % 2 = 32 % 2 = 0 ✓
132 % 2 = 0 ✓
→ 开启 Multicast on B, kNumMulticast=2, LOAD_BLOCK_N=64
```

**Step 3 — Stage 数：**
```
每 stage SMEM:
  A: 128 × 128 × 1 = 16384 B
  B:  64 × 128 × 1 =  8192 B
  SFA: 128 × 4     =   512 B
  SFB: 128 × 4     =   512 B
  小计: 25600 B/stage

固定开销:
  CD: 128 × 128 × 2 = 32768 B
  Barriers + TMEM ptr ≈ 800 B
  小计: ~33568 B

可用: 232448 - 33568 = 198880 B
max_stages = 198880 / 25600 ≈ 7
→ 最终选择: num_stages = 7
```

**Step 4 — TMEM：**
```
kNumMWaves = 128 / 128 = 1
kNumAccumTmemCols = 2 × 1 × 128 = 256
kNumSFATmemCols = 128 / 32 = 4
kNumSFBTmemCols = 128 / 32 = 4
总列数 = 256 + 4 + 4 = 264 ≤ 512
→ kNumEpilogueStages = 2 ✓
```

---

## 深入：Swizzle、对齐与 UTCCP 的硬件约束

### swizzle_cd_mode 的含义

> **源码：** [`common.hpp` — `get_swizzle_mode()`](../csrc/jit_kernels/heuristics/common.hpp)
> **源码：** [`sm100.hpp` — `get_smem_cd_size()`](../csrc/jit_kernels/heuristics/sm100.hpp)

`swizzle_cd_mode` 是 **输出矩阵 C/D 在 shared memory 中使用的 swizzle 模式的字节宽度**（128 / 64 / 32 / 16）。

```cpp
// common.hpp — get_swizzle_mode()
static int get_swizzle_mode(const int& block_size, const size_type_t& elem_size) {
    for (const int& mode: {128, 64, 32, 16}) {
        if ((block_size * static_cast<int>(elem_size)) % mode == 0)
            return mode;
    }
}
```

选择逻辑：找到能被 `block_n * element_size` 整除的**最大** swizzle 粒度。例如 `BLOCK_N=128, cd_elem_size=2(BF16)` → `128*2=256`，能被 128 整除 → `swizzle_cd_mode = 128`。

这个值直接对应 PTX 文档中 **Shared Memory Descriptor**（bits 61-63）的 swizzle 模式编码：

| 编码值 | 含义 |
|--------|------|
| 0 | No swizzling |
| 1 | 128-Byte with 32B atomic swizzling |
| 2 | 128-Byte swizzling |
| 4 | 64-Byte swizzling |
| 6 | 32-Byte swizzling |

`swizzle_cd_mode` 的用途：
1. **计算 SMEM_CD 的大小**：`get_smem_cd_size()` 返回 `min(block_m, 128) * swizzle_cd_mode * 2`
2. **创建 TMA descriptor** 时传入，告诉 TMA 硬件如何在 SMEM 中排布数据
3. **MMA 指令的 shared memory descriptor** 中编码 swizzle 模式

### 为什么 SMEM 各区域必须 1024 字节对齐？

> **参考：** [PTX ISA §9.7.16.4.1 — Shared Memory Descriptor, Table 41](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#shared-memory-descriptor)

```cpp
// sm100_fp8_gemm_1d1d.cuh
DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0
    and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
    "Shared memory of A/B must be aligned to 1024 bytes");

extern __shared__ __align__(1024) uint8_t smem_buffer[];
```

**这是硬件要求，不是经验值。** PTX ISA 文档 §9.7.16.4.1 的 **Table 41** 明确规定了 swizzle 模式对 shared memory 起始地址的对齐要求：

| Swizzling mode | Starting address of the repeating pattern |
|---|---|
| **128-Byte swizzle** | **1024-Byte boundary** |
| 64-Byte swizzle | 512-Byte boundary |
| 32-Byte swizzle | 256-Byte boundary |

当使用 128B swizzle 模式时，shared memory 中每个矩阵 tile 的起始地址必须对齐到 **1024 字节边界**。

在 DeepGEMM 的 SMEM 布局中：
```
[SMEM_CD | A_stage_0 | A_stage_1 | ... | B_stage_0 | B_stage_1 | ... | SF | barriers]
```

每个区域的起始地址 = 前面所有区域大小之和。如果 `SMEM_CD_SIZE` 不是 1024 的倍数，那么 `A_stage_0` 的起始地址就不在 1024 字节边界上，128B swizzle 的 TMA load 和 MMA 指令就会产生**未定义行为**。同理，每个 stage 的 A/B 大小也必须是 1024 的倍数，否则后续 stage 的起始地址会偏移。

### 为什么 kNumUTCCPAlignedElems = 128？

> **参考：** [PTX ISA §9.7.16.9.2 — tcgen05.cp](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions-tcgen05-cp)
> **源码：** [`copy_sm100.hpp` — `SM100_UTCCP_4x32dp128bit_1cta`](../third-party/cutlass/include/cute/arch/copy_sm100.hpp)

**这也是硬件要求，不是经验值。** PTX 文档中 `tcgen05.cp` 指令的 `.shape` 选项：

```
.shape = { .128x256b, .4x256b, .128x128b, .64x128b, .32x128b }
```

DeepGEMM 使用的是 `.32x128b.warpx4` 模式（对应 `SM100_UTCCP_4x32dp128bit`），其 PTX 指令为：

```asm
// copy_sm100.hpp — SM100_UTCCP_4x32dp128bit_1cta::copy()
tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], sdesc;
```

这条指令的含义：
- **32 lanes × 128 bits** = 32 × 16 bytes = **512 bytes** 的数据从 SMEM 拷贝到 TMEM
- `.warpx4` 表示数据被 multicast 到 4 个 warp

Scale factor 是 `uint32_t`（4 bytes），每个 scale factor 对应一个 M/N 维度的元素。一次 `tcgen05.cp.32x128b.warpx4` 拷贝的数据量：

```
32 lanes × 128 bits / 32 bits per element = 32 × 4 = 128 个 uint32_t 元素
```

所以 `kNumUTCCPAlignedElems = 128` 正好是**一次 `tcgen05.cp.32x128b.warpx4` 指令能处理的 scale factor 元素数量**。

代码中的使用方式：

```cpp
// sm100_fp8_gemm_1d1d.cuh — SF 拷贝循环
for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
    auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
    replace_smem_desc_addr(sf_desc, smem_ptr);
    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
}
```

每次循环处理 128 个 SF 元素，对应 TMEM 中 4 列（`128 / 32 = 4`，因为 TMEM 每列有 32 个 lane，每个 lane 存一个 uint32_t）。

### 总结：三个值的性质

| 问题 | 答案 | 性质 |
|---|---|---|
| `swizzle_cd_mode` | C/D 矩阵的 swizzle 字节宽度（128/64/32/16），用于 TMA descriptor 和 MMA shared memory descriptor | 软件配置 |
| `% 1024 == 0` 对齐 | PTX ISA 规定 128B swizzle 模式的 repeating pattern 必须从 **1024 字节边界** 开始 | **硬件要求** |
| `kNumUTCCPAlignedElems = 128` | `tcgen05.cp.32x128b.warpx4` 一次拷贝 32 lanes × 128 bits = 128 个 uint32_t SF 元素 | **硬件要求** |

---

## 深入：SMEM_CD 的 M 维度为什么是 min(block_m, 128)？

> **源码：** [`sm100.hpp` — `get_smem_cd_size()`](../csrc/jit_kernels/heuristics/sm100.hpp)
> **源码：** [`sm100_fp8_gemm_1d1d.cuh` — `STORE_BLOCK_M` 定义](../deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh)

```cpp
// sm100.hpp
static int get_smem_cd_size(...) {
    constexpr static int layout_ad_m = 128;
    return std::min(block_m, layout_ad_m) * swizzle_cd_mode * 2;
}

// sm100_fp8_gemm_1d1d.cuh
constexpr uint32_t LAYOUT_AD_M = 128;
constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
```

核心原因：**TMEM 固定只有 128 行（lanes）**。

- 当 `BLOCK_M ≤ 128` 时，整个 tile 的 M 维度可以一次性放进 TMEM，epilogue 一次就能把所有 M 行的结果写出去
- 当 `BLOCK_M > 128`（如 `BLOCK_M = 256`）时，TMEM 一次只能容纳 128 行的累加结果，需要**分多个 M-wave** 处理

epilogue 写回时，每次只处理 `min(BLOCK_M, 128)` 行——不需要为整个 `BLOCK_M` 分配 SMEM_CD 空间，因为：

1. **TMEM 一次只输出 128 行**，多出来的行会在下一个 M-wave 处理
2. **SMEM_CD 是复用的**：第一个 wave 的 128 行写完后，同一块 SMEM_CD 可以被下一个 wave 复用
3. 这样做**节省了宝贵的 shared memory**，把更多空间留给 A/B 的 pipeline stages

| BLOCK_M | STORE_BLOCK_M | kNumMWaves | SMEM_CD 行为 |
|---|---|---|---|
| 64 | 64 | 1 | 一次写 64 行，SMEM_CD 只需 64 行 |
| 128 | 128 | 1 | 一次写 128 行，SMEM_CD 需 128 行 |
| 256 | 128 | 2 | 分 2 个 wave，每次写 128 行，SMEM_CD 只需 128 行（复用） |

`× 2` 则是因为 `kNumTMAStoreStages = 2`，即 epilogue 使用**双缓冲**（double buffering）来 overlap TMEM→SMEM 的拷贝和 SMEM→GMEM 的 TMA store。
