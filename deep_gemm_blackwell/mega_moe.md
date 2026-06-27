# DeepGEMM Mega MoE Kernel 详解（SM100 / Blackwell）

> **核心源码（路径相对 DeepGEMM 仓库根）：**
> - Python 前端（buffer 分配 / 权重变换 / 两个 kernel 封装）：[`deep_gemm/mega/__init__.py`](../../DeepGEMM/deep_gemm/mega/__init__.py)
> - C++ API 与分发：[`csrc/apis/mega.hpp`](../../DeepGEMM/csrc/apis/mega.hpp)
> - 启发式 / 配置（tiling、wave、流水深度）：[`csrc/jit_kernels/heuristics/mega_moe.hpp`](../../DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp)
> - JIT launcher / TMA 描述符：[`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`](../../DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp)
> - **主 kernel（FP8×FP4）**：[`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`](../../DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)
> - BF16 变体：[`deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh`](../../DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh)
> - 调度器：[`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`](../../DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)
> - workspace / pool 布局：[`deep_gemm/include/deep_gemm/layout/mega_moe.cuh`](../../DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh)
> - 对称内存映射 / NVLink barrier：[`layout/sym_buffer.cuh`](../../DeepGEMM/deep_gemm/include/deep_gemm/layout/sym_buffer.cuh)、[`comm/barrier.cuh`](../../DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh)
> - 测试 / 调用范例：[`tests/test_mega_moe.py`](../../DeepGEMM/tests/test_mega_moe.py)
>
> 本文先讲 Mega MoE 是什么、整体流程，再逐层拆解 warp 专精、三层 ring buffer、调度器 wave、底层 PTX 原语，最后用一节专门澄清几个常见疑惑（含 N 切 vs K 切、为什么 dispatch 与 GEMM 落在不同 SM）。

---

## 0. TL;DR

Mega MoE 是一个 **持久化（persistent）的 Blackwell 单 mega-kernel**，把整条 EP MoE 前向：

```
EP dispatch → Linear1(gate/up) → SwiGLU → Linear2(down) → EP combine
```

**融合进一次 kernel launch**，用 **warp 专精 + 异步原语（TMA / tcgen05 / mbarrier）** 让 NVLink 通信和 Tensor Core 计算全程重叠。

- **支持后端**：2 个数据类型后端，均仅 SM100：
  - `bf16xbf16` → `bf16_mega_moe`
  - `fp8xfp4`（默认）→ `fp8_fp4_mega_moe`（FP8 e4m3 激活 × FP4 e2m1 权重，UE8M0 scale）
- **只支持** `swiglu` 激活、`arch_major == 10`（Blackwell）。
- **通信模型**：EP over PyTorch symmetric memory（`torch.distributed._symmetric_memory`），**NVLink 直接映射对端显存**（非 NVSHMEM/IBGDA），最多 72 rank（NVL72/GB200）。
- **主要收益**：① 融合消除中间结果 HBM 往返（省 IO 带宽）② 通信-计算重叠（隐藏 NVLink 延迟）③ FP8/FP4 低精度减字节 ④ 单 launch 省启动/全局同步 ⑤ ring buffer 控显存使大 batch 可跑。

---

## 1. 对照基线：它在跟什么比

未融合标准做法 = 5 次独立 kernel（见 `tests/test_mega_moe.py:175-202`）：

```
dispatch →[写HBM] GEMM1 →[写HBM] SwiGLU →[写HBM] GEMM2 →[写HBM] combine
   每个箭头都有一次"写 HBM + 读回 HBM"，每步之间还要全局同步、各自 launch
```

Mega MoE 把这 5 步压进 **一次 launch**，中间结果（尤其 L1 输出）**不落 HBM**。下文所有优化都是相对这个基线。

---

## 2. 端到端流程（全在一个 kernel 内）

主机侧先把 `x`(FP8)、`x_sf`(UE8M0)、`topk_idx`、`topk_weights` 写进 symmetric buffer；权重一次性 host 变换（见 §6）。然后 kernel 内部 5 阶段：

```
 ┌─ A. DISPATCH (跨 rank 路由)  sm100_fp8_fp4_mega_moe.cuh:331-660
 │   1) 数 per-expert token 计数 (atomicAdd_block)
 │   2) 原子预留全局槽位 → 把源 token 索引写到"目标rank"的对称buffer (all-to-all 元数据)
 │   3) ── nvlink_barrier ── 等所有 rank 写完计数
 │   4) PULL: 轮询本地expert的token → round-robin 选源rank均衡 (:463-511)
 │         TMA 从远端拉 token+SF+weight → 存进 L1 ring (本地HBM)
 │         写 TokenSrcMetadata{rank,token,topk} 给 combine 用 (:589)
 └────────────────────────┬──────────────────────────────
                          │ L1 token ring (FP8, HBM, 跨SM)
 ┌─ B. LINEAR1 grouped GEMM (FP8激活 × FP4权重)
 │     A·W1 → TMEM, N=2*intermediate (gate 与 up 沿 N 拼接)
 └────────────────────────┬──────────────────────────────
                          │ 从 TMEM 读 gate/up 对
 ┌─ C. SwiGLU (融在 L1 epilogue, 不落地!) :938-1131
 │     silu(gate)*up*routing_weight → per-token amax
 │     → 重量化成 FP8 e4m3 + 新 UE8M0 SF → 直接写进 L2 ring
 └────────────────────────┬──────────────────────────────
                          │ L2 token ring (FP8, HBM, 跨SM)
 ┌─ D. LINEAR2 grouped GEMM (FP8激活 × FP4权重)
 │     h·W2 → TMEM, N=hidden
 └────────────────────────┬──────────────────────────────
                          │ L2 epilogue 转 BF16
 ┌─ E. COMBINE (跨 rank 聚合) :1133-1384
 │   1) 按 TokenSrcMetadata 把结果 scatter 回"源rank"的 combine buffer (:1231)
 │   2) ── nvlink_barrier ──
 │   3) 每个本地token: TMA拉回它的 top-k partial → FP32累加 → BF16 → 写 y
 └────────────────────────┬──────────────────────────────
                          ▼  输出 y (BF16)
```

**关键点：L1 输出从不以 BF16 落 HBM** —— 留在片上 TMEM，epilogue 直接 SwiGLU + 重量化成 FP8，写进 L2 输入 ring（`:993-1101`）。基线里"L1输出写HBM→SwiGLU读写HBM→L2读HBM"这三趟大数组 IO 全部省掉。

---

## 3. SM 分配：所有 SM 同构，专精在 warp 层

> **结论：没有"某些 SM 只做 GEMM、某些只做 dispatch"。每个 SM 跑同一个 CTA，内部都有完整的一套 warp 角色。专精发生在"一个 SM 内部的 warp 之间"，不是 SM 与 SM 之间。**

启动配置（`sm100_fp8_fp4_mega_moe.hpp:220-222`）：`grid = num_sms` 个 CTA、`cluster = 2`、`__launch_bounds__(kNumThreads, 1)`（每 SM 常驻 1 个 CTA）。

每个 CTA 内按 `warp_idx`（`:79`）分角色：

| warp 角色 | 条件 | 职责 |
|---|---|---|
| Dispatch warps | `warp_idx < kNumDispatchWarps` (`:331`) | 路由计数 + 跨rank拉token + workspace清理 |
| TMA-load A warp | `== kNumDispatchWarps` (`:661`) | 搬激活 A + SFA：HBM→smem |
| TMA-load B warp | `== +1` (`:722`) | 搬权重 B + SFB：HBM→smem |
| MMA warp | `== +2` (`:765`，仅 leader CTA) | 发 `tcgen05` UMMA |
| idle warp | `== +3` (`:878`) | 仅 reg dealloc |
| Epilogue warps | `>= +kNumMMANonEpilogueWarps` (`:882`) | SwiGLU/重量化、L2 scatter、combine 归约 |

寄存器倾斜（`setmaxnreg`）：dispatch/producer `warpgroup_reg_dealloc`（少寄存器），epilogue `warpgroup_reg_alloc<208>`（多寄存器），`:333/663/884`。

**工作怎么在 SM 间分**——不是分 SM，而是每个阶段各自 grid-stride 自取：

| 阶段 | 切分依据 | 代码 |
|---|---|---|
| dispatch 拉 token | 全局 token 下标 `token_idx = sm_idx*W + warp; += 全局warp数` | `:432` |
| L1/L2 GEMM | 全局 block 下标 `block_idx = blockIdx.x; += kNumSMs` | scheduler `:62,160` |

→ **每个 SM"什么都做一点"**，只是做不同 token/block 的那一份。负载均衡靠各阶段独立的 grid-stride，而不是把任务静态切给不同 SM。

---

## 4. 三层 ring buffer（环套环）

Mega MoE 有 **三个不同层级的环形缓冲**，解决不同层级的生产者/消费者解耦：

| 环 | 物理位置 | 装什么 | 控制原语 | 作用 |
|---|---|---|---|---|
| **operand ring** | smem（片上，CTA内） | GEMM 的 A/B tile | `full/empty` mbarrier + phase + `expect_tx` | TMA 搬数 ↔ MMA 计算 重叠 |
| **accum ring** | TMEM（片上，CTA内） | MMA 累加结果 | `tmem_full/tmem_empty` mbarrier + `tcgen05.commit` | MMA ↔ SwiGLU/epilogue 重叠 |
| **L1/L2 token ring** | **HBM（全局，跨SM）** | 一波 expert 的 token | **显存计数器 + `red.release.gpu`/`ld.acquire.gpu`** | **限显存=一波；满了背压上游** |

### 4.1 operand ring（smem，TMA → MMA）

`kNumStages` 个 smem 格子装 GEMM operand。生产者 TMA-load warp，消费者 MMA warp：

```
TMA-A/B warp:                       MMA warp:
  empty_barriers[s].wait(phase^1)     full_barriers[s].wait(phase)
  cp.async.bulk.tensor 搬 A/B          tcgen05.mma 算这格
  full_barriers[s]                     tcgen05.commit ──自动──►
    .arrive_and_expect_tx(N字节)          empty_barriers[s].arrive
    └ TMA硬件搬完N字节自动点亮            └ Tensor Core算完自动放空这格
```

- **`expect_tx`**：生产者只声明"这格会进 N 字节"，TMA 硬件搬完自己把屏障减到位 → 无线程轮询。
- **`tcgen05.commit`**：Tensor Core 算完自己点亮 empty → MMA warp 不等结果。
- **相位 phase**：`wait(phase)` vs `wait(phase^1)` 差一位天然错开；写指针绕回 stage0 时 `phase ^= 1`（`:295-302`），同一屏障地址无限复用不重建。

### 4.2 accum ring（TMEM，MMA → epilogue，2 格双缓冲）

`kNumEpilogueStages = 2`。MMA 算完写 TMEM accum，epilogue 来读：MMA 在算 accum1 的同时 epilogue 处理 accum0，两者重叠。屏障 `tmem_full/tmem_empty`（`:804/817/926/953`）。

### 4.3 L1/L2 token ring（HBM，dispatch → GEMM，跨 SM）——本文重点

物理上是 HBM 里 `kNumRingBlocks × BLOCK_M × hidden` 的二维数组。dispatch 拉来的 token 按"逻辑 block 编号 % 槽数"塞进去（`:933` `ring_block_idx = pool_block_idx % kNumRingBlocks`）。

**为什么需要它（省显存）**：不用 ring 就得把全部 expert 的 token 都物化进 HBM（大 batch 几十 GB / OOM）。用 ring 只开固定槽位，逻辑 block 用取模循环复用，**任意时刻 HBM 里只活着"一波"的 token**。

**背压（back-pressure）**：覆盖一个环格前要等它上一轮被消费完——
- dispatch 覆盖前等 `l1_empty_count`（`:529-532`）
- L1 epilogue 写 L2 前等 `l2_empty_count`（`:940-942`）

跨 SM 同步用显存计数器，不是 mbarrier：`red.release.gpu.add` 写（`:595` `l1_full_count`）、`ld.acquire.gpu` 读自旋（`:532/686/942`）。

```
        L1 token ring (HBM, kNumRingBlocks 槽, 全卡共享)
        ┌────────┬────────┬────────┬────────┐
        │ slot0  │ slot1  │ slot2  │ slot3  │ ──绕回(% kNumRingBlocks)──┐
        └────────┴────────┴────────┴────────┘                          │
逻辑block: 0       1        2        3        4(复用slot0,需等)...        │
   dispatch写(red.release.gpu) ───l1_full_count──► L1 GEMM读(ld.acquire) ┘
   L1 GEMM用完 ───l1_empty_count──► 放行 dispatch 覆盖
```

---

## 5. 调度器：Wave 调度

> 源码：`scheduler/mega_moe.cuh`；wave 大小启发式：`heuristics/mega_moe.hpp:134-185`

### 5.1 一个 block(tile) 是什么

每个 expert 是独立 grouped GEMM（M=该expert的token数，N/K固定）。输出按 `BLOCK_M × BLOCK_N` 切，每块=一个 block。`m_block_idx = block_idx / kNumL1BlockNs`，`n_block_idx = block_idx % kNumL1BlockNs`（N 是内层快维，`:122,158`）。

### 5.2 Wave = 把 expert 分批

```
所有 expert: [E0 E1 E2 E3 | E4 E5 E6 E7 | ...]
              └─ wave 0 ──┘ └─ wave 1 ──┘
一波内执行顺序: 先这波【所有 expert 的 L1】, 再这波【所有 expert 的 L2】
  wave0: [E0.L1 E1.L1 E2.L1 E3.L1] → [E0.L2 E1.L2 E2.L2 E3.L2] → wave1...
```

- 先 L1 后 L2（`get_next_block` `:155-178`）：L2 输入=L1输出经SwiGLU，必须 L1 先完。
- L1→L2 转换时把 expert 指针倒回波首：`set_expert_idx(align_down(expert-1, kNumExpertsPerWave))`（`:165`）。

### 5.3 一波内 grid-stride 摊给所有 SM

把这波所有 block 拉平成一维，CTA 从 `blockIdx.x` 起步、`+= kNumSMs` 跨步领取（`:160,172`）。`fetch_next_l1_block`（`:118-131`）定位 flat block_idx 落在哪个 expert：若 `m_block_idx ≥ 当前expert的m_blocks` 就减掉整个 expert 的 block 数再看下一个。

### 5.4 wave 大小 = 显存 ↔ 喂满SM 双约束

```
get_num_experts_per_wave (heuristics/mega_moe.hpp):
  num_max_experts_per_wave: 一波 pool token ≤ num_ring_tokens 的最大值 (:140-144, 显存上限)
  num_min_...to_fill_sms:   能喂满所有 SM 所需的最少 expert 数 (:156, imbalanceFactor=2)
  最终 = min(两者), 再做尾波最小化微调 (:170-184)
```

显存预算（`__init__.py:84-96`）：prefill(≥6144 token) ~8GB、固定每波 1 expert；decode ~18GB、heuristic 自选。

> **wave 解决什么**：把 ring buffer 占用从"所有 expert"降到"一波 expert"。tile 级流水（overlap）和 wave（显存复用）是正交的两件事——一波内各 expert 的 tile 仍是流水重叠的，先L1后L2只是为了让 ring 生产/消费节奏对齐、footprint 限制在一波。

---

## 6. 数据布局与量化

- **激活 FP8 e4m3**，per-32 **UE8M0** SF，recipe 强制 `(1,1,32)`（`mega.hpp:182,205`）。
- **权重 FP4 e2m1**，host 打包存储（k//2 int8），**在 smem 里解包成 8-bit**；权重 SF 是 UE8M0、MN-major、TMA 对齐。
- **Host 权重变换**（`__init__.py:133-151`）：
  - L1 权重做 granularity-8 的 gate/up 交错：`[g0..7,u0..7,g8..15,...]`（`_interleave_weights`），让 epilogue 用一条 `SM100_TMEM_LOAD_16dp256b1x` 相邻读出 gate/up 对。
  - SF 做 `(4,32)` 转置（`_transpose_sf_for_utccp`）以匹配 UTCCP 写进 TMEM 的布局。
- **SwiGLU 输出重量化**：SF 只存指数字节 `sf >> 23`（`:1097-1100`），即原始 UE8M0。

---

## 7. 底层 PTX 原语速查（overlap 的真正引擎）

> "提升 SM 利用率"靠的不是多 kernel/stream，而是 **warp 专精 + mbarrier 异步生产者/消费者队列 + 硬件异步发起并回填屏障**。

| 控制目标 | 原语（PTX） | 位置 |
|---|---|---|
| warp 角色分配 | `canonical_warp_idx_sync()` + 分支链 | `:79,331/661/722/765/882` |
| 寄存器倾斜 | `setmaxnreg.inc/dec.sync.aligned.u32` | `:333/663/884` |
| 建屏障 | `mbarrier.init.shared::cta.b64` | barrier.h:397 |
| 等数据/等空槽 | `mbarrier.try_wait.parity.shared::cta.b64`（自旋） | `:695/742/825/926` |
| **TMA完成自动到达** | `mbarrier.arrive.expect_tx.shared::cta.b64` | `:714/757` |
| 异步搬 2D operand（2-CTA multicast） | `cp.async.bulk.tensor.2d.cta_group::2...mbarrier::complete_tx::bytes` | copy_sm100_tma:91 |
| 异步搬 1D（跨rank拉token/combine） | `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` | tma.cuh:63 |
| 异步写回 | `cp.async.bulk...bulk_group` + `commit_group`/`wait_group` | tma.cuh:79/91 |
| 异步 block-scaled MMA | `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale` | tcgen05.cuh:61 |
| **MMA完成→直接到达屏障** | `tcgen05.commit.cta_group::2.mbarrier::arrive::one...multicast::cluster.b64` | barrier.h:848 |
| 分配 tensor memory | `tcgen05.alloc.cta_group::2.sync.aligned` | tmem_allocator:139 |
| SF 拷进 TMEM | `tcgen05.cp.cta_group::2.32x128b.warpx4`（UTCCP） | copy_sm100:510 |
| 从 TMEM 读累加器 | `tcgen05.ld.sync.aligned.16x256b.x1.b32` | copy_sm100:628 |
| named barrier（跨角色warp） | `bar.sync` / `barrier.sync`（`sync_aligned`/`sync_unaligned`） | utils.cuh:22-28 |
| 2-CTA cluster 同步 | `barrier.cluster.arrive.relaxed/wait` | barrier.cuh:14 |
| 选单 lane 发指令 | `elect.sync` | cluster_sm90:189 |
| 跨SM ring 计数器 | `red.release.gpu.add` / `ld.acquire.gpu` | ld_st.cuh:220/171 |
| 跨SM grid sync | `atom.release.gpu.add` + `ld.acquire.gpu` | barrier.cuh:21 |
| 跨rank NVLink barrier | `red.release.sys.add` / `ld.acquire.sys` | ld_st.cuh:224/234 |

**真正控制 overlap 的三组原语**：
1. `setmaxnreg`（warp 专精的寄存器倾斜）
2. `mbarrier.arrive.expect_tx` + `mbarrier.try_wait.parity`（带相位的异步生产者/消费者队列）
3. `cp.async.bulk(.tensor)` + `tcgen05.mma`/`tcgen05.commit.mbarrier::arrive`（搬运/计算异步发起，硬件直接回填屏障）

---

## 8. 优化总结：省了哪几种开销

| 开销类型 | 怎么省的 | 关键性 |
|---|---|---|
| **HBM IO（带宽）** | 融合 → 中间结果不落地；FP8/FP4 → 字节更少 | ★ 最大 |
| **通信延迟（NVLink）** | comm/compute 重叠；FP8 dispatch 减半流量 | ★ 很大 |
| **launch + 级间同步** | 5 launch→1，tile 级握手替代全局 barrier | 中 |
| **算力/SM 利用率** | TMA+tcgen05+mbarrier 异步流水，tensor core 满载 | 中 |
| **显存容量** | ring buffer 流式，占用只跟"一波"挂钩 | 使能项 |

**面向阶段**：prefill + decode 都支持，各有独立调参路径。
- prefill（token多、compute-bound）：主要吃"融合省IO + FP8/FP4高吞吐 + SM满载"；每波1 expert 就够喂满SM。
- decode（token少、comm/latency-bound）：招牌特性"dispatch/combine 与计算融合重叠"边际收益最突出；wave heuristic 一波多塞 expert 凑满 SM。

---

## 9. 常见疑惑澄清（Q&A）

### Q1. ring buffer 是什么？是不是 token 多了就轮询、效率下降？

Ring buffer = 一块**固定大小、首尾相接循环复用**的内存。它是**滑动窗口**，不是串行排队：只要生产者填充和消费者腾空速度匹配，窗口一直滑、不产生等待，`%` 和 `ld.acquire` 自旋只在真卡住时才空转。

token 多 → 总时间长，这是 **算力上限（FLOPs）** 决定的，**与 ring 无关**——去掉 ring 也不会更快，只会 OOM。只要 ring 够深（设计就是为此，prefill ~8GB / decode ~18GB），GEMM 全程满载，**吞吐几乎无损失**。

两种等待性质相反：
- **dispatch 等 GEMM（背压）** → GEMM 满载 → **好**，无损失（token 多时通常是这种）。
- **GEMM 等 dispatch（饿死）** → SM 空转 → 坏，靠深 ring 避免。

真实小开销只有：wave 边界的流水填充/排空气泡（heuristic 尽量减波数/满尾波）、HBM 带宽（通常被计算掩盖）。

### Q2. SwiGLU 是某个 GEMM 算完就直接做、不等所有 expert 吗？

是，而且粒度比"一个 expert"更细——**按 tile 流水**。某个 tile 的 MMA 一完成（`tmem_full` 信号 `:926`），epilogue warp 立刻对这个 tile 做 SwiGLU（`:993` 起），不等同 expert 其它 tile，更不等别的 expert。GEMM/SwiGLU/L2 在不同 warp 上同时跑。

### Q3. 为什么 L1/L2 token ring 要跨 SM？

因为 **拉 token 的 SM ≠ 算 GEMM 的 SM**：
- dispatch 按"全局 token 下标"切（`:432`），所有 SM 一起拉，谁命中谁拉。
- GEMM 按"全局 block 下标"grid-stride 切，谁算哪块由 stride 决定。
- 两套切分互不相干 → 某 token 由 SM3 拉、却由 SM9 去算。

生产者 SM 和消费者 SM 不同，握手只能放在全卡共享的 HBM、用 `red.release.gpu`/`ld.acquire.gpu` 计数器（mbarrier 只在单 SM 内有效）。

### Q4. 为什么 dispatch 与 GEMM 必须用不同 SM 划分（不能让一个 SM 包圆）？三个硬约束：

1. **一份 token 被多个 SM 同时用**（最决定性）：同一 m_block 的 token 喂给该 expert 的所有 n_block tile，而这些 n_block tile 被 grid-stride 摊到不同 SM。→ token 不能归属单个消费 SM，只能放 HBM 共享。
2. **GEMM 的 block 划分依赖 dispatch 的计数结果**：每 expert 收多少 token（→ m_block 数 → block 编号）要等 dispatch 跨 rank 计数完成（`fetch_expert_recv_count` `:185-199` 自旋等所有 rank 报数）。拉之前还不知道消费 SM 是谁，只能用与 block 无关的 token 切分。
3. **两阶段瓶颈不同、最优切分是不同函数**：dispatch 是 NVLink-bound，要"token 在各源 rank 间均衡"（round-robin）；GEMM 是 compute-bound，要"tile 在 SM 间均衡"（抗 MoE 路由不均）。强行统一会让其中一个不均衡。

### Q5. 同一份 token 在 N 维度会分给不同 SM 吗？

会。固定 (expert, m_block)，它的 `kNumL1BlockNs` 个 n_block 拿到连续 flat 编号，经 `% kNumSMs` 落到**不同 SM**。
- **2-CTA cluster 缓解但不消除**：调度器约束 N block 数为偶（`:37-41`），cluster 内相邻 2 个 SM 共享同一 m_block、A 走 multicast（`LOAD_BLOCK_M = BLOCK_M/2`），cluster 内只读一次 token；但 **cluster 之间** 仍各读一遍同样的 token。重复读次数 ≈ `kNumL1BlockNs / 2`。

### Q6. 划分到多个 SM 不是要 reduce partial sum 吗？

**不是。这是 N 切 vs K 切的混淆。**
- 输出 `C[M,N] = A[M,K] @ B[K,N]`。**K 是被求和的累加维。**
- **沿 N 切**（本 kernel 做法）：每个 SM 算输出的**不同列** `C[m0, n_i] = A[m0,:](全K) @ B[:, n_i]`。各 SM 产出的列互不重叠 → **各写各的地址 → 零 reduce**。共享的只是输入读取（A[m0] 多读几遍），不是输出累加。
- **沿 K 切（split-K）**：两个 SM 各算半个 K 的 partial dot product，再相加 → 这才需要跨 SM reduce。**本 kernel 把整个 K 放在一个 cluster 内 K-loop 完成，刻意不做 split-K，避免 reduce。**

为什么不让一个 cluster 包掉某 m0 的全部 N？① 一个 tile 受 TMEM/smem/寄存器限制只能 `BLOCK_M × BLOCK_N`，N=2*intermediate 太大必须切；② 摊到更多 SM 是"零代价换并行"（N 切不需要 reduce），否则只用 2 个 SM 串行算 N，prefill 时浪费 146 个 SM。

### Q7. dispatch 拉完喂 MMA、算的同时拉下一批 → overlap，对吗？没有 round-robin 以前怎么做？

方向对，但中间还隔一跳：dispatch **不直接喂 MMA**，而是写进 HBM 的 L1 ring，GEMM 自己的 TMA-A warp 再从 ring 搬到 smem 给 MMA：

```
dispatch warp → L1 ring(HBM) → GEMM 的 TMA-A warp → MMA warp → epilogue
(NVLink:远端HBM→smem→本地HBM)   (HBM→smem)         (smem→TMEM)
```

且是**连续流水**（不是一批批交替）：任意时刻 dispatch 在拉 b4、MMA 在算 b2、epilogue 在处理 b1，靠 ring 的 full/empty 计数器解耦。

**round-robin 解决"拉得快不快"**（NVLink 负载均衡）。朴素做法按源 rank 分段拉（先拉完 rank0 的、再 rank1），同一时刻只有一条 NVLink 链路在忙、带宽浪费、还可能把某 rank 打成热点；round-robin（`:463-511` iterative min-peeling）跨源 rank 交错拉，多条链路同时满，dispatch 吞吐拉满才能真正把通信藏进计算。
> 注：这里描述的是 round-robin 要解决的"朴素替代做法"，非断言仓库历史上某具体旧版本；如需真实 git 演进可查 log。

### Q8. round-robin 和 ring buffer 各针对什么？

- **round-robin**：针对 **NVLink 通信带宽**——拉取在各源 rank 间交错，避免单 rank 热点，多链路同时满。发生在 dispatch pull（生产侧）。
- **ring buffer**：针对 **HBM 显存容量**——固定环 + 取模复用 + 背压，占用只跟"一波"挂钩，不 OOM。发生在 L1/L2 token pool（消费侧）。
- 一句话：round-robin 让 token"拉得均匀拉得快"（网络层），ring buffer 让拉进来的 token"存得下不爆显存"（内存层）。

---

## 10. BF16 变体差异

`sm100_bf16_mega_moe.cuh` 结构与 FP8/FP4 完全相同（同样 warp 专精、scheduler、ring、dispatch/combine over symmetric memory），区别：
- BF16 token/intermediate 布局，**无 SF buffer**、无 UTCCP/SFA/SFB TMEM 列。
- L1 epilogue 做 SwiGLU 后**直接写 BF16**（无 FP8 重量化、无 amax 归约）。
- 权重变换只交错 L1 gate/up，L2 不变（`__init__.py:148-150`）。
