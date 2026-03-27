# RTP-LLM Code Review 已知问题模式

## 概述

本文档记录 RTP-LLM 项目中 Code Review 时常见的问题模式，按主题分类。是 `rtp-llm-code-review.skill.md` 的详细参考。

---

## A. 兼容性与配置

### A.1 配置项缺少默认值

**模式**：新增配置项（环境变量、命令行参数、GptInitModelParameters 字段）未提供默认值，导致旧部署启动失败。

**检查点**：
- 新字段在 `GptInitModelParameters` 中是否有默认值
- Python 端 `getattr(config, 'new_field', default)` 是否有 fallback
- 环境变量是否用 `os.environ.get('KEY', default)` 读取

### A.2 stub_source 符号链接

**模式**：PR 中包含 `stub_source` 符号链接的变更（通常是 `git checkout` 导致的意外重置）。

**检查点**：diff 中不应出现 `stub_source` 的变更。

### A.3 pybind11 绑定与 Python 调用不匹配

**模式**：C++ pybind11 绑定接受 `const torch::Tensor&`，但 Python 端可能传入 `None`。pybind11 不会自动将 `None` 转为空 tensor，直接抛 `TypeError`。

**典型案例**：`update_kv_cache_offset(kv_cache_block_id_device)` 在 `kv_cache_block_id_device` 为 `None` 时崩溃。

**修复模式**：
```python
# Python 端加 None 检查
if value is not None:
    cpp_binding_call(value)
```

或在 C++ 端使用 `std::optional<torch::Tensor>` 或 `py::none` 处理。

---

## B. 正确性与逻辑

### B.1 CUDA Graph capture/replay tensor 一致性

**模式**：CUDA Graph capture 时使用的 tensor 地址在 replay 时必须保持不变。如果 replay 路径创建了新 tensor 或改变了 tensor 的 data_ptr，graph 会使用过期的地址。

**检查点**：
- `prepare_cuda_graph` 方法中是否正确复用 capture 时的 tensor
- `CudaGraphRunner.cc` 中的 `kv_cache_block_id_device = kv_cache_kernel_block_id_device` 赋值是否保持一致
- `fillParams` 是否更新了 graph 内部引用的 tensor 内容而非替换 tensor

### B.2 KV Cache Block ID 两组字段

**模式**：RTP-LLM 维护两组 block ID：
- `kv_cache_kernel_block_id_*`：用于 attention 计算（kernel 直接使用）
- `kv_cache_block_id_*`：用于 cache store（写入 KV cache）

**检查点**：
- 改动是否正确区分了这两组字段
- Graph 模式下是否正确赋值（`CudaGraphRunner.cc` 第 357/594 行）
- Python 端 `prepare_cuda_graph` 是否使用了正确的字段

### B.3 TP/EP 并行下的数据一致性

**模式**：Tensor Parallel (TP) 和 Expert Parallel (EP) 下，某些数据需要在所有 rank 间同步。

**检查点**：
- 新增的共享状态是否有 broadcast/allreduce
- 配置变更是否在所有 rank 上一致
- 随机数种子是否正确同步

### B.4 ROCm HIPGraph 特殊处理

**模式**：ROCm 平台的 HIPGraph 与 CUDA Graph 行为不完全一致，特别是 RCCL collectives 在 capture 模式下的限制。

**检查点**：
- ROCm 特定代码路径是否有正确的条件编译
- HIPGraph capture 状态管理是否正确
- aiter（ROCm attention 实现）的 `prepare_cuda_graph` 是否处理了 `None` 值

---

## C. 性能

### C.1 热路径日志

**模式**：在 per-forward、per-token 的热路径中添加日志打印，导致性能严重下降。

**检查点**：
- `forward()` 方法内部不应有无条件的 `logging.info/debug`
- 使用 `if logger.isEnabledFor(logging.DEBUG)` 守卫
- CUDA kernel 内部不应有 `printf`（除非 debug 模式）

### C.2 不必要的内存分配

**模式**：在热路径中创建临时 tensor 或 Python 对象，触发 GC 或 CUDA malloc。

**检查点**：
- 是否可以复用已有 buffer
- `torch.empty` / `torch.zeros` 是否可以移到初始化阶段
- Python list/dict 创建是否可以避免

### C.3 Kernel Launch 参数

**模式**：CUDA kernel 的 grid/block 配置不合理，导致 GPU 利用率低。

**检查点**：
- block size 是否为 32 的倍数（warp size）
- grid size 是否覆盖了所有数据
- shared memory 使用是否合理

---

## D. 多平台兼容

### D.1 CUDA/ROCm/PPU 同步

**模式**：在一个平台上的改动未同步到其他平台。

**检查点**：
- `cpp/devices/cuda_impl/` 的改动是否需要同步到 `rocm_impl/`
- `models_py/modules/factory/attention/` 下的 CUDA 实现是否有对应的 ROCm 实现
- 新增的 C++ op 是否在所有平台的 BUILD 文件中注册

### D.2 条件编译

**模式**：平台特定代码缺少正确的条件编译宏。

**检查点**：
- `#ifdef USING_ROCM` / `#ifdef USING_CUDA` 是否正确
- Python 端 `if device_type == 'rocm':` 分支是否完整
- Bazel `select()` 是否覆盖了所有平台

---

## E. 测试覆盖

### E.1 新功能缺少测试

**模式**：新增功能或修复 bug 没有对应的测试用例。

**检查点**：
- 新增的 Python 函数/类是否有单元测试
- 新增的 C++ op 是否有对应的 test target
- Bug fix 是否有回归测试

### E.2 Smoke 测试 Golden 数据

**模式**：模型行为变更后 smoke 测试的 golden 数据未更新。

**检查点**：
- 改动是否影响模型输出（attention 算法、数值精度）
- 是否需要更新 `smoke_actual` golden JSON
- 非确定性改动是否需要放宽 golden 比较阈值

---

## F. 代码组织

### F.1 无关改动混入

**模式**：PR 中混入了与主要功能无关的格式化改动（空格对齐、import 重排等）。

**检查点**：
- 格式化改动是否应该单独 PR
- 是否有超过 30 行的纯格式化 diff

### F.2 Commit 原子性

**模式**：单个 commit 包含多个不相关的改动。

**检查点**：
- 每个 commit 是否自包含
- commit message 是否准确反映改动内容
- 是否可以 cherry-pick 单个 commit

---

## G. 安全与敏感信息

### G.1 内部路径泄露

**模式**：代码或注释中包含内部 NAS 路径、内部 URL、员工信息。

**检查点**：
- 开源仓库代码中不应包含 `alibaba-inc.com` 等内部域名
- 模型路径不应硬编码内部存储地址
- 注释中不应包含员工花名/工号

---

## 按 PR 类型的 Review 侧重

| PR 类型 | 重点关注 | 常见问题 |
|---------|---------|---------|
| Attention Kernel | B.1, B.2, C.1, C.3, D.1 | Graph 一致性、block ID 混淆 |
| 配置/参数变更 | A.1, A.3, B.3 | 缺默认值、绑定不匹配 |
| 模型新增 | E.1, E.2, D.1 | 缺测试、缺多平台支持 |
| 性能优化 | C.1, C.2, C.3, E.2 | 引入 bug、golden 失效 |
| Bug Fix | B.*, E.1 | 缺回归测试 |
| 重构 | F.1, F.2, A.2 | 混入无关改动 |
| 分布式 | B.3, D.1, E.1 | TP/EP 不一致 |

---

## 变更记录

- 2026-03-27: 初始版本，基于 RTP-LLM 项目 .claude/skills/code-review/SKILL.md 和实际 review 经验整理
