---
name: rtp-llm-build-test
description: "RTP-LLM 构建与测试指南。MUST USE for any task involving RTP-LLM bazelisk build, bazelisk test, 编译, 构建, 本地测试, 远程测试, smoke test, 单元测试, compilation error, build failure. Covers environment setup, platform configs (CUDA/ROCm/PPU), local/remote execution, cache management, and common error resolution."
---

# RTP-LLM Build & Test

> 你是 RTP-LLM 构建与测试专家。当用户需要编译、构建、运行测试时应用此知识。

---

## 项目结构

```
RTP-LLM/                          # 内源仓库根目录
├── github-opensource/             # 开源 submodule（所有 bazelisk 命令在此目录执行）
│   ├── .bazelrc                   # 平台配置（CUDA/ROCm/PPU）
│   ├── .bazeliskrc                # bazelisk 版本配置
│   ├── stub_source -> internal_source  # 必须指向 internal_source
│   ├── rtp_llm/                   # 主代码目录
│   │   ├── cpp/                   # C++ 核心代码
│   │   ├── models_py/             # Python 模型代码
│   │   └── test/                  # 测试目录
│   └── internal_source/           # 内部配置
│       └── .cicd_bazelrc          # CI/远程执行配置
└── build_logs/                    # 构建日志输出目录
```

**关键约束**：所有 `bazelisk` 命令必须在 `github-opensource/` 目录下执行。

---

## 前置检查（每次构建前必做）

```bash
cd <project_root>/github-opensource

# 1. 检查 stub_source 指向（必须是 internal_source）
ls -la stub_source
# 如果指向 open_source，修复：
ln -sfn internal_source stub_source

# 2. 杀掉残留进程
pkill -f 'entry.py|start_server' 2>/dev/null
pkill -f 'bazel.*github-opensource' 2>/dev/null

# 3. 确认 /dev/shm 空间充足
df -h /dev/shm
```

**常见陷阱**：`git checkout -- .` 会重置 `stub_source → open_source`，导致 Bazel 挂起。

---

## 平台配置

| 平台 | `--config` | Cache 目录 |
|------|-----------|-----------|
| CUDA 12.x | `--config=cuda12_9` | `~/.cache/bazel_cuda12_cache` |
| ROCm | `--config=rocm` | `~/.cache/bazel_rocm_cache` |
| PPU | `--config=ppu` | `~/.cache/bazel_ppu_cache` |

---

## 构建命令

### 本地构建（带远程缓存加速）

```bash
cd <project_root>/github-opensource

bazelisk --output_user_root=~/.cache/bazel_cuda12_cache build \
  <targets> \
  --config=cuda12_9 \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --jobs=200 --verbose_failures
```

### 仅编译测试（不运行）

```bash
bazelisk --output_user_root=~/.cache/bazel_cuda12_cache test \
  <targets> \
  --config=cuda12_9 \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --build_tests_only=1 --jobs=200
```

---

## 测试命令

### 本地测试（带远程缓存）

```bash
cd <project_root>/github-opensource

bazelisk --output_user_root=~/.cache/bazel_cuda12_cache test \
  //rtp_llm/test/xxx:target_name \
  --config=cuda12_9 \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --test_timeout=3000 --jobs=200 \
  --run_under=//rtp_llm/test/utils:gpu_lock
```

### 远程执行测试

仅当用户明确要求远程执行时，额外加 `--config=daily_aone_bazel_remote`：

```bash
bazelisk --output_user_root=~/.cache/bazel_cuda12_cache test \
  //rtp_llm/test/xxx:target_name \
  --config=cuda12_9 \
  --config=daily_aone_bazel_cache \
  --config=daily_aone_bazel_remote \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --run_under=//rtp_llm/test/utils:gpu_lock \
  --test_timeout=3000 --jobs=200
```

### 稳定性测试（重复 N 次）

```bash
bazelisk --output_user_root=~/.cache/bazel_cuda12_cache test \
  //internal_source/rtp_llm/test/smoke:target_name \
  --config=cuda12_9 \
  --config=daily_aone_bazel_cache \
  --config=daily_aone_bazel_remote \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --run_under=//rtp_llm/test/utils:gpu_lock \
  --runs_per_test=10 --test_timeout=3000 --jobs=200
```

---

## 常用 Bazel 参数

| 参数 | 用途 |
|------|------|
| `--jobs=200` | 并行编译任务数 |
| `--keep_going` | 遇错继续编译 |
| `--verbose_failures` | 详细错误输出 |
| `--test_timeout=3000` | 测试超时（秒） |
| `--runs_per_test=N` | 重复测试 N 次 |
| `--build_tests_only=1` | 仅编译不运行 |
| `--test_tag_filters=-manual,-multi_device` | 排除标签 |

---

## 远程缓存 vs 远程执行

| 配置 | 含义 |
|------|------|
| `--config=daily_aone_bazel_cache` | 读写远端编译缓存（编译在本地） |
| `--config=daily_aone_bazel_remote` | 远程执行（编译和测试都在远端） |

**默认总是带 cache 参数**，仅用户明确要求时才加 remote。

---

## 常见编译错误

| 错误 | 原因 | 修复 |
|------|------|------|
| `undeclared inclusion of 'header.h'` | BUILD 缺少 dep | 添加对应 target 到 `deps` |
| `no such target '//path:target'` | target 不存在 | 检查 BUILD 文件路径 |
| `circular dependency` | 循环依赖 | 使用前向声明或拆分 |
| `Could not find any cuda.h` | 宿主机无 CUDA | 必须在容器内执行 |
| Bazel 挂起不动 | `stub_source → open_source` | `ln -sfn internal_source stub_source` |

---

## Docker 容器执行

构建和测试必须在 Docker 容器内执行：

```bash
# 查找容器
docker ps --format '{{.Names}}\t{{.Image}}\t{{.ID}}' | grep <keyword>

# 在容器内执行
docker exec -u <user> <container> bash -c "cd <project_root>/github-opensource && <command>"

# 后台执行长任务
docker exec -u <user> -d <container> bash -c "<command> > <project_root>/build_logs/task.log 2>&1; echo \$? > <project_root>/build_logs/task.signal"
```

---

## 详细文档

完整的缓存管理、测试产物获取、Cache 污染修复等详见：`rtp-llm/RTP-LLM_Build_Test_Complete_Guide.md`
