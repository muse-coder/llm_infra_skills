# RTP-LLM 构建与测试完整指南

## 概述

RTP-LLM 是阿里巴巴开源的 LLM 推理引擎（类似 vLLM/SGLang），使用 Bazel 构建系统，通过 bazelisk 管理版本。支持 CUDA、ROCm、PPU 三个平台。

本文档是 `rtp-llm-build-test.skill.md` 的详细参考，覆盖完整的构建测试流程、缓存管理、容器操作和排错指南。

---

## 仓库结构详解

```
RTP-LLM/                              # 内源仓库（foundation_models/RTP-LLM）
├── github-opensource/                 # 开源 submodule（alibaba/rtp-llm）
│   ├── .bazelrc                       # 平台配置定义
│   ├── .bazeliskrc                    # bazelisk 版本
│   ├── BUILD                          # 根 BUILD 文件
│   ├── WORKSPACE                      # 外部依赖声明
│   ├── stub_source -> internal_source # 符号链接（关键！）
│   ├── deps/                          # 依赖 submodule
│   ├── rtp_llm/
│   │   ├── cpp/                       # C++ 核心（attention, kernels, devices）
│   │   │   ├── devices/              # 设备抽象层（CudaDevice, ROCmDevice）
│   │   │   ├── kernels/              # CUDA/HIP kernels
│   │   │   └── model_rpc/            # gRPC 模型服务
│   │   ├── models_py/                # Python 模型实现
│   │   │   ├── modules/              # 模块工厂（attention, embedding）
│   │   │   └── bindings/             # pybind11 绑定
│   │   └── test/                     # 测试目录
│   │       ├── smoke/                # Smoke 测试（端到端）
│   │       └── utils/                # 测试工具（gpu_lock 等）
│   └── internal_source/              # 内部配置
│       ├── .cicd_bazelrc             # CI 远程执行配置
│       └── rtp_llm/test/smoke/       # 内部 smoke 测试
└── build_logs/                        # 构建日志目录（需手动创建）
```

### stub_source 机制

`stub_source` 是一个符号链接，决定 Bazel 从哪里加载内部配置：
- `stub_source -> internal_source`：正确，使用内部配置和远程缓存
- `stub_source -> open_source`：错误，Bazel 会尝试从 GitHub 下载依赖，导致挂起

**触发重置的操作**：
- `git checkout -- .`
- `git reset --hard`
- `git stash` + `git stash pop`

**修复命令**：
```bash
cd <project_root>/github-opensource
ln -sfn internal_source stub_source
```

---

## Docker 容器操作详解

### 查找容器

```bash
# 按用户名查找
docker ps --format '{{.Names}}\t{{.Image}}\t{{.ID}}' | grep <username>

# 按镜像关键词查找
docker ps --format '{{.Names}}\t{{.Image}}' | grep -i cuda

# 检查已停止的容器
docker ps -a --format '{{.Names}}\t{{.Status}}' | grep <container>

# 启动已停止的容器
docker start <container>
```

### 容器内执行

```bash
# 交互式执行
docker exec -u <user> <container> bash -c "<command>"

# 后台执行（长时间任务）
docker exec -u <user> -d <container> bash -c "\
  cd <project_root>/github-opensource && \
  <command> > <project_root>/build_logs/<name>.log 2>&1; \
  echo \$? > <project_root>/build_logs/<name>.signal"
```

### 容器分支同步

容器内的 git 仓库是独立的工作目录，不会自动同步宿主机改动：

```bash
# 检查容器内分支
docker exec -u <user> <container> bash -c "\
  cd <project_root> && git rev-parse --abbrev-ref HEAD && \
  cd github-opensource && git rev-parse --abbrev-ref HEAD"

# 切换分支
docker exec -u <user> <container> bash -c "\
  cd <project_root> && git fetch origin <branch> && git checkout <branch> && \
  cd github-opensource && git fetch origin <branch> && git checkout <branch>"

# 同步单个文件
docker cp /host/path/to/file <container>:/container/path/to/file
```

---

## 平台配置详解

### .bazelrc 中的平台定义

| 平台 | `--config` 值 | GPU 架构 | 说明 |
|------|--------------|---------|------|
| CUDA 12.x | `cuda12_9` | SM 70-90 | 默认平台，覆盖 V100-H100 |
| ROCm | `rocm` | gfx90a/gfx942 | AMD MI250/MI300 |
| PPU | `ppu` | 自定义 | 燧原科技 |

### 远程执行配置（.cicd_bazelrc）

| 配置 | 端点 | 用途 |
|------|------|------|
| `daily_aone_bazel_cache` | alibaba-inc.com | 日常开发机读写远端缓存 |
| `daily_aone_bazel_remote` | alibaba-inc.com | 日常开发机远程执行 |
| `online_aone_bazel_cache` | vipserver | CI Pod 读写远端缓存 |
| `online_aone_bazel_remote` | vipserver | CI Pod 远程执行 |

### 认证

所有远程操作需要认证 header：
```
--remote_header=x-aone-bazel-api-key=ai-infra-cicd
```

---

## 缓存管理

### Cache 目录约定

```
/home/<username>/.cache/bazel_<arch>_cache
```

| 平台 | 路径 |
|------|------|
| CUDA | `/home/<username>/.cache/bazel_cuda12_cache` |
| ROCm | `/home/<username>/.cache/bazel_rocm_cache` |
| PPU | `/home/<username>/.cache/bazel_ppu_cache` |

### Cache 污染修复

如果在宿主机上错误运行了 bazel（比如用 fake CUDA toolkit），会污染共享的 bazel output_base：

```bash
# 1. 杀掉宿主机上的 bazel server
pkill -u $(whoami) -f 'bazel.*github-opensource'

# 2. 删除宿主机上的 bazel state
rm -rf /dev/shm/<username>/.cache/bazel/

# 3. 在容器内清理
docker exec -u <user> <container> bash -c "\
  pkill -u <user> -f bazel; sleep 1; \
  rm -rf /home/<user>/.cache/bazel_cuda12_cache; \
  mkdir -p /home/<user>/.cache/bazel_cuda12_cache"

# 4. 重新在容器内执行构建
```

### 清理 Bazel 缓存

```bash
# 完全清理（删除所有构建产物）
bazelisk --output_user_root=~/.cache/bazel_cuda12_cache clean --expunge

# 仅清理当前 workspace 的构建产物
bazelisk --output_user_root=~/.cache/bazel_cuda12_cache clean
```

---

## 测试产物获取

远程执行的测试产出（smoke_actual、fingerprint 等）会打包到 `test.outputs/outputs.zip`：

```bash
# 查找 test outputs
docker exec -u <user> <container> bash -c "\
  ls <cache_dir>/*/execroot/rtp_llm/bazel-out/k8-opt/testlogs/<test_path>/run_*/test.outputs/"

# 解压到本地分析
docker cp <container>:<path>/test.outputs/outputs.zip /tmp/outputs.zip
cd /tmp && unzip -q outputs.zip

# smoke_actual 在 smoke_actual/<golden_path>/*.query_N.json
```

---

## 构建进度指标

| 输出 | 含义 |
|------|------|
| `Loading: X packages loaded` | 正在加载 package |
| `[X / Y] Compiling ...` | 正在编译 |
| `Target //...:target up-to-date` | 编译成功 |
| `FAILED` | 编译失败 |

**注意**：CUDA kernel `.cu` 文件编译可能需要 10+ 分钟。

---

## 常见编译错误详解

### 1. undeclared inclusion

```
error: undeclared inclusion of 'some_header.h'
```
**原因**：BUILD 文件中缺少对应的 `deps` 依赖。
**修复**：找到 header 所在的 BUILD target，添加到 `deps` 列表。

### 2. no such target

```
ERROR: no such target '//path/to:target'
```
**原因**：target 不存在或路径错误。
**修复**：检查 BUILD 文件中是否定义了该 target，或路径是否正确。

### 3. no such package

```
ERROR: no such package 'path/to'
```
**原因**：模块被移动或重命名。
**修复**：更新路径到新位置。

### 4. Label is duplicated

```
ERROR: Label '//path:target' is duplicated in the 'deps' attribute
```
**原因**：deps 中有重复条目。
**修复**：删除重复的 dep。

### 5. circular dependency

```
ERROR: cycle in dependency graph
```
**原因**：两个 target 互相依赖。
**修复**：使用前向声明或抽取共享库。

### 6. Could not find cuda.h

```
ERROR: Could not find any cuda.h matching version
```
**原因**：在宿主机上运行 bazel，宿主机没有 CUDA toolkit。
**修复**：必须在 Docker 容器内执行。

### 7. Bazel 挂起不动

**原因**：`stub_source` 指向 `open_source`，Bazel 尝试从 GitHub 下载依赖。
**修复**：
```bash
ln -sfn internal_source stub_source
```

### 8. Stale bazel server

```
ERROR: Failed to connect to bazel server
```
**原因**：残留的 bazel server 进程。
**修复**：
```bash
pkill -f 'bazel.*github-opensource'
```

---

## GPU Lock 机制

对于 GPU 密集型测试（如 smoke test），使用 `gpu_lock` 避免多个测试同时占用 GPU 导致 OOM：

```bash
--run_under=//rtp_llm/test/utils:gpu_lock
```

`gpu_lock` 是一个 wrapper 脚本，确保同一时间只有一个测试使用 GPU。在共享开发机上运行 smoke test 时**必须**使用。

---

## 测试类型

| 类型 | 路径 | 说明 |
|------|------|------|
| 单元测试 | `//rtp_llm/test/...` | C++/Python 单元测试 |
| Smoke 测试 | `//internal_source/rtp_llm/test/smoke/...` | 端到端模型推理测试 |
| 多卡测试 | tag: `multi_device` | 需要多 GPU 的分布式测试 |

### 排除特定测试

```bash
--test_tag_filters=-manual,-multi_device,-gpu_only
```

---

## 变更记录

- 2026-03-27: 初始版本，基于 RTP-LLM 项目 .claude/skills/test-execution/SKILL.md 整理
