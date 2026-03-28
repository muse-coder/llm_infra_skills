# RTP-LLM 本地 Smoke Test 复现指南（ROCm / AMD GPU）

> 适用环境：AMD Instinct MI308X，Docker 容器内，rtp_llm 已通过 pip 安装到系统 Python

---

## 一、测试方式速查

### 方式 1：单次 curl 测试（LLM 推理，`/generate` 端点）

适合快速验证服务是否正常，使用原生 `/generate` 端点（非 OpenAI 格式）：

```bash
curl -XPOST http://localhost:8066 \
  -d '{"prompt": "以下是一段关于人工智能的介绍：人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，AI的全称是什么？", "generate_config": {"max_new_tokens": 10, "do_sample": false}}'
```

> **端口说明**：`/generate` 端点默认在 `8066`；OpenAI 兼容端点（`/v1/chat/completions`）在 `8088`。

### 方式 2：OpenAI 格式 curl 测试

```bash
curl -s http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_32b","messages":[{"role":"user","content":"AI的全称是什么？"}],"max_tokens":50}'
```

### 方式 3：多 Batch 并发测试（`batch_test.py`）

使用本目录的 `batch_test.py`，支持多批次并发请求，并利用 **Prefix Cache** 优化（所有请求共享长公共前缀）。

**依赖安装**：
```bash
pip install aiohttp
```

**基本用法**：
```bash
python3 batch_test.py --url http://localhost:8066
```

**完整参数**：
```bash
python3 batch_test.py \
  --url http://localhost:8066 \   # 服务地址（/generate 端点）
  --batch-size 4 \                # 每批请求数量
  --num-batches 3 \               # 批次数量（共发 batch_size × num_batches 个请求）
  --concurrency 4                 # 最大并发数
```

**脚本功能**：
- 内置 12 个不同问题，共享约 1500 字符的公共前缀，用于触发 Prefix Cache
- 多批次之间有 0.5s 间隔，模拟真实流量
- 输出每个请求的耗时、回答，以及整体吞吐量统计

---

## 二、环境前提

### 必须在 Docker 容器内执行

所有启动命令必须在容器内运行，宿主机缺少必要的 GPU 驱动和库。

```bash
# 确认当前在容器内
hostname   # 应显示容器名，如 e01-cn-fsw49jf8p02
```

### 关键路径

| 组件 | 路径 |
|------|------|
| Python | `/opt/conda310/bin/python3.10` |
| rtp_llm 包 | `/usr/local/lib/python3.10/site-packages/rtp_llm/` |
| C++ 共享库 | `/usr/local/lib/python3.10/site-packages/rtp_llm/libs/` |
| PyTorch lib | `/opt/conda310/lib/python3.10/site-packages/torch/lib/` |

### 必须设置的环境变量

```bash
export PYTHONPATH=/usr/local/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/opt/conda310/lib/python3.10/site-packages/torch/lib:/usr/local/lib/python3.10/site-packages/rtp_llm/libs:${LD_LIBRARY_PATH}
```

> **原因**：`libth_transformer` 等 C++ 扩展依赖 torch/lib 中的 `.so`，不设置会导致 `ImportError`。

---

## 二、通用启动模板

```bash
REUSE_CACHE=1 \
FT_SERVER_TEST=1 \
PYTHONPATH=/usr/local/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/opt/conda310/lib/python3.10/site-packages/torch/lib:/usr/local/lib/python3.10/site-packages/rtp_llm/libs:${LD_LIBRARY_PATH} \
HIP_VISIBLE_DEVICES=<卡号> \
/opt/conda310/bin/python3.10 -m rtp_llm.start_server \
  --model_type <model_type> \
  --tokenizer_path <模型路径> \
  --checkpoint_path <模型路径> \
  [其他参数...] \
  --start_port 8088
```

### 后台运行（推荐）

```bash
nohup bash -c '<上述命令>' > /tmp/server.log 2>&1 &
echo "PID: $!"
```

### 监控启动进度

```bash
tail -f /tmp/server.log
# 看到以下日志说明启动成功：
# Application startup complete.
# Uvicorn running on socket ('0.0.0.0', 8088)
```

---

## 三、Smoke Test 复现方式

Smoke test 有两种复现方式，按需选择：

| 方式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **方式一：bazelisk test** | 完整复现 CI 流程，验证 golden 对比 | 自动启动/关闭服务，结果与 CI 一致 | 需要 bazel 环境，编译耗时长 |
| **方式二：本地启动服务** | 快速调试、验证推理是否正常 | 启动快，灵活，可交互调试 | 不做 golden 对比，需手动发请求 |

---

### 方式一：bazelisk test（完整 CI 复现）

在容器内的 `github-opensource` 目录下执行，会自动启动服务、发送测试请求、对比 golden 文件。

**前置检查**（确保 `stub_source -> internal_source`）：
```bash
ls -la /home/moudi.mou/RTP-LLM/github-opensource/stub_source
# 应显示 stub_source -> internal_source，否则执行：
# ln -sfn internal_source /home/moudi.mou/RTP-LLM/github-opensource/stub_source
```

**Roberta Embedding（bge-m3）**：
```bash
cd /home/moudi.mou/RTP-LLM/github-opensource && \
bazelisk --output_user_root=/home/moudi.mou/.cache/bazel_rocm_cache test \
  //rtp_llm/test/smoke:roberta_ptpc \
  --config=rocm \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --test_timeout=3000 --jobs=200 \
  > /home/moudi.mou/RTP-LLM/build_logs/roberta_smoke.log 2>&1
```

**Qwen3-32B FP8**：
```bash
cd /home/moudi.mou/RTP-LLM/github-opensource && \
bazelisk --output_user_root=/home/moudi.mou/.cache/bazel_rocm_cache test \
  //rtp_llm/test/smoke:qwen3_32b_ptpc \
  --config=rocm \
  --config=daily_aone_bazel_cache \
  --remote_header=x-aone-bazel-api-key=ai-infra-cicd \
  --test_timeout=3000 --jobs=200 \
  > /home/moudi.mou/RTP-LLM/build_logs/qwen3_smoke.log 2>&1
```

**监控进度**：
```bash
tail -f /home/moudi.mou/RTP-LLM/build_logs/qwen3_smoke.log
# 成功：Target //rtp_llm/test/smoke:qwen3_32b_ptpc  PASSED
# 失败：FAILED，查看 test.log 定位原因
```

---

### 方式二：本地手动启动服务

手动启动服务后，用 curl 或 `batch_test.py` 发送请求验证推理结果。详见下方各 Case 的具体命令。

---

## 四、Smoke Test 案例（方式二：本地启动服务）

### Case 1：Roberta Embedding（bge-m3）

**模型类型**：Embedding，**端点**：`/v1/embeddings`

#### 启动命令

```bash
nohup bash -c '
REUSE_CACHE=1 \
FT_SERVER_TEST=1 \
PYTHONPATH=/usr/local/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/opt/conda310/lib/python3.10/site-packages/torch/lib:/usr/local/lib/python3.10/site-packages/rtp_llm/libs:${LD_LIBRARY_PATH} \
HIP_VISIBLE_DEVICES=0 \
/opt/conda310/bin/python3.10 -m rtp_llm.start_server \
  --model_type roberta \
  --tokenizer_path /home/moudi.mou/models/bge-m3 \
  --checkpoint_path /home/moudi.mou/models/bge-m3 \
  --seq_size_per_block 16 \
  --use_aiter_pa 1 \
  --use_asm_pa 1 \
  --load_python_model 1 \
  --act_type FP16 \
  --start_port 8088
' > /tmp/roberta_server.log 2>&1 &
```

#### 测试命令

```bash
curl -s http://localhost:8088/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-m3","input":"Hello world"}'
```

#### 预期响应

```json
{"object":"list","data":[{"object":"embedding","index":0,"embedding":[...]}],"model":"","usage":{"prompt_tokens":3,"total_tokens":3}}
```

---

### Case 2：Qwen3-32B FP8（单卡）

**模型类型**：LLM，**端点**：`/v1/chat/completions`，**加载时间**：约 1~2 分钟

#### 启动命令

```bash
nohup bash -c '
REUSE_CACHE=1 \
FT_SERVER_TEST=1 \
PYTHONPATH=/usr/local/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/opt/conda310/lib/python3.10/site-packages/torch/lib:/usr/local/lib/python3.10/site-packages/rtp_llm/libs:${LD_LIBRARY_PATH} \
HIP_VISIBLE_DEVICES=0 \
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
/opt/conda310/bin/python3.10 -m rtp_llm.start_server \
  --model_type qwen_3 \
  --tokenizer_path /home/moudi.mou/models/Qwen/Qwen3-32B-FP8-Dynamic/ \
  --checkpoint_path /home/moudi.mou/models/Qwen/Qwen3-32B-FP8-Dynamic/ \
  --quantization FP8_PER_CHANNEL_COMPRESSED \
  --use_swizzleA 1 \
  --use_asm_pa 1 \
  --disable_flash_infer 1 \
  --warm_up 0 \
  --use_aiter_pa 1 \
  --seq_size_per_block 16 \
  --load_python_model 1 \
  --act_type BF16 \
  --test_block_num 1000 \
  --reserver_runtime_mem_mb 10000 \
  --start_port 8088
' > /tmp/qwen3_server.log 2>&1 &
```

#### 测试命令

```bash
curl -s http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3_32b","messages":[{"role":"user","content":"Hello, what is 2+3?"}],"max_tokens":50,"temperature":0.1}'
```

#### 预期响应（含 thinking）

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "<think>\nOkay, the user is asking what 2 plus 3 is..."
    },
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 17, "total_tokens": 67, "completion_tokens": 50},
  "aux_info": {
    "first_token_cost_time": 2058.381,
    "cost_time": 4223.907
  }
}
```

#### 实测性能（MI308X 单卡）

| 指标 | 数值 |
|------|------|
| 首 token 延迟 | ~2058 ms |
| 总耗时（50 tokens） | ~4224 ms |
| Decode 速度 | ~11.8 tok/s |

---

## 四、常见问题与解决

### OOM：`HIP out of memory`

**现象**：
```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 250.00 MiB.
GPU 0 has a total capacity of 191.98 GiB of which 0 bytes is free.
```

**原因**：`--reserver_runtime_mem_mb` 设置过大。CI smoke test 配置（如 `70000`）是为多卡 TP 模式设计的，单卡运行时该值会直接占用单卡显存导致 OOM。

**解决**：单卡运行时将 `--reserver_runtime_mem_mb` 减小到 `10000` 或更低：
```bash
--reserver_runtime_mem_mb 10000
```

### ImportError：找不到 libth_transformer

**现象**：
```
ImportError: libth_transformer.so: cannot open shared object file
```

**解决**：确保 `LD_LIBRARY_PATH` 包含 torch/lib 和 rtp_llm/libs：
```bash
export LD_LIBRARY_PATH=/opt/conda310/lib/python3.10/site-packages/torch/lib:/usr/local/lib/python3.10/site-packages/rtp_llm/libs:${LD_LIBRARY_PATH}
```

### Health check 失败

**现象**：
```
ERROR backend_server process manager is not available
ERROR Health checks failed
```

**排查**：查看 backend 进程的具体报错：
```bash
grep "ERROR\|Exception\|Traceback" /tmp/server.log | tail -30
```

通常是 OOM 或模型路径不存在导致 backend 进程提前退出。

### 服务启动后 curl 无响应

**原因**：模型还在加载中（32B 模型约需 1~2 分钟）。

**判断方法**：
```bash
# 看到这行才说明真正就绪
grep "Application startup complete" /tmp/server.log
```

---

## 五、清理服务

```bash
pkill -9 -f "rtp_llm"
```

---

## 六、关键参数说明

| 参数 | 说明 |
|------|------|
| `REUSE_CACHE=1` | 复用已有的 KV cache，加速重启 |
| `FT_SERVER_TEST=1` | 测试模式标志，跳过部分生产检查 |
| `HIP_VISIBLE_DEVICES` | 指定使用哪张 GPU（单卡填单个数字） |
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | 减少显存碎片，FP8 大模型推荐开启 |
| `--load_python_model 1` | 使用 Python 侧模型加载（ROCm 环境必须） |
| `--use_aiter_pa 1` | 启用 aiter paged attention（ROCm 优化） |
| `--use_asm_pa 1` | 启用汇编级 paged attention kernel |
| `--disable_flash_infer 1` | 禁用 flash_infer（ROCm 环境不支持） |
| `--reserver_runtime_mem_mb` | 预留给 KV cache 以外的运行时显存（MB），单卡建议 10000 |
| `--test_block_num` | KV cache block 数量上限 |
| `--warm_up 0` | 跳过 warmup，加快启动速度 |
