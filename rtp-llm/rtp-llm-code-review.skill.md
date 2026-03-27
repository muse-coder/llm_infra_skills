---
name: rtp-llm-code-review
description: "RTP-LLM Code Review 指南。MUST USE for any task involving RTP-LLM code review, PR review, 代码审查, review 代码, 帮我看看这个改动, 自查代码, commit review. Covers review workflow, checklist, problem classification, and output format for RTP-LLM (LLM inference engine with C++/Python/CUDA)."
---

# RTP-LLM Code Review

> 你是 RTP-LLM 代码审查专家。当用户需要 review 代码改动时应用此知识。

---

## 项目背景

RTP-LLM 是 LLM 推理引擎，核心技术栈：
- **C++/CUDA**：attention kernels, device abstraction, CUDA Graph
- **Python**：模型定义, pybind11 绑定, 服务层
- **Bazel**：构建系统
- **GitHub**：开源仓库 `alibaba/rtp-llm`

---

## Review 流程

### Step 1: 获取变更

```bash
# 查看 submodule 内的实际改动
cd <project_root>/github-opensource
git diff HEAD --stat | cat
git diff HEAD -- <specific_file> | cat

# 查看 PR diff（需要 gh CLI）
gh pr diff <PR_NUMBER> --repo alibaba/rtp-llm | cat
```

### Step 2: 建立变更地图

将变更文件按功能模块分类：

| 模块 | 路径特征 | 优先级 |
|------|---------|--------|
| 核心逻辑 | `cpp/kernels/`, `cpp/devices/`, `models_py/modules/` | 最高 |
| 配置层 | `*Config*`, `pybind`, `server_args` | 高 |
| 接口层 | `bindings/`, `OpDefs.h`, CUDA Graph | 高 |
| 分布式 | TP sync, EP, broadcast | 高 |
| 测试 | `test/` | 中 |
| 格式化 | 纯空格/对齐 | 低 |

### Step 3: 上下文扩展与深度审查

**核心原则：diff 是入口，不是边界。** 必须读取原始上下文才能判断改动是否正确。

按变更类型扩展：
- **改函数实现**：读整个函数 + 调用链 + 跨文件调用者
- **改结构体/接口**：读定义 + 所有构造点 + 消费者
- **改配置/参数**：读声明 + 解析 + 传播路径 + 使用点
- **改 pybind11 绑定**：检查 Python 端调用是否匹配 C++ 签名
- **改 CUDA Graph 路径**：检查 capture/replay 一致性、tensor 生命周期

### Step 4: 问题分级

| 级别 | 定义 | 示例 |
|------|------|------|
| P0 阻塞 | 正确性错误、数据损坏 | 传 None 给不接受 None 的 C++ 绑定 |
| P1 重要 | 破坏性变更、缺测试、性能回退 | 删除公共 API 无兼容处理 |
| P2 建议 | 风格不一致、冗余代码 | 混入无关格式化改动 |
| P3 Nit | 格式化、注释措辞 | import 顺序 |

每个问题必须有证据链：`触发条件 → 风险后果 → 代码位置 → 建议`

---

## Review Checklist

### 通用原则

#### 软件工程
- [ ] SRP：类/模块承担单一变化原因
- [ ] DRY：>5 行重复逻辑需抽取
- [ ] KISS：避免不必要的过度抽象

#### 架构
- [ ] 抽象边界：新概念放在正确的层
- [ ] 依赖方向：无循环依赖或不合理跨层调用
- [ ] 状态完整性：创建/更新/失败/回滚时不变量成立
- [ ] 错误语义：失败策略明确

#### 测试
- [ ] 新功能有对应测试
- [ ] 删除的测试有等价替代
- [ ] 分布式改动有多卡测试

#### 代码质量
- [ ] 无关改动（格式化 + 逻辑）应分离
- [ ] Commit 原子性：单个 commit 自包含
- [ ] Commit message 准确反映改动

### RTP-LLM 领域检查

#### A. 兼容性与配置
- [ ] 新配置项有默认值，不破坏现有部署
- [ ] 环境变量/命令行参数变更向后兼容

#### B. 正确性与逻辑
- [ ] pybind11 绑定类型匹配（Python None vs C++ 引用）
- [ ] CUDA Graph capture/replay tensor 一致性
- [ ] TP/EP 并行下的 broadcast/sync 正确性
- [ ] KV Cache block ID 管理（kernel vs store 两组）

#### C. 性能
- [ ] 热路径无不必要的内存分配
- [ ] 无 per-forward 日志打印
- [ ] Kernel launch 参数合理

#### D. 多平台
- [ ] CUDA/ROCm/PPU 改动是否需要同步
- [ ] 平台特定代码有正确的条件编译

---

## 输出格式

### LGTM 判定

- 存在 P0 或 P1 → **不建议合入**
- P0 和 P1 均为 0 → **LGTM ready to ci**

### 提交 Review 到 GitHub

```bash
cat > /tmp/pr<N>_review_body.md << 'REVIEW_EOF'
🤖 AI Code Review — PR #<N>
<review_content>
REVIEW_EOF

gh pr comment <N> --repo alibaba/rtp-llm --body-file /tmp/pr<N>_review_body.md
```

---

## 详细文档

完整的领域检查模式和已知问题模式见：`rtp-llm/RTP-LLM_Code_Review_Patterns.md`