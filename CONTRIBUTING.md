# 贡献规范

本文档说明如何向 `llm_infra_skills` 添加新的 Skill 或知识库文件。适用于 AI Agent 和人工贡献者。

---

## 文件类型说明

| 类型 | 后缀 | 大小限制 | 用途 |
|------|------|---------|------|
| **Skill 文件** | `*.skill.md` | < 300 行 | 精炼操作指令，由 opencode 自动注入 agent prompt |
| **知识库文件** | `*.md`（不带 `.skill`） | 不限 | 详细参考文档，agent 通过 `Read` 工具按需读取 |

**选择原则**：
- 需要 agent 每次任务都自动获得的核心知识 → Skill 文件
- 详细的参数说明、完整代码示例、排错手册 → 知识库文件
- 两者可以配合使用：Skill 文件引用知识库文件路径，agent 按需 Read

---

## 目录结构规范

每个领域一个目录，目录名使用小写短横线：

```
llm_infra_skills/
├── megatron/      # Megatron-LM 训练、量化、蒸馏
├── vllm/          # vLLM 推理部署
├── trt-llm/       # TensorRT-LLM
├── infra/         # GPU 调试、NCCL、集群运维
└── ...            # 新领域按需添加
```

新增领域时直接创建对应目录，无需修改配置。

---

## 命名规则

```
# Skill 文件
{领域}-{主题}.skill.md
示例：megatron-qad.skill.md, vllm-serving.skill.md

# 知识库文件
{Topic}_{类型}.md  或  描述性名称
示例：QAD_Megatron_Complete_Guide.md, NCCL_Troubleshooting.md
```

---

## Skill 文件规范

### YAML Frontmatter（必须）

```yaml
---
name: {skill-name}          # 与文件名一致（不含 .skill.md）
description: "..."          # 关键：opencode 用此字段决定何时推荐该 skill
                            # 应包含：适用场景、关键词、触发条件
---
```

`description` 写作要点：
- 以 "MUST USE for..." 开头，明确触发条件
- 列出关键技术词（模型名、框架名、操作类型）
- 说明 skill 覆盖的范围

### 内容结构（推荐）

```markdown
---
name: {name}
description: "..."
---

# {领域} 完整知识

> 角色定义：你是 X 专家，在处理 Y 任务时应用此知识。

---

## 核心概念
（简明扼要，不超过 20 行）

## 完整流程
（分步骤，每步包含可直接运行的命令）

## 关键约束
（CRITICAL 级别的限制，必须放在显眼位置）

## 常用参数
（表格或代码块，直接可用）

## 排错指南
（常见错误 → 原因 → 解决方案）

## 关键文件索引
（源码路径，方便 agent 定位）
```

### 质量标准

- 所有命令必须是**实际可运行**的，不能是伪代码
- 关键约束必须注明**根本原因**（哪个文件哪一行）
- 必须包含**排错章节**，覆盖已知的坑
- 内容来源于**实际源码阅读**，不能基于猜测
- 控制在 **300 行以内**（超出部分移入知识库文件）

参考示例：`megatron/megatron-qad.skill.md`

---

## 知识库文件规范

知识库文件无需 YAML frontmatter，直接用 Markdown 写：

```markdown
# {主题} 完整指南

## 概述
（背景和适用范围）

## 详细说明
（可以很长，包含完整代码、参数表、架构图等）

## 参数参考
（完整参数列表，带说明和默认值）

## 已知问题
（详细的问题描述、复现步骤、解决方案）

## 变更记录
（重要发现按时间记录）
```

知识库文件通常由 Skill 文件引用，例如在 Skill 末尾写：

```markdown
## 详细文档
完整参数说明见：`megatron/QAD_Megatron_Complete_Guide.md`
```

---

## 贡献流程

### 新增 Skill

1. 确认目标领域目录存在（不存在则创建）
2. 创建 `{领域}-{主题}.skill.md`，填写 frontmatter 和内容
3. 在 opencode 中测试：`/{skill-name}` 或 `task(load_skills=["{skill-name}"], ...)`
4. 确认 skill 被正确加载后，commit 并 push

```bash
git add megatron/new-skill.skill.md
git commit -m "feat(megatron): add {topic} skill"
git push
```

### 更新已有文件

- 追加新发现时，在文件末尾添加，不要删除已有内容
- 修正错误时，直接修改对应行，commit message 说明原因
- 重大重构需在 commit message 中说明影响范围

### Commit Message 格式

```
feat({领域}): add {topic} skill/guide
fix({领域}): correct {具体错误}
update({领域}): add {新发现} to {文件名}
```

---

## Agent 贡献注意事项

- **不要猜测**：所有技术细节必须来自实际读取源码或运行命令的结果
- **不要重复**：添加前先检查是否已有相关内容
- **追加而非覆盖**：更新知识库时使用 Append，不要覆盖已有内容
- **记录根因**：发现 bug 或限制时，必须定位到具体文件和行号
- **保持简洁**：Skill 文件是精华，不是百科全书
