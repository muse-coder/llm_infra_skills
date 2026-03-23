# LLM Infra Skills & Knowledge Base

跨 session 复用的 AI Agent 知识库。包含 Skill（注入 agent prompt 的操作指令）和知识库（详细参考文档）。

## 目录结构

```
llm_infra_skills/
├── README.md                  # 本文件
├── CONTRIBUTING.md            # Agent 贡献规范（必读）
│
├── megatron/                  # Megatron-LM 相关
│   ├── megatron-qad.skill.md  # QAD 量化蒸馏 Skill
│   └── QAD_Megatron_Complete_Guide.md  # QAD 完整知识库
│
├── vllm/                      # (待添加) vLLM 部署相关
├── trt-llm/                   # (待添加) TensorRT-LLM 相关
└── infra/                     # (待添加) GPU/NCCL/集群调试
```

## 两种文件类型

| 类型 | 命名规则 | 大小限制 | 用途 |
|------|---------|---------|------|
| **Skill** | `*.skill.md` | < 300 行 | 精炼指令，自动注入 agent prompt |
| **知识库** | `*.md`（不带 `.skill`） | 不限 | 详细参考文档，agent 按需 Read |

## 配置方式

在 `~/.config/opencode/opencode.json` 中添加:

```json
"skills": {
    "sources": [
        {
            "path": "/Users/moudi/Desktop/llm_infra/infer/llm_infra_skills",
            "recursive": true,
            "glob": "**/*.skill.md"
        }
    ]
}
```

重启 opencode 后所有 `*.skill.md` 自动被发现。

## 使用方式

```typescript
// 委托任务时加载 skill
task(category="deep", load_skills=["megatron-qad"], prompt="...")

// 或者手动触发
/megatron-qad
```

## 如何贡献

见 [CONTRIBUTING.md](./CONTRIBUTING.md)
