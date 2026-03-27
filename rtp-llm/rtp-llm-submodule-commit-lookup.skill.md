# RTP-LLM Submodule Commit 查找规则

## 适用场景

当用户提到某个 commit hash、要求查看 commit 内容、分析 commit diff 时，必须遵循本规则。

## 背景

RTP-LLM 项目使用 git submodule 结构：

- **主仓库**：`/home/moudi.mou/RTP-LLM`（内部仓库 `foundation_models/RTP-LLM`）
- **子模块**：`/home/moudi.mou/RTP-LLM/github-opensource`（开源仓库 `alibaba/rtp-llm`）

主仓库和子模块有各自独立的 git 历史。一个 commit 可能存在于主仓库，也可能存在于子模块中。

## 查找规则

当用户提供一个 commit hash 时，**必须同时在两个目录中查找**：

```bash
# 1. 先在主仓库查找
cd /home/moudi.mou/RTP-LLM && git log <commit_hash> -1 --format="%H %s" 2>/dev/null | cat

# 2. 再在子模块查找
cd /home/moudi.mou/RTP-LLM/github-opensource && git log <commit_hash> -1 --format="%H %s" 2>/dev/null | cat
```

两条命令应**并行执行**，哪个返回结果就在哪个目录下继续操作（git show、git diff 等）。

## 注意事项

- 不要只在一个目录查找失败就报错，必须两个都尝试
- 查看 diff 时使用 `git show <hash> --no-color | cat`，避免分页器阻塞
- 如果 diff 输出被截断，使用 `read_file` 读取保存的临时文件获取完整内容
- 代码文件的实际路径需要加上对应仓库的前缀：
  - 主仓库的文件：`/home/moudi.mou/RTP-LLM/<path>`
  - 子模块的文件：`/home/moudi.mou/RTP-LLM/github-opensource/<path>`
