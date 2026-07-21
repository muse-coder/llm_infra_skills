# FlashAttention-4 / Blackwell 知识库

## 文档索引

- [`FA4_Blackwell_Scheduling_Unified.md`](FA4_Blackwell_Scheduling_Unified.md)：推荐先读；统一整理 varlen、GQA/Pack-GQA、LPT、L2 swizzle、CLC、当前支持状态，并在最后集中比较 1CTA 与 2CTA。
- [`FA4_hd128_1CTA_CLC_LPT.md`](FA4_hd128_1CTA_CLC_LPT.md)：hd128 通用 SM100 forward 的 1CTA、varlen、causal、GQA、LPT、CLC，以及 PR #2218/#2346。
- [`FA4_varlen_GQA_causal_CLC_LPT.md`](FA4_varlen_GQA_causal_CLC_LPT.md)：hd256 2CTA varlen GQA causal CLC 的支持边界、L2 locality 与建议实现路线。
- [`FlashInfer_Blackwell_Decode_Attention.md`](FlashInfer_Blackwell_Decode_Attention.md)：FlashInfer decode attention 实现谱系；重点解释 B200/B300 上的 CuTe DSL GQA 与 TRTLLM-Gen、Flash-Decoding、GQA、KV split、Multi-CTA/CGA、KV layout、page size 和开源边界。
- [`vLLM_Qwen35_PagedKV_NHD_HND_Prefill_Decode.md`](vLLM_Qwen35_PagedKV_NHD_HND_Prefill_Decode.md)：vLLM/Qwen3.5 中 Paged KV、NHD/HND、prefill/decode、prefix cache 与 FlashInfer/TRTLLM-Gen 的系统侧衔接。

阅读时注意区分：

```text
upstream PR 合并时状态
后续 PR/bugfix 后状态
当前 muse branch 状态
agent_space benchmark 实际启用的配置
```
