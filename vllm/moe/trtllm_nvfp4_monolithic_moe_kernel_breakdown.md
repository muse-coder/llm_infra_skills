# TrtLlmNvFp4ExpertsMonolithic — NvFP4 MoE 全链路分析

> 代码版本: vLLM main + FlashInfer main (2026-05)

---

## 1. 什么是 Monolithic

"Monolithic" 是 **vLLM 侧的接口模式**：一次调用接收原始 `router_logits`，内部完成 routing → FC1 → activation → FC2 → finalize，返回最终输出。vLLM 不需要在外部执行 routing / permute / finalize。

**不等于单个 CUDA kernel。** FlashInfer C++ 内部仍然是 `Routing::Runner.run()` + `MoE::Runner.run()` 的多阶段 pipeline，至少两次 kernel launch。

与之对应的 **Modular** 接口接收已完成 routing 的 `topk_ids + topk_weights`，仅执行 expert 计算。用于需要外部控制 routing 的场景（all2all、EPLB）。

两者共享同一套 C++ launcher/runner，GEMM kernel 相同。

---

## 2. Backend 选择

`select_nvfp4_moe_backend` 在 `moe_backend="auto"` 时按以下顺序尝试，第一个 `is_supported_config=True` 的即选中：

```
FLASHINFER_TRTLLM → FLASHINFER_CUTEDSL → FLASHINFER_CUTEDSL_BATCHED
→ FLASHINFER_CUTLASS → VLLM_CUTLASS → MARLIN → EMULATION
```

`FLASHINFER_TRTLLM` 排首位时，`backend_to_kernel_cls` 返回 `[Monolithic, Modular]`，优先 Monolithic。

### 设备兼容性

| Backend | vLLM `_supports_current_device` | SM120 命中? |
|---|---|---|
| FLASHINFER_TRTLLM | `family(100)` only | **No** |
| FLASHINFER_CUTEDSL | `family(100)` | **No** |
| FLASHINFER_CUTLASS | `family(100/110/120)` + `cc(90)` | **Yes** ← SM120 默认 |
| VLLM_CUTLASS | `family(100/110/120)` | Yes（被上面抢先） |

> C++ 侧 `init_common` 检查 `major == 10 || major == 12`，允许 SM12x。但 vLLM Python 侧只放行 `family(100)`，SM120 被跳过。

### Monolithic 限制条件

以下情况 Monolithic 不可用，自动降级 Modular：
- `use_all2all_kernels = True`（EP with all2all）
- `enable_eplb = True`（Expert Load Balancing）
- 不支持 chunking、expert_map

---

## 3. 调用链

```
MoERunner._apply_quant_method()
  │  is_monolithic=True → 跳过外部 routing
  ▼
FusedMoEKernelMonolithicImpl.apply()
  ├─ prepare_finalize.prepare()     → NvFP4 量化: a1q [T, H/2] uint8 + scale [T, H/16] fp8
  ├─ TrtLlmNvFp4ExpertsMonolithic.apply()
  │     └─ flashinfer.fused_moe.trtllm_fp4_block_scale_moe(...)
  │           ├─ AutoTuner.choose_one() → (tile_N, config)
  │           └─ TVM-FFI → C++ trtllm_fp4_block_scale_moe()
  │                 ├─ FP4BlockScaleLauncher 构建 & 选择 tile_N
  │                 └─ launcher->run()
  │                       ├─ Routing::Runner.run()     ← routing GPU kernel
  │                       └─ MoE::Runner.run()         ← GEMM GPU kernels (TRT-LLM cubins)
  └─ prepare_finalize.finalize()    → 直接返回
```

Modular 路径差异：vLLM 先 `select_experts()` 得到 topk，打包为 `(expert_id << 16) | weight_bf16.view(int16)` 的 int32，调用 `trtllm_fp4_block_scale_routed_moe()`。

---

## 4. 数据格式

### 权重（概念 shape，预处理前）

以 DeepSeek-V3 `H=7168, inter=2048, E=256, top_k=8` 为例：

| 张量 | Shape | Dtype |
|---|---|---|
| w1 (FC1 gate_up) | `[E, 2*inter, H/2]` | uint8 (FP4 packed) |
| w1_scale | `[E, 2*inter, H/16]` | fp8_e4m3 |
| w2 (FC2 down) | `[E, H, inter/2]` | uint8 |
| w2_scale | `[E, H, inter/16]` | fp8_e4m3 |

> 实际 kernel 看到的布局不同。vLLM `prepare_nvfp4_moe_layer_for_fi_or_cutlass(FLASHINFER_TRTLLM)` 会做 256 对齐 padding、w1/w3 reorder、FP4 shuffle + block_scale_interleave。

### Scale 融合

```python
# process_weights_after_loading:
w13_weight_scale_2 *= w13_input_scale       # 融合 activation scale 到 weight scale
w2_weight_scale_2  *= w2_input_scale

# g1_scale_c 计算:
g1_scale_c = g1_alphas * a2_gscale          # gated (SwiGLU)
g1_scale_c = a2_gscale.clone()              # non-gated (ReLU²)
```

传给 FlashInfer：`output1_scale_scalar` = g1_scale_c，`output1_scale_gate_scalar` = g1_alphas，`output2_scale_scalar` = g2_alphas。

`g1_scale_c` 注册为 layer parameter，EPLB 重排时会一起迁移。

---

## 5. C++ 执行流程

`FP4BlockScaleLauncher::run()` 内部：

1. **check_routing** — 校验 logits/bias shape、top_k 范围
2. **prepare_routing** — 分配 workspace（permutation indices、histogram、CTA schedule 等）
3. **Routing::Runner.run()** — GPU routing kernel，按 `RoutingMethodType` 执行路由算法
4. **check_moe** — 校验权重 dtype (uint8)、scale dtype (fp8_e4m3)
5. **prepare_moe** — 创建 `MoE::Runner`，选择 tactic，分配 gemm1/gemm2 中间 buffer
6. **MoE::Runner.run()** — TRT-LLM Gen cubins 执行 FC1 → Activation → FC2 → Finalize

### Tile 自动调优

- 支持 tile: BF16 `{8,16,32,64}`，FP4/FP8 `{8,16,32,64,128,256}`
- 启发式: `tile_N = nextPowerOfTwo(num_tokens * top_k / num_local_experts)`
- Tactic = `[tile_N, config_index]`，AutoTuner 首次 profiling 后缓存

---

## 6. Routing 支持

`_supports_routing_method()` 显式列出：

| 方法 | 值 | 支持 |
|---|---|---|
| Default (Softmax→TopK) | 0 | **No** |
| Renormalize | 1 | Yes |
| DeepSeekV3 | 2 | Yes |
| Llama4 | 3 | Yes |
| RenormalizeNaive | 4 | Yes |
| TopK | 5 | **No** |
| SigmoidRenorm | 6 | Yes |
| MiniMax2 | 7 | Yes |
| Simulated | 102 | Yes |

不支持的 routing method 会导致 `is_supported_config` 返回 False，自动降级到下一个 backend。

激活函数仅支持 `SILU (Swiglu)` 和 `RELU2_NO_MUL (Relu2)`。

---

## 7. 与 FLASHINFER_CUTLASS 的对比

| 维度 | FLASHINFER_TRTLLM | FLASHINFER_CUTLASS |
|---|---|---|
| GEMM kernel | TRT-LLM Gen precompiled cubins | CUTLASS 3.x warp-specialized MMA |
| 接口模式 | Monolithic（router_logits 输入） | Modular（topk_ids + topk_weights 输入） |
| Routing | C++ 内部 `Routing::Runner` | 外部 Python 侧 `select_experts` |
| 中间 buffer | launcher 内部分配 | vLLM 分配 workspace |
| Weight 预处理 | shuffle + block_scale_interleave | swizzle |
| vLLM 设备检查 | `family(100)` only | `family(100/110/120)` + `cc(90)` |

**性能差异不大。** 两者的核心计算（FP4 GEMM）都是高度优化的 Tensor Core kernel。Monolithic 的主要收益是减少 Python 层调度和中间 tensor 分配的开销（微秒级），不影响 GEMM 本身的计算/带宽瓶颈。Decode 受 HBM 带宽限制，Prefill 受计算吞吐限制——两个 backend 面对的瓶颈相同。

---

## 8. 关键源文件

| 文件 | 角色 |
|---|---|
| `vllm/.../fused_moe/experts/trtllm_nvfp4_moe.py` | Monolithic/Modular 定义、scale 融合 |
| `vllm/.../fused_moe/oracle/nvfp4.py` | Backend 选择、weight 预处理 |
| `vllm/.../fused_moe/modular_kernel.py` | MonolithicImpl 框架 |
| `flashinfer/fused_moe/core.py` | Python API、AutoTuner |
| `csrc/trtllm_fused_moe_kernel_launcher.cu` | C++ launcher、FP4BlockScaleLauncher |
