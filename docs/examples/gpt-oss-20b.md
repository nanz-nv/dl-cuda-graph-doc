<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# GPT-OSS 20B: Dropless Small MoE without Paged Stashing

:::{note}

NVIDIA's MLPerf Training v6.0 GPT-OSS 20B implementation uses full-iteration CUDA graphs for token-dropless MoE training. Device-initiated grouped GEMM and sync-free HybridEP dispatch make the expert region capturable. The recipe does not use paged stashing: its saved activations stay in capacity-sized buffers.
:::

## Overview

GPT-OSS 20B is a small mixture-of-experts (MoE) language model, in contrast to the 671B DeepSeek-V3: it has 32 routed experts per layer and top-4 routing, which activates about 3.6B parameters per token. Each token selects its experts at run time, so the number of tokens assigned to each expert changes with every microbatch. Token-dropless training preserves every assignment, which makes per-expert token counts dynamic inside an otherwise fixed-size workload.

Two components make that workload capturable in a single CUDA graph: **device-initiated grouped GEMM** reads per-expert token counts from GPU memory, and **sync-free dispatch** writes tokens into statically sized buffers. Together they keep the dynamic counts on the GPU for the whole forward and backward pass.

This page covers the same dropless MoE problem as the [DeepSeek-V3 671B example](deepseek-v3-paged-stash.md), at small-MoE scale. DeepSeek-V3 adds paged stashing because expert parallelism raises the worst-case number of tokens a GPU can receive, and with it the capacity-sized activations that GPU must hold for backward. GPT-OSS 20B trains with no model parallelism in the configuration described here, so each GPU receives only its own tokens' expert assignments. It needs just the capture-enabling components and a modest capacity factor that covers quantization padding. Skipping paged stashing avoids its performance overhead: the stash and restore copies.

## MLPerf Training GPT-OSS 20B Benchmark

The [NVIDIA MLPerf Training v6.0 implementation](https://github.com/mlcommons/training_results_v6.0/tree/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo) trains the model with Megatron-Bridge, Megatron-Core, PyTorch, and Transformer Engine (TE).

**Model**:

- 24 layers, hidden size 2880, 64 attention heads with 8 key/value groups, sequence length 8192
- 32 routed experts per layer with top-4 routing and an expert FFN size of 2880
- Sliding-window attention (window 127) on alternating layers
- No auxiliary load-balancing loss; the router makes real, unconstrained decisions (`MOE_ROUTER_FORCE_LOAD_BALANCING=False`)

**Training Setup** (the [GB300 2×4 configuration](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh)):

- **Hardware configuration**: 2 GB300 nodes with four GPUs each (8 GPUs)
- **Distributed training**: TP=1, PP=1, CP=1, EP=1, so all 8 GPUs are data parallel
- **Batch**: micro-batch size 3, global batch size 24
- **Mixed precision**: MXFP8
- **Expert execution**: HybridEP dispatch and TE fused grouped MLP operations
- **Activation storage**: Capacity-sized buffers; paged stashing is off (`MOE_PAGED_STASH` defaults to `False`)
- **Quality target**: Log perplexity 3.34 on a C4 validation subset

The submission's other configurations scale this recipe across more GPUs and systems; this page focuses on the GB300 2×4 configuration.

**CUDA Graph Challenge**:

Different tokens choose different experts, so the number of tokens each expert processes is not known until the router runs. An eager implementation can read these counts on the CPU, then size tensors and select GEMM shapes accordingly. A CUDA graph cannot repeat that host decision on every replay: buffer addresses and kernel launch parameters are fixed at capture.

The solution keeps counts on the GPU and allocates buffers for a fixed capacity. See the [DeepSeek-V3 page](deepseek-v3-paged-stash.md#essential-modifications-for-cuda-graph-compatibility) for the detailed mechanisms; the sections below show how this recipe applies them.

## Integration Approach: Full-Iteration Graphing

The implementation captures the forward/backward schedule with Megatron's `FullCudaGraphWrapper`.

- **Implementation**: [NVIDIA GPT-OSS 20B submission](https://github.com/mlcommons/training_results_v6.0/tree/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo)
- **Software environment**: [Submitted Dockerfile](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/Dockerfile), based on PyTorch NGC 26.04, Megatron-Core/Bridge `26.04-alpha.rc2`, TE `v2.15`, and HybridEP from DeepEP, the same stack as the DeepSeek-V3 submission

### Capture Scope

**Captured in the forward/backward graph**:

- The model's forward and backward pass for the microbatch
- MoE routing outputs, HybridEP dispatch, permutation, grouped expert computation, and combine

**Outside that graph**:

- Data loading and preparation of microbatch inputs
- The optimizer update, which this recipe leaves eager (it does not set `OPTIMIZER_CG`)

This differs from the DeepSeek-V3 recipe, which graphs the optimizer update separately. A single graph replay covers the entire forward/backward computation.

### Wrapper Integration and Warmup

The recipe's [`callback_warmup.py`](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/callback_warmup.py) wraps the forward/backward function in `FullCudaGraphWrapper` when `cuda_graph_impl` is `local` and the scope includes `full_iteration`. The wrapper runs a number of eager warmup steps (`CUDA_GRAPH_WARMUP_STEPS`, default `2`) before it captures, so lazy initialization completes outside the graph. The benchmark also runs its own warmup on synthetic data before the timed region. As in the DeepSeek-V3 recipe, `PagedStashRunner` sits around the graph wrapper to check for receive-budget overflow after each forward/backward pass (see section 4 below).

### Configuration and Setup

| Purpose | Setting | Value |
|---------|---------|-------|
| Full forward/backward graph | `CUDA_GRAPH_IMPLEMENTATION` / `CUDA_GRAPH_SCOPE` | `local` / `full_iteration` |
| Graph warmup | `CUDA_GRAPH_WARMUP_STEPS` | `2` (default) |
| Dispatcher | `MOE_TOKEN_DISPATCHER_TYPE` / `MOE_FLEX_DISPATCHER_BACKEND` | `flex` / `hybridep` (defaults) |
| Fused grouped expert path | `USE_TE_OPS` | `True` |
| CuTe DSL fused grouped MLP | `NVTE_CUTEDSL_FUSED_GROUPED_MLP` | `1` |
| Static receive capacity | `MOE_EXPERT_RANK_CAPACITY_FACTOR` | `2` (default from `config_common_cg.sh`) |
| Paged stashing | `MOE_PAGED_STASH` | `False` (default) |

The configuration scripts [`config_common_cg.sh`](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/config_common_cg.sh) and the [model YAML](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/conf/gpt_oss.yaml) define these values, and [`pretrain.py`](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/pretrain.py) maps them into the Megatron configuration.

To reproduce, follow the submission [README](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/README.md) to build the container and prepare the data (the dataset is shared with the Llama 3.1 8B benchmark). From the implementation directory:

```bash
source config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh
# CONT, DATADIR and LOGDIR must match your setup.
# Add the account/partition arguments required by your Slurm cluster.
sbatch -N "${DGXNNODES}" --time="${WALLTIME}" run.sub
```

## Essential Modifications for CUDA Graph Compatibility

### 1. Device-Initiated Grouped GEMM

Host-initiated grouped GEMM needs CPU-side per-expert token counts to choose GEMM shapes, which blocks capture. Device-initiated grouped GEMM reads the shapes from GPU memory, so one captured launch handles whatever split of tokens the router produced on that replay.

The recipe enables the TE operation-fuser path (`USE_TE_OPS=True`) and the CuTe DSL fused grouped MLP (`NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`). With the operation fuser enabled, the model interleaves the gated-linear-unit weights in blocks of 32 (`moe_mlp_glu_interleave_size=32`) for the fused kernel. See [section 1 of the DeepSeek-V3 page](deepseek-v3-paged-stash.md#1-device-initiated-grouped-gemm) for how shape metadata flows through the expert MLP.

### 2. Sync-Free Dispatch with Static Receive Buffers

The recipe selects the HybridEP backend and sets `MOE_EXPERT_RANK_CAPACITY_FACTOR`. The dispatcher turns that factor into a static budget for the permuted token buffer:

```text
budget = tokens_per_rank × top_k × capacity_factor   (rounded up to the quantization alignment)
```

With a static budget, HybridEP no longer needs the CPU to read the received-token count to size its buffers, and the received-token count can stay on the GPU. Here `tokens_per_rank` is the micro-batch size times the sequence length (3 × 8192 = 24576), so a factor of 1.0 corresponds to 98,304 permuted rows per layer.

### 3. Quantization Padding and the Capacity Factor

MXFP8 grouped GEMM requires each expert's token count to be a multiple of an alignment size. The recipe sets `moe_router_padding_for_quantization=True`, which pads every expert's token segment up to that multiple. Padding adds rows beyond `tokens_per_rank × top_k`, so a capacity factor of exactly 1.0 can overflow even without any routing imbalance. The factor must leave at least enough headroom for the padding.

Once that minimum is met, any larger value is extra headroom that costs memory. This differs from the DeepSeek-V3 recipe at EP=32, where the factor also has to absorb imbalance across expert ranks.

### 4. Overflow Detection with `PagedStashRunner`

This recipe uses `PagedStashRunner` even though paged stashing is off. Megatron-Bridge's training loop wraps the forward/backward function in it whenever `moe_expert_rank_capacity_factor` is set, and the runner's job here is to detect receive-budget overflow, including overflow caused by quantization padding. If a step exceeds the receive budget, the runner discards the step's gradients and reruns the same microbatches eagerly without the static limit, preserving dropless semantics. The [DeepSeek-V3 page](deepseek-v3-paged-stash.md#5-overflow-recovery-before-the-optimizer-update) describes this recovery in detail.

The recipe's capacity factors sit above the padding requirement, so overflow is not expected in normal training. The check still runs after each forward/backward pass, and the eager rerun covers the case where padding pushes a step over its budget.

## Performance Impact

The table compares three configurations on GB300 with 2 nodes × 4 GPUs (global batch size 24). TFLOPS is per GPU, and speedup is relative to the slowest configuration:

| Configuration | TFLOPS | Speedup |
|---------------|--------|---------|
| No CUDA graph, no CuTe DSL | 612 | 1.00× |
| No CUDA graph, CuTe DSL device-initiated grouped GEMM | 718 | 1.17× |
| **Full-iteration CUDA graph, CuTe DSL device-initiated grouped GEMM** | **1180** | **1.93×** |

The CuTe DSL device-initiated grouped GEMM improves eager execution by 1.17×. Capturing the forward/backward pass in a full-iteration graph then adds a further 1.64×, which brings the combined speedup to 1.93×.

## Key Lessons

1. **Device-initiated grouped GEMM removes host shape queries.** Expert kernels read GPU token counts, so one captured launch handles any split of tokens across experts.
2. **Dispatch must be sync-free as well.** A static receive budget lets HybridEP feed the captured grouped MLP without a host read of the received-token count.
3. **Quantization padding sets the floor for the capacity factor.** Padding adds rows beyond `tokens × top_k`, so the factor must exceed 1.0 by enough to cover it; larger values only add memory headroom.
4. **Paged stashing is a memory optimization, not a prerequisite.** With no pipeline parallelism and no expert parallelism, capacity-sized saved activations fit, and the capacity factor controls how much memory they use.
5. **Both pieces contribute.** In this recipe, device-initiated grouped GEMM gives 1.17× and full-iteration graphing adds 1.64× on top, for 1.93× overall.

## References

**MLPerf Training Resources**:

- [NVIDIA GPT-OSS 20B v6.0 implementation](https://github.com/mlcommons/training_results_v6.0/tree/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo) - Submitted configuration, launcher, and environment
- [GB300 2×4 configuration](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/config_GB300_2x4x3xtp1pp1cp1ep1_mxfp8.sh) - The configuration this page describes
- [Submission README](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/gpt_oss_20b/implementations/nemo/README.md) - Container, data, and launch instructions

**Design and Implementation**:

- [Megatron Core paper, section 4.3.7](https://arxiv.org/html/2603.07685v1#S4.SS3.SSS7): device-initiated kernels and paged stashing.
- [Megatron-LM HybridEP token dispatcher](https://github.com/NVIDIA/Megatron-LM/blob/6e7e6ed435e41af04fc41e7007ef862049153fdc/megatron/core/transformer/moe/token_dispatcher.py): static permuted-token budget and overflow detection.

## What's Next?

**Explore more CUDA graph examples**:

- **[DeepSeek-V3 671B](deepseek-v3-paged-stash.md)** - Add paged stashing and overflow recovery for expert-parallel, pipelined dropless MoE
- **[Llama 3.1 405B](llama-31-405b.md)** - Full-iteration graphing for a dense transformer
- **[GPT-3 175B](gpt-3-175b.md)** - Per-layer graphing with pipeline parallelism and FP8

**Continue learning**:

- **[Writing Sync-Free Code](../torch-cuda-graph/sync-free-code.md)** - Keep shape metadata and execution decisions on the GPU
- **[Memory Issues](../troubleshooting/memory-issues.md)** - Diagnose graph memory pools, fragmentation, and buffer lifetimes
- **[Partial Graphing for Dropless MoE](../torch-cuda-graph/handling-dynamic-patterns.md#example-2-moe-models-with-token-dropless)** - Graph surrounding static layers when the expert implementation requires host-side sizing
- **[Transformer Engine and Megatron-LM CUDA Graphs](../torch-cuda-graph/te-megatron-cuda-graphs.md)** - Framework capture and replay infrastructure
- **[Troubleshooting](../troubleshooting/introduction.md)** - Debug capture failures, memory issues, and numerical errors
