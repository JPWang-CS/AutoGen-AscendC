---
name: Project Structure
description: Repo layout, operator directories, and platform architecture mappings for the AscendC workspace
type: reference
---

## Top-Level Layout

```
m:\Desktop\tmp\AgentTest\AscendC\
  project/              -- Active development projects (A5 migration work)
  ops-transformer_AI/   -- Reference operator codebase (git submodule, large repo)
  template/             -- Operator templates (aiv_only, aic_only, aiv_aic_mixed)
  skills/               -- Knowledge base markdown files for AscendC development
  .claude/              -- Agent configuration and memory
```

## ops-transformer_AI Operator Modules

| Module | Description | Key Operators |
|---|---|---|
| attention/ | Attention operators | attention_pioneer (FlashAttention), attention_update, fused_causal_conv1d, gather_pa_kv_cache, scatter_pa_cache/kv_cache |
| gmm/ | Grouped MatMul operators | grouped_matmul, grouped_matmul_finalize_routing (GMMFR), grouped_matmul_add, grouped_matmul_swiglu_quant, quant_grouped_matmul_dequant |
| ffn/ | Feed-Forward Network | ffn, ffn_worker_batching, ffn_worker_scheduler |
| moe/ | Mixture of Experts | moe_compute_expert_tokens, moe_finalize_routing_v2, moe_gating_top_k, moe_init_routing, moe_re_routing, moe_inplace_index_add, moe_masked_scatter |
| mhc/ | Multi-Head Cross-attention | mhc_pre, mhc_post, mhc_post_backward, mhc_sinkhorn |
| mc2/ | Multi-Chip Communication | matmul_all_reduce, all_gather_matmul_v2, allto_all_matmul, quant_batch_matmul_v3, moe_distribute_combine |
| posembedding/ | Position Embedding | apply_rotary_pos_emb, dequant_rope_quant_kvcache, interleave_rope, kv_rms_norm_rope_cache, norm_rope_concat |
| common/ | Shared utilities | -- |
| cmake/ | Build scripts | -- |
| scripts/ | Dev/test scripts | -- |

## Standard Operator File Structure

Each operator follows this convention:
```
op_name/
  op_host/
    op_name_def.cpp            -- Operator definition (inputs/outputs/attributes)
    op_name_infershape.cpp     -- Output shape inference
    op_name_tiling.h/.cpp      -- Tiling calculation (arch32 for A2/A3)
    op_tiling/arch35/          -- A5-specific tiling (optional)
    op_api/                    -- aclnn C API wrapper
    config/
      ascend910b/              -- A2 (910B) config
      ascend910_93/            -- A3 (910C) config
      ascend950/               -- A5 (950) config (optional)
  op_kernel/
    op_name.cpp/.h             -- Kernel implementation (arch32, used by A2/A3)
    op_name_apt.cpp            -- APT entry point (arch35, used by A5)
    arch35/                    -- A5-specific kernel headers (optional)
  op_graph/                    -- Graph IR definition
  examples/                    -- Usage examples
  tests/                       -- Unit and integration tests
  docs/                        -- API documentation
```

## Template Types

- `aiv_only/` -- Vector-only operators (AIV)
- `aic_only/` -- Cube-only operators (AIC)
- `aiv_aic_mixed/` -- Mixed Cube+Vector operators (most complex)

## Platform Architecture Mapping

| Platform | Chip | Architecture Dir | Config Dir | Kernel Entry |
|---|---|---|---|---|
| A2 | 910B | arch32 (default) | ascend910b | op_name.cpp |
| A3 | 910C | arch32 (default) | ascend910_93 | op_name.cpp |
| A5 | 950 | arch35 | ascend950 | op_name_apt.cpp |

## Skills Knowledge Base

Located in `skills/` directory:
- ascendc_programming_model.md
- ascendc_platform_differences.md
- ascendc_operator_structure.md
- ascendc_api_guide.md
- gmmfr_deterministic_analysis.md
