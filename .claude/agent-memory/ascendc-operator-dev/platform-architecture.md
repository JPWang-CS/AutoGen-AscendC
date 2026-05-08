---
name: Platform Architecture
description: Key architectural differences between A2/A3 and A5 platforms that affect AscendC operator development
type: reference
---

## Architecture Mapping

| Platform | Chip | Arch Tag | Tiling Dir | Kernel Entry | Programming Model |
|---|---|---|---|---|---|
| A2 | 910B | arch32 | (default) | op_name.cpp | Native AscendC API |
| A3 | 910C | arch32 | (default) | op_name.cpp | Native AscendC API (compatible with A2) |
| A5 | 950 | arch35 | arch35/ | op_name_apt.cpp | Cgmct Builder + AscendC API |

## Critical A5 vs A2/A3 Differences

1. **Programming Framework**: A5 uses Cgmct (C++ Gemm Template) Builder pattern; A2/A3 use native AscendC MatmulImpl API
2. **Kernel Entry**: A5 uses `_apt.cpp` files; A2/A3 use `.cpp` files
3. **Core Synchronization**: A5 SyncAll() requires strict 1:1 pairing (deadlock risk otherwise)
4. **L0C Size**: A5 has 256 KB L0C vs A2/A3's 128 KB
5. **Data Types**: A5 has no INT4; uses FP4_E2M1 instead. A5 adds FP8_E4M3FN, FP8_E5M2, HIFLOAT8, FP4_E2M1, FP8_E8M0 (MX scale)
6. **Tiling**: A5 tiling is in `op_tiling/arch35/` subdirectory; A2/A3 tiling is at `op_host/` level

## Code Pattern for Multi-Platform Support

A5 detection in kernel: `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310`
- This guard is used in `_apt.cpp` files (A5 arch35)
- A2/A3 kernel files use different `__CCE_AICORE__` values

## A5 Config Presence

Many operators in ops-transformer_AI have `config/ascend950/` directories with binary JSON and simplified key INI files, indicating A5 support. Operators with ascend950 config include: attention_update, batch_mat_mul_v3, mat_mul_v3, all_gather_matmul_v2, allto_all_matmul, moe_compute_expert_tokens, moe_finalize_routing_v2, moe_gating_top_k, moe_init_routing, apply_rotary_pos_emb, mhc_post_backward, mhc_sinkhorn, and many more.

## Epilogue Template Pattern (Cgmct)

A5 operators using Cgmct can swap Epilogue classes as template parameters:
```cpp
using BlockEpilogueDequant = BlockEpilogueDequantFinalizeRouting<CType, C1Type, ...>;
using GmmKernel = KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue, BlockEpilogueDequant, BlockScheduler>;
```
This enables Plan B style modifications (new Epilogue class without touching existing code).
