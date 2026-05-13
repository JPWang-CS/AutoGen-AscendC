---
name: InitBuffer AIC Protection Requirement
description: TPipe::InitBuffer must be guarded with ASCEND_IS_AIV on A5/950 and all platforms with separate AIC/AIV cores
type: reference
---

# TPipe::InitBuffer Must Be Protected on AIC Cores

## Rule
All `TPipe::InitBuffer` calls must be wrapped in `if ASCEND_IS_AIV { ... }` or placed inside a function that has an early `if ASCEND_IS_AIC { return; }` guard.

## Why
`TPipe::InitBuffer` allocates UB (Unified Buffer) queue memory. AIC (Cube) cores do not have UB. Executing InitBuffer on AIC cores causes undefined behavior -- typically a segmentation fault because the AIC core has no valid UB address space.

This was the root cause of a segfault in GMMFR deterministic mode on A5 (950). The A3 reference implementation (grouped_matmul_finalize_routing.h InitUbBuffer()) has `if ASCEND_IS_AIC { return; }` at line 181, but the A5 port missed this guard.

## How to Apply
When writing any Kernel code that uses TPipe::InitBuffer (for TQue, TQueBind, TBuf), ensure:
1. The InitBuffer call is inside `if ASCEND_IS_AIV { ... }`, OR
2. The enclosing function has an early `if ASCEND_IS_AIC { return; }`

Check ALL buffer initializations in new code, not just the first one. In the GMMFR case, the queBind InitBuffer was missed even though other buffer init locations were properly guarded.

## Reference
- A3 correct pattern: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing.h` line 179-182
- A5 bug (fixed): `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` line 167-169
- Prologue correct pattern: `gmm/common/cgmct/prologue/block_prologue_finalize_routing.h` line 100-103
