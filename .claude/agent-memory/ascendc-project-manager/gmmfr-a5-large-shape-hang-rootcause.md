---
name: GMMFR A5 Large Shape Hang Root Cause
description: Root cause analysis of A5 deterministic mode hanging on large shapes (groupM > windowSizeRows)
type: project
---

## Problem
A5 GMMFR deterministic mode hangs on large shapes: groupNum=16, batch=256, topK=16, m=4096, k=2048, n=7168.

## Root Cause
**deterWorkspaceSize hardcoded at 96MB is insufficient for N=7168.**

Calculation chain:
- deterWorkspaceSize_ = 96MB (hardcoded upper limit in tiling)
- windowSizeRows = 96MB / (7168 * sizeof(float)) = 96MB / 28672 = **3512 rows**
- groupM = batch * topK = 256 * 16 = **4096 rows per group**
- **4096 > 3512**: Prerequisite `groupM <= windowSizeRows` is violated

The kernel documentation explicitly states: "Prerequisite: single group M <= windowSizeRows".
When violated, VectorSequentialWrite writes past workspace bounds -> hardware hang on A5.

## Key Files in the Chain
1. Tiling: `grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` (L467-470)
2. Host kernel: `grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` (L119-120)
3. Cycling kernel: `common/cgmct/kernel/kernel_gmm_fr_deterministic.h` (L384 group-boundary check)
4. Epilogue write: `common/cgmct/epilogue/block_epilogue_dequant_sequential_write_deterministic.h` (L587 wsWriteOffset, L295 DataCopyPad)
5. Aggregation: `grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h`

## Fix Direction
- P0: Tiling must validate groupM <= windowSizeRows; dynamically adjust deterWorkspaceSize or reject invalid shapes
- P0: Design group-internal row splitting for cases where max groupM > windowSizeRows
- P1: Add runtime assert at host kernel entry
