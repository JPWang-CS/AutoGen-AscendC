---
name: GMMFR Deterministic Migration
description: Active project status for GMMFR deterministic feature migration from A3 to A5 (W8A8/INT8 only)
type: project
---

## Current Status: In Development (Plan A code generated, Plan B designed)

- **Source Platform**: A3 (910C) / arch32
- **Target Platform**: A5 (950) / arch35
- **Scope**: Only W8A8 (INT8 x INT8) PerToken full quantization path
- **Out of Scope**: FP8/FP4/HIFLOAT8, MX format, Weight Quant (pseudo-quantization)
- **Estimated Effort**: ~5 days (2 people parallel)

## Problem

GMMFR Finalize Routing uses SetAtomicAdd for multi-core result accumulation. Due to non-deterministic parallel execution order, floating-point addition is non-associative, causing small numerical differences across runs.

## Solution Strategy

Three-phase "sliding window + delayed aggregation":
1. Cube+Vector compute normally, but write dequantized results to workspace (not yGm)
2. When accumulated rows reach windowSize, trigger FRDeterministic
3. FRDeterministic: SyncAll -> assign row ownership by outRow % coreNumVec -> SetAtomicAdd to yGm -> SyncAll

## Two Plans

### Plan A (Modify existing Epilogue)
- 8 files (6 modified + 1 new + 1 test)
- Adds deterministic branch inside existing BlockEpilogueDequantFinalizeRouting
- Medium regression risk (modifies shared Cgmct code)

### Plan B (New Epilogue, preferred)
- 6 files (4 modified + 2 new)
- Creates new BlockEpilogueDequantOnly Epilogue class (INT8 only)
- Zero regression risk for non-deterministic path (doesn't touch existing Epilogue)
- Key insight: Epilogue is a template parameter in Cgmct, so it can be swapped

## Files to Modify (Plan B)

### New files:
- `op_kernel/arch35/block_epilogue_dequant_only.h` -- New Epilogue for INT8 dequant only
- `op_kernel/arch35/gmm_fr_deterministic_a5.h` -- A5 deterministic aggregation function

### Modified files:
- `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` -- Add deterministicFlag + deterWorkspaceSize
- `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` -- Declare deterministic members
- `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` -- Implement deterministic tiling
- `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` -- Add if/else deterministic branch

## Key Data Types (INT8 path only)

```
Input: x=INT8, weight=INT8(FRACTAL_NZ), scale=FLOAT/BF16, pertoken_scale=FLOAT, bias=BF16
Intermediate: Cube output = INT32 (INT8 x INT8)
Output: y=FP32 (INT32 -> dequant -> FP32)
```

## Acceptance Criteria

1. deterministicFlag=1 correctly computed in A5 Tiling
2. Same INT8 input produces bit-wise identical output across 10 runs on A5
3. Max relative error between deterministic and non-deterministic < 1e-3
4. Only W8A8/INT8 PerToken path supports deterministic; other paths unaffected
5. Zero regression for non-deterministic mode
6. A2/A3/A5 CI all pass

## A5 Sync Warning

A5 SyncAll() requires strict 1:1 pairing. If a core skips SyncAll(), it will deadlock. All conditional paths must ensure SyncAll() is called on every core.
