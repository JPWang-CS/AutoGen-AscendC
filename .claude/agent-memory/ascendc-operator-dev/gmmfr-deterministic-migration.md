---
name: GMMFR Deterministic Migration
description: Active project status for GMMFR deterministic feature migration from A3 to A5 (W8A8/INT8 only)
type: project
---

## Current Status: Sliding Window Implementation Complete

- **Source Platform**: A3 (910C) / arch32
- **Target Platform**: A5 (950) / arch35
- **Scope**: Only W8A8 (INT8 x INT8) PerToken full quantization path
- **Out of Scope**: FP8/FP4/HIFLOAT8, MX format, Weight Quant (pseudo-quantization)

## Problem

GMMFR Finalize Routing uses SetAtomicAdd for multi-core result accumulation. Due to non-deterministic parallel execution order, floating-point addition is non-associative, causing small numerical differences across runs.

## Solution: External Multi-Round Loop + Offset Passing (Plan D)

Sliding window with group-boundary alignment. Workspace reused across rounds. Never degrades.

### Architecture
1. Tiling: compute windowSize = workspaceSize / (N * sizeof(float)), never degrade
2. Outer while loop: group groups into rounds that fit within windowSize
3. Each round: new Cgmct Kernel instance with offset inputs + preOffsetInit
4. Each round: FRDeterministicA5 aggregates workspace -> yGm with globalRowOffset

### Files Modified (5 files)

1. `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` -- added windowSize, totalM
2. `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` -- replaced degradation with windowSize calc
3. `common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` -- added preOffsetInit to GMMTiling
4. `op_kernel/arch35/gmm_fr_deterministic_a5.h` -- added globalRowOffset param
5. `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` -- sliding window multi-round loop

### Key Design Decisions
- GMMTiling preOffsetInit: for cumulative groupListType=0, preOffset_ must be initialized to the accumulated value of skipped groups
- Prologue batch=0 for subsequent rounds: Prologue initOutput zeros 0 rows, effectively a no-op
- Epilogue SequentialWrite writes from workspace row 0 each round (accumulatedGroupOffset_ resets per kernel instance)
- NZ weight offset: CeilDiv(n, 32) * CeilDiv(k, 16) * 512 per group (non-transposed)

## Acceptance Criteria

1. deterministicFlag=1 correctly computed in A5 Tiling
2. Same INT8 input produces bit-wise identical output across 10 runs on A5
3. Max relative error between deterministic and non-deterministic < 1e-3
4. Only W8A8/INT8 PerToken path supports deterministic; other paths unaffected
5. Zero regression for non-deterministic mode
6. A2/A3/A5 CI all pass

## A5 Sync Warning

A5 SyncAll() requires strict 1:1 pairing. If a core skips SyncAll(), it will deadlock. All conditional paths must ensure SyncAll() is called on every core.
