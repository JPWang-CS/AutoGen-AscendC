---
name: GMMFR Deterministic Tiling Analysis
description: Detailed analysis of A5 GMMFR deterministic mode Tiling layer for coredump investigation
type: reference
---

# GMMFR A5 Deterministic Tiling Analysis (2026-05-12)

## Key Files
- A5 Tiling: `gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`
- TilingData: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`
- Kernel entry: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`
- FRDeterministicA5: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h`
- Epilogue SeqWrite: `gmm/common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h`
- Cgmct Kernel: `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h`
- A3 Tiling (reference): `gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_base_tiling.cpp`

## Architecture
- A3 (arch32): Kernel-internal sliding window via VectorSync, no windowSize/totalM in Tiling
- A5 (arch35): Tiling-computed windowSize/totalM, external sliding window loop in pertoken_dequant.h

## Tiling Execution Order (TilingBaseClass)
1. GetPlatformInfo
2. GetShapeAttrsInfo -> AnalyzeAttrs -> AnalyzeDtype -> AnalyzeInputs
3. DoOpTiling (windowSize/totalM computed here)
4. DoLibApiTiling (CalBasicBlock/CalL1Tiling computed here, baseM/baseN set)
5. GetTilingKey
6. PostTiling
7. GetWorkspaceSize

## Key Parameters
- windowSizeRows = deterWorkspaceSize(96MB/64MB) / (nSize * sizeof(float))
- totalM = static_cast<uint32_t>(inputParams_.mSize) -- potential truncation from uint64_t
- deterWorkspaceSize_: uint32_t, max 96MB
- inputParams_.nSize: uint64_t

## Per-Round SyncAll Count: 4
1. Cgmct Kernel SyncAll<false>() (after Prologue)
2. Caller SyncAll() (after gmm(params))
3. FRDeterministicA5 SyncAll() (aggregation start)
4. FRDeterministicA5 SyncAll() (aggregation end)

Empty rounds (roundM==0) also have exactly 4 SyncAll calls.

## Findings
- Tiling logic is fundamentally correct in normal paths
- uint32_t totalM truncation risk (LOW, MoE M usually < 4B)
- No single obvious Tiling bug found that would cause coredump
- Coredump likely from runtime factors: workspace size mismatch, groupList data inconsistency, or specific shape edge cases
- Epilogue uses accumulatedGroupOffset_ (from UpdateGlobalBuffer logit offset) for workspace addressing -- correctly resets to 0 each round
