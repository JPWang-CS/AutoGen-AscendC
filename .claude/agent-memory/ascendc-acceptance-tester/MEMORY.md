# AscendC Acceptance Tester Memory

## Project Structure
- `project/` -- active operator development projects (currently only `gmmfr_deterministic_a5`)
- `ops-transformer_AI/` -- upstream operator library with existing tests and golden implementations
- `skills/` -- accumulated testing knowledge repository

## Key Findings
- (2026-05-09) GMMFR deterministic fix verified: PASS. 3 files modified, root cause (scatter+AtomicAdd) fixed
- (2026-05-08) GMMFR initial migration: CONDITIONAL PASS. 5 files modified, 3 critical bugs, 3 major risks found

## GMMFR Deterministic Fix (2026-05-09) -- VERIFIED PASS
- Root cause: Epilogue scatter+AtomicAdd caused non-deterministic float accumulation order
- Fix: SequentialWrite Epilogue (no scatter, no AtomicAdd) + Prologue->yGm + Aggregation via FRDeterministicA5
- 3 files modified: sequential_write.h (new), pertoken_dequant.h, gmm_fr_deterministic_a5.h
- SyncAll pairing: 3 per core type (1 in Kernel operator(), 2 in FRDeterministicA5)
- New file: `gmm/common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h`

## Common A3->A5 Migration Pitfalls
- Workspace size MUST be explicitly accumulated in A5 tiling framework
- A5 SyncAll requires ALL cores (AIC+AIV) -- strict 1:1 pairing required
- Cgmct Epilogue scatter is root cause of non-determinism in finalize_routing
- deterBufferOffset calculation differs between A3 and A5
- TilingKey routing provides implicit safety net but explicit checks are better

## Testing Patterns
- Tiling tests use CSV-driven parameterized tests (48-column CSV format)
- OpAPI L2 tests verify `TestGetWorkspaceSize` return codes (parameter validation only)
- Kernel UT uses `ICPU_RUN_KF` for CPU simulation with no numerical comparison
- No existing test framework supports `GetDeterministic()` simulation

## A5 GMMFR File Locations (in ops-transformer_AI/gmm/grouped_matmul_finalize_routing/)
- Tiling data: `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`
- Tiling header: `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h`
- Tiling impl: `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`
- Deterministic kernel: `op_kernel/arch35/gmm_fr_deterministic_a5.h`
- PerToken dequant: `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`
- SequentialWrite Epilogue: `gmm/common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h`
- Original Epilogue: `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` (NOT modified)
- Kernel framework: `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` (NOT modified)
- Prologue: `gmm/common/cgmct/prologue/block_prologue_finalize_routing.h` (NOT modified)
- A3 prototypes: `project/gmmfr_deterministic_a5/plan_a_code/a3_prototype/`

## SyncAll Counting Pattern (A5)
- Kernel operator() has 1 SyncAll<false>() before group loop
- FRDeterministicA5 has 2 SyncAll() (AIC dummy + AIV real)
- Total per core type: 3 -- must match between AIC and AIV
- Always count SyncAll in both code paths when reviewing A5 code

- [GMMFR Testing Details](gmmfr-testing-details.md)
