# AscendC Acceptance Tester Memory

## Project Structure
- `project/` -- active operator development projects (currently only `gmmfr_deterministic_a5`)
- `ops-transformer_AI/` -- upstream operator library with existing tests and golden implementations
- `skills/` -- accumulated testing knowledge repository

## Key Findings (2026-05-08)
- GMMFR deterministic feature (W8A8/INT8 PerToken) migrated from A3 (arch32) to A5 (arch35)
- A5 (arch35) uses Cgmct Builder framework, completely different from A3's native AscendC API
- Acceptance report: `project/gmmfr_deterministic_a5/acceptance_report.md` -- verdict: CONDITIONAL PASS
- 5 files modified, 3 critical bugs, 3 major risks found

## Common A3->A5 Migration Pitfalls
- Workspace size MUST be explicitly accumulated in A5 tiling framework (BUG-04)
- A5 SyncAll may require ALL cores (AIC+AIV) -- verify per-platform (WARNING-03)
- Cgmct Epilogue scatter behavior is opaque -- y-address redirection needs verification (RISK-01)
- Sliding window (VectorSync) may be needed for large M scenarios (BUG-08)
- deterBufferOffset calculation differs between A3 and A5 (BUG-05)
- TilingKey routing provides implicit safety net but explicit checks are better (BUG-02)

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
- A3 prototypes: `project/gmmfr_deterministic_a5/plan_a_code/a3_prototype/`

- [GMMFR Testing Details](gmmfr-testing-details.md)
