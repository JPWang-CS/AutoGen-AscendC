---
name: GMMFR Testing Details
description: Detailed testing status and gaps for GMMFR deterministic A5 migration
type: reference
---

# GMMFR Testing Details

## Relevant File Paths

### Project Directory
- `project/gmmfr_deterministic_a5/project_plan.md` -- main migration plan (v3.0, W8A8/INT8 only)
- `project/gmmfr_deterministic_a5/plan_b.md` -- alternative plan (new Epilogue, zero regression risk)
- `project/gmmfr_deterministic_a5/plan_a_code/` -- Plan A code modifications

### Upstream Tests
- `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/tests/ut/op_host/test_grouped_matmul_finalize_routing_tiling.cpp` -- CSV-driven tiling tests (48-column format)
- `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/tests/ut/op_kernel/test_grouped_matmul_finalize_routing.cpp` -- Kernel UT (arch32 only, 5 cases, no golden comparison)
- `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/tests/assets/golden.py` -- CPU golden reference
- `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/tests/ut/op_kernel/grouped_matmul_finalize_routing_data/gen_data.py` -- test data generation

### Key Source Files for Deterministic Feature
- `op_kernel/grouped_matmul_finalize_routing.h` (arch32) -- A3 deterministic: FRDeterministic(), VectorSync(), lines 653-687
- `op_host/grouped_matmul_finalize_routing_base_tiling.cpp` -- A3 DeterministicTilingProcess()
- `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` -- A5 INT8 PerToken path (target for modification)
- `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` -- A5 tiling data structure (needs deterministic fields)
- `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` -- A5 tiling calculation (needs deterministic branch)

## Critical Gaps
1. ZERO deterministic test coverage anywhere in the codebase
2. No A5 (arch35) Kernel UT at all
3. Tiling CSV format does not support GetDeterministic() simulation
4. All existing L2 tests are parameter-validation only, no end-to-end numerical verification
5. Kernel UT has no golden comparison -- only verifies no crash

## Acceptance Criteria (from project_plan.md)
1. A5 Tiling correctly calculates deterministicFlag=1 and deterWorkspaceSize
2. Same INT8 input executed 10x on A5 produces bit-wise identical output
3. Deterministic vs non-deterministic output max relative error < 1e-3
4. Only W8A8/INT8 PerToken path supports deterministic, other paths unaffected
5. Non-deterministic mode zero regression
6. A2/A3/A5 all-platform CI passes
