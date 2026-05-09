---
name: GMMFR A5 Status
description: Current status of the GMMFR deterministic feature A5 adaptation project
type: project
---

## Project: gmmfr_deterministic_a5

**Location**: `m:\Desktop\tmp\AgentTest\AscendC\project\gmmfr_deterministic_a5\`
**Target Platform**: A5 (Ascend 950), arch35
**Status**: Code implemented and synced to ops-transformer_AI, but hash not deterministic (as of 2026-05-09)

### Critical Finding (2026-05-09)

Actual code in ops-transformer_AI diverges from plan_a_code documentation:
- `pertoken_dequant.h` still uses original `BlockEpilogueDequantFinalizeRouting` (scatter+AtomicAdd), NOT the new `BlockEpilogueDequantSequentialWrite`
- `BlockEpilogueDequantSequentialWrite` exists in repo but is NOT used in the deterministic branch
- The actual approach: both Prologue and Epilogue write to `deterBuffer` (workspace+16MB), then aggregation iterates 0..batch rows and direct-writes from workspace to yGm
- The Epilogue scatter+AtomicAdd on workspace means workspace accumulates results by outRow correctly
- Aggregation reads `deterBufferGm[outRow * N + nOffset]` (by outRow) and writes `yGm[outRow * N + nOffset]`

### Root Cause of Non-Determinism (User Report)
- Precision correct (error ratio 0.000000)
- Hash changes each run (7a3886de, 669a8652, 0b2d37c2, etc.)
- Indicates non-deterministic floating point accumulation order still exists somewhere

### Code Structure (plan_a_code/)
- `a3_prototype/` -- A3 deterministic prototype (reference only, NOT for A5 use)
- `op_kernel/arch35/` -- A5 Kernel side: deterministic branches, aggregation functions
  - `gmm_fr_deterministic_a5.h` -- A5-specific deterministic aggregation (SyncAll-based)
  - `grouped_matmul_finalize_routing_pertoken_dequant.h` -- Modified with deterministic branch
- `op_host/op_tiling/arch35/` -- A5 Tiling side: deterministic tiling logic

### Key Documentation
- `project_plan.md` -- Overall project plan
- `code_analysis_optimized_plan.md` -- Optimized plan after code analysis
- `plan_b.md` -- Alternative plan B (SequentialWrite Epilogue -- exists but unused)
- `plan_a_code/README.md` -- Code modification guide
- `acceptance_report.md` -- Acceptance report with known BUGs (BUG-04 fixed, BUG-05/BUG-08 need evaluation)
- `server_verification_checklist.md` -- Server verification steps

### Known BUGs (from acceptance_report.md)
- BUG-04: FIXED - GetWorkspaceSize() now adds deterWorkspaceSize to total
- BUG-05: FIXED - deterBufferOffset = SYS_WORKSPACE_SIZE (16MB) instead of usedCoreNum*baseM*baseN
- BUG-08: EVALUATED - overflow protection added in Tiling (requiredDeterSize check), auto-downgrade to non-deterministic
- WARNING-03: RESOLVED - AIC cores now call SyncAll twice (strict pairing)

### Unresolved Issues
- SequentialWrite Epilogue exists but is NOT used -- actual code uses original AtomicAdd scatter Epilogue on workspace
- Need to investigate why hash is still non-deterministic despite deterministic branch
