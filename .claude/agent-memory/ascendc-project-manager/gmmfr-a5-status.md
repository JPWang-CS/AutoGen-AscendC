---
name: GMMFR A5 Status
description: Current status of the GMMFR deterministic feature A5 adaptation project
type: project
---

## Project: gmmfr_deterministic_a5

**Location**: `m:\Desktop\tmp\AgentTest\AscendC\project\gmmfr_deterministic_a5\`
**Target Platform**: A5 (Ascend 950), arch35
**Status**: COMPLETED -- precision verified, deterministic hash verified (2026-05-09)

### Final Implementation (Plan B: SequentialWrite Epilogue)

Three files modified/created:
1. `gmm/common/cgmct/epilogue/block_epilogue_dequant_sequantial_write.h` -- NEW, sequential write epilogue
2. `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` -- MODIFIED, deterministic branch
3. `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h` -- NEW, A5 aggregation function

### Data Flow (Deterministic Mode)
```
Phase 1: Prologue -> yGm (zeros + residual)
Phase 2: Epilogue (SequentialWrite) -> workspace (dequant only, no AtomicAdd)
Phase 3: Aggregation (FRDeterministicA5) -> yGm += workspace (AtomicAdd, single writer per outRow)
```

### Key Design Decisions
- Prologue writes to yGm (NOT workspace) to avoid conflict with Epilogue
- Epilogue uses accumulatedGroupOffset_ for absolute row addressing across groups
- No AtomicAdd in Epilogue; AtomicAdd only in Aggregation with single-writer-per-row guarantee
- totalM uses matmulTiling_.M (total input rows) not batch (output rows)
- Separate GMMTilingDeterministic type needed (nested struct type incompatibility)

### Verification Results
- Precision: error ratio 0.000000
- Determinism: hash consistent across multiple runs

### Key Documentation
- `gmmfr_deterministic_a5_modification_guide.md` -- Complete technical design document (v2.0, full A3->A5 migration guide)
  - Covers: Tiling layer, Kernel layer, aclnn layer, Cgmct framework layer
  - Includes: design rationale, A3 vs A5 comparison, call chain tracing, code analysis with line numbers
  - Key sections: accumulatedGroupOffset_ mechanism, GMMTiling type isolation, AIC SyncAll pairing

### Resolved Issues (Historical)
- BUG-04: FIXED - GetWorkspaceSize() adds deterWorkspaceSize
- BUG-05: FIXED - deterBufferOffset = SYS_WORKSPACE_SIZE (16MB)
- BUG-08: FIXED - overflow protection, auto-downgrade
- WARNING-03: RESOLVED - AIC cores call SyncAll twice
- Prologue/Epilogue workspace conflict: FIXED - Prologue targets yGm not workspace
