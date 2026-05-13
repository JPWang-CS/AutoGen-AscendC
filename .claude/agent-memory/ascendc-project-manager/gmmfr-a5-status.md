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

### Workspace Overflow Protection Status (2026-05-11)

**Tiling Layer (quant_tiling.cpp DoOpTiling)**: IMPLEMENTED (degradation approach)
- `requiredDeterSize = inputParams_.mSize * inputParams_.nSize * sizeof(float)`
- If `requiredDeterSize > deterWorkspaceSize_` (96MB or 64MB): auto-downgrade to non-deterministic + OP_LOGW
- GetWorkspaceSize() adds `deterWorkspaceSize_` to `SYS_WORKSPACE_SIZES` (16MB)

**Kernel Layer (pertoken_dequant.h)**: NO SLIDING WINDOW (yet)
- A3 has sliding window via VectorSync; A5 processes all rows at once (`curM = totalM`)
- Relies on Tiling-layer overflow protection to prevent OOB writes
- `deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE` (16MB offset)
- Workspace total = SYS_WORKSPACE_SIZES(16MB) + deterWorkspaceSize_(96/64MB) = max 112MB

**Key Limitation**: A5 workspace max = 112MB. This limits deterministic mode to M*N*4 <= 96MB.
  Example: N=4096 => max M=6144 rows. Larger M silently downgrades to non-deterministic.

### Sliding Window Migration (2026-05-11 -- IN PROGRESS)

**Requirement**: Migrate A3 sliding window to A5. No degradation allowed for W8A8 deterministic.
**Approach**: Plan D -- outer multi-round loop + offset passing
- 5 files to modify (1 Cgmct minimal change: preOffsetInit field + 1 line)
- Window boundary aligns with group boundary (no mid-group interrupt)
- Design doc: `project/gmmfr_deterministic_a5/a5_gmmfr_deterministic_sliding_window_design.md`

**Key Design Points**:
- GMMTiling preOffsetInit: for cumulative groupList mode (type=0), init to groupList[groupStart-1]
- Prologue skip: batch=0, sharedInputOffset=0, sharedInputLen=0 for rounds after first
- NZ weight offset: transB=false => CeilDiv(N,32) * CeilDiv(K,16) * 512 per group
- Each round: new Cgmct Kernel instance, workspace writes from row 0
- SyncAll: 3 per round (1 Kernel internal + 2 FRDeterministicA5)

**Task Pipeline**: PM(done) -> ops-expert(done) -> operator-dev(done) -> code-review(done) -> acceptance-tester(pending)

### Segfault Investigation (2026-05-12) -- ACTIVE BUG

**Symptom**: groupNum=8, batch=72, topK=8, m=576, k=1024, n=2048. Printed "72 36" then Segfault.

**Root Cause Analysis** (highest to lowest probability):

1. **`deterPipe.InitBuffer` on AIC core** (FIXED): `TPipe::InitBuffer` is a Vector-side API.
   Already wrapped with `if ASCEND_IS_AIV { ... }` at lines 168-170.

2. **SyncAll pairing mismatch**: FRDeterministicA5 has 2 SyncAll (both AIC+AIV participate).
   Kernel internal has 1 SyncAll. Plus 1 extra SyncAll after `gmm(params)` at line 267.
   Total per round: 4 SyncAll. Need to verify all cores participate in all SyncAll calls.
   Potential deadlock/hang if counts mismatch.

3. **Epilogue VectorSequentialWrite workspace OOB**: accumulatedGroupOffset_ may be wrong.
   Low probability for small shapes (m=576, n=2048, workspace=64MB+).

4. **Tiling layer windowSize / single group overflow**: Tiling now checks single group M > windowSizeRows
   and returns GRAPH_FAILED. But the check may not cover all edge cases (e.g., negative groupM values).

5. **NZ weight offset calculation**: `singleGroupWeightNZ` uses CeilDiv-based formula.
   Need to verify consistency with Cgmct kernel's internal `UpdateOffset` formula (same CeilDiv params).

**CRITICAL OBSERVATION**: The `gmm(params)` call at line 266 runs the full Cgmct Kernel which
includes its OWN SyncAll internally (in Kernel::operator()). Then line 267 calls another SyncAll()
explicitly. Then FRDeterministicA5 calls 2 more SyncAll. That's 4 SyncAll per round total.
If the Kernel's internal SyncAll doesn't have all cores participating (AIC vs AIV paths differ),
this could cause coredump.

**Key files involved in sliding window changes**:
- arch35/grouped_matmul_finalize_routing_pertoken_dequant.h (A5 entry + sliding window loop)
- arch35/gmm_fr_deterministic_a5.h (A5 aggregation with SyncAll)
- common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h (public kernel: preOffsetInit + NZ fix)
- common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h (new file)
- arch35/grouped_matmul_finalize_routing_tiling_data.h (windowSize, totalM fields)
- op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp (tiling: window size calc)

**Additional code context discovered during 2026-05-12 investigation**:
- A3 quant kernel entry: op_kernel/grouped_matmul_finalize_routing.h (QuantGroupMatmul class)
- A3 quant kernel .cpp: op_kernel/grouped_matmul_finalize_routing.cpp (A8W8_IMPL macro, __CCE_AICORE__==220)
- A3 utils: op_kernel/grouped_matmul_finalize_routing_utils.h (MNConfig, SyncConfig, DETER_UB_SIZE=12KB)
- A5 entry (deterministic): op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h
- A5 entry (non-deterministic MX): op_kernel/arch35/grouped_matmul_finalize_routing.h (cgmct builder)
- Tiling data struct: op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h (GMMFinalizeRoutingDataParams)
- Tiling key: op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h (ATRANS/BTRANS/SCALETYPE/ROWINDEXTYPE)
- A3 uses __CCE_AICORE__==220 guard; A5 uses cgmct framework (arch35 builder with DAV3510 arch)

### Deep Code Review Results (2026-05-11)

5-dimension review completed. Key findings:

**CRITICAL (1)**:
- R3-1: Single group M exceeding windowSize causes workspace overflow, no runtime check
  - Fix: Add maxGroupM check in Tiling DoOpTiling(), or assert in Kernel inner loop
  - Design doc Section 8.2 acknowledges this as ISSUE-2

**HIGH (3)**:
- R4-2: Kernel::End() WaitForVector timing with FRDeterministicA5 -- confirmed correct after analysis
- R2-1/R2-3: totalM calculation and DataCopyPad differences -- design-level, functionally equivalent

**No deadlock/hang/deadloop risks found.**
- SyncAll pairing verified: 3 per round (1 Kernel + 2 FRDeterministicA5), all AIC+AIV participate
- While loop always advances groupStart by at least 1 per iteration
- queBind no resource contention (kernel-local, passed by reference)

**Files reviewed**:
1. grouped_matmul_finalize_routing_tiling_data.h -- struct layout correct
2. grouped_matmul_finalize_routing_quant_tiling.cpp -- tiling logic correct
3. kernel_gmm_finalize_routing_pertoken_dequant.h -- preOffsetInit field correct
4. gmm_fr_deterministic_a5.h -- SyncAll pairing correct, globalRowOffset correct
5. grouped_matmul_finalize_routing_pertoken_dequant.h -- sliding window logic correct, offset chain correct

### Resolved Issues (Historical)
- BUG-04: FIXED - GetWorkspaceSize() adds deterWorkspaceSize
- BUG-05: FIXED - deterBufferOffset = SYS_WORKSPACE_SIZE (16MB)
- BUG-08: PARTIAL - Tiling overflow protection done; no sliding window (design choice)
- WARNING-03: RESOLVED - AIC cores call SyncAll twice
- Prologue/Epilogue workspace conflict: FIXED - Prologue targets yGm not workspace
