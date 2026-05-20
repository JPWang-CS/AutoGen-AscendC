# AscendC Acceptance Tester Memory

## Project Structure
- `project/` -- active operator development projects (currently only `gmmfr_deterministic_a5`)
- `ops-transformer_AI/` -- upstream operator library with existing tests and golden implementations
- `skills/` -- accumulated testing knowledge repository

## Key Findings
- (2026-05-19) V5 Critical fix verification: **CONDITIONAL PASS**. 3 files modified (kernel_gmm_fr_deterministic.h, sequential_write_deterministic.h, gmm_fr_deterministic_a5.h). Major refactor: SyncAll cycling -> AIV-only inline cycling. Cycling check: `workspaceBaseRow + tileM > windowSizeRows`. workspaceBaseRow: tile-level increment. wsWriteOffset: removed logitBase. myVecId: `GetBlockIdx()/GetTaskRation()`. 2 annotation issues found (ISSUE-C1: comment/code mismatch, ISSUE-C2: stale header comment).
- (2026-05-19) Large shape precision fix verification: **PASS**. 2 files modified (sequential_write_deterministic.h, kernel_gmm_fr_deterministic.h). Fix1: WaitFlag<MTE3_V>(0/1) after ping-pong loop. Fix2: pre-group + workspaceBaseRow>0 guard, post-group aggregation after +=groupM. All boundary cases verified.
- (2026-05-11) Sliding window full implementation re-verification: **PASS**. 5 files modified (tiling_data.h, tiling.cpp, kernel framework, deterministic_a5.h, pertoken_dequant.h). All 5 check items passed. ISSUE-2 (单group溢出) retained as MEDIUM risk.
- (2026-05-11) Sliding window ISSUE-1 fix re-test: **PASS**. prologueSharedInputOffset/PrologueSharedInputLen/prologueBatch 三个参数后续轮正确清零，Prologue空跑不破坏yGm
- (2026-05-11) Sliding window implementation: initial **FAIL**, ISSUE-1 fixed and re-verified PASS. 1 MEDIUM risk remaining (ISSUE-2: 单group溢出)
- (2026-05-11) DETER_UB_SIZE 12KB->8KB verified: PASS. UB margin improved from ~25KB to ~33KB, no performance impact for N<=4096
- (2026-05-09) GMMTiling type fix verified: PASS. Deterministic branch now uses GmmKernelDeterministic::GMMTiling instead of GmmKernel::GMMTiling
- (2026-05-09) GMMFR deterministic fix verified: PASS. 3 files modified, root cause (scatter+AtomicAdd) fixed
- (2026-05-08) GMMFR initial migration: CONDITIONAL PASS. 5 files modified, 3 critical bugs, 3 major risks found

## A5 UB Budget Analysis (248KB total)
- SequentialWrite Epilogue: ~199KB (VECIN + VECOUT, max offset at ~199KB)
- Prologue: ~136KB (VECCALC + VECIN, time-multiplexed with Epilogue, shares same UB)
- FRDeterministicA5 queBind: 16KB (2 buffers * 8KB)
- **Active total**: ~215KB (33KB margin)
- Prologue and Epilogue are time-multiplexed -- they share the same UB space, not additive
- Only 4 files in the entire project reference DETER_UB_SIZE (2 arch35, 2 arch32)

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
- **GMMTiling type mismatch**: Same class template with different Epilogue params produces DIFFERENT GMMTiling types in C++. Always use the correct kernel's own GMMTiling (e.g., `GmmKernelDeterministic::GMMTiling` for deterministic branch, NOT `GmmKernel::GMMTiling`). Fields are identical but types are incompatible.
- **Prologue skip pitfall**: When passing batch=0 to skip Prologue in multi-round, MUST also set sharedInputOffset=0 and sharedInputLen=0. Otherwise Prologue clears yGm data and unsigned underflow in `n*(0-tail)`. Prologue code has no batch==0 guard -- it checks sharedInputLen first.

## Sliding Window Design Pattern (A5 GMMFR)
- Window boundaries align to group boundaries (not within groups, unlike A3)
- Each round: new Kernel instance + FRDeterministicA5 aggregation
- workspace reused each round (written from row 0, safe after SyncAll #3)
- Prologue only runs in round 1 (batch>0); round 2+ must use batch=0 AND sharedInputLen=0
- SyncAll per round: 3 (1 Kernel + 2 FRDeterministicA5), AIC+AIV strictly paired
- Single group exceeding windowSize causes workspace overflow (no runtime check)

## Workspace Cycling Aggregation Triggers (A5 GMMFR) -- V5 AIV-only tile-level
- **Inline tile-level**: `workspaceBaseRow + tileM > windowSizeRows` -- inside tile while loop, AIV-only (no SyncAll)
  - workspaceBaseRow is tile-level increment (not group-level)
  - Aggregation: AIV-only per-core partitioning (outRow % coreNum) + AtomicAdd
  - After aggregation: globalRowOffset += workspaceBaseRow; workspaceBaseRow = 0
  - Prerequisite: single tile M <= windowSizeRows
- **Final aggregation**: SyncAll + FRDeterministicA5 + SyncAll, after group loop, handles remaining rows
- **SyncAll total**: 3 (1 init + 2 final aggregation) -- no SyncAll inside tile loop

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
- SequentialWrite Epilogue (deterministic): `gmm/common/cgmct/epilogue/block_epilogue_dequant_sequential_write_deterministic.h`
- Deterministic kernel framework: `gmm/common/cgmct/kernel/kernel_gmm_fr_deterministic.h`
- Original Epilogue: `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` (NOT modified)
- Kernel framework: `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` (NOT modified)
- Prologue: `gmm/common/cgmct/prologue/block_prologue_finalize_routing.h` (NOT modified)
- A3 prototypes: `project/gmmfr_deterministic_a5/plan_a_code/a3_prototype/`

## A5 Ping-Pong Event Stream Pattern
- In ping-pong DataCopy loops (MTE3 writes), the last iteration's SetFlag<MTE3_V> is never consumed by a subsequent WaitFlag inside the loop
- MUST drain both ping-pong buffers (WaitFlag<MTE3_V>(0) and WaitFlag<MTE3_V>(1)) AFTER the loop exits
- WaitFlag is idempotent: waiting on an already-completed flag returns immediately, so draining both is safe
- This is the root cause pattern for "聚合时读到不完整的 workspace 数据" -- MTE3 writes still in-flight when aggregation reads GM

## SyncAll Counting Pattern (A5 V5)
- Kernel operator() has 1 SyncAll<false>() before group loop
- Final aggregation has 2 SyncAll() (1 before + 1 after FRDeterministicA5)
- Total per core type: 3 -- must match between AIC and AIV
- V5 change: NO SyncAll inside tile loop (was SyncAll in V4.1, now AIV-only)
- Always count SyncAll in both code paths when reviewing A5 code

- [GMMFR Testing Details](gmmfr-testing-details.md)
