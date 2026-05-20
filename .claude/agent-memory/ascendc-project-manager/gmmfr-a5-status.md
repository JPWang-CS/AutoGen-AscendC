---
name: GMMFR A5 Status
description: Current status of the GMMFR deterministic feature A5 adaptation project
type: project
---

## Project: gmmfr_deterministic_a5

**Location**: `m:\Desktop\tmp\AgentTest\AscendC\project\gmmfr_deterministic_a5\`
**Target Platform**: A5 (Ascend 950), arch35
**Status**: ACTIVE DEBUGGING -- large-shape hang fixed (root cause identified), precision issue under investigation

### Current Implementation: Workspace Cycling (v3, group-boundary)

**Core Architecture**: Cycling Kernel modifies Cgmct group loop internally.
- Group-boundary cycling (NOT tile-level, because SyncAll is global barrier)
- Prerequisite: single groupM <= windowSizeRows
- workspaceBaseRow tracks window usage, reset to 0 after each aggregation
- LOGIT_OFFSETS never reset (global, for logit/rowIndex reads)
- epilogueOffset[4]=group-inner mOffset, [5]=workspaceBaseRow

**Key Files (committed to source repo)**:
1. `gmm/common/cgmct/kernel/kernel_gmm_fr_deterministic.h` -- Cycling Kernel (NEW)
2. `gmm/common/cgmct/epilogue/block_epilogue_dequant_sequential_write_deterministic.h` -- Cycling Epilogue (NEW)
3. `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h` -- Aggregation (MODIFIED, has printf)
4. `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` -- Entry (MODIFIED)
5. `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` -- Tiling (MODIFIED)

### Active Issues (2026-05-19)

**ISSUE-1 (CRITICAL - HANG)**: groupM > windowSizeRows causes workspace overflow -> hardware hang
- Root cause: deterWorkspaceSize hardcoded at 96MB. For N=7168, windowSizeRows=3512 but groupM=4096.
- Fix direction: Tiling must validate groupM <= windowSizeRows; dynamic workspace sizing or group-internal splitting
- Status: ROOT CAUSE IDENTIFIED, fix pending

**ISSUE-2 (HIGH - PRECISION)**: Large-shape precision errors
- User reports precision issues on large shapes. Debug printf code added but results not yet analyzed.
- Possible root causes: workspace OOB data corruption, Epilogue yGlobal_ rebase error, GetBlockIdx() semantics
- Status: UNDER INVESTIGATION, printf code present in gmm_fr_deterministic_a5.h and kernel_gmm_fr_deterministic.h

### Historical Milestones
- (2026-05-13) Per-group (groupNum=1) W8A8 deterministic: PASS
- (2026-05-11) Sliding Window full implementation re-verification: PASS
- (2026-05-09) GMMFR deterministic fix: PASS, precision error 0.000000
- (2026-05-08) Initial migration: CONDITIONAL PASS, 3 critical bugs fixed
