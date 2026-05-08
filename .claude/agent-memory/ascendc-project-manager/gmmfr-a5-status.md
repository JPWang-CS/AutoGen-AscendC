---
name: GMMFR A5 Status
description: Current status of the GMMFR deterministic feature A5 adaptation project
type: project
---

## Project: gmmfr_deterministic_a5

**Location**: `m:\Desktop\tmp\AgentTest\AscendC\project\gmmfr_deterministic_a5\`
**Target Platform**: A5 (Ascend 950), arch35
**Status**: Plan A code implementation exists, not yet merged upstream

### Code Structure (plan_a_code/)
- `a3_prototype/` -- A3 deterministic prototype (reference only, NOT for A5 use)
- `op_kernel/arch35/` -- A5 Kernel side: deterministic branches, aggregation functions
  - `gmm_fr_deterministic_a5.h` -- A5-specific deterministic aggregation
  - `grouped_matmul_finalize_routing.h` -- Modified with deterministic branch
  - `grouped_matmul_finalize_routing_pertoken_dequant.h` -- Modified with deterministic branch
  - `weight_quant_basic_block/` -- Weight quant sub-module deterministic adaptation
- `op_host/op_tiling/arch35/` -- A5 Tiling side: deterministic tiling logic

### Key Documentation
- `project_plan.md` -- Overall project plan
- `code_analysis_optimized_plan.md` -- Optimized plan after code analysis
- `plan_b.md` -- Alternative plan B
- `plan_a_code/README.md` -- Code modification guide with change markers

### Risks
- Test file `test_gmmfr_split.py` has been deleted but not staged
- Code follows correct A3/A5 separation pattern (arch35 directory isolation)
