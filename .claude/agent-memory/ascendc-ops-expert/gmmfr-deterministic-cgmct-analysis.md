# GMMFR Deterministic Migration: Cgmct Framework Analysis

## Key Files
- Epilogue: `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h`
- Kernel: `gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h`
- BlockMmad: `gmm/common/cgmct/block/block_mmad_multi_block.h`
- Prologue: `gmm/common/cgmct/prologue/block_prologue_finalize_routing.h`
- Arch35 Kernel: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`
- Arch35 Deterministic: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h`
- Arch35 Tiling: `gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`
- A3 Prototype Kernel: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing.h`
- A3 Prototype Utils: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_utils.h`
- A3 Base Tiling: `gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_base_tiling.cpp`

## Critical Differences: Cgmct vs A3 Prototype

### Cube Output Destination
- A3: `mm.GetTensorC(mmOutGm[workspaceOffset])` -> workspace (GM)
- Cgmct: `mmadOp_(..., l0cOutUb_, ...)` -> UB (via GetTensorC(ubCmatrix))
- **Cgmct does NOT use workspace for Cube intermediate results**

### Workspace Layout
- SYS_WORKSPACE_SIZE = 16MB (system workspace for AscendC底层)
- A3 workspace = system + `CV_PARALL_NUM(4) * coreNum * baseM * baseN * sizeof(int32_t)` + deterBuffer
- A5/Arch35 workspace = SYS_WORKSPACE_SIZE(16MB) + deterWorkspaceSize(64~96MB)

### Epilogue Scatter Address
- Cgmct Epilogue: `yGlobal_[outRow * n_ + yOffset]` -- scatter by output row index
- A3 deterministic mode: writes to `mmQuantOutGm[vecAParams.yGmOffset1 - (lowBoundM - windowSize) * n]` -- sequential by input row

## Known Issues (as of analysis)
1. **BUG-05**: deterBufferOffset should be SYS_WORKSPACE_SIZE, not `usedCoreNum * baseM * baseN * sizeof(int32_t)`
2. **RISK-01**: Epilogue scatter writes by outRow, but FRDeterministicA5 reads by mOffset -- address mismatch is fatal
