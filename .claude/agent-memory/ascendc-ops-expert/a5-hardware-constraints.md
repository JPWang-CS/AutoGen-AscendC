# A5 (950) Hardware Constraints for GMMFR Deterministic

## Key Hardware Parameters
- **L2 Cache**: 128MB (NOT 96MB as sometimes incorrectly stated)
- **UB Size**: 248KB per core

## DETER_WORK_SPACE_SIZE Logic
The workspace size is NOT derived from L2 size directly. The logic is:
- If `l2Size > 96MB`: use 96MB workspace
- If `l2Size <= 96MB`: use 64MB workspace
- On A5 (L2=128MB): selects 96MB workspace -- this is CORRECT and intentional

The workspace is allocated on HBM/Global Memory, not L2. L2 only caches Global Memory access.
96MB < 128MB L2, so no L2 thrashing risk for the workspace.

## UB Budget for SequentialWrite Epilogue + FRDeterministicA5
- SequentialWrite Epilogue UB: ~198KB (SEQ_MAX_SINGLE_MNS=128*256, multiple ping-pong buffers)
- FRDeterministicA5 queBind: 24KB (BUFFER_NUM=2, DETER_UB_SIZE=12KB)
- Cgmct framework overhead: ~10-15KB
- Total: ~237-242KB out of 248KB -- TIGHT, may need DETER_UB_SIZE reduction to 8KB

## Sliding Window Decision
- A3 uses windowed sync: `windowSize = deterWorkspaceSize / (n * sizeof(float))`
- A5 current approach: `windowSize = totalM` (one-shot, with Tiling-level overflow protection)
- Decision: NOT introducing sliding window for A5. Rationale:
  1. Typical MoE scenarios have M*N*4 < 96MB
  2. Windowing requires major Cgmct framework changes (group boundary splitting, sync injection)
  3. Degradation to non-deterministic is acceptable for extreme cases

## Source Files
- A5 Tiling: `gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` (lines 458-484)
- A3 Tiling: `gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_base_tiling.cpp` (lines 413-425)
- A3 Kernel windowing: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing.h` (VectorSync, lines 622-650; FRDeterministic, lines 653-687)
- A5 Epilogue UB: `gmm/common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` (Init, lines 194-224)
- A5 Aggregation: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h`
