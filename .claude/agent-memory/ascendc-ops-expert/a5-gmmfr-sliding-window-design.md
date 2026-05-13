# A5 GMMFR Deterministic Sliding Window Design

## Key Architecture Decision
- A5 uses Cgmct framework where Kernel::operator() processes all groups in one loop
- Unlike A3 (hand-written loop with VectorSync window check), A5 cannot interrupt mid-group
- Solution: multi-round outer loop in pertoken_dequant.h, each round creates new Cgmct Kernel instance with offset pointers

## Files Modified (5 files total, 1 Cgmct minimal change)
1. tiling_data.h - add windowSize, totalM fields
2. quant_tiling.cpp - remove degradation, compute windowSize
3. kernel_gmm_finalize_routing_pertoken_dequant.h - add preOffsetInit to GMMTiling (1 line in InitParamsAndTensor)
4. gmm_fr_deterministic_a5.h - add globalRowOffset param
5. grouped_matmul_finalize_routing_pertoken_dequant.h - replace deterministic branch with multi-round loop

## Cgmct Framework Insight
- groupList cumulative mode (type=0): preOffset_ must be initialized to groupList[groupStart-1]
- groupList absolute mode (type=1): no preOffset_ adjustment needed
- Prologue skip: pass batch=0 for rounds after the first (InitOutputWithZeros returns on size=0)
- Epilogue workspace reuse: Kernel starts from groupIdx=0, accumulatedGroupOffset_=0, writes to workspace row 0
- Scheduler: new instance each Kernel::operator() call, no state carryover

## NZ Weight Offset Calculation
- transB=false (W8A8): singleGroupOffset = CeilDiv(N,32) * CeilDiv(K,16) * 512
- transB=true: CeilDiv(K,32) * CeilDiv(N,16) * 512

## Design doc location
`M:\...\ops-transformer_AI\docs\a5_gmmfr_deterministic_sliding_window_design.md`
