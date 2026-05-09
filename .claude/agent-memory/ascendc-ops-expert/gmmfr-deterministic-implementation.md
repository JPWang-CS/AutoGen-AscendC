# GMMFR A5 Deterministic Implementation Notes

## Architecture (2026-05-09)

### Data Flow (Deterministic Mode)

```
Phase 1: Prologue -> yGm (zeros + residual)
Phase 2: Epilogue (SequentialWrite) -> workspace (dequant only, no AtomicAdd)
Phase 3: Aggregation (FRDeterministicA5) -> yGm += workspace (AtomicAdd, single writer per outRow)
```

### Key Design Decisions

1. **Prologue writes to yGm, NOT to workspace**
   - Avoids conflict with Epilogue writing to workspace
   - yGm = residual + sum(dequant) via Phase 3 AtomicAdd
   - Workspace only contains dequant intermediate results (clean)

2. **Epilogue uses absolute mOffset addressing**
   - `accumulatedGroupOffset_` captures cumulative row offset across groups
   - Updated in `UpdateGlobalAddr` from `SEQ_LOGIT_INDEXS` (which accumulates across groups)
   - Write address: `(accumulatedGroupOffset_ + offsetM + i) * n_ + yOffset`
   - Each address globally unique (no cross-core, no cross-group overlap)

3. **No AtomicAdd in Epilogue**
   - Since Prologue doesn't write to workspace, workspace is clean
   - Direct DataCopyPad write suffices
   - Different from original VectorAtomicProcess which needs AtomicAdd for scatter overlap

4. **Aggregation uses AtomicAdd to yGm**
   - yGm already initialized by Prologue (zeros + residual)
   - Multiple mOffset values can map to same outRow (routing)
   - AtomicAdd is deterministic: each outRow owned by exactly one core (outRow % coreNumVec)

### Critical: Cgmct offsetM is group-relative

The Cgmct kernel framework computes `mOffset` (row offset) relative to each group, not globally.
This means without correction, different groups would write to overlapping workspace addresses.

Fix: `accumulatedGroupOffset_` in `BlockEpilogueDequantSequentialWrite` captures the cumulative
logit offset from `UpdateGlobalAddr`. This offset is set by the kernel's `IDX_LOGIT_OFFSETS`
in `baseOffset_`, which accumulates across groups via `UpdateOffset()`.

### Files Modified

| File | Change |
|------|--------|
| `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` | VectorSequentialWrite: use abs mOffset addressing, no AtomicAdd; add accumulatedGroupOffset_ member |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | Det branch: use GmmKernelDeterministic with SequentialWrite epilogue; Prologue -> yGm; call FRDeterministicA5 |
| `op_kernel/arch35/gmm_fr_deterministic_a5.h` | Updated docs; logic unchanged (already correct for mOffset workspace layout) |

### SyncAll Count

- Deterministic: Cgmct kernel 1 + FRDeterministicA5 2 = 3 SyncAll per core
- Non-deterministic: Cgmct kernel 1 = 1 SyncAll per core
- Both correctly paired (all cores take same branch based on shared tiling data)
