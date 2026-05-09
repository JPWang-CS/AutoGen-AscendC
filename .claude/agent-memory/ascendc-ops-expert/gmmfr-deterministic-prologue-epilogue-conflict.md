# GMMFR Deterministic Prologue/Epilogue Workspace Conflict

## Root Cause (2026-05-08)

In deterministic mode, Prologue and Epilogue were both pointed to the same `deterBuffer` workspace.
Prologue writes residual data to `deterBuffer[sharedInputOffset * N ... batch * N)`, then
Epilogue's `VectorSequentialWrite` overwrites the same addresses with dequant results (direct write,
no AtomicAdd). Result: residual is completely lost, causing ~97.5% error rate.

## Fix

Change Prologue's `yGmAddr` from `deterBuffer` to original `y` (final output buffer).
This way:
1. Prologue writes residual directly to `y` (same as non-deterministic mode)
2. Epilogue writes dequant results to `deterBuffer` (workspace only)
3. FRDeterministicA5 uses AtomicAdd to accumulate workspace -> y: `y = residual + dequant_sum`

Only one line changed in `grouped_matmul_finalize_routing_pertoken_dequant.h`:
```cpp
// Before: {share_input, deterBuffer, ...}
// After:  {share_input, y, ...}
```

## Key Lesson

When migrating from non-deterministic (AtomicAdd scatter) to deterministic (sequential write) mode:
- **Never** let Prologue and Epilogue share the same workspace buffer
- Prologue should always write to the final output buffer (`y`), because it initializes residual
- Only Epilogue needs workspace for deferred aggregation
- The workspace (`deterBuffer`) should ONLY contain dequant intermediate results, not residual

## Files Involved

- `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` -- Params construction (fix location)
- `common/cgmct/prologue/block_prologue_finalize_routing.h` -- Prologue writes to `yGmAddr_`
- `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` -- VectorSequentialWrite
- `op_kernel/arch35/gmm_fr_deterministic_a5.h` -- FRDeterministicA5 aggregation
