"""
Fix Critical #1: Remove ASCEND_IS_AIV guard around FRDeterministicA5 calls
in kernel_gmm_fr_deterministic.h so that both AIC and AIV enter the function,
keeping SyncAll counts matched.
"""
import sys

filepath = 'M:/Desktop/tmp/AgentTest/AscendC/ops-transformer_AI/gmm/common/cgmct/kernel/kernel_gmm_fr_deterministic.h'

with open(filepath, 'rb') as f:
    data = f.read()

# Verify CRLF
assert data.count(b'\r\n') > 0, 'Expected CRLF line endings'

# === Fix 1: Cycling aggregation (lines 397-411) ===
old_block1 = (
    b'                    SyncAll();\r\n'
    b'                    if ASCEND_IS_AIV {\r\n'
    b'                        GMMFRDeterministic::SyncConfig syncConfig{};\r\n'
    b'                        syncConfig.curM = cyclingRowOffset;\r\n'
    b'                        syncConfig.lowBoundM = cyclingRowOffset;\r\n'
    b'                        syncConfig.windowSize = cyclingRowOffset;\r\n'
    b'                        syncConfig.baseN = params.cyclingParams.baseN;\r\n'
    b'\r\n'
    b'                        GMMFRDeterministic::FRDeterministicA5<float,\r\n'
    b'                            typename BlockEpilogueDequantFinalizeRouting::DataTypeRowIndex>(\r\n'
    b'                            syncConfig, deterBufferGm, yGmAgg, rowIndexGm, aggQueBind,\r\n'
    b'                            params.cyclingParams.coreNum, params.cyclingParams.n,\r\n'
    b'                            globalRowOffset);\r\n'
    b'                    }\r\n'
    b'                    SyncAll();'
)

new_block1 = (
    b'                    SyncAll();\r\n'
    b'                    // FRDeterministicA5 is called without ASCEND_IS_AIV guard so that\r\n'
    b'                    // both AIC and AIV enter the function. Inside, AIC executes 2x SyncAll\r\n'
    b'                    // and returns; AIV performs the actual aggregation with 2x SyncAll.\r\n'
    b'                    // This keeps SyncAll counts matched (AIC: 2 outer + 2 inner = 4,\r\n'
    b'                    // AIV: 2 outer + 2 inner = 4).\r\n'
    b'                    GMMFRDeterministic::SyncConfig syncConfig{};\r\n'
    b'                    syncConfig.curM = cyclingRowOffset;\r\n'
    b'                    syncConfig.lowBoundM = cyclingRowOffset;\r\n'
    b'                    syncConfig.windowSize = cyclingRowOffset;\r\n'
    b'                    syncConfig.baseN = params.cyclingParams.baseN;\r\n'
    b'\r\n'
    b'                    GMMFRDeterministic::FRDeterministicA5<float,\r\n'
    b'                        typename BlockEpilogueDequantFinalizeRouting::DataTypeRowIndex>(\r\n'
    b'                        syncConfig, deterBufferGm, yGmAgg, rowIndexGm, aggQueBind,\r\n'
    b'                        params.cyclingParams.coreNum, params.cyclingParams.n,\r\n'
    b'                        globalRowOffset);\r\n'
    b'                    SyncAll();'
)

count1 = data.count(old_block1)
print('Cycling aggregation block found %d time(s)' % count1)
if count1 != 1:
    print('ERROR: Expected exactly 1 occurrence of cycling aggregation block')
    sys.exit(1)
data = data.replace(old_block1, new_block1, 1)

# === Fix 2: Final aggregation (lines 467-481) ===
old_block2 = (
    b'            SyncAll();\r\n'
    b'            if ASCEND_IS_AIV {\r\n'
    b'                GMMFRDeterministic::SyncConfig syncConfig{};\r\n'
    b'                syncConfig.curM = cyclingRowOffset;\r\n'
    b'                syncConfig.lowBoundM = cyclingRowOffset;\r\n'
    b'                syncConfig.windowSize = cyclingRowOffset;\r\n'
    b'                syncConfig.baseN = params.cyclingParams.baseN;\r\n'
    b'\r\n'
    b'                GMMFRDeterministic::FRDeterministicA5<float,\r\n'
    b'                    typename BlockEpilogueDequantFinalizeRouting::DataTypeRowIndex>(\r\n'
    b'                    syncConfig, deterBufferGm, yGmAgg, rowIndexGm, aggQueBind,\r\n'
    b'                    params.cyclingParams.coreNum, params.cyclingParams.n,\r\n'
    b'                    globalRowOffset);\r\n'
    b'            }\r\n'
    b'            SyncAll();'
)

new_block2 = (
    b'            SyncAll();\r\n'
    b'            // FRDeterministicA5 called without ASCEND_IS_AIV guard for SyncAll pairing.\r\n'
    b'            GMMFRDeterministic::SyncConfig syncConfig{};\r\n'
    b'            syncConfig.curM = cyclingRowOffset;\r\n'
    b'            syncConfig.lowBoundM = cyclingRowOffset;\r\n'
    b'            syncConfig.windowSize = cyclingRowOffset;\r\n'
    b'            syncConfig.baseN = params.cyclingParams.baseN;\r\n'
    b'\r\n'
    b'            GMMFRDeterministic::FRDeterministicA5<float,\r\n'
    b'                typename BlockEpilogueDequantFinalizeRouting::DataTypeRowIndex>(\r\n'
    b'                syncConfig, deterBufferGm, yGmAgg, rowIndexGm, aggQueBind,\r\n'
    b'                params.cyclingParams.coreNum, params.cyclingParams.n,\r\n'
    b'                globalRowOffset);\r\n'
    b'            SyncAll();'
)

count2 = data.count(old_block2)
print('Final aggregation block found %d time(s)' % count2)
if count2 != 1:
    print('ERROR: Expected exactly 1 occurrence of final aggregation block')
    sys.exit(1)
data = data.replace(old_block2, new_block2, 1)

with open(filepath, 'wb') as f:
    f.write(data)

print('kernel_gmm_fr_deterministic.h: Both fixes applied successfully.')
