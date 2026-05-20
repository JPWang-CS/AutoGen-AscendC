"""
Fix Critical #2: Replace GetBlockIdx() with
GetBlockIdx() * GetTaskRation() + GetSubBlockIdx()
in gmm_fr_deterministic_a5.h so that AIV cores get
unique identifiers in the range [0, coreNumVec).
"""
import sys

filepath = 'M:/Desktop/tmp/AgentTest/AscendC/ops-transformer_AI/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h'

with open(filepath, 'rb') as f:
    data = f.read()

# Check line endings
has_crlf = data.count(b'\r\n') > 0
eol = b'\r\n' if has_crlf else b'\n'
print('File uses CRLF: %s' % has_crlf)

# The old line (line 85): "        if (outRow % coreNumVec != GetBlockIdx()) {"
old_line = b'        if (outRow % coreNumVec != GetBlockIdx()) {' + eol

# The new lines: compute myVecId, then use it
new_lines = (
    b'        uint64_t myVecId = GetBlockIdx() * GetTaskRation() + GetSubBlockIdx();' + eol +
    b'        if (outRow % coreNumVec != myVecId) {' + eol
)

count = data.count(old_line)
print('GetBlockIdx line found %d time(s)' % count)
if count != 1:
    print('ERROR: Expected exactly 1 occurrence')
    sys.exit(1)

data = data.replace(old_line, new_lines, 1)

with open(filepath, 'wb') as f:
    f.write(data)

print('gmm_fr_deterministic_a5.h: Fix applied successfully.')
