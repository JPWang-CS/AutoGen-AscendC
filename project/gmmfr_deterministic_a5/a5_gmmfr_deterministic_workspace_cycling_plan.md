# A5 GMMFR W8A8 确定性模式 -- Workspace Cycling 实现方案 (v2)

> 平台：A5 (Ascend 950, DAV3510, arch35)
> 日期：2026-05-14 (v2: 修复 3 个 Critical 问题)

---

## 1. 问题

当前 A5 确定性模式中，当 `totalM * N * sizeof(float) > deterWorkspaceSize`（64-96MB）时，Tiling 层将 `deterministicFlag` 置 0，退化为非确定性模式。

用户要求：不允许退化，通过 workspace cycling 重复使用固定大小的 workspace。

---

## 2. 验收发现的 3 个 Critical 问题 (v1 缺陷)

### Critical 1: 跨 group cycling 地址不匹配

**现象**：cycling reset 后，`UpdateOffset(groupIdx)` 推进 `LOGIT_OFFSETS` 一个 group 的 M，
导致 `accumulatedGroupOffset_` 不从 0 开始。但 `FRDeterministicA5` 始终从 `workspace[0]` 读，地址不匹配。

**根因**：reset 机制试图通过 `LOGIT_OFFSETS -= accumulatedRows` 回退全局偏移，
期望下一轮 `UpdateOffset` 自然对齐回 0。但这要求 cycling 边界恰好在 group 边界上，
实际上 cycling 是 **tile 级别** 的，可以在 group 中间触发。

### Critical 2: group 内 cycling offset 越界

**现象**：Epilogue 的 `offsetM` 是 tile 在 group 内的绝对位置（如第 5 个 tile 的 offsetM=1024）。
cycling reset 只改了 `accumulatedGroupOffset_`，`absRowBase = accumulatedGroupOffset_ + offsetM`
仍等于 1024，超过 `windowSizeRows`（假设 1000），workspace 越界写入。

**根因**：`offsetM` 来自 `CoordClass::GetQuantOffset`，计算为 `mTileIdx * l1M + mSplitOffset`，
这是 tile 在**当前 group 内**的行偏移，不受 cycling reset 影响。
cycling 发生在 group 中间时，group 内后半部分的 tile 的 offsetM 可能超过 workspace
当前窗口内剩余行数。

### Critical 3: 写后检查

**现象**：`accumulatedRows += tileM` 和越界检查在 Epilogue 写入 workspace **之后**，
最后一个 tile 可能已经超出 workspace 边界。

**根因**：当前代码流程是：
```
Epilogue(tile) -> workspace   // 先写
accumulatedRows += tileM      // 再计数
if (accumulatedRows >= windowSizeRows) { ... }  // 再检查
```
如果 `accumulatedRows` 在写入前已经接近 `windowSizeRows`，这个 tile 的写入可能越界。

---

## 3. 根因分析：统一的地址模型

### 3.1 原始（非 cycling）确定性模式的地址模型

在非 cycling 模式下，workspace 一次性容纳所有 group 的全部行。地址模型简单：

```
Epilogue 写地址:
  absRowBase = LOGIT_OFFSETS(跨 group 累积) + offsetM(group 内 tile 偏移)
  workspace[absRowBase * N + nOffset]

FRDeterministicA5 读地址:
  workspace[mOffset * N + nOffset]     // mOffset 从 0 到 totalM-1
  rowIndex[globalRowOffset + mOffset]   // 映射到 yGm 行
```

两者一致，因为 `absRowBase` 就是从 0 开始递增的全局行号。

### 3.2 Cycling 模式需要的地址模型

Cycling 模式下，workspace 是一个固定大小的滑动窗口。关键约束：

1. **Epilogue 写地址必须在 [0, windowSizeRows) 范围内**
2. **FRDeterministicA5 读地址必须与 Epilogue 写地址一致**
3. **rowIndex 查找必须使用全局行号（不随 cycling 变化）**

核心矛盾：
- `LOGIT_OFFSETS`（= `accumulatedGroupOffset_`）是**全局**累积的
- `offsetM` 是**group 内**的
- workspace 窗口要求**窗口内**地址

### 3.3 重新审视 offsetM 的来源

在 Kernel 的 tile 循环中（cycling kernel 第 369-373 行）：

```cpp
blockOffset_ = coord.template GetQuantOffset<PERTOKEN_MODE>(
    Get<CYC_IDX_M_TILEIDXS>(tileIdx), ...);
```

`GetQuantOffset` 计算（coord_utils.h 第 216-226 行）：
```
mOffset = mTileIdx * l1M + mSplitOffset   // group 内偏移
Get<5>(offset) = mOffset * n + nOffset     // C_OFFSETS = y
```

Kernel 提取：
```cpp
int64_t y = Get<CYC_IDX_C_OFFSETS>(blockOffset_);  // = mOffset * n + nOffset
int64_t mOffset = y / n;                            // group 内行偏移
```

传递给 Epilogue 的 `epilogueOffset` 最后两个元素都是 `mOffset`（group 内）。

Epilogue 的 `operator()` 中：
```
logitOffset = Get<SEQ_CYCLING_LOGIT_INDEXS>(blockCoord) + mOffset
            = accumulatedGroupOffset_ + mOffset         (AIV 侧看到的)
```

`VectorSequentialWrite` 中：
```
absRowBase = accumulatedGroupOffset_ + offsetM    // offsetM 就是 logitOffset
```

因此 `absRowBase = accumulatedGroupOffset_ + group 内 mOffset`，这是**全局行号**。

---

## 4. 修订后的解决方案

### 4.1 核心设计：引入 cyclingRowOffset 追踪变量

**核心思想**：不再试图通过 reset `LOGIT_OFFSETS` 来间接控制 Epilogue 的写地址，
而是在 Kernel 层引入一个明确的 **cycling 窗口内偏移量** `cyclingRowOffset`，
直接告知 Epilogue 当前 tile 应该写入 workspace 的哪个位置。

**放弃 LOGIT_OFFSETS reset 方案**：`LOGIT_OFFSETS` 同时被用于 Epilogue 的多个功能
（logit 读取、rowIndex 读取、x1Scale 读取），不能简单地减回。即使 reset 了 `LOGIT_OFFSETS`，
Epilogue 的 `offsetM` 仍然是 group 内偏移，无法反映 cycling 窗口内的位置。

### 4.2 新增变量

```cpp
// Kernel 层新增（kernel_gmm_fr_deterministic_cycling.h）
uint64_t cyclingRowOffset = 0;  // 当前 cycling 窗口内已使用的行数
                                 // Epilogue 将 workspace[cyclingRowOffset...] 作为写入起点
```

### 4.3 Epilogue 写地址：引入 cyclingBaseRow

**方案**：修改 Epilogue 的 `VectorSequentialWrite`，
将 `accumulatedGroupOffset_` 的语义从"全局累积偏移"改为"cycling 窗口内基准偏移"。

Kernel 通过 `UpdateGlobalBuffer` 传递 `cyclingBaseRow` 给 Epilogue：

```cpp
// Kernel::UpdateGlobalBuffer 中，AIV 侧:
// 替换原来传递 LOGIT_OFFSETS 给 Epilogue 的方式
// 新增：传递 cyclingRowOffset 作为 Epilogue 写入的基准行

if ASCEND_IS_AIV {
    AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
        0L,
        Get<CYC_IDX_BIAS_OFFSETS>(baseOffset_),
        Get<CYC_IDX_X2SCALE_OFFSETS>(baseOffset_),
        Get<CYC_IDX_X1SCALE_OFFSETS>(baseOffset_),
        static_cast<int64_t>(cyclingRowOffset),   // Epilogue 写入基准 = cycling 窗口内偏移
        static_cast<int64_t>(cyclingRowOffset)};   // logit/rowIndex 仍使用全局偏移（通过 logitGlobal_ 的 base）
    epilogueDequantOp_.UpdateGlobalAddr(vecBaseOffset);
}
```

但是，这里有一个问题：`logitGlobal_` 和 `rowIndexGlobal_` 的基地址也需要正确设置。
它们的 base 需要 `LOGIT_OFFSETS`（全局偏移），但 `accumulatedGroupOffset_` 需要 `cyclingRowOffset`。

**解决**：将 Epilogue 的 `UpdateGlobalAddr` 拆分语义：

1. `accumulatedGroupOffset_`（用于 workspace 写地址）= cycling 窗口内基准
2. `logitGlobal_` 和 `rowIndexGlobal_` 的 base 仍使用 `LOGIT_OFFSETS`（全局偏移）

### 4.4 修订后的 Epilogue UpdateGlobalAddr

**问题**：当前 Epilogue 的 `UpdateGlobalAddr` 中，
`logitGlobal_` 和 `rowIndexGlobal_` 的 base 和 `accumulatedGroupOffset_` 都来自同一个
`SEQ_CYCLING_LOGIT_INDEXS`。需要拆分。

**方案**：在 Epilogue 中新增一个方法或参数来单独设置 workspace 写入基准行。

具体做法：在 Kernel 层，直接修改传给 Epilogue 的 `vecBaseOffset` 的第 4/5 元素语义。

看 Epilogue 的 `UpdateGlobalAddr`（cycling 版本第 242-260 行）：

```cpp
accumulatedGroupOffset_ = static_cast<uint64_t>(Get<SEQ_CYCLING_LOGIT_INDEXS>(baseOffset));
logitGlobal_.SetGlobalBuffer(logitGmAddr + Get<SEQ_CYCLING_LOGIT_INDEXS>(baseOffset));
rowIndexGlobal_.SetGlobalBuffer(rowIndexGmAddr + Get<SEQ_CYCLING_LOGIT_INDEXS>(baseOffset));
```

三个用途都用了同一个值。我们需要区分：
- `logitGlobal_` 和 `rowIndexGlobal_` 的偏移需要是**全局**的（LOGIT_OFFSETS）
- `accumulatedGroupOffset_` 需要是**cycling 窗口内**的（cyclingRowOffset）

**解决方案**：修改 Epilogue 的 `UpdateGlobalAddr` 接口，
使用 `baseOffset` 的不同索引来分别传递这两个值。

当前 `BlockCoord`（6 元素）的索引分配：
```
0: SEQ_CYCLING_Y_IDXS       (yOffset, 未使用)
1: SEQ_CYCLING_BIAS_IDXS    (bias 偏移)
2: SEQ_CYCLING_X2SCALE_IDXS (x2Scale 偏移)
3: SEQ_CYCLING_X1SCALE_IDXS (x1Scale 偏移)
4: SEQ_CYCLING_LOGIT_INDEXS (logit/rowIndex/accumulatedGroupOffset)
5: 第 5 元素 (也传 logit 偏移)
```

**改造**：重新定义 Epilogue 的 `BlockCoord` 语义：

```
0: SEQ_CYCLING_Y_IDXS       (yOffset, 未使用)
1: SEQ_CYCLING_BIAS_IDXS    (bias 偏移)
2: SEQ_CYCLING_X2SCALE_IDXS (x2Scale 偏移)
3: SEQ_CYCLING_X1SCALE_IDXS (x1Scale 偏移)
4: SEQ_CYCLING_LOGIT_INDEXS (logit/rowIndex 的全局偏移 -- LOGIT_OFFSETS)
5: SEQ_CYCLING_WORKSPACE_ROW_BASE (workspace 写入基准行 -- cyclingRowOffset)
```

Epilogue 的 `UpdateGlobalAddr` 改为：

```cpp
if ASCEND_IS_AIV {
    // workspace 写地址基准 = cycling 窗口内偏移（第 5 元素）
    accumulatedGroupOffset_ = static_cast<uint64_t>(Get<5>(baseOffset));

    // logit 和 rowIndex 仍使用全局偏移（第 4 元素）
    logitGlobal_.SetGlobalBuffer(logitGmAddr + Get<SEQ_CYCLING_LOGIT_INDEXS>(baseOffset));
    rowIndexGlobal_.SetGlobalBuffer(rowIndexGmAddr + Get<SEQ_CYCLING_LOGIT_INDEXS>(baseOffset));

    // 其他不变...
    x1ScaleGlobal_.SetGlobalBuffer(x1ScaleGmAddr + Get<SEQ_CYCLING_X1SCALE_IDXS>(baseOffset));
    x2ScaleGlobal_.SetGlobalBuffer(x2ScaleGmAddr + Get<SEQ_CYCLING_X2SCALE_IDXS>(baseOffset));
    biasGlobal_.SetGlobalBuffer(biasGmAddr + Get<SEQ_CYCLING_BIAS_IDXS>(baseOffset));
}
```

**但是**：Epilogue 的 `operator()` 中也使用了 `blockCoord` 的第 4/5 元素：
```cpp
uint64_t logitOffset = Get<SEQ_CYCLING_LOGIT_INDEXS>(blockCoord) + mOffset;
```

这里的 `blockCoord` 是 **tile 级别的** `epilogueOffset`（由 Kernel 的 tile 循环设置），
不是 `baseOffset`（group 级别的）。`blockCoord` 和 `baseOffset` 是不同的数据。

看 Kernel 代码（第 396-401 行）：
```cpp
epilogueOffset{nOffset,
    Get<CYC_IDX_BIAS_OFFSETS>(blockOffset_),
    Get<CYC_IDX_X2SCALE_OFFSETS>(blockOffset_),
    Get<CYC_IDX_X1SCALE_OFFSETS>(blockOffset_),
    mOffset, mOffset};
```

`epilogueOffset` 的第 4 和第 5 元素都是 `mOffset`（group 内 tile 偏移）。
Epilogue `operator()` 用第 4 元素 `mOffset + subBlockOffset` 来读 logit 和写 workspace。

所以实际上 Epilogue 中有两层偏移：
1. `UpdateGlobalAddr` 设置的 `baseOffset` -- group 级别基准
2. `operator()` 接收的 `blockCoord` -- tile 级别偏移

**问题核心**：`logitGlobal_` 的 base 需要**全局偏移**来读取正确的 logit 数据，
但 `accumulatedGroupOffset_` 需要**cycling 窗口内偏移**来写入正确的 workspace 地址。

这两个需求可以解耦：

```cpp
// UpdateGlobalAddr 中:
logitGlobal_.SetGlobalBuffer(logitGmAddr + Get<SEQ_CYCLING_LOGIT_INDEXS>(baseOffset));
// 这里的 baseOffset 第 4 元素 = LOGIT_OFFSETS (全局累积)

rowIndexGlobal_.SetGlobalBuffer(rowIndexGmAddr + Get<SEQ_CYCLING_LOGIT_INDEXS>(baseOffset));
// rowIndex 也不受 cycling 影响，使用全局偏移

accumulatedGroupOffset_ = static_cast<uint64_t>(Get<5>(baseOffset));
// 第 5 元素改为传递 cyclingRowOffset（cycling 窗口内偏移）
```

**这样不需要改变 Epilogue 的 `operator()` 逻辑**，因为 `operator()` 中的
`logitOffset = Get<SEQ_CYCLING_LOGIT_INDEXS>(blockCoord) + mOffset` 中的 `blockCoord`
是 tile 级别的 `mOffset`（不变），而 `logitGlobal_` 的 base 已经正确设置为 `LOGIT_OFFSETS`。

`VectorSequentialWrite` 中 `absRowBase = accumulatedGroupOffset_ + offsetM`，
其中 `offsetM` 是从 `logitOffset` 来的（group 内 tile 偏移 + subBlockOffset），
`accumulatedGroupOffset_` 现在是 `cyclingRowOffset`。

**地址计算验证**：
```
workspace 写地址 = (cyclingRowOffset + group 内 tile mOffset) * N + nOffset
```

只要 `cyclingRowOffset + 当前 tile 的 mOffset < windowSizeRows`，就不会越界。

### 4.5 Kernel 层 cyclingRowOffset 的维护

```cpp
// Kernel::operator() 中
uint64_t cyclingRowOffset = 0;      // cycling 窗口内已用行数
uint64_t globalRowOffset = 0;       // 全局已处理行数（用于 rowIndex 查找）

for each group:
    UpdateGroupParams(groupIdx)     // LOGIT_OFFSETS 累加（不变）

    // 每个 group 开始时，UpdateGlobalBuffer 传递当前 cyclingRowOffset 给 Epilogue
    UpdateGlobalBuffer(params)      // 修改：第 5 元素传 cyclingRowOffset

    for each tile:
        // 写前检查（Critical 3 修复）
        uint64_t tileRows = Get<MNK_M>(singleShape);
        if (cyclingRowOffset + tileRows > windowSizeRows) {
            // 当前 tile 放不下，先聚合已写入的数据
            Aggregate(cyclingRowOffset, globalRowOffset);
            cyclingRowOffset = 0;
            // 聚合后需要重新设置 Epilogue 的 workspace 基准
            UpdateCyclingBase(params, cyclingRowOffset);
        }

        // AIC: MMAD -> L0C
        // AIV: Epilogue -> workspace[cyclingRowOffset + tile mOffset]

        cyclingRowOffset += tileRows;

// 最后一批
if (cyclingRowOffset > 0):
    Aggregate(cyclingRowOffset, globalRowOffset);
```

### 4.6 关键问题：Cycling 跨 Group 边界

Cycling 发生在 tile 级别，可能在 group 中间触发。聚合后：
- `cyclingRowOffset = 0`（重置）
- 但当前 group 的 `LOGIT_OFFSETS` 不变（因为 logit/rowIndex 仍需全局偏移来读）
- 当前 group 的 tile 循环继续，下一个 tile 的 `mOffset`（group 内偏移）仍然正确

**重新设置 Epilogue 基准**：聚合后需要通知 Epilogue `cyclingRowOffset` 变为 0。
这通过在聚合后调用 `UpdateGlobalBuffer` 实现（只更新第 5 元素）。

但有一个**效率问题**：聚合后调用完整的 `UpdateGlobalBuffer` 会重新设置所有 GlobalTensor
的 base，包括 `logitGlobal_`、`x1ScaleGlobal_` 等。这些值在 group 内是不变的，
重复设置无害但冗余。为简洁起见，可以直接调用完整 `UpdateGlobalBuffer`。

### 4.7 UpdateGlobalBuffer 修改

```cpp
// Kernel::UpdateGlobalBuffer (修改后)
__aicore__ inline void UpdateGlobalBuffer(const Params &params, uint64_t cyclingBase)
{
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer(aGmAddr + Get<CYC_IDX_A_OFFSETS>(baseOffset_));
        bGlobal_.SetGlobalBuffer(bGmAddr + Get<CYC_IDX_B_OFFSETS>(baseOffset_));
    }
    if ASCEND_IS_AIV {
        AscendC::Coord<...> vecBaseOffset{
            0L,
            Get<CYC_IDX_BIAS_OFFSETS>(baseOffset_),
            Get<CYC_IDX_X2SCALE_OFFSETS>(baseOffset_),
            Get<CYC_IDX_X1SCALE_OFFSETS>(baseOffset_),
            Get<CYC_IDX_LOGIT_OFFSETS>(baseOffset_),    // logit/rowIndex 全局偏移
            static_cast<int64_t>(cyclingBase)};          // workspace 写入基准
        epilogueDequantOp_.UpdateGlobalAddr(vecBaseOffset);
    }
}
```

新增参数 `cyclingBase`，每次调用时传入当前的 `cyclingRowOffset`。

### 4.8 完整的 Cycling Kernel 主循环

```cpp
__aicore__ inline void operator()(const Params &params)
{
    // Init 阶段（不变）
    ...

    uint64_t windowSizeRows = params.cyclingParams.windowSizeRows;
    uint64_t cyclingRowOffset = 0;   // cycling 窗口内已用行数
    uint64_t globalRowOffset = 0;    // 全局行偏移（用于 rowIndex）

    for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
        if (!UpdateGroupParams(params, groupIdx)) {
            continue;
        }

        int64_t m = Get<MNK_M>(problemShape_);
        int64_t n = Get<MNK_N>(problemShape_);
        int64_t k = Get<MNK_K>(problemShape_);
        TupleShape resProblemShape{m, n, k, 0};
        bs.UpdateNextProblem(resProblemShape);
        epilogueDequantOp_.UpdateNextProblem(resProblemShape);
        UpdateGlobalBuffer(params, cyclingRowOffset);  // 传递 cyclingRowOffset
        CoordClass coord(m, n, k, baseM, baseN, baseK);

        BlockCoord tileIdx;
        while (bs.GetTileIdx(tileIdx)) {
            BlockShape singleShape = bs.GetBlockShape(tileIdx);
            blockOffset_ = coord.template GetQuantOffset<PERTOKEN_MODE>(...);
            uint64_t tileRows = static_cast<uint64_t>(Get<MNK_M>(singleShape));

            // === 写前检查（Critical 3 修复）===
            if (cyclingRowOffset + tileRows > windowSizeRows) {
                // 先聚合已写入的数据，再处理当前 tile
                if ASCEND_IS_AIC {
                    if (isVecSetSyncCom_) WaitForVector();
                }
                SyncAll();
                if ASCEND_IS_AIV {
                    FRDeterministicA5(... cyclingRowOffset ... globalRowOffset ...);
                }
                SyncAll();
                globalRowOffset += cyclingRowOffset;
                cyclingRowOffset = 0;
                isVecSetSyncCom_ = false;
                // 通知 Epilogue 新的 cycling 基准
                UpdateGlobalBuffer(params, cyclingRowOffset);
            }

            // AIC: MMAD
            if ASCEND_IS_AIC {
                if (isVecSetSyncCom_) WaitForVector();
                mmadOp_(...);
                NotifyVector();
            }
            isVecSetSyncCom_ = true;

            // AIV: Epilogue -> workspace
            if ASCEND_IS_AIV {
                epilogueDequantOp_(epilogueShape, epilogueOffset);
                NotifyCube();
            }

            cyclingRowOffset += tileRows;
        }
    }

    // 最终聚合
    if ASCEND_IS_AIC {
        if (isVecSetSyncCom_) WaitForVector();
    }
    if (cyclingRowOffset > 0) {
        SyncAll();
        if ASCEND_IS_AIV {
            FRDeterministicA5(... cyclingRowOffset ... globalRowOffset ...);
        }
        SyncAll();
    }
}
```

### 4.9 地址计算完整验证

假设场景：3 个 group，M = [500, 500, 500]，baseM = 256，windowSizeRows = 800，N = 1024

**Group 0 (M=500):**
- `UpdateOffset(0)`: LOGIT_OFFSETS = 0
- `UpdateGlobalBuffer(0)`: cyclingBase = 0
  - Epilogue: `accumulatedGroupOffset_ = 0`
  - `logitGlobal_ base = 0`

Tile 0 (mOffset=0, rows=256):
- 写前检查: 0 + 256 <= 800, 通过
- Epilogue: `absRowBase = 0 + 0 = 0` -> `workspace[0..255 * N]`
- cyclingRowOffset = 256

Tile 1 (mOffset=256, rows=244, tail):
- 写前检查: 256 + 244 = 500 <= 800, 通过
- Epilogue: `absRowBase = 0 + 256 = 256` -> `workspace[256..499 * N]`
- cyclingRowOffset = 500

**Group 1 (M=500):**
- `UpdateOffset(1)`: LOGIT_OFFSETS = 0 + 500 = 500
- `UpdateGlobalBuffer(500)`: cyclingBase = 500
  - 但等等！500 > 0 不等于 cycling 窗口内偏移...

**问题发现**：上面的设计中，`cyclingRowOffset` 只在 cycling 触发时重置为 0。
但在 group 边界处，`UpdateGlobalBuffer` 被调用时传入 `cyclingRowOffset`（当前窗口内偏移），
这是正确的。但 `logitGlobal_` 的 base 需要 `LOGIT_OFFSETS`（全局偏移）来读取 logit。

在 Epilogue 的 `UpdateGlobalAddr` 中：
```
accumulatedGroupOffset_ = Get<5>(baseOffset)  // = cyclingRowOffset = 500
logitGlobal_ base = Get<4>(baseOffset)         // = LOGIT_OFFSETS = 500
```

Tile 0 (mOffset=0, rows=256):
- 写前检查: 500 + 256 = 756 <= 800, 通过
- Epilogue: `absRowBase = 500 + 0 = 500` -> `workspace[500..755 * N]`
- logit 读: `logitGlobal_[0]` = 全局偏移 500 处的 logit 数据 -- 正确!
- cyclingRowOffset = 756

Tile 1 (mOffset=256, rows=244):
- 写前检查: 756 + 244 = 1000 > 800, **触发 cycling**
- 聚合 756 行: FRDeterministicA5(756, globalRowOffset=0)
  - 读 workspace[0..755], rowIndex[0..755]
- globalRowOffset = 756, cyclingRowOffset = 0
- UpdateGlobalBuffer(0): cyclingBase = 0
  - `accumulatedGroupOffset_ = 0`
  - `logitGlobal_ base = 500` (LOGIT_OFFSETS 没变)

  **问题**: logitGlobal_ base 仍然是 500，
  但下一个 tile 的 `logitOffset = Get<4>(blockCoord) + mOffset`
  其中 `blockCoord` 第 4 元素是 tile 的 group 内 mOffset = 256。

  `CopyInLogit` 中: `DataCopyPad(logitUb, logitGlobal_[offsetM], ...)`
  其中 offsetM = logitOffset = 256。

  实际读的是 `logitGmAddr + 500 + 256 = logitGmAddr + 756`。
  这是全局第 756 行的 logit -- 正确!

- Epilogue: `absRowBase = 0 + 256 = 256` -> `workspace[256..499 * N]`
  **等等，group 1 的第 2 个 tile 在 workspace 中从 256 开始写，
  但 group 1 的第 1 个 tile 已经写到了 500-755。**
  **聚合后 workspace 被清空（逻辑上），所以 0-499 的区域可以被重用。**
  **写入 workspace[256..499] 不会与任何有效数据冲突 -- 正确!**

- cyclingRowOffset = 0 + 244 = 244

**验证 FRDeterministicA5 读地址一致性**：

聚合 1（756 行，globalRowOffset=0）：
- FRDeterministicA5 读 workspace[0..755 * N]
- 写 yGm: rowIndex[0 + mOffset] -> outRow

聚合 2（244 行，globalRowOffset=756）：
- FRDeterministicA5 读 workspace[0..243 * N]
  **等等！聚合后 cyclingRowOffset 被重置为 0，
  下一个 tile 写到 workspace[256..499]，
  但 group 1 的第 2 个 tile 只有 244 行。**
  **在下一个 group 继续之前，workspace 使用范围是 [0, 244)**
  **但刚才写的是 workspace[256..499]？不对，验证一下：**

  cycling 后 cyclingRowOffset = 0。
  Tile 1 (rows=244) 在 cycling 后处理：
  - `UpdateGlobalBuffer(0)`: accumulatedGroupOffset_ = 0
  - Epilogue: `absRowBase = 0 + 256 = 256`
    -> `workspace[256..499 * N]`

  **问题**: cyclingRowOffset 在写前检查后是 0，但 Epilogue 实际写入 workspace[256..499]。
  cyclingRowOffset += 244 后变为 244。
  **下次聚合时 curM = 244，但实际写入的是 workspace[256..499]，
  而 FRDeterministicA5 会读 workspace[0..243]，地址不匹配！**

**这是一个新发现的 Critical 问题**：

`cyclingRowOffset` 追踪的是**连续写入的起始位置**，
但 Epilogue 的 `absRowBase = accumulatedGroupOffset_ + group 内 mOffset`。
当 cycling 发生在 group 中间时，group 内后半部分的 tile 的 mOffset 不从 0 开始，
导致 Epilogue 写入 workspace 的中间位置，而不是从 cyclingRowOffset 开始。

**根因**：Epilogue 的 `offsetM`（来自 `logitOffset`）是 group 内的绝对行偏移，
cycling reset 只改了 `accumulatedGroupOffset_`，
但 `offsetM = group 内 mOffset` 不受影响。

例如上面的 group 1 tile 1：mOffset = 256（group 内第 2 个 tile），
cycling 后 accumulatedGroupOffset_ = 0，
absRowBase = 0 + 256 = 256，不是从 0 开始。

---

## 5. 最终解决方案：引入 cyclingRowOffset 到 Epilogue

### 5.1 方案总结

上面的分析表明，**仅靠 LOGIT_OFFSETS reset 或 cyclingRowOffset 追踪都不够**，
因为 Epilogue 的 `offsetM` 是 group 内的 tile 偏移，不受 Kernel 层任何 reset 影响。

**根本解决方案**：在 Epilogue 的 `VectorSequentialWrite` 中，
不使用 `accumulatedGroupOffset_ + offsetM` 作为写地址，
而是改用 **Kernel 传入的 cycling 窗口内行偏移**。

具体做法：

1. **Kernel 层**维护 `cyclingRowOffset`（cycling 窗口内已用行数）
2. **Epilogue 层**新增 `SetCyclingBaseRow(uint64_t baseRow)` 方法，
   在每个 tile 调用前设置当前 tile 在 workspace 中的写入起始行
3. **VectorSequentialWrite** 使用 `cyclingBaseRow + i` 而不是
   `accumulatedGroupOffset_ + offsetM + i`

但这要求**每个 tile 调用前**都设置一次，性能开销较大。

### 5.2 更优方案：修改 Kernel 传给 Epilogue 的 epilogueOffset

回顾 Kernel 传给 Epilogue 的 `epilogueOffset`（6 元素 tuple）：

```cpp
epilogueOffset{nOffset,           // 0: yOffset
    biasOffset,                    // 1: bias 偏移
    x2ScaleOffset,                 // 2: x2Scale 偏移
    x1ScaleOffset,                 // 3: x1Scale 偏移
    mOffset,                       // 4: logit 偏移（group 内）
    mOffset}                       // 5: 也传 mOffset（用于 workspace 写地址）
```

Epilogue `operator()` 中：
```cpp
logitOffset = Get<4>(blockCoord) + mOffset;  // logit 读取偏移
// ...
VectorSequentialWrite(... logitOffset + i*32, yOffset, yLocal);
```

`VectorSequentialWrite` 中：
```cpp
absRowBase = accumulatedGroupOffset_ + offsetM;
// offsetM = logitOffset（来自 Epilogue operator() 的循环计算）
```

**关键洞察**：`offsetM` 传到 `VectorSequentialWrite` 时，
其值是 `logitOffset + i * SEQ_MAX_OUTPUT_M_UBS`，
其中 `logitOffset = Get<4>(blockCoord) + subBlockMOffset`
= group 内 mOffset + subBlock 内偏移。

**修改方案**：在 Kernel 层，计算 tile 在 **cycling 窗口内**的绝对行号，
替换 `mOffset` 传给 Epilogue 的第 5 元素。

但第 4 元素仍传 group 内 mOffset（因为 logit/rowIndex/x1Scale 需要正确的 group 内偏移来读取）。

```cpp
// Kernel tile 循环中:
int64_t y = Get<CYC_IDX_C_OFFSETS>(blockOffset_);
int64_t groupInnerMOffset = y / n;     // group 内偏移
int64_t nOffset = y - groupInnerMOffset * n;

// 计算 cycling 窗口内的 workspace 行偏移
// cyclingRowOffset 是窗口内已用行数
// 当前 tile 在 workspace 中的写入起始行 = cyclingRowOffset
// 而不是 groupInnerMOffset（group 内的绝对位置）

AscendC::Std::tuple<...> epilogueOffset{
    nOffset,
    Get<CYC_IDX_BIAS_OFFSETS>(blockOffset_),
    Get<CYC_IDX_X2SCALE_OFFSETS>(blockOffset_),
    Get<CYC_IDX_X1SCALE_OFFSETS>(blockOffset_),
    groupInnerMOffset,                     // 4: logit 读取用 group 内偏移
    static_cast<int64_t>(cyclingRowOffset) // 5: workspace 写地址用 cycling 窗口内偏移
};
```

同时，修改 Epilogue 的 `operator()` 中 `logitOffset` 的计算，
只使用第 4 元素（group 内偏移）来读 logit。

**但** `VectorSequentialWrite` 中的 `offsetM` 也需要改为使用第 5 元素（cycling 偏移）。

当前 Epilogue `operator()` 中调用 `VectorSequentialWrite` 时传的 offsetM 是 `logitOffset`：
```cpp
VectorSequentialWrite(singleN_, repeatTimesLine_,
    logitOffset + i * SEQ_CYCLING_MAX_OUTPUT_M_UBS, yOffset, yLocal);
```

这里 `logitOffset` 被同时用于 logit 读取和 workspace 写地址。
需要拆分。

### 5.3 最终方案：修改 Epilogue operator() 和 VectorSequentialWrite

**Epilogue operator() 修改**：

```cpp
// 新增：从 blockCoord 提取 workspace 写入基准行
uint64_t workspaceBaseRow = Get<5>(blockCoord);  // = cyclingRowOffset
uint64_t logitBase = Get<4>(blockCoord);          // = group 内 mOffset

// logit 读取偏移仍然使用 group 内偏移
uint64_t logitOffset = logitBase + mOffset;

// ...

// VectorSequentialWrite 调用改为传 workspace 基准行
for (uint32_t i = 0; i < loopNumY; i++) {
    // ...
    // 旧: VectorSequentialWrite(... logitOffset + i * 32, yOffset, yLocal)
    // 新: workspace 写入从 workspaceBaseRow 开始
    uint64_t wsRowOffset = workspaceBaseRow + i * SEQ_CYCLING_MAX_OUTPUT_M_UBS;
    VectorSequentialWrite(singleN_, repeatTimesLine_,
        wsRowOffset, yOffset, yLocal);
    // ...
}
```

**VectorSequentialWrite 修改**：

```cpp
// 旧:
uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;

// 新: offsetM 现在直接是 cycling 窗口内的行偏移（由 Kernel 计算好传入）
uint64_t absRowBase = offsetM;
```

**去掉 accumulatedGroupOffset_**：不再需要这个变量。
workspace 写地址完全由 Kernel 传入的 `cyclingRowOffset` 控制。

### 5.4 完整地址验证（使用最终方案）

场景：3 个 group，M = [500, 500, 500]，baseM = 256，windowSizeRows = 800

**Group 0 (M=500, LOGIT_OFFSETS=0):**
- `UpdateGlobalBuffer(cyclingRowOffset=0)`

Tile 0 (groupInnerMOffset=0, rows=256):
- 写前检查: 0 + 256 <= 800, 通过
- epilogueOffset[4] = 0, epilogueOffset[5] = 0 (cyclingRowOffset)
- Epilogue: workspaceBaseRow = 0, absRowBase = 0
  -> workspace[0..255 * N]
- logit 读: logitGlobal_[0] (base=0 + offset=0) -- 正确
- cyclingRowOffset = 256

Tile 1 (groupInnerMOffset=256, rows=244):
- 写前检查: 256 + 244 = 500 <= 800, 通过
- epilogueOffset[4] = 256, epilogueOffset[5] = 256 (cyclingRowOffset)
- Epilogue: workspaceBaseRow = 256, absRowBase = 256
  -> workspace[256..499 * N]
- logit 读: logitGlobal_[256] (base=0 + offset=256) -- 正确
- cyclingRowOffset = 500

**Group 1 (M=500, LOGIT_OFFSETS=500):**
- `UpdateGlobalBuffer(cyclingRowOffset=500)`

Tile 0 (groupInnerMOffset=0, rows=256):
- 写前检查: 500 + 256 = 756 <= 800, 通过
- epilogueOffset[4] = 0, epilogueOffset[5] = 500 (cyclingRowOffset)
- Epilogue: workspaceBaseRow = 500, absRowBase = 500
  -> workspace[500..755 * N]
- logit 读: logitGlobal_[0] (base=500 + offset=0) -> 全局偏移 500 -- 正确
- cyclingRowOffset = 756

Tile 1 (groupInnerMOffset=256, rows=244):
- 写前检查: 756 + 244 = 1000 > 800, **触发 cycling**
- 聚合: FRDeterministicA5(curM=756, globalRowOffset=0)
  - 读 workspace[0..755 * N]
  - rowIndex[0..755] -> yGm
- globalRowOffset = 756, cyclingRowOffset = 0
- UpdateGlobalBuffer(cyclingRowOffset=0)

  重新处理 Tile 1:
- 写前检查: 0 + 244 <= 800, 通过
- epilogueOffset[4] = 256, epilogueOffset[5] = 0 (cyclingRowOffset)
- Epilogue: workspaceBaseRow = 0
  **但 Epilogue 的 operator() 计算 wsRowOffset = 0 + i*32:**
  - 对于 i=0: wsRowOffset = 0, absRowBase = 0
    -> workspace[0..31 * N] -- 从 0 开始写

  **等等，这里 workspaceBaseRow = 0 是 cyclingRowOffset，
  但 tile 的行不是从 0 开始的，tile 有 244 行。
  wsRowOffset = 0 + 0*32 = 0 (前 32 行)
  wsRowOffset = 0 + 1*32 = 32 (第 33-64 行)
  ...
  wsRowOffset = 0 + 7*32 = 224 (第 225-244 行)
  全部在 workspace[0..243 * N] -- 正确!**

- logit 读: logitGlobal_[256] (base=500 + offset=256) -> 全局偏移 756 -- 正确
- cyclingRowOffset = 244

**Group 2 (M=500, LOGIT_OFFSETS=1000):**
- `UpdateGlobalBuffer(cyclingRowOffset=244)`

Tile 0 (groupInnerMOffset=0, rows=256):
- 写前检查: 244 + 256 = 500 <= 800, 通过
- epilogueOffset[5] = 244
- Epilogue: workspaceBaseRow = 244, absRowBase = 244
  -> workspace[244..499 * N]
- cyclingRowOffset = 500

Tile 1 (groupInnerMOffset=256, rows=244):
- 写前检查: 500 + 244 = 744 <= 800, 通过
- epilogueOffset[5] = 500
- Epilogue: workspaceBaseRow = 500
  -> workspace[500..743 * N]
- cyclingRowOffset = 744

**最终聚合**: FRDeterministicA5(curM=744, globalRowOffset=756)
- 读 workspace[0..743 * N]
- rowIndex[756..1499] -> yGm

**总验证**: 756 + 744 = 1500 = 500+500+500 = totalM -- 正确!

---

## 6. 逐文件改动（修订版）

### 6.1 修改：kernel_gmm_fr_deterministic_cycling.h

**路径**: `op_kernel/arch35/kernel_gmm_fr_deterministic_cycling.h`

**改动 1**: `UpdateGlobalBuffer` 签名和实现

```cpp
// 新增参数 cyclingBase
__aicore__ inline void UpdateGlobalBuffer(const Params &params, uint64_t cyclingBase)
{
    if ASCEND_IS_AIC {
        aGlobal_.SetGlobalBuffer(aGmAddr + Get<CYC_IDX_A_OFFSETS>(baseOffset_));
        bGlobal_.SetGlobalBuffer(bGmAddr + Get<CYC_IDX_B_OFFSETS>(baseOffset_));
    }
    if ASCEND_IS_AIV {
        AscendC::Coord<...> vecBaseOffset{
            0L,
            Get<CYC_IDX_BIAS_OFFSETS>(baseOffset_),
            Get<CYC_IDX_X2SCALE_OFFSETS>(baseOffset_),
            Get<CYC_IDX_X1SCALE_OFFSETS>(baseOffset_),
            Get<CYC_IDX_LOGIT_OFFSETS>(baseOffset_),       // 4: logit/rowIndex 全局偏移
            static_cast<int64_t>(cyclingBase)};             // 5: workspace 写入基准
        epilogueDequantOp_.UpdateGlobalAddr(vecBaseOffset);
    }
}
```

**改动 2**: 去掉 LOGIT_OFFSETS reset

不再需要 `Get<CYC_IDX_LOGIT_OFFSETS>(baseOffset_) -= accumulatedRows;`

**改动 3**: epilogueOffset 第 5 元素改为 cyclingRowOffset

```cpp
// 旧:
epilogueOffset{..., mOffset, mOffset};

// 新:
epilogueOffset{...,
    mOffset,                                // 4: logit 读取用 group 内偏移
    static_cast<int64_t>(cyclingRowOffset)}; // 5: workspace 写入用 cycling 窗口内偏移
```

**改动 4**: 写前检查（移到 tile 处理前）

```cpp
uint64_t tileRows = static_cast<uint64_t>(Get<MNK_M>(singleShape));

// 写前检查
if (cyclingRowOffset + tileRows > windowSizeRows) {
    // 聚合已写入的数据
    if ASCEND_IS_AIC { if (isVecSetSyncCom_) WaitForVector(); }
    SyncAll();
    if ASCEND_IS_AIV {
        FRDeterministicA5(... cyclingRowOffset ... globalRowOffset ...);
    }
    SyncAll();
    globalRowOffset += cyclingRowOffset;
    cyclingRowOffset = 0;
    isVecSetSyncCom_ = false;
    UpdateGlobalBuffer(params, cyclingRowOffset);
}

// 然后处理当前 tile
// AIC: MMAD
// AIV: Epilogue

cyclingRowOffset += tileRows;
```

### 6.2 修改：block_epilogue_dequant_sequential_write_cycling.h

**路径**: `op_kernel/arch35/block_epilogue_dequant_sequential_write_cycling.h`

**改动 1**: 去掉 `accumulatedGroupOffset_` 变量

不再需要此变量，workspace 写地址完全由 Kernel 传入。

**改动 2**: `UpdateGlobalAddr` 简化

不再设置 `accumulatedGroupOffset_`，只设置各 GlobalTensor 的 base：
`logitGlobal_` 和 `rowIndexGlobal_` 的 base 仍使用第 4 元素（LOGIT_OFFSETS）。

**改动 3**: `operator()` 中 workspace 写地址计算

```cpp
// 从 blockCoord 提取两个独立偏移
uint64_t logitBase = Get<SEQ_CYCLING_LOGIT_INDEXS>(blockCoord);    // 4: group 内 mOffset
uint64_t workspaceBaseRow = Get<5>(blockCoord);                     // 5: cyclingRowOffset
uint64_t logitOffset = logitBase + mOffset;

// ...
// VectorSequentialWrite 调用传 workspace 基准行
for (uint32_t i = 0; i < loopNumY; i++) {
    // ...
    uint64_t wsWriteOffset = workspaceBaseRow + i * SEQ_CYCLING_MAX_OUTPUT_M_UBS;
    VectorSequentialWrite(singleN_, repeatTimesLine_, wsWriteOffset, yOffset, yLocal);
}
```

**改动 4**: `VectorSequentialWrite` 简化

```cpp
// 旧:
uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;

// 新: offsetM 已是 cycling 窗口内的绝对行号（由 Kernel 计算）
uint64_t absRowBase = offsetM;
```

### 6.3 gmm_fr_deterministic_a5.h

不变（已有 `globalRowOffset` 参数）。

### 6.4 grouped_matmul_finalize_routing_quant_tiling.cpp

不变（删除溢出降级逻辑）。

### 6.5 grouped_matmul_finalize_routing_pertoken_dequant.h

不变。

---

## 7. 三个 Critical 问题的修复总结

| Critical | 根因 | 修复 |
|----------|------|------|
| 1: 跨 group 地址不匹配 | LOGIT_OFFSETS reset 方案依赖 cycling 恰好在 group 边界，实际在 tile 级 | 放弃 LOGIT_OFFSETS reset，改用独立 cyclingRowOffset 变量 |
| 2: group 内 offset 越界 | Epilogue 的 offsetM 是 group 内偏移，cycling reset 后 absRowBase 仍超过 windowSizeRows | Kernel 直接传入 cycling 窗口内偏移（epilogueOffset[5]=cyclingRowOffset），不依赖 group 内偏移 |
| 3: 写后检查 | 先写再检查，越界后才触发聚合 | 移到 tile 处理前检查（写前检查），确保当前 tile 不会超出 workspace |

---

## 8. SyncAll 时序

```
每次聚合的 SyncAll 调用:

1. AIC: WaitForVector()     -- CrossCore 确保最后一个 tile 的 AIV 完成
2. SyncAll()                 -- 全核同步，确保 workspace 写入完成
3. AIV: FRDeterministicA5()  -- 内部两次 SyncAll（AIC dummy + AIV 聚合）
   AIC: SyncAll() + SyncAll()  -- 匹配 AIV 的两次
   AIV: SyncAll() + [聚合] + SyncAll()

注意: 实际 SyncAll 配对需要 AIC/AIV 每条路径的调用次数完全相同。
```

---

## 9. 不修改的文件

| 文件 | 原因 |
|------|------|
| `common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` | 共享框架不动 |
| `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` | 共享 Epilogue 不动 |
| `tiling_data.h` | 已有字段，无需新增 |
| `block_prologue_finalize_routing.h` | Prologue 只在初始化阶段运行一次 |
| `block_scheduler_gmm_aswt_with_tail_split.h` | Scheduler 在 CyclingKernel 内正常使用 |

---

## 10. 改动文件汇总

| # | 文件 | 改动类型 | 改动量 |
|---|------|---------|--------|
| 1 | `op_kernel/arch35/kernel_gmm_fr_deterministic_cycling.h` | 修改 | ~40 行改动（epilogueOffset、写前检查、UpdateGlobalBuffer） |
| 2 | `op_kernel/arch35/block_epilogue_dequant_sequential_write_cycling.h` | 修改 | ~20 行改动（operator、VectorSequentialWrite、去掉 accumulatedGroupOffset_） |
| 3 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | 不变 | 已有 globalRowOffset |
| 4 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 修改 | ~5 行（删降级） |
| 5 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 不变 | 入口代码不变 |

**不修改任何 `common/cgmct/` 下的共享框架文件。**

---

## 11. 验证场景

| 场景 | 预期行为 |
|------|---------|
| 小规模（不溢出）: totalM=2048, N=1024 | 不触发 cycling，与当前行为一致 |
| Group 边界触发: 总行数恰好在 group 边界超过 windowSize | group 结束时触发聚合 |
| Group 中间触发: 单个 group 跨越 windowSize | group 中间 tile 触发聚合，cyclingRowOffset 重置 |
| 连续大 group: 每个 group 接近 windowSize | 每个 group 可能触发多次聚合 |
| 空 group: 某些 group M=0 | 正确跳过，cyclingRowOffset 不变 |
| 大规模多轮: totalM=32768, N=4096 | 多次 cycling，最终聚合正确 |
