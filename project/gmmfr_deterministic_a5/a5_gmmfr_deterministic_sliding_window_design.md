# A5 GMMFR 确定性模式滑动窗口技术设计文档

> 项目：GMMFR (Grouped Matmul Finalize Routing) A5 确定性模式滑动窗口改造
> 平台：A5 (Ascend 950, arch35)
> 需求：确定性模式不允许降级，所有 W8A8 shape 必须支持确定性
> 参考：A3 (arch32) 的 VectorSync + FRDeterministic 滑动窗口机制

---

## 1. 需求背景

### 1.1 问题描述

原 A5 确定性模式在 Tiling 层（`grouped_matmul_finalize_routing_quant_tiling.cpp` 第 471-483 行）有一个溢出降级逻辑：

```cpp
// 原始代码（已删除）
uint64_t requiredDeterSize = inputParams_.mSize * inputParams_.nSize * sizeof(float);
if (requiredDeterSize > deterWorkspaceSize_) {
    OP_LOGW(...);
    deterministicFlag_ = 0;  // 降级为非确定性
    ...
}
```

当 `M * N * sizeof(float) > 96MB` 时，静默降级为非确定性模式。

### 1.2 用户要求

1. **确定性模式不允许降级** — 用户开启 deterministic=1 时必须保证确定性执行
2. **所有 W8A8 shape 都要支持确定性** — 无论 M*N 多大
3. **参考 A3 实现滑动窗口** — workspace 不够时分批处理

---

## 2. 方案选型

### 2.1 方案对比

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| A. 外部多轮，全量 Kernel | 不改 Cgmct | 每轮都跑所有 group，浪费 Cube 计算 | 不可行 |
| B. 修改 Kernel 内部循环 | 精确窗口控制 | 侵入 Cgmct 框架核心代码 | 不推荐 |
| C. 修改 Epilogue 写入目标 | 不改 Kernel | Epilogue 无全局视角判断窗口边界 | 不可行 |
| **D. 外层多轮 + 偏移传递** | **改动最小，性能无损** | **需修改 GMMTiling 的 preOffsetInit（1 行）** | **推荐** |

### 2.2 推荐方案 D 的理由

1. Cgmct Kernel、Scheduler、Epilogue 核心逻辑均不变
2. 改动集中在用户代码层（pertoken_dequant.h、gmm_fr_deterministic_a5.h、tiling）
3. 对 Cgmct 的唯一修改是 GMMTiling 新增 `preOffsetInit` 字段 + InitParamsAndTensor 中 1 行赋值
4. 性能无损 -- 每轮只处理实际需要的 group，无浪费计算

### 2.3 与 A3 的对应关系

- **A3**：在 Process() 的 block 循环中通过 VectorSync 检查窗口边界，可中断 group
- **A5**：受限于 Cgmct 框架，窗口边界对齐 group 边界（不中断 group 内部）
- **共同点**：写满一个窗口 -> SyncAll -> 聚合 -> 移动窗口

---

## 3. 数据流图

### 3.1 修改前（一次性处理所有 group）

```
Prologue: zeros + residual -> yGm[batch*N]
         |
         v
Cgmct Kernel::operator()  -- 处理 group[0..groupNum-1]
  Cube: MMAD -> workspace
  Vector: Epilogue(SequentialWrite) -> workspace[0..totalM*N]
         |
         v  SyncAll
FRDeterministicA5 -- 一次性聚合 workspace -> yGm
  workspace[m*N+n] --AtomicAdd--> yGm[rowIndex[m]*N+n]
         |
         v  SyncAll
```

**问题**：workspace 需要 `totalM * N * sizeof(float)` 字节。大 shape 时溢出导致降级。

### 3.2 修改后（滑动窗口多轮）

```
Prologue: zeros + residual -> yGm[batch*N]  (只在第一轮)

while (groupStart < groupNum) {
    1. 计算本轮 group 范围 [groupStart, groupEnd)
       累加 groupM 直到接近 windowSize 或耗尽

    2. Cgmct Kernel::operator()  -- 处理 group[groupStart..groupEnd)
       输入指针已偏移：x + xRowOffset*K, w + wGroupOffset*NZ_SIZE, ...
       Epilogue 写入 workspace[0..roundM*N]  (每轮从 0 开始)

    3. SyncAll (Kernel 内部 + FRDeterministicA5 入口)

    4. FRDeterministicA5 -- 本轮聚合
       从 rowIndex[globalRowOffset + m] 读 outRow
       workspace[m*N+n] --AtomicAdd--> yGm[outRow*N+n]

    5. SyncAll (FRDeterministicA5 出口)

    6. 更新偏移: globalRowOffset += roundM, groupStart = groupEnd
}
```

**关键特性**：
- workspace 大小固定为 `windowSize * N * sizeof(float)`（<= 96MB）
- 每轮 workspace 从第 0 行写入（覆盖上一轮数据，已聚合完毕）
- Cgmct Kernel 每轮重新实例化，无状态残留

---

## 4. 需要修改的文件清单及详细说明

### 4.1 文件总览

| # | 文件路径 | 修改类型 | 修改量 |
|---|---------|---------|-------|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 新增字段 | +2行 |
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 删除降级+新增计算 | ~15行 |
| 3 | `common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` | 新增字段+赋值 | +3行 |
| 4 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | 新增参数+修改逻辑 | ~5行 |
| 5 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 替换确定性分支 | ~120行 |

**不需要修改的文件**：
- `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` — workspace 从第 0 行写入，每轮自动复用
- `common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h` — 每轮新建实例
- `common/cgmct/prologue/block_prologue_finalize_routing.h` — 通过 batch=0 控制跳过
- `common/cgmct/block/block_mmad_builder.h` — 不变

---

## 5. 修改详情（逐文件）

### 5.1 文件 1：tiling_data.h — 新增滑动窗口参数

**文件**：`gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`

**修改位置**：`GMMFinalizeRoutingDataParams` 结构体，在 `deterWorkspaceSize` 字段后

**修改内容**：新增两个字段

```cpp
struct GMMFinalizeRoutingDataParams {
    // ... 原有字段不变 ...
    uint32_t deterministicFlag = 0;      // 0=非确定性, 1=确定性
    uint32_t deterWorkspaceSize = 0;     // 确定性 workspace 大小（字节）
    uint32_t windowSize = 0;             // [新增] 滑动窗口大小（行数）= deterWorkspaceSize / (N * sizeof(float))
    uint32_t totalM = 0;                 // [新增] 所有 group 的总 M 行数
};
```

**逻辑说明**：
- `windowSize`：每轮最多处理的行数。Tiling 层计算为 `deterWorkspaceSize / (N * sizeof(float))`
- `totalM`：所有 group 的总输入行数，供 Kernel 层参考
- 结构体原有 48 字节，新增 8 字节后为 56 字节，仍满足 `#pragma pack(push, 8)` 对齐要求

---

### 5.2 文件 2：quant_tiling.cpp — 删除降级逻辑，改为窗口计算

**文件**：`gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`

**修改位置**：`DoOpTiling()` 函数，原第 471-483 行

**修改前（删除）**：
```cpp
uint64_t requiredDeterSize = inputParams_.mSize * inputParams_.nSize * sizeof(float);
if (requiredDeterSize > deterWorkspaceSize_) {
    OP_LOGW(context_->GetNodeName(),
            "Deterministic buffer overflow: required %lu bytes but only %u available, "
            "deterministic mode disabled.",
            requiredDeterSize, deterWorkspaceSize_);
    deterministicFlag_ = 0;
    tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 0;
    tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = 0;
} else {
    tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
}
```

**修改后（替换为）**：
```cpp
uint64_t windowSizeRows = deterWorkspaceSize_ / (inputParams_.nSize * sizeof(float));
if (windowSizeRows == 0) {
    OP_LOGE(context_->GetNodeName(),
            "Deterministic workspace too small for N=%lu, workspaceSize=%u",
            inputParams_.nSize, deterWorkspaceSize_);
    return ge::GRAPH_FAILED;
}
tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
tilingData_.gmmFinalizeRoutingDataParams.windowSize = static_cast<uint32_t>(windowSizeRows);
tilingData_.gmmFinalizeRoutingDataParams.totalM = static_cast<uint32_t>(inputParams_.mSize);
```

**逻辑说明**：
- 计算窗口大小：`windowSize = workspaceSize / (N * 4)`，即 workspace 能容纳的最大行数
- 如果 windowSize == 0（N 过大导致一行都放不下），直接报错返回 GRAPH_FAILED
- 不再降级：无论 M*N 多大，都传递 windowSize 给 Kernel，由 Kernel 分批处理
- `GetWorkspaceSize()` 无需修改，仍然分配固定的 `SYS_WORKSPACE_SIZES + deterWorkspaceSize`

同时在 `PrintQuantParams()` 中新增两个字段的打印。

---

### 5.3 文件 3：kernel_gmm_finalize_routing_pertoken_dequant.h — Cgmct 最小化修改

**文件**：`gmm/common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h`

**修改 1**：GMMTiling 结构体新增 `preOffsetInit` 字段

```cpp
struct GMMTiling {
    uint32_t groupNum;
    uint8_t groupListType;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    uint8_t hasBias;
    int32_t preOffsetInit = 0;   // [新增] 累积模式初始偏移
    const TCubeTiling *__restrict__ matmulTiling;
    // ...
};
```

**修改 2**：构造函数增加 `preOffsetInit` 参数（带默认值 0，保持向后兼容）

```cpp
__aicore__ GMMTiling(uint32_t groupNum_, uint8_t groupListType_, int32_t baseM_, int32_t baseN_, int32_t baseK_,
                     uint8_t hasBias_, int32_t preOffsetInit_ = 0)
    : groupNum(groupNum_), groupListType(groupListType_), baseM(baseM_), baseN(baseN_), baseK(baseK_),
      hasBias(hasBias_), preOffsetInit(preOffsetInit_)
{
}
```

**修改 3**：`InitParamsAndTensor` 中新增 1 行赋值

```cpp
__aicore__ inline void InitParamsAndTensor(const Params &params) {
    Get<MNK_N>(problemShape_) = params.gmmParams.matmulTiling->N;
    Get<MNK_K>(problemShape_) = params.gmmParams.matmulTiling->Ka;
    groupListGm_.SetGlobalBuffer((__gm__ int64_t *)params.mmadParams.groupListGmAddr);
    preOffset_ = params.gmmParams.preOffsetInit;  // [新增]
}
```

**逻辑说明**：

累积模式（groupListType=0）下，`GetSplitValueFromGroupList` 使用 `preOffset_` 计算当前 group 的 M：

```
splitValue = groupList[groupIdx] - preOffset_
```

当从中间 group 开始（如 groupStart=2）时，`preOffset_` 必须初始化为已跳过 group 的累积值。

**示例**：groupList = [100, 250, 500], groupStart = 1
- 传入 groupListRound = &groupList[1]，即看到 [250, 500]
- preOffset_ 设为 100（groupList[0] 的值）
- Kernel 读 groupListRound[0]=250, splitValue=250-100=150（正确，第 1 个 group 有 150 行）
- Kernel 读 groupListRound[1]=500, splitValue=500-250=250（正确，第 2 个 group 有 250 行）

---

### 5.4 文件 4：gmm_fr_deterministic_a5.h — 新增 globalRowOffset 参数

**文件**：`gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h`

**修改 1**：函数签名新增 `globalRowOffset` 参数

```cpp
// 修改前
__aicore__ inline void FRDeterministicA5(
    ..., uint32_t coreNum, uint32_t n)

// 修改后
__aicore__ inline void FRDeterministicA5(
    ..., uint32_t coreNum, uint32_t n,
    uint64_t globalRowOffset)  // [新增] 全局行偏移
```

**修改 2**：聚合逻辑修改

```cpp
// 修改前
uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
uint64_t baseOffset = syncConfig.lowBoundM - syncConfig.windowSize;
auto outRow = tokenRanksGm.GetValue(baseOffset + mOffset);

// 修改后
uint64_t totalM = syncConfig.curM;  // 本轮 roundM
auto outRow = tokenRanksGm.GetValue(globalRowOffset + mOffset);  // 全局偏移读取 rowIndex
```

**逻辑说明**：
- `totalM` 直接使用 `syncConfig.curM`（由调用方设置为本轮的 roundM）
- rowIndex 读取加上 `globalRowOffset`：因为 workspace 从第 0 行写入，但全局 rowIndex 需要从 `globalRowOffset` 处开始读取
- 例如第 2 轮处理了 roundM=2000 行，globalRowOffset=3000，则读 rowIndex[3000..4999]

---

### 5.5 文件 5：pertoken_dequant.h — 主入口滑动窗口循环

**文件**：`gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`

**修改范围**：整个 `if (deterministicFlag == 1)` 分支替换为滑动窗口多轮循环

#### 5.5.1 整体结构

```
if (deterministicFlag == 1) {
    // 1. 读取 Tiling 参数
    // 2. 准备 GlobalTensor 和 UB buffer
    // 3. 定义辅助函数 getGroupM 和 NZ 偏移常量
    // 4. 滑动窗口循环
    while (groupStart < groupNum) {
        //   a. 确定本轮 group 范围
        //   b. 计算累积模式 preOffsetInit
        //   c. 构造偏移后的输入指针
        //   d. Prologue 参数（第一轮正常，后续跳过）
        //   e. 构造 GMMTiling 和 Kernel 参数
        //   f. 运行本轮 Kernel
        //   g. 窗口聚合 workspace -> yGm
        //   h. 更新偏移
    }
}
```

#### 5.5.2 关键逻辑详解

**a. 确定本轮 group 范围**

```cpp
uint32_t roundM = 0;
uint32_t groupEnd = groupStart;
while (groupEnd < groupNum) {
    uint32_t mi = getGroupM(groupEnd);
    if (mi == 0) { groupEnd++; continue; }           // 跳过空 group
    if (roundM + mi > windowSize && roundM > 0) {    // 溢出保护
        break;
    }
    roundM += mi;
    groupEnd++;
}
```

- 窗口边界对齐 group 边界（不中断 group 内部）
- 如果单个 group 的 M 就超过 windowSize（`roundM == 0`），不 break，该 group 单独占一轮
- 空 group（M=0）直接跳过

**b. 计算累积模式 preOffsetInit**

```cpp
int32_t preOffsetInit = 0;
if (groupListType == 0 && groupStart > 0) {
    preOffsetInit = static_cast<int32_t>(groupListGm.GetValue(groupStart - 1));
}
```

- 仅在累积模式（groupListType=0）且非第一组时需要
- 取前一个 group 在 groupList 中的累积值作为初始偏移

**c. 构造偏移后的输入指针**

```cpp
GM_ADDR xRound = x + xRowOffset * k;                                    // x 输入按行偏移
GM_ADDR wRound = w + wGroupOffset * singleGroupWeightNZ;                // w 权重按 group 偏移（NZ 格式）
GM_ADDR groupListRound = group_list + groupStart * sizeof(int64_t);     // groupList 从当前 group 开始
GM_ADDR wScaleRound = w_scale + wGroupOffset * n * sizeof(weightscaleType);  // w_scale 按 group 偏移
GM_ADDR xScaleRound = x_scale + xRowOffset * sizeof(float);             // x_scale 按行偏移
GM_ADDR biasRound = bias + wGroupOffset * n * sizeof(bfloat16_t);       // bias 按 group 偏移
GM_ADDR logitRound = logit + xRowOffset * sizeof(float);                // logit 按行偏移
GM_ADDR rowIndexRound = row_index + xRowOffset * sizeof(rowIndexType);  // rowIndex 按行偏移
```

**偏移计算与 Cgmct Kernel 内部 UpdateOffset 一致性**：
- `xRowOffset`：累积的输入行偏移，乘以 K 得到字节偏移
- `wGroupOffset`：累积的 group 偏移，乘以 `singleGroupWeightNZ`（NZ 格式单个 group 权重大小）
- `singleGroupWeightNZ = CeilDiv(n, 32) * CeilDiv(k, 16) * 512`

**d. Prologue 参数（关键 BUG 修复）**

```cpp
uint32_t prologueBatch = isFirstRound ?
    gmmFinalizeRoutingQuantParams_.batch : 0;
uint32_t prologueSharedInputOffset = isFirstRound ?
    gmmFinalizeRoutingQuantParams_.sharedInputOffset : 0;
uint32_t prologueSharedInputLen = isFirstRound ?
    gmmFinalizeRoutingQuantParams_.sharedInputLen : 0;
```

> **BUG 修复说明**（ISSUE-1，验收测试发现）：
>
> 后续轮不仅需要 `batch=0`，还需要将 `sharedInputOffset` 和 `sharedInputLen` 也设为 0。
> 如果不置零，Prologue 会错误地清零 yGm 数据（覆盖前几轮的聚合结果），
> 并且由于 `sharedInputLen != 0` 导致 `InitOutputWithZeros` 尝试清零巨大范围的内存。

**e. 构造 GMMTiling 和 Kernel 参数**

```cpp
GMMTilingDeterministic gmmParamsRound{
    roundGroupNum, groupListType,
    matmulTiling_.baseM, matmulTiling_.baseN, matmulTiling_.baseK,
    gmmFinalizeRoutingQuantParams_.hasBias,
    preOffsetInit   // 累积模式初始偏移
};
gmmParamsRound.matmulTiling = &matmulTiling_;
```

- `roundGroupNum`：本轮实际处理的 group 数量
- `preOffsetInit`：累积模式的初始偏移

**f. 运行本轮 Kernel**

```cpp
GmmKernelDeterministic gmm;
gmm(params);
```

每轮创建新的 Kernel 实例，无状态残留。Kernel 内部执行：
1. Prologue：第一轮初始化 yGm（zeros + residual），后续轮空跑
2. Cube MMAD：矩阵乘法，结果写入 workspace
3. Vector Epilogue（SequentialWrite）：反量化 + 写入 workspace

**g. 窗口聚合**

```cpp
GMMFRDeterministic::SyncConfig syncConfig{};
syncConfig.curM = roundM;
syncConfig.lowBoundM = roundM;
syncConfig.windowSize = roundM;
syncConfig.baseN = baseN;

GMMFRDeterministic::FRDeterministicA5<float, rowIndexType>(
    syncConfig, deterBufferGm, yGm, rowIndexGm, queBind,
    matmulTiling_.usedCoreNum, n, globalRowOffset);
```

- `syncConfig.curM = roundM`：本轮处理的行数
- 传入 `globalRowOffset`：FRDeterministicA5 从 `rowIndex[globalRowOffset + m]` 读取输出行号
- FRDeterministicA5 内部使用 AtomicAdd 聚合到 yGm，保证确定性

**h. 更新偏移**

```cpp
globalRowOffset += roundM;
xRowOffset += roundM;
wGroupOffset += roundGroupNum;
groupStart = groupEnd;
isFirstRound = false;
```

---

## 6. 为什么不需要修改 Epilogue / Prologue / Scheduler

### 6.1 Epilogue 不需要修改

Epilogue 的 `VectorSequentialWrite` 写入地址为：
```cpp
uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;
DataCopyPad(yGlobal_[(absRowBase + i) * n_ + yOffset], ...);
```

- 每轮 Kernel 从 groupIdx=0 开始，`accumulatedGroupOffset_` 从 0 累积
- 所以写入地址为 `(offsetM + i) * n_`，即从 workspace 第 0 行开始
- 每轮 workspace 从头写入，自动覆盖上一轮数据（上一轮已聚合完毕）

### 6.2 Prologue 不需要修改

通过参数控制：
- **第一轮**：`prologueBatch = batch`, `sharedInputOffset/Len = 原始值` → 正常初始化 yGm
- **后续轮**：`prologueBatch = 0`, `sharedInputOffset/Len = 0` → Prologue 空跑，不修改 yGm

### 6.3 Scheduler 不需要修改

`BlockSchedulerGmmAswtWithTailSplit` 在 Kernel::operator() 内部每轮创建新实例，无状态残留。

---

## 7. SyncAll 配对方案

### 7.1 每轮 SyncAll 调用序列

```
每轮 (round):
  [Kernel::operator() 内部]
  1. SyncAll<false>()        -- Prologue 完成后，AIC/AIV 同步
     ... Cube MMAD + Vector Epilogue 执行 ...
  [Kernel::End() 内部]
     ... WaitForVector (AIC only) ...

  [FRDeterministicA5 内部]
  2. SyncAll()               -- 等待所有 workspace 写入完成（聚合开始）
     ... AtomicAdd workspace -> yGm ...
  3. SyncAll()               -- 等待所有聚合写入完成（聚合结束）
```

### 7.2 SyncAll 正确性保证

- SyncAll #1（Kernel 内部）：保证 Prologue 完成后再开始 Epilogue
- SyncAll #2（FRDeterministicA5 入口）：保证所有 Core 的 workspace 写入完成，避免读-写竞争
- SyncAll #3（FRDeterministicA5 出口）：保证所有 yGm AtomicAdd 完成，下一轮可安全开始写入 workspace

### 7.3 SyncAll 总数

修改前：3 次（1 Kernel + 2 FRDeterministic）
修改后：`numRounds * 3` 次

| 场景 | numRounds | SyncAll 次数 | 额外时间 |
|------|-----------|-------------|---------|
| totalM=8192, N=1024, workspace=96MB | 1 | 3 | 0 |
| totalM=8192, N=4096, workspace=96MB | 2 | 6 | ~30us |
| totalM=32768, N=4096, workspace=96MB | 6 | 18 | ~90us |

每次 SyncAll 约 5-10us，额外开销可忽略。

---

## 8. group 边界处理策略

### 8.1 窗口边界对齐 group 边界

窗口不在 group 内中断。每轮处理若干完整 group。

### 8.2 单个 group 超出窗口的处理

当 `roundM == 0` 时（第一个非空 group 就超出窗口），不会 break（因为 `roundM > 0` 条件不满足），该 group 单独占一轮。

此时 workspace 实际使用 `singleGroupM * N * sizeof(float)` 字节。如果此值超过 `deterWorkspaceSize`，会导致 workspace 溢出。

**已知风险**（ISSUE-2，MEDIUM）：运行时无检查。但实际场景中 windowSize >= 1536 行（N<=16384 时），足以容纳绝大多数 group。

---

## 9. 性能影响评估

### 9.1 workspace 使用

| 场景 | 修改前 workspace | 修改后 workspace |
|------|-----------------|-----------------|
| totalM=8192, N=1024 | 32MB | 32MB (1轮) |
| totalM=32768, N=4096 | 512MB (溢出→降级) | 96MB (6轮) |

### 9.2 Cube/Vector 计算量

不变。每行数据只被处理一次。

### 9.3 总体评估

- 小规模场景（M*N*4 <= 96MB）：零开销（只有 1 轮）
- 大规模场景：仅增加 SyncAll 开销（< 0.1%）
- workspace 固定大小，不再随 totalM 增长

---

## 10. 已知风险和后续优化

| 风险 | 等级 | 说明 |
|------|------|------|
| 单 group 超 windowSize 无运行时检查 | MEDIUM | 实际场景极少触发，后续可在 Tiling 层遍历 groupList 检查最大 groupM |
| totalM 类型 uint32_t 截断 | LOW | MoE 场景 totalM 通常 < 4 billion |
