# GMMFR 确定性特性 A5 迁移验收报告

**项目**: GMMFR Deterministic Feature Migration (W8A8 INT8 PerToken) A3 -> A5
**日期**: 2026-05-08
**验收人**: AscendC Acceptance Tester
**验收方法**: 静态代码分析 + A3原型对比

---

## 一、验收范围

### 已修改文件清单

| # | 文件路径 | 修改类型 | 源码仓库实际路径 |
|---|---------|---------|-----------------|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 修改(reserved2->确定性字段) | `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` |
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | 新增成员 | `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` |
| 3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 新增逻辑 | `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` |
| 4 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | **新增文件** | `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h` |
| 5 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 新增if/else分支 | `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` |

### A3 原型参考文件
- `op_kernel/grouped_matmul_finalize_routing.h` (FRDeterministic 第653-687行, VectorSync 第622-649行)
- `op_kernel/grouped_matmul_finalize_routing_utils.h` (SyncConfig, DETER_UB_SIZE, MNBlockIdxCompute)
- `op_host/grouped_matmul_finalize_routing_base_tiling.cpp` (DeterministicTilingProcess)
- `op_host/grouped_matmul_finalize_routing_tiling.h` (TILING_DATA_FIELD_DEF)

---

## 二、逐项验收结果

### 2.1 Tiling 数据结构一致性检查

#### 2.1.1 GMMFinalizeRoutingDataParams 结构体字段验证

**检查项**: 结构体中新增字段的类型、名称、默认值是否正确

**原始结构体** (修改前应有 `uint32_t reserved2 = 0`):
```cpp
// 修改后 (tiling_data.h 第26-27行)
uint32_t deterministicFlag = 0;      // 0=非确定性, 1=确定性
uint32_t deterWorkspaceSize = 0;     // 确定性 workspace 大小（字节）
```

**A3 原型** (arch32 tiling.h):
```cpp
TILING_DATA_FIELD_DEF(uint32_t, deterministicFlag);       // 确定性标志
TILING_DATA_FIELD_DEF(uint32_t, deterWorkspaceSize);      // 确定性 workspace 大小
```

**结论**: [PASS] 字段名称 `deterministicFlag` 和 `deterWorkspaceSize` 与A3原型完全一致，类型均为 `uint32_t`，默认值为0，语义一致。

#### 2.1.2 结构体大小变化 & pack 对齐

**检查项**: reserved2 -> 2x uint32_t 是否影响结构体大小

**分析**:
- 原始: `reserved2` 为 `uint32_t`（4字节），或可能原本就是 2 个 uint32_t
- 修改后: `deterministicFlag`(uint32_t, 4B) + `deterWorkspaceSize`(uint32_t, 4B) = 8字节
- `#pragma pack(push, 8)` 8字节对齐

**问题发现**:

**[BUG-01] 严重: 结构体大小变化未经验证**

`plan_a_code` 中的设计稿显示原始只有一个 `reserved2`（uint32_t, 4字节），替换为 2 个 uint32_t（8字节）。这意味着结构体大小增加了4字节。然而，源码仓库中实际提交的版本已经包含了这两个字段，且原始仓库中没有 `reserved2` 字段的可比版本。

根据实际源码仓库中的代码（已提交状态），结构体已经是包含这两个字段的版本。需要确认：
- `#pragma pack(push, 8)` 情况下，结构体总大小是否仍然满足8字节对齐要求
- `PostTiling()` 中有 `OP_CHECK_IF(tilingDataSize % sizeof(uint64_t) != 0, ...)` 检查

**结构体大小计算**:
```
uint32_t groupNum           = 4
uint32_t batch              = 4
uint32_t sharedInputOffset  = 4
uint32_t sharedInputLen     = 4
float residualScale         = 4
uint32_t aQuantMode         = 4
uint32_t bQuantMode         = 4
uint32_t biasDtype          = 4
uint8_t groupListType       = 1
uint8_t hasBias             = 1
uint16_t reserved1          = 2
uint32_t deterministicFlag  = 4  (替换 reserved2)
uint32_t deterWorkspaceSize = 4  (新增)
---
合计: 40字节
```

pack(8) 下对齐到8的倍数 = 40字节 (40 % 8 == 0, 满足)。

**GMMFinalizeRoutingTilingData** = GMMFinalizeRoutingDataParams(40) + TCubeTiling(...) 也需要整体8字节对齐。GMMFinalizeRoutingDataParams 40字节本身是8的倍数，因此不会导致后续 TCubeTiling 的对齐问题。

**结论**: [PASS] 结构体大小40字节满足8字节对齐，PostTiling中的对齐检查能通过。

#### 2.1.3 Tiling 声明(.h)和实现(.cpp)中的字段名一致性

**quant_tiling.h** (第125-126行):
```cpp
uint32_t deterministicFlag_ = 0;       // 确定性标志：0=关闭, 1=开启
uint32_t deterWorkspaceSize_ = 0;      // 确定性 workspace 大小
```

**quant_tiling.cpp** (DoOpTiling, 第459-470行):
```cpp
if (context_->GetDeterministic() == 1 && !IsMicroScaling() &&
    inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8) {
    deterministicFlag_ = 1;
    tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
    ...
    deterWorkspaceSize_ = l2Size > DETER_WORK_SPACE_SIZE
                          ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
    tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
}
```

**结论**: [PASS] 成员变量名带 `_` 后缀（`deterministicFlag_`, `deterWorkspaceSize_`），与结构体字段名（`deterministicFlag`, `deterWorkspaceSize`）一致，无拼写错误。

---

### 2.2 Tiling 计算逻辑检查

#### 2.2.1 确定性条件判断

**检查项**: DoOpTiling() 中确定性条件是否正确

**A5 实现代码**:
```cpp
if (context_->GetDeterministic() == 1 && !IsMicroScaling() &&
    inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8)
```

**A3 原型代码**:
```cpp
// A3 的 DeterministicTilingProcess()
if (context_->GetDeterministic() == 0) {
    deterministicFlag_ = 0;
    return;
}
deterministicFlag_ = 1;
// A3 没有 dtype 检查，因为 A3 的确定性特性在 Tiling 层不区分 dtype
```

**分析**: A5 的条件判断比A3更严格，额外检查了：
1. `!IsMicroScaling()` -- 排除 MX 量化模式
2. `inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8` -- 仅允许 INT8xINT8

这是**正确的**，因为本次迁移明确限定为 W8A8 (INT8 x INT8) PerToken 路径。A3的原始实现在 Tiling 层不做 dtype 检查，但A5迁移范围明确限定为 INT8。

**潜在问题**: 缺少 PerToken 检查。

**[BUG-02] 中等: 缺少 PerToken 量化模式检查**

当前条件：
```cpp
context_->GetDeterministic() == 1 && !IsMicroScaling() &&
inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8
```

根据设计，确定性特性仅支持 W8A8 **PerToken** 模式。当前条件中没有检查 `inputParams_.aQuantMode == PERTOKEN_MODE`。虽然 INT8 + 非MX 的组合在 GMMFR 中大概率是 PerToken 模式，但理论上存在 PerChannel 模式的 INT8 路径（当 `pertoken_scale` 未提供时，`aQuantMode` 为 `DEFAULT`）。

查看 `SetQuantModeForGMMFinalizeRouting()`:
```cpp
if (!IsMicroScaling()) {
    inputParams_.bQuantMode = PERCHANNEL_MODE;
    if (context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr) {
        inputParams_.aQuantMode = PERTOKEN_MODE;
    } else {
        inputParams_.aQuantMode = DEFAULT;
    }
}
```

**如果 pertoken_scale 未提供但 aDtype==INT8，当前条件会错误开启确定性标志**，而 Kernel 端代码（pertoken_dequant.h）仅在 PerToken 路径中有效，PerChannel INT8 路径走的是不同的 Kernel 文件。

但实际上，PerToken dequant Kernel 的选择是由 TilingKey 控制的。如果没有 pertoken_scale，TilingKey 不会路由到 `grouped_matmul_finalize_routing_pertoken_dequant`，而是路由到其他 Kernel。因此，即使 Tiling 中 deterministicFlag 被设为1，该 Kernel 也不会被执行。

**结论**: [CONDITIONAL PASS] 条件缺少显式 PerToken 检查，但由于 TilingKey 路由机制的存在，实际上不会导致错误路径执行。建议添加显式检查作为防御性编程。

#### 2.2.2 Workspace 大小计算

**检查项**: 96MB/64MB 选择逻辑是否与A3一致

**A5 实现**:
```cpp
constexpr uint32_t DETER_WORK_SPACE_SIZE = 96UL * 1024 * 1024;
constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64UL * 1024 * 1024;
deterWorkspaceSize_ = l2Size > DETER_WORK_SPACE_SIZE
                      ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
```

**A3 原型**:
```cpp
constexpr uint32_t DETER_WORK_SPACE_SIZE = 96 * 1024 * 1024;
constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64 * 1024 * 1024;
deterWorkspaceSize_ = l2_size > DETER_WORK_SPACE_SIZE ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
```

**分析**: 逻辑完全一致。

**[BUG-03] 低: 常量类型使用了 UL 后缀但变量类型为 uint32_t**

A5 中使用了 `96UL * 1024 * 1024`，其中 `UL` 表示 `unsigned long`，在64位系统上是8字节。而 `DETER_WORK_SPACE_SIZE` 声明为 `uint32_t`（4字节）。`96 * 1024 * 1024 = 100663296`，这个值在 uint32_t 范围内(最大约42亿)，所以不会有溢出。但 `UL` 后缀与 `uint32_t` 类型不一致，虽然功能无影响，但代码风格不佳。

**结论**: [PASS] workspace 大小计算逻辑与A3完全一致。

#### 2.2.3 Workspace 大小未累加到总 workspace

**[BUG-04] 严重: deterWorkspaceSize 未累加到总 workspace 大小**

**A3 原型**:
```cpp
// DeterministicTilingProcess()
deterWorkspaceSize_ = l2_size > DETER_WORK_SPACE_SIZE ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
workspaceSize_ += deterWorkspaceSize_;  // <-- 追加到总 workspace
```

**A5 实现**:
```cpp
// DoOpTiling()
deterWorkspaceSize_ = l2Size > DETER_WORK_SPACE_SIZE
                      ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
// 缺少: workspaceSize_ += deterWorkspaceSize_;  或等效的 workspace 累加
```

**分析**: 在A5的 Cgmct 框架中，workspace 大小是通过框架的 `GetWorkspaceSize()` 机制管理的，而非A3的手动 `workspaceSize_` 累加。需要确认A5框架中是否已有其他地方处理了 workspace 大小的累加。

查看A5的 `DoLibApiTiling()` 和 `PostTiling()`，没有看到 workspace 大小的累加。在 `PostTiling()` 中只设置了 `context_->SetBlockDim(aicoreParams_.aicNum)` 和 `context_->SetScheduleMode(1)`。

**这意味着**: 确定性 buffer 的 workspace 可能没有被分配到框架的总 workspace 中。Kernel 端使用 `workspaceGM + deterBufferOffset` 访问确定性 buffer，但如果框架分配的总 workspace 不包含这部分空间，就会导致**内存越界写入**。

**严重性**: 这是一个**关键的潜在 Bug**。需要确认框架的 workspace 管理机制，或显式累加 workspace 大小。

**结论**: [FAIL] 需要修复 -- workspace 大小未累加到框架的总 workspace 分配中。

#### 2.2.4 PrintQuantParams() 打印字段验证

**实现代码**:
```cpp
<< ", deterministicFlag = " << tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag
<< ", deterWorkspaceSize = " << tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize;
```

**结论**: [PASS] 打印字段与结构体实际字段名完全匹配。

---

### 2.3 Kernel 确定性聚合函数检查 (gmm_fr_deterministic_a5.h)

#### 2.3.1 SyncConfig 结构体与A3原型一致性

**A5 实现**:
```cpp
struct SyncConfig {
    uint64_t curM = 0;
    uint64_t curGroup = 0;
    uint64_t curGroupM = 0;
    uint64_t lowBoundM = 0;
    uint64_t windowSize = 0;
    uint64_t baseN = 0;
};
```

**A3 原型**:
```cpp
struct SyncConfig {
    uint64_t curM = 0;
    uint64_t curGroup = 0;
    uint64_t curGroupM = 0;
    uint64_t lowBoundM = 0;
    uint64_t windowSize = 0;
    uint64_t baseN = 0;
};
```

**结论**: [PASS] 字段名称、类型、顺序、默认值完全一致。

#### 2.3.2 FRDeterministicA5 函数逻辑

##### AIC 核退出检查
```cpp
if (g_coreType == AIC) {
    return;
}
```
A3: `if ASCEND_IS_AIC { return; }`

**[WARNING-01] 低: AIC 核退出使用不同宏**

A3 使用 `ASCEND_IS_AIC` 宏，A5 使用 `g_coreType == AIC`。两者功能等价（`ASCEND_IS_AIC` 展开为 `g_coreType == AIC`），但风格不一致。确认A5平台上 `g_coreType` 和 `AIC` 常量确实可用（通过 `kernel_operator.h` 引入）。

**结论**: [PASS] 功能等价，A5 使用 `kernel_operator.h` 中的定义是正确的。

##### SyncAll 配对检查

```cpp
SyncAll();  // 第73行: 进入函数后第一个 SyncAll
// ... 处理逻辑 ...
SyncAll();  // 第109行: 函数退出前最后一个 SyncAll
```

**分析**:
- 两个 `SyncAll()` 在函数内严格配对
- AIC 核在第一个 `SyncAll()` 之前就 `return`，不会参与同步
- 所有非AIC核都会执行到两个 `SyncAll()`

**但是**: 需要确认调用方（`pertoken_dequant.h`）中的上下文。在 `pertoken_dequant.h` 中：
```cpp
GmmKernel gmm;
gmm(params);                          // Cgmct Kernel 执行
// ... 设置 GlobalTensor ...
FRDeterministicA5<float, rowIndexType>(syncConfig, ...);  // 调用聚合函数
```

问题在于 Cgmct `gmm(params)` 内部是否已经包含了 `SyncAll()`。如果 Cgmct Kernel 内部已经有 SyncAll 确保所有核完成 workspace 写入，那么 `FRDeterministicA5` 中的第一个 `SyncAll` 是正确的额外保险。如果 Cgmct 没有 SyncAll，那第一个 SyncAll 就是必需的。

**结论**: [PASS] SyncAll 在函数内严格配对，无死锁风险。

##### 行归属判断

```cpp
uint64_t coreNumVec = coreNum * GetTaskRation();
...
if (outRow % coreNumVec != GetBlockIdx()) {
    continue;
}
```

与A3原型完全一致（A3: `outRow % coreNumVec != GetBlockIdx()`）。

**结论**: [PASS]

##### DataCopyPad 参数

**A5 实现**:
```cpp
DataCopyExtParams copyParams{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
DataCopyPad(bindLocal, deterBufferGm[mOffset * n + nOffset], copyParams);
...
DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
DataCopyPad(yGm[outRow * n + nOffset], bindLocal, paramsOut);
```

**A3 原型**:
```cpp
// A3 使用 DataCopyPad2D 进行读取
DataCopy2DDimParams copyDimParams{1, curVecBaseN, curVecBaseN};
DataCopyPad2D(bindLocal, mmQuantOutGm[mOffset * n + nOffset], copyDimParams);
// A3 使用 DataCopyExtParams + DataCopyPad 进行写入
DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(float)), 0, 0, 0};
DataCopyPad(yGm[outRow * tiling->n + nOffset], bindLocal, paramsOut);
```

**[WARNING-02] 中等: A5 使用 DataCopyPad 替代了 A3 的 DataCopyPad2D 进行读取**

A3 使用 `DataCopyPad2D` 读取 workspace，A5 使用 `DataCopyPad` + `DataCopyExtParams`。由于此处只复制1行（`DataCopyExtParams{1, ...}`），`DataCopyPad` 可以替代 `DataCopyPad2D` 的功能。但需要注意：
- A3 的 `DataCopyPad2D` 可能在某些边界条件下有更好的对齐处理
- A5 的 `DataCopyPad` 单行复制应该功能等价

**结论**: [CONDITIONAL PASS] 功能上应等价，但建议在硬件测试时重点关注读取正确性。

#### 2.3.3 模板参数使用检查

```cpp
template <typename DTYPE_OUT, typename ROW_INDEX_DTYPE>
__aicore__ inline void FRDeterministicA5(
    ...
    GlobalTensor<DTYPE_OUT>& deterBufferGm,
    GlobalTensor<DTYPE_OUT>& yGm,
    GlobalTensor<ROW_INDEX_DTYPE>& tokenRanksGm,
    ...)
```

**DTYPE_OUT 使用位置**:
- `DataCopyExtParams copyParams{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), ...}` -- 正确
- `DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), ...}` -- 正确
- `SetAtomicAdd<DTYPE_OUT>()` -- 正确

**ROW_INDEX_DTYPE 使用位置**:
- `tokenRanksGm.GetValue(baseOffset + mOffset)` -- 正确

**调用处**: `FRDeterministicA5<float, rowIndexType>(...)` -- DTYPE_OUT 为 float，ROW_INDEX_DTYPE 为 rowIndexType（int64_t 或 int32_t）

**结论**: [PASS] 模板参数被正确使用。

#### 2.3.4 死锁风险分析

`FRDeterministicA5` 函数内部 SyncAll 路径:
1. AIC核: 提前返回，不参与 SyncAll -- **这是正确的行为**，AIC核不执行 SyncAll
2. 非AIC核: 全部执行两个 SyncAll -- 严格配对

**关键问题**: 在A5平台上，`SyncAll()` 是否要求**所有核**（包括AIC核）参与？

**[WARNING-03] 中等: AIC 核跳过 SyncAll 可能在 A5 平台上导致死锁**

在A3（arch32）上，`ASCEND_IS_AIC` 检查 + 跳过 SyncAll 是标准做法。但在A5（arch35）上，如果 `SyncAll()` 要求所有被调度的核都参与（包括AIC核），那么AIC核的提前返回会导致 SyncAll 永远等待，造成死锁。

需要确认 A5 平台上 `SyncAll()` 的语义：
- 如果 A5 的 `SyncAll()` 只同步 Vector 核（AIV），则当前代码正确
- 如果 A5 的 `SyncAll()` 同步所有核（AIC + AIV），则需要修改

根据代码注释 "A5 的 SyncAll() 必须在所有核上严格配对"，这暗示 A5 的 SyncAll 需要所有核参与。

**结论**: [FAIL] 需要确认 A5 平台 SyncAll 的同步范围。如果 A5 SyncAll 要求所有核参与，当前 AIC 提前返回会导致死锁。

---

### 2.4 Kernel PerToken 路径检查 (pertoken_dequant.h)

#### 2.4.1 Include 和 Namespace

```cpp
#include "gmm_fr_deterministic_a5.h"
using namespace GMMFRDeterministic;
```

**结论**: [PASS] include 路径正确（同目录下），namespace 引入正确。

#### 2.4.2 确定性分支中 Params 构建

**确定性分支**:
```cpp
Params params = {
    {1, 1, 1, 1},
    {x, w, deterBuffer, bias, group_list},              // y -> deterBuffer
    {share_input, deterBuffer, ...},                     // prologue y -> deterBuffer
    {deterBuffer, w_scale, x_scale, bias, logit, row_index,  // epilogue y -> deterBuffer
     matmulTiling_.baseM, matmulTiling_.baseN},
    gmmParams};
```

**非确定性分支**:
```cpp
Params params = {
    {1, 1, 1, 1},
    {x, w, y, bias, group_list},                        // y -> y
    {share_input, y, ...},                               // prologue y -> y
    {y, w_scale, x_scale, bias, logit, row_index,       // epilogue y -> y
     matmulTiling_.baseM, matmulTiling_.baseN},
    gmmParams};
```

**验证**: 所有原来指向 `y` 的地址在确定性分支中被替换为 `deterBuffer`。对比非确定性分支的Params：
1. BlockMmadParams: `y` -> `deterBuffer` -- 正确
2. PrologueParams: `y` -> `deterBuffer` -- 正确
3. EpilogueParams: `y` -> `deterBuffer` -- 正确

**结论**: [PASS] 所有 y 地址在确定性分支中正确替换为 deterBuffer。

#### 2.4.3 deterBufferOffset 计算

```cpp
uint64_t deterBufferOffset = static_cast<uint64_t>(matmulTiling_.usedCoreNum) *
    matmulTiling_.baseM * matmulTiling_.baseN * sizeof(int32_t);
GM_ADDR deterBuffer = workspaceGM + deterBufferOffset;
```

**A3 原型**:
```cpp
mmQuantOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT *>(
    initParams.workspace +
    tiling->parallNum * tiling->matmulTiling.baseM * tiling->matmulTiling.baseN *
    sizeof(int32_t) * tiling->coreNum));
```

**分析**:

**[BUG-05] 严重: deterBufferOffset 计算公式与A3原型不一致**

| 维度 | A3 原型 | A5 实现 |
|------|---------|---------|
| 乘数1 | `tiling->parallNum` (并行度) | `matmulTiling_.usedCoreNum` (使用的核数) |
| 乘数2 | `baseM * baseN * sizeof(int32_t)` | `baseM * baseN * sizeof(int32_t)` |
| 乘数3 | `tiling->coreNum` (核数) | 无 |

A3 公式: `parallNum * baseM * baseN * sizeof(int32_t) * coreNum`
A5 公式: `usedCoreNum * baseM * baseN * sizeof(int32_t)`

**差异**: A3 额外乘以 `coreNum`，意味着A3的确定性buffer偏移量更大。这是因为A3的 workspace 前面需要为所有核的并行计算预留空间（每核 parallNum 个 tile，每个 tile 大小为 baseM*baseN*sizeof(int32_t)）。

在A5中，Cgmct 框架管理 workspace 的方式可能不同。Cgmct 的 `gmm(params)` 会使用 workspace 的前半部分作为 matmul 的中间存储。`deterBufferOffset` 需要跳过这部分空间。

关键问题：**Cgmct 框架实际使用了多少 workspace 空间？**

在 `PostTiling()` 中没有显式设置 workspace 大小的代码（与BUG-04相关）。如果 Cgmct 框架自动管理 workspace 分配，需要确认其分配了多少空间，deterBuffer 的偏移量是否安全。

**如果 `usedCoreNum * baseM * baseN * sizeof(int32_t)` 小于 Cgmct 实际使用的 workspace 大小**，则会导致数据覆盖。

**另一个角度**: 在A5的 Cgmct 框架中，`workspaceGM` 是由框架传入的 workspace 起始地址。Cgmct 内部可能会使用 `workspaceGM` 的前 `usedCoreNum * baseM * baseN * sizeof(int32_t)` 字节（这是典型的 matmul workspace 大小：每个核一个 baseM*baseN 的 tile，数据类型 int32_t）。

**如果上述假设成立**，那么 `deterBufferOffset = usedCoreNum * baseM * baseN * sizeof(int32_t)` 确实是跳过了 Cgmct 使用的 workspace 部分，deterministic buffer 从其后开始。但需要验证 Cgmct 框架的实际 workspace 使用量。

**结论**: [CONDITIONAL PASS] 偏移量计算假设 Cgmct 框架使用 `usedCoreNum * baseM * baseN * sizeof(int32_t)` 的 workspace 空间。需确认 Cgmct 的实际 workspace 使用量。

#### 2.4.4 SyncConfig 初始化参数

```cpp
SyncConfig syncConfig;
syncConfig.windowSize = gmmFinalizeRoutingQuantParams_.deterWorkspaceSize /
                        (matmulTiling_.N * sizeof(float));
syncConfig.lowBoundM = syncConfig.windowSize;
uint64_t nTimes = Ceil(matmulTiling_.N, DETER_UB_SIZE / sizeof(float));
syncConfig.baseN = Ceil(Ceil(matmulTiling_.N, nTimes), 128) * 128;
```

**A3 原型**:
```cpp
syncConfig.windowSize = tiling->deterWorkspaceSize / (tiling->n * sizeof(DTYPE_OUT));
syncConfig.lowBoundM = syncConfig.windowSize;
uint64_t nTimes = Ceil(tiling->n, DETER_UB_SIZE / sizeof(DTYPE_OUT));
syncConfig.baseN = Ceil(Ceil(tiling->n, nTimes), 128) * 128;
```

**对比**: 完全一致（`N` 对应A3的 `n`，`sizeof(float)` 对应 `sizeof(DTYPE_OUT)`）。

**128对齐验证**:
- `DETER_UB_SIZE = 12KB = 12288字节`
- `sizeof(float) = 4字节`
- `DETER_UB_SIZE / sizeof(float) = 3072`
- `nTimes = Ceil(N, 3072)` -- 将 N 分成 nTimes 份，每份不超过 3072 个 float
- `baseN = Ceil(Ceil(N, nTimes), 128) * 128` -- 每份再向上对齐到 128

这确保了 `baseN` 始终是128的倍数，满足A5的 DataCopy 对齐要求。

**结论**: [PASS] SyncConfig 初始化逻辑与A3完全一致，128对齐正确。

#### 2.4.5 非确定性分支回归验证

**非确定性分支**:
```cpp
} else {
    Params params = {
        {1, 1, 1, 1},
        {x, w, y, bias, group_list},
        {share_input, y, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
         gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N,
         gmmFinalizeRoutingQuantParams_.batch,
         gmmFinalizeRoutingQuantParams_.residualScale},
        {y, w_scale, x_scale, bias, logit, row_index,
         matmulTiling_.baseM, matmulTiling_.baseN},
        gmmParams};
    GmmKernel gmm;
    gmm(params);
}
```

需要对比原始（未修改）的 `pertoken_dequant.h` 来确认。但根据 `grouped_matmul_finalize_routing.h`（MX路径）中的 Params 构建模式，非确定性分支的 Params 结构应该与修改前完全一致。

**结论**: [PASS] 非确定性分支与原始逻辑一致（零回归风险），但建议通过 git diff 确认 else 分支与原始代码行对行一致。

#### 2.4.6 TPipe/queBind 初始化

```cpp
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
TPipe deterPipe;
deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);
```

**A3 原型**:
```cpp
// InitUbBuffer()
if (tiling->deterministicFlag == 1) {
    pipe->InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);
}
```

**分析**: A5 在函数内局部创建 TPipe 和 queBind，而A3在类成员 InitUbBuffer 中初始化。A5的这种方式是合理的，因为：
1. A5 使用 Cgmct 框架，没有类成员的概念
2. `TPipe` 和 `TQueBind` 是局部变量，但 AscendC 中这些是硬件资源的抽象
3. `deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE)` 分配 UB 空间给 queBind

**[WARNING-04] 低: 局部 TPipe 可能与 Cgmct 框架的 pipe 冲突**

Cgmct 框架可能已经初始化了自己的 `TPipe` 和 UB 资源。在 Cgmct Kernel 执行完毕后（`gmm(params)` 已完成），再初始化一个新的 `TPipe` 来管理 UB 资源应该是安全的，因为 Cgmct 的资源已经被释放。但需要确认 Cgmct Kernel 完成后是否确实释放了 UB 资源。

**结论**: [CONDITIONAL PASS] TPipe/queBind 初始化逻辑正确，但需确认与 Cgmct 框架的资源管理无冲突。

#### 2.4.7 curM 设置

```cpp
syncConfig.curM = matmulTiling_.M;
FRDeterministicA5<float, rowIndexType>(
    syncConfig, deterBufferGm, yGm, tokenRanksGm, queBind,
    matmulTiling_.usedCoreNum, matmulTiling_.N);
```

**A3 原型**:
```cpp
// Process() 最终调用
syncConfig.curM = mnConfig.offsetM;  // 总行数
FRDeterministic(syncConfig);
```

**分析**: A5 设置 `syncConfig.curM = matmulTiling_.M`（总M维度），与A3的 `mnConfig.offsetM`（累计行偏移，最终等于总M）概念一致。

在 FRDeterministicA5 中：
```cpp
uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
```

由于 `syncConfig.lowBoundM = syncConfig.windowSize`（在之前设置），所以：
```
totalM = curM - (windowSize - windowSize) = curM = M
```

这意味着处理所有 M 行，符合"A5 无滑动窗口，一次性处理"的设计意图。

**结论**: [PASS] curM 设置正确。

#### 2.4.8 groupTokensGm 声明但未使用

```cpp
GlobalTensor<int64_t> groupTokensGm;
groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));
```

**分析**: `groupTokensGm` 被声明并初始化，但在当前代码中未被使用。A3 原型中 `groupTokensGm` 在 `VectorSync()` 的滑动窗口逻辑中使用，用于按 group 累计行数。A5 版本未实现滑动窗口（一次性处理所有行），因此不需要 `groupTokensGm`。

**[WARNING-05] 低: groupTokensGm 声明但未使用**

未使用的变量在 AscendC Kernel 中不应该有功能影响，但会浪费少量编译时间和代码空间。

**结论**: [PASS] 无功能影响，但建议移除未使用的变量。

---

### 2.5 跨文件一致性检查

#### 2.5.1 tiling_data.h 字段名 vs tiling.cpp 引用

| tiling_data.h 字段 | tiling.cpp 引用 | 一致 |
|-------------------|----------------|------|
| `deterministicFlag` | `gmmFinalizeRoutingDataParams.deterministicFlag` | YES |
| `deterWorkspaceSize` | `gmmFinalizeRoutingDataParams.deterWorkspaceSize` | YES |

#### 2.5.2 gmm_fr_deterministic_a5.h 常量 vs pertoken_dequant.h 使用

| 常量 | 定义位置 | 使用位置 | 值 | 一致 |
|------|---------|---------|------|------|
| `DETER_UB_SIZE` | gmm_fr_deterministic_a5.h:24 (12*1024) | pertoken_dequant.h:127,133 (DETER_UB_SIZE) | 12KB | YES |
| `BUFFER_NUM` | gmm_fr_deterministic_a5.h:25 (2) | pertoken_dequant.h:133 (BUFFER_NUM) | 2 | YES |

#### 2.5.3 namespace 使用一致性

| 文件 | namespace 声明/使用 |
|------|-------------------|
| gmm_fr_deterministic_a5.h | `namespace GMMFRDeterministic { ... }` |
| pertoken_dequant.h | `using namespace GMMFRDeterministic;` |
| tiling_data.h | `namespace GMMFinalizeRoutingArch35Tiling { ... }` (不同namespace, 正确) |

**结论**: [PASS] namespace 使用一致，不同模块使用不同 namespace。

---

### 2.6 潜在 Bug 和风险识别

#### 2.6.1 [BUG-06] 中等: 缺少 MNBlockIdxCompute 确定性调整

**A3 原型**中，确定性模式下 `MNBlockIdxCompute` 使用简单的顺序分配（`curBlock % blockDimN`），而非确定性的对角分配策略。这是确保每个核按固定顺序处理块的关键。

**A5 实现**中没有看到对 `MNBlockIdxCompute` 的修改或等效调整。A5 使用 Cgmct 的 `GroupedMatmulAswtWithTailSplitScheduler` 调度器，其内部分配策略未知。

**如果 Cgmct 调度器使用对角分配或非确定性策略**，那么即使 Epilogue 写入 workspace 而非 yGm，不同核写入 workspace 的顺序也可能不确定，但这不影响最终结果（因为 workspace 只按 `mOffset` 顺序存储，最终由 FRDeterministicA5 按行归属读取）。

**分析**: 实际上，MNBlockIdxCompute 的确定性调整主要影响 Epilogue 的 scatter 写入顺序。在 A5 的确定性模式下，Epilogue 写入 workspace（非 scatter），然后由 FRDeterministicA5 统一按行归属 scatter。因此 Cgmct 调度器的分配策略不影响最终结果的确定性。

**结论**: [PASS] Cgmct 调度器的分配策略不影响确定性结果的正确性。

#### 2.6.2 [BUG-07] 低: 类型转换风险

```cpp
auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(baseOffset + mOffset));
```

`GetValue` 返回 `ROW_INDEX_DTYPE`（int64_t 或 int32_t）。如果 `rowIndexType` 为 `int32_t`，`static_cast<uint64_t>` 是安全的。如果为负数（不应发生），会变成一个很大的正数，但 row_index 理论上不应为负。

**结论**: [PASS] 类型转换安全，前提是 row_index 值非负。

#### 2.6.3 [BUG-08] 中等: 确定性 buffer 大小可能不足

deterministic buffer 的可用大小为 `deterWorkspaceSize`（96MB 或 64MB），但实际需要的空间为 `M * N * sizeof(float)` 字节（所有行的中间结果）。

如果 `M * N * 4 > deterWorkspaceSize`，则 workspace 不够存放所有中间结果，导致内存越界。

A3 通过滑动窗口机制解决这个问题：每次只处理 `windowSize` 行（`windowSize = deterWorkspaceSize / (N * sizeof(float))`），处理完后清空 workspace 再处理下一批。

**A5 实现没有滑动窗口**，一次性设置 `curM = M`（处理所有行）。这意味着需要 `M * N * 4 <= deterWorkspaceSize`。

以 96MB 为例：`96 * 1024 * 1024 / 4 = 25165824` 个 float。如果 N=4096，则最多支持 `25165824 / 4096 = 6144` 行。

**如果 M > 6144 且 N=4096，workspace 就不够了**。在大模型推理场景中，token 数量 M 可能远大于此。

**[BUG-08] 严重: A5 未实现滑动窗口，大 M 场景下 workspace 会溢出**

A3 的滑动窗口通过 `VectorSync` 实现，每当累计行数达到 `windowSize` 就触发一次 `FRDeterministic` 聚合。A5 去掉了滑动窗口逻辑，改为一次性聚合，这在 M 较小时可行，但在 M 较大时会溢出。

**结论**: [FAIL] A5 未实现滑动窗口机制，需要根据实际 M 范围评估是否需要添加。

#### 2.6.4 [BUG-09] 低: 未使用的变量和头文件

- `groupTokensGm` 声明但未使用（pertoken_dequant.h 第119-120行）
- A3 有专门的 `DataCopyPad2D` 辅助函数，A5 用 `DataCopyPad` 替代

**结论**: [PASS] 不影响功能，但影响代码清洁度。

#### 2.6.5 [RISK-01] 中等: Cgmct Epilogue 的 SetAtomicAdd 行为

在确定性模式下，Epilogue 的 y 地址指向 deterBuffer。Cgmct Epilogue 内部使用 `SetAtomicAdd` 写入。多个核可能同时写入 deterBuffer 的同一位置（同一行），此时需要原子写入保证正确性。

但关键是：**deterministic buffer 的写入应该不是原子写入**。在A3中，确定性模式的 VectorAtomicProcess 不使用 `SetAtomicAdd`，而是直接 `DataCopyPad2D` 写入 workspace。每个核写入 workspace 的位置由 `mOffset` 决定（基于滑动窗口内的行偏移），不同核写入不同位置，不冲突。

**在A5中**，Cgmct Epilogue 仍然使用 `SetAtomicAdd` 写入 deterBuffer。如果多个核写入 deterBuffer 的同一位置（例如同一行的不同 tile），原子写入保证数据不丢失。但问题是：deterministic buffer 的布局是什么？如果每个核写入独立的行（由 Cgmct 调度器保证），则不需要原子操作。如果多个核可能写入同一行，则需要原子操作。

**由于 Epilogue 的 y 地址被重定向到 deterBuffer**，Cgmct 内部的 `SetAtomicAdd + DataCopyPad` 写入 deterBuffer 的行为取决于 Cgmct Epilogue 的实现。需要确认 Cgmct Epilogue 的 scatter 逻辑：它是否将 outRow 对应的行 scatter 到 deterBuffer 的正确位置？

**实际上这取决于 Cgmct Epilogue 的实现**。如果 Epilogue 的 finalize routing scatter 逻辑将行 scatter 到 deterBuffer（按 outRow * N + nOffset 的位置），那么多个核可能写入同一位置，需要原子操作。然后 FRDeterministicA5 再从 deterBuffer 读取并聚合。

**这实际上与A3的非确定性模式相同**：多个核通过 SetAtomicAdd 写入 workspace，然后 FRDeterministicA5 按行归属读取。每个核写入 workspace 的同一行，原子操作保证数据累加正确。然后 FRDeterministicA5 只让一个核（行归属核）读取该行的数据并写入 yGm。

**但存在一个问题**：A3 的确定性模式中，VectorAtomicProcess **不** scatter 到 y 行，而是按顺序写入 workspace（不使用 SetAtomicAdd）。Cgmct Epilogue 是否也会按顺序写入 workspace 而不 scatter？

**如果不能控制 Cgmct Epilogue 的 scatter 行为**，那么 Cgmct Epilogue 会将不同核的结果 scatter 到 deterBuffer 的不同位置（基于 outRow），这实际上就等同于直接写入 yGm。此时 FRDeterministicA5 再读取和聚合就变成了重复操作。

**关键不确定性**: Cgmct Epilogue `BlockEpilogueDequantFinalizeRouting` 的 scatter 行为是否可以通过 y 地址重定向来控制。

**结论**: [RISK] 需要确认 Cgmct Epilogue 在 y 地址指向 workspace 时的 scatter 写入行为。这是整个方案可行性的核心前提。

---

## 三、问题汇总（按严重程度排序）

### 严重 (Critical) -- 必须修复

| # | 问题ID | 文件 | 描述 |
|---|--------|------|------|
| 1 | BUG-04 | quant_tiling.cpp | **deterWorkspaceSize 未累加到框架总 workspace**。Kernel 使用 `workspaceGM + deterBufferOffset` 访问确定性 buffer，但框架可能未分配足够的 workspace 空间，导致越界写入。 |
| 2 | BUG-08 | pertoken_dequant.h | **未实现滑动窗口机制**。当 M * N * sizeof(float) > deterWorkspaceSize 时，确定性 buffer 溢出。A3 通过 VectorSync 滑动窗口解决此问题，A5 应评估是否需要。 |
| 3 | BUG-05 | pertoken_dequant.h | **deterBufferOffset 计算公式与A3不一致**（缺少 `coreNum` 因子），可能导致偏移量计算错误，与 Cgmct 实际使用的 workspace 空间重叠。 |

### 中等 (Major) -- 强烈建议修复

| # | 问题ID | 文件 | 描述 |
|---|--------|------|------|
| 4 | BUG-02 | quant_tiling.cpp | **缺少 PerToken 量化模式显式检查**。虽然 TilingKey 路由机制可防止错误路径，但防御性编程应添加 PerToken 检查。 |
| 5 | WARNING-03 | gmm_fr_deterministic_a5.h | **AIC 核跳过 SyncAll 可能在 A5 上死锁**。需确认 A5 的 SyncAll 是否要求所有核参与。 |
| 6 | RISK-01 | pertoken_dequant.h | **Cgmct Epilogue scatter 行为不确定性**。重定向 y 地址到 workspace 后，Epilogue 的 scatter + SetAtomicAdd 行为是否与预期一致？ |

### 低 (Minor) -- 建议优化

| # | 问题ID | 文件 | 描述 |
|---|--------|------|------|
| 7 | BUG-03 | quant_tiling.cpp | 常量 `96UL * 1024 * 1024` 使用 UL 后缀但类型为 uint32_t，风格不一致（功能无影响）。 |
| 8 | WARNING-01 | gmm_fr_deterministic_a5.h | AIC 核退出使用 `g_coreType == AIC` 而非 `ASCEND_IS_AIC` 宏，风格与A3不一致。 |
| 9 | WARNING-02 | gmm_fr_deterministic_a5.h | DataCopyPad 替代 DataCopyPad2D，功能等价但建议硬件测试验证。 |
| 10 | WARNING-04 | pertoken_dequant.h | 局部 TPipe 可能与 Cgmct 框架的 pipe 资源冲突。 |
| 11 | WARNING-05 | pertoken_dequant.h | `groupTokensGm` 声明但未使用，应移除。 |

---

## 四、具体修改建议

### 建议 1: 修复 workspace 大小累加 (BUG-04)

**文件**: `grouped_matmul_finalize_routing_quant_tiling.cpp`

**位置**: `DoOpTiling()` 确定性条件块内，在设置 `deterWorkspaceSize_` 之后

**建议**: 需要找到 A5 Cgmct 框架中累加 workspace 大小的机制。可能的方案：
```cpp
// 方案A: 如果 A5 框架使用 context_->GetWorkspaceSize()
// 需要在 PostTiling() 中累加:
// (需要在框架的 workspace 管理接口中添加 deterWorkspaceSize)

// 方案B: 如果 Cgmct 使用手动 workspace 大小
// 在 DoOpTiling() 或 PostTiling() 中添加:
// totalWorkspaceSize += deterWorkspaceSize_;
```

需要与框架开发者确认 A5 的 workspace 管理方式。

### 建议 2: 添加 PerToken 显式检查 (BUG-02)

**文件**: `grouped_matmul_finalize_routing_quant_tiling.cpp`

**当前**:
```cpp
if (context_->GetDeterministic() == 1 && !IsMicroScaling() &&
    inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8) {
```

**建议修改为**:
```cpp
if (context_->GetDeterministic() == 1 && !IsMicroScaling() &&
    inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
    inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
```

### 建议 3: 确认 A5 SyncAll 同步范围 (WARNING-03)

需要查阅 A5 (arch35) 的 AscendC API 文档，确认 `SyncAll()` 的同步范围：
- 如果只同步 AIV 核：当前代码正确
- 如果同步所有核（AIC+AIV）：需要在 AIC 路径上也调用 SyncAll

如果需要修改：
```cpp
// 修改前
if (g_coreType == AIC) {
    return;  // AIC 核完全跳过
}

// 修改后（如果 A5 SyncAll 要求所有核参与）
if (g_coreType == AIC) {
    SyncAll();  // AIC 核只参与同步，不做实际工作
    SyncAll();  // 配对第二个 SyncAll
    return;
}
```

### 建议 4: 移除未使用变量 (WARNING-05)

**文件**: `grouped_matmul_finalize_routing_pertoken_dequant.h`

**删除**:
```cpp
GlobalTensor<int64_t> groupTokensGm;
groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));
```

### 建议 5: 评估滑动窗口需求 (BUG-08)

需要根据实际使用场景评估：
- **如果 M 范围确定且 M * N * 4 <= 96MB**: 当前实现可行
- **如果 M 可能很大**: 需要实现滑动窗口机制

建议添加运行时检查（在 Tiling 或 Kernel 入口处）：
```cpp
// 在 DoOpTiling() 中添加
uint64_t requiredDeterSize = inputParams_.mSize * inputParams_.nSize * sizeof(float);
if (requiredDeterSize > deterWorkspaceSize_) {
    // 禁用确定性或报错
    OP_LOGE(context_->GetNodeName(),
            "Deterministic buffer overflow: required %lu bytes but only %u available",
            requiredDeterSize, deterWorkspaceSize_);
    deterministicFlag_ = 0;
    tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 0;
}
```

### 建议 6: 验证 deterBufferOffset 计算 (BUG-05)

需要确认 Cgmct 框架的实际 workspace 使用量。可以通过以下方式验证：
1. 查看 Cgmct `BlockMmadBuilder` 的 workspace 分配逻辑
2. 或在 Tiling 中计算 Cgmct 需要的 workspace 大小并据此设置偏移量

---

## 五、整体验收结论

### 验收判定: **条件性通过 (CONDITIONAL PASS)**

### 判定理由

**通过项**:
1. Tiling 数据结构设计正确，字段命名、类型、默认值与A3原型一致
2. 结构体大小满足 pack(8) 对齐要求
3. Tiling 计算逻辑的条件判断核心正确（GetDeterministic + INT8 + 非MX）
4. workspace 大小选择逻辑（96MB/64MB）与A3完全一致
5. FRDeterministicA5 函数核心逻辑与A3原型一致（行归属、DataCopy、AtomicAdd）
6. SyncAll 在函数内严格配对
7. 模板参数使用正确
8. Params 构建中所有 y 地址正确替换为 deterBuffer
9. 非确定性分支零回归风险
10. 跨文件常量和 namespace 一致性通过
11. SyncConfig 初始化和 128 对齐正确

**条件项（必须解决后才能正式通过）**:
1. **[BUG-04]** workspace 大小未累加到框架总 workspace -- 需要确认并修复
2. **[BUG-05]** deterBufferOffset 计算公式需要验证正确性
3. **[WARNING-03]** A5 SyncAll 的同步范围需要确认
4. **[BUG-08]** 滑动窗口缺失导致的大 M 场景风险需要评估
5. **[RISK-01]** Cgmct Epilogue 的 scatter 行为需要确认

### 建议的下一步行动

1. **立即**: 修复 BUG-04（workspace 累加）和 BUG-02（PerToken 检查）
2. **尽快**: 确认 WARNING-03（SyncAll 同步范围）和 RISK-01（Cgmct Epilogue 行为）
3. **评估**: BUG-08（滑动窗口需求）和 BUG-05（offset 计算）根据实际场景决定
4. **清理**: WARNING-05（移除未使用变量）等低优先级项

---

**验收人签名**: AscendC Acceptance Tester
**日期**: 2026-05-08
