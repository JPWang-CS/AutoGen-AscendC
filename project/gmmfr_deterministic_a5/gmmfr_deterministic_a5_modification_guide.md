# GMMFR 确定性特性 A3 -> A5 迁移完整技术设计文档

## 1. 概述

### 1.1 GMMFR 算子简介

GroupedMatmulFinalizeRouting (GMMFR) 是一个融合算子，将分组矩阵乘法 (GroupedMatmul) 和 MoE FinalizeRouting 操作融合在一起。其核心计算流程为：

1. 对每个 expert group 执行 INT8 x INT8 矩阵乘法
2. 对矩阵乘法结果进行反量化 (dequant)
3. 用 logit 权重乘以反量化结果
4. 根据 rowIndex 进行 scatter，将结果累加到最终输出 y 中

在 MoE (Mixture of Experts) 场景中，多个 token 会被路由到不同的 expert，最终需要将所有 expert 的输出按照路由索引聚合到统一的输出张量中。

### 1.2 A3 vs A5 架构差异对确定性实现的影响

| 维度 | A3 (910C, arch32) | A5 (950, arch35) |
|------|-------------------|-------------------|
| 编程模型 | 原生 AscendC，手动管理 group 循环和同步 | Cgmct Builder 框架，模板化 Epilogue/Prologue 注入 |
| 核心架构 | AIC + AIV 协作，AIC Core 可跳过 SyncAll | AIC + AIV 协作，SyncAll 要求严格 1:1 配对 |
| Epilogue 写入 | 内联代码，可直接 if-else 分支 | 模板注入，运行时无法分支，需通过类型系统区分 |
| mOffset 语义 | 全局偏移（用户代码直接管理跨 group 累加） | group 内偏移（框架管理 group 循环，UpdateGlobalAddr 重置地址） |
| Tiling 系统 | 自定义 flat tiling struct，手动 set/get | GMMFinalizeRoutingTilingData struct，通过 memcpy 传递 |

这些差异决定了 A5 的确定性实现不能简单复制 A3 的代码，而必须适配 Cgmct 框架的编程范式。

### 1.3 Cgmct Builder 框架 vs A3 原生 AscendC

A3 的实现 (`grouped_matmul_finalize_routing.h`) 使用原生 AscendC 编程模型：
- 手动管理 group 循环 (`for groupIdx = 0; groupIdx < groupNum; groupIdx++`)
- 手动管理 Cube/Vector 同步 (`CrossCoreSetFlag/CrossCoreWaitFlag`)
- 在 `VectorAtomicProcess` 中可以直接用 `if (deterministicFlag == 1)` 做分支

A5 的实现使用 Cgmct (Common Grouped Matmul Components and Templates) 框架：
- 框架管理 group 循环 (在 `KernelGmmFinalizeRoutingPertokenDequant::operator()` 中)
- 框架管理 AIC/AIV 同步 (`NotifyVector/WaitForCube`)
- Epilogue 作为模板参数注入，运行时无法切换行为

这意味着 A5 必须通过 **类型隔离** (不同 Epilogue 类型 -> 不同 Kernel 实例化) 来区分确定性和非确定性模式。

### 1.4 迁移策略总览

A5 的确定性迁移采用"三阶段数据流"策略：

```
Phase 1 (Prologue):  初始化 yGm 为零 + 写入 sharedInput residual -> yGm
Phase 2 (Epilogue):  反量化 + logit 乘法 -> SequentialWrite -> workspace (deterBuffer)
Phase 3 (Aggregation): workspace -> rowIndex 映射 -> AtomicAdd -> yGm (每行唯一写者)
```

核心思路：将原来的 scatter + AtomicAdd (非确定性) 拆分为 sequential write + 延迟聚合，消除并发写入的不确定性。

---

## 2. 设计思路

### 2.1 确定性原理

#### 为什么 AtomicAdd scatter 不确定性？

在非确定性模式下，Epilogue 的 `VectorAtomicProcess` 执行如下操作：

```cpp
// block_epilogue_dequant_finalize_routing.h, line 263-274
SetAtomicAdd<float>();
for (uint32_t i = 0; i < curVecBaseM; i++) {
    auto outRow = static_cast<uint64_t>(rowIndexGlobal_.GetValue(offsetM + i));
    DataCopyPad(yGlobal_[outRow * n_ + yOffset], yLocal[i * alignN_], paramsOut);
}
SetAtomicNone();
```

当多个 core 同时对同一个 `outRow` 执行 AtomicAdd 时，浮点数加法的累加顺序取决于 core 的执行时序。由于不同 core 的执行速度不完全一致，导致最终的浮点累加结果在不同运行之间可能产生微小差异。这就是非确定性的根源。

#### 为什么 SequentialWrite + 延迟聚合可以保证确定性？

SequentialWrite 的核心思想是：

1. **写入阶段确定性**：每个 core 按 mOffset（输入行序号）写入 workspace，而不是按 outRow（输出行序号）scatter 写入 yGm。由于每个 mOffset 在所有 core 之间是唯一的（每个输入行只被一个 core 处理），因此不存在并发写入冲突，无需 AtomicAdd。

2. **聚合阶段确定性**：在 `FRDeterministicA5` 中，按 `outRow % coreNumVec` 分配行所有权，每个输出行只有一个 core 负责聚合。聚合时使用 AtomicAdd 是安全的，因为此时只有一个写者，AtomicAdd 退化为普通写入。

### 2.2 A3 原型回顾

A3 的确定性实现在 `grouped_matmul_finalize_routing.h` 中，核心逻辑如下：

#### A3 的三阶段

1. **Prologue** (`PreProcess`, line 240-283)：将 yGm 初始化为零并写入 sharedInput residual 到 yGm
2. **SequentialWrite** (`VectorAtomicProcess` 确定性分支, line 377-382)：

```cpp
if (tiling->deterministicFlag == 1) {
    DataCopy2DDimParams dimParams{vecAParams.curVecBaseM, vecAParams.curVecBaseN, vecAParams.alignBaseN};
    DataCopyPad2D(mmQuantOutGm[vecAParams.yGmOffset1 - (syncConfig.lowBoundM - syncConfig.windowSize) * tiling->n],
        yLocal, dimParams, tiling->n);
    vecOutQueue.FreeTensor(yLocal);
    return;
}
```

A3 直接写入 `mmQuantOutGm`（确定性专用 workspace），使用 `DataCopyPad2D` 按 mOffset 顺序写入。`syncConfig.lowBoundM - syncConfig.windowSize` 提供了 workspace 的起始偏移。

3. **聚合** (`FRDeterministic`, line 647-681)：

```cpp
SyncAll();
uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
    auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(...));
    if (outRow % coreNumVec != GetBlockIdx()) { continue; }
    // DataCopyPad2D 读 workspace -> AtomicAdd 写 yGm
}
SyncAll();
```

A3 的 AIC Core 可以跳过 SyncAll (`if ASCEND_IS_AIC { return; }` 在 FRDeterministic 入口处)。

#### A3 的窗口化同步机制

A3 使用 `SyncConfig::windowSize` 实现滑动窗口同步。`windowSize` 由 `deterWorkspaceSize / (n * sizeof(float))` 计算得出，即 workspace 能容纳的最大行数。当累加的输入行数达到窗口大小时，触发一次 FRDeterministic 聚合并清空 workspace，然后继续处理下一批。

### 2.3 A5 适配挑战

#### 2.3.1 Cgmct 框架管理 group 循环，mOffset 是 group 内偏移

在 A3 中，用户代码直接管理 group 循环：
```cpp
// A3: 用户手动管理
for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
    mnConfig.offsetM = ...; // 全局偏移，跨 group 累加
}
```

在 A5 的 Cgmct 框架中，group 循环由 `KernelGmmFinalizeRoutingPertokenDequant::operator()` 管理（line 356-361）。框架在每次切换 group 时调用 `UpdateGlobalBuffer` 重置地址偏移。Epilogue 接收到的 `mOffset`（通过 `blockCoord` 传递）是 **当前 group 内的偏移**，而非全局偏移。

这意味着 A5 的 Epilogue 无法直接用 mOffset 计算 workspace 地址。解决方案是引入 `accumulatedGroupOffset_`，在 `UpdateGlobalAddr` 中捕获框架传递的累积 logit 偏移（`SEQ_LOGIT_INDEXS`），从而恢复全局行索引。

相关代码 (`block_epilogue_dequant_sequential_write.h`, line 252)：
```cpp
accumulatedGroupOffset_ = static_cast<uint64_t>(Get<SEQ_LOGIT_INDEXS>(baseOffset));
```

框架中 `UpdateGlobalBuffer`（`kernel_gmm_finalize_routing_pertoken_dequant.h`, line 228-229）传递给 Epilogue 的 `vecBaseOffset` 中，`SEQ_LOGIT_INDEXS` 位实际上等于 `IDX_LOGIT_OFFSETS`，即跨 group 累积的 logit 偏移量，恰好就是全局输入行偏移。

#### 2.3.2 Cgmct Epilogue 使用模板注入，无法运行时分支

A3 可以在 `VectorAtomicProcess` 中直接用 `if (deterministicFlag == 1)` 做分支切换。Cgmct 框架的 Epilogue 是通过模板参数注入的：

```cpp
// kernel_gmm_finalize_routing_pertoken_dequant.h, line 134
BlockEpilogueDequantFinalizeRouting epilogueDequantOp_;
```

运行时无法切换 Epilogue 的行为。解决方案是：

1. 创建新的 Epilogue 类 `BlockEpilogueDequantSequentialWrite`
2. 创建对应的 Kernel 实例化 `GmmKernelDeterministic`
3. 在 Kernel 入口通过运行时 `if-else` 选择不同的 Kernel 实例

#### 2.3.3 A5 SyncAll 严格 1:1 配对

A3 的 `FRDeterministic` 函数中，AIC Core 直接 `return` 跳过（line 649-651）：
```cpp
if ASCEND_IS_AIC {
    return;  // A3: AIC Core 跳过 SyncAll
}
SyncAll();
```

在 A5 (arch35) 上，`SyncAll` 要求所有参与的 core 必须成对调用，不允许部分 core 跳过。因此 A5 的 `FRDeterministicA5` 中，AIC Core 必须也调用 `SyncAll`（虽然不做实际聚合工作）：

```cpp
// gmm_fr_deterministic_a5.h, line 67-71
if (g_coreType == AIC) {
    SyncAll();  // AIC Core 必须参与 SyncAll 配对
    SyncAll();
    return;
}
```

这里有两次 `SyncAll`：第一次对应 VEC Core 的第一次 `SyncAll`（line 74），第二次对应 VEC Core 结尾的 `SyncAll`（line 111）。

#### 2.3.4 GMMTiling 类型系统隔离

Cgmct 框架的 `GMMTiling` 是 `KernelGmmFinalizeRoutingPertokenDequant` 的嵌套类型。由于不同 Epilogue 模板参数会产生不同的 Kernel 实例化，每个 Kernel 实例化都有自己的 `GMMTiling` 类型。这些类型在 C++ 类型系统中是不同的，不能互相赋值。

因此，确定性模式必须使用 `GmmKernelDeterministic::GMMTiling` 类型，而非 `GmmKernel::GMMTiling`：

```cpp
// grouped_matmul_finalize_routing_pertoken_dequant.h, line 110-117
using GMMTilingDeterministic = typename GmmKernelDeterministic::GMMTiling;
GMMTilingDeterministic gmmParamsDeter{...};
```

### 2.4 三阶段数据流设计

#### Phase 1: Prologue -> yGm (zeros + residual)

Prologue (`BlockPrologueFinalizeRouting`) 由 Cgmct 框架在所有 AIV Core 上执行。它完成两件事：

1. 将 yGm 的全部 `batch * N` 元素初始化为零
2. 在 sharedInput 对应的位置写入 residual（经过 scale 缩放后的 sharedInput）

无论是否确定性模式，Prologue 的行为完全相同。关键点是 **Prologue 写入的是 yGm（最终输出）**，而非 workspace。这保证了在确定性模式下，workspace 只包含 Epilogue 的 dequant 结果，不存在与 Prologue 的数据冲突。

#### Phase 2: Epilogue -> workspace (SequentialWrite)

在确定性模式下，Epilogue (`BlockEpilogueDequantSequentialWrite`) 的写入目标是 workspace（`deterBuffer`），而非 yGm。

每个 core 按 mOffset 顺序写入 workspace：
```
workspace[(accumulatedGroupOffset_ + offsetM + i) * n_ + yOffset]
```

其中：
- `accumulatedGroupOffset_`：跨 group 的累积行偏移（在 `UpdateGlobalAddr` 中更新）
- `offsetM`：当前 tile 在 group 内的 M 维偏移
- `i`：tile 内的行索引
- `n_`：N 维大小
- `yOffset`：N 维偏移

由于 workspace 地址的计算基于输入行序号（mOffset），而非输出行序号（outRow），不同 core 写入的地址互不重叠。

#### Phase 3: Aggregation -> yGm (AtomicAdd, 每行唯一写者)

在 Epilogue 完成后，Kernel 入口调用 `FRDeterministicA5` 进行延迟聚合：

1. 等待所有 core 完成 workspace 写入（`SyncAll`）
2. 遍历所有输入行 `mOffset = 0..totalM-1`
3. 读取 `rowIndex[mOffset]` 得到 outRow
4. 通过 `outRow % coreNumVec == GetBlockIdx()` 判断当前 core 是否拥有该行
5. 拥有该行的 core 从 workspace 读取数据，AtomicAdd 到 yGm
6. 等待所有 core 完成聚合（`SyncAll`）

由于行所有权机制保证了每个 outRow 只有一个 core 写入，AtomicAdd 在这里退化为普通写入，保证了确定性。

---

## 3. 修改文件总览

| 序号 | 文件路径 | 修改类型 | 说明 |
|------|----------|----------|------|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | **新增字段** | Tiling 数据结构增加 `deterministicFlag` 和 `deterWorkspaceSize` |
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | **新增成员** | Tiling 类增加 `deterministicFlag_` 和 `deterWorkspaceSize_` 成员 |
| 3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | **逻辑修改** | `DoOpTiling` 增加确定性条件判断，`GetWorkspaceSize` 增加 workspace 计算 |
| 4 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | **新增分支** | Kernel 入口增加确定性 if-else 分支，定义 `GmmKernelDeterministic` 类型 |
| 5 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | **新文件** | A5 确定性聚合函数 `FRDeterministicA5` |
| 6 | `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` | **新文件** | 确定性 Epilogue，使用 SequentialWrite 替代 AtomicAdd scatter |

> 注意：以下文件用于对比参考，未修改：
> - `common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` -- Cgmct 框架 Kernel（理解调用链）
> - `common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` -- 原始 Epilogue（对比参考）
> - `common/cgmct/prologue/block_prologue_finalize_routing.h` -- Prologue（理解数据流）

---

## 4. Tiling 层修改详解

### 4.1 Tiling 数据结构 (tiling_data.h)

**文件**：`op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`

在 `GMMFinalizeRoutingDataParams` 结构体中新增两个字段：

```cpp
// line 39-40
uint32_t deterministicFlag = 0;      // 0=非确定性, 1=确定性
uint32_t deterWorkspaceSize = 0;     // 确定性 workspace 大小（字节）
```

**设计决策**：
- 这两个字段放在 `GMMFinalizeRoutingDataParams` 的末尾，位于 `reserved2` 之后
- 使用 `#pragma pack(push, 8)` 保证结构体对齐
- `deterministicFlag` 为 0 表示非确定性（默认），为 1 表示确定性
- `deterWorkspaceSize` 在非确定性模式下为 0，在确定性模式下为实际分配的 workspace 大小

**与 A3 tiling 的对比**：

A3 的 tiling 结构（`grouped_matmul_finalize_routing_tiling.h`）也有相同的两个字段，但 A3 的 tiling 是通过 `tilingData_.set_deterministicFlag()` 和 `tilingData_.set_deterWorkspaceSize()` 设置的，而 A5 的 tiling 使用扁平结构体直接赋值：

```cpp
// A3 (base_tiling.cpp, line 454-455)
tilingData_.set_deterministicFlag(deterministicFlag_);
tilingData_.set_deterWorkspaceSize(deterWorkspaceSize_);

// A5 (quant_tiling.cpp, line 463-464, 482)
tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
```

### 4.2 Tiling 声明 (quant_tiling.h)

**文件**：`op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h`

在 `GroupedMatmulFinalizeRoutingQuantTiling` 类中新增两个私有成员（line 127-128）：

```cpp
uint32_t deterministicFlag_ = 0;       // 确定性标志：0=关闭, 1=开启
uint32_t deterWorkspaceSize_ = 0;      // 确定性 workspace 大小
```

这两个成员不直接写入 Tiling 数据结构，而是在 `DoOpTiling()` 中经过条件判断后写入 `tilingData_`，并在 `GetWorkspaceSize()` 中用于计算总 workspace。

### 4.3 Tiling 实现 (quant_tiling.cpp)

**文件**：`op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`

#### 4.3.1 确定性条件判断逻辑

在 `DoOpTiling()` 方法中（line 458-484），新增确定性 Tiling 处理：

```cpp
// line 458-484
if (context_->GetDeterministic() == 1 && !IsMicroScaling() &&
    inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
    inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
    deterministicFlag_ = 1;
    tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
    // ... workspace 大小计算 ...
}
```

**条件分解**：

| 条件 | 含义 | 原因 |
|------|------|------|
| `context_->GetDeterministic() == 1` | Python 层传递确定性标志 | 用户通过 `torch.use_deterministic_algorithms(True)` 设置 |
| `!IsMicroScaling()` | 非 MX 微缩放模式 | 确定性仅支持 K-C/T-C 量化模式 |
| `aDtype == DT_INT8 && bDtype == DT_INT8` | W8A8 量化 | 确定性仅支持 INT8 x INT8 |
| `aQuantMode == PERTOKEN_MODE` | PerToken 反量化 | 需要 per-token scale 进行反量化 |

**与 A3 的对比**：A3 的条件判断在 `DeterministicTilingProcess()` 中（`grouped_matmul_finalize_routing_base_tiling.cpp`, line 413-425），只检查 `context_->GetDeterministic()` 和 dtype 检查（在外层完成），不做量化模式检查。

#### 4.3.2 workspace 大小计算

```cpp
// line 464-483
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
uint64_t l2Size = 0;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
constexpr uint32_t DETER_WORK_SPACE_SIZE = 96UL * 1024 * 1024;       // 96 MB
constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64UL * 1024 * 1024; // 64 MB
deterWorkspaceSize_ = l2Size > DETER_WORK_SPACE_SIZE
                      ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
```

workspace 大小取决于 L2 cache 大小：
- L2 > 96MB：使用 96MB workspace
- L2 <= 96MB：使用 64MB workspace

**溢出保护**（line 472-483）：

```cpp
uint64_t requiredDeterSize = inputParams_.mSize * inputParams_.nSize * sizeof(float);
if (requiredDeterSize > deterWorkspaceSize_) {
    OP_LOGW(context_->GetNodeName(), "Deterministic buffer overflow: ...");
    deterministicFlag_ = 0;
    tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 0;
    tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = 0;
} else {
    tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
}
```

如果所需确定性 buffer（M * N * sizeof(float)）超过可用 workspace，则自动降级为非确定性模式并打印警告日志。

**与 A3 的对比**：A3 的 workspace 计算在 `DeterministicTilingProcess()` 中（`grouped_matmul_finalize_routing_base_tiling.cpp`, line 419-424），逻辑相同但不做溢出保护（A3 在 Kernel 层通过窗口化机制处理溢出）。

#### 4.3.3 GetWorkspaceSize override

```cpp
// line 559-570
ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t totalWorkspace = GmmConstant::SYS_WORKSPACE_SIZES;
    if (deterministicFlag_ == 1 && deterWorkspaceSize_ > 0) {
        totalWorkspace += static_cast<size_t>(deterWorkspaceSize_);
    }
    workspaces[0] = totalWorkspace;
    return ge::GRAPH_SUCCESS;
}
```

总 workspace = 系统预留空间 (`SYS_WORKSPACE_SIZES`) + 确定性 workspace。

**与 A3 的对比**：A3 在 `DeterministicTilingProcess()` 中直接 `workspaceSize_ += deterWorkspaceSize_`，逻辑相同。

### 4.4 调用链：Python -> aclnn -> Tiling -> Kernel 的参数传递路径

```
Python: torch.use_deterministic_algorithms(True)
    |
    v
GE (Graph Engine): 设置 deterministic 标志到 gert::TilingContext
    |
    v
aclnn: aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSize()
    |-- 不直接处理 deterministic 标志
    |-- 由框架自动将 deterministic 标志传递给 TilingContext
    |
    v
Tiling: context_->GetDeterministic() 读取标志
    |-- DoOpTiling(): 判断条件，设置 deterministicFlag_ 和 deterWorkspaceSize_
    |-- GetWorkspaceSize(): 计算总 workspace
    |-- PostTiling(): 将 tilingData_ 通过 memcpy 写入 raw tiling data
    |
    v
Kernel: 从 tilingGM 读取 tilingData
    |-- gmmFinalizeRoutingQuantParams_.deterministicFlag 判断分支
    |-- if (flag == 1) -> GmmKernelDeterministic (SequentialWrite Epilogue)
    |-- else -> GmmKernel (AtomicAdd Epilogue)
```

关键点：aclnn 层 (`aclnn_grouped_matmul_finalize_routing.cpp`) 不包含任何 `deterministic` 相关代码。确定性标志由 Python/GE 层直接设置到 `TilingContext`，Tiling 层通过 `context_->GetDeterministic()` 读取。这是 CANN 框架的标准做法。

---

## 5. Kernel 层修改详解

### 5.1 确定性 Epilogue (block_epilogue_dequant_sequential_write.h)

**文件**：`common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h`

#### 5.1.1 设计：为什么新建文件而非修改原始 Epilogue

选择新建文件而非修改原始 `BlockEpilogueDequantFinalizeRouting` 的原因：

1. **Cgmct 框架要求**：Epilogue 是模板参数，不同行为需要不同类型。Cgmct 的 `KernelGmmFinalizeRoutingPertokenDequant` 通过模板参数 `BlockEpilogueDequantFinalizeRouting_` 注入 Epilogue。要改变 Epilogue 的写入行为，必须提供不同的类型。

2. **最小侵入性**：新建文件避免了修改框架公共代码（原始 Epilogue 被其他算子共享），降低回归风险。

3. **清晰的关注点分离**：确定性逻辑和非确定性逻辑在代码层面完全隔离，便于维护和测试。

#### 5.1.2 accumulatedGroupOffset_ 跨 group 寻址机制

这是 A5 确定性实现中最重要的设计决策之一。

**问题**：Cgmct 框架在每次切换 group 时，通过 `UpdateGlobalBuffer` 重置 Epilogue 的全局地址。传递给 Epilogue `UpdateGlobalAddr` 的 `vecBaseOffset` 中，`SEQ_LOGIT_INDEXS` 是跨 group 累积的 logit 偏移。但 Epilogue 内部的 `mOffset`（在 `operator()` 中计算）只是当前 tile 在当前 group 内的偏移。

**解决方案**：在 `UpdateGlobalAddr` 中捕获累积偏移：

```cpp
// line 247-252
accumulatedGroupOffset_ = static_cast<uint64_t>(Get<SEQ_LOGIT_INDEXS>(baseOffset));
logitGlobal_.SetGlobalBuffer((__gm__ float *)params_->logitGmAddr + Get<SEQ_LOGIT_INDEXS>(baseOffset));
rowIndexGlobal_.SetGlobalBuffer((__gm__ DataTypeRowIndex *)params_->rowIndexGmAddr +
                                Get<SEQ_LOGIT_INDEXS>(baseOffset));
```

然后在 `VectorSequentialWrite` 中使用：

```cpp
// line 317
uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;
```

**数据流推导**：

框架中 `UpdateGlobalBuffer` 的调用链：
1. `operator()` 遍历 group：`for (groupIdx = 0; groupIdx < groupNum; groupIdx++)`
2. `UpdateGroupParams` 更新偏移：`Get<IDX_LOGIT_OFFSETS>(baseOffset_) += m;`
3. `UpdateGlobalBuffer` 构造 `vecBaseOffset`：`Get<SEQ_LOGIT_INDEXS>(vecBaseOffset) = Get<IDX_LOGIT_OFFSETS>(baseOffset_)`
4. `epilogueDequantOp_.UpdateGlobalAddr(vecBaseOffset)` 传递给 Epilogue

因此 `SEQ_LOGIT_INDEXS` 实际上就是到当前 group 为止的累积输入行数，即全局 mOffset 偏移。

#### 5.1.3 VectorSequentialWrite vs VectorAtomicProcess 的详细对比

**原始 VectorAtomicProcess**（`block_epilogue_dequant_finalize_routing.h`, line 263-274）：

```cpp
SetAtomicAdd<float>();
DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
for (uint32_t i = 0; i < curVecBaseM; i++) {
    auto outRow = static_cast<uint64_t>(rowIndexGlobal_.GetValue(offsetM + i));
    DataCopyPad(yGlobal_[outRow * n_ + yOffset], yLocal[i * alignN_], paramsOut);
}
SetAtomicNone();
```

特征：
- 使用 `SetAtomicAdd<float>()` 开启原子加
- 按 **outRow** (输出行) scatter 写入
- 多个 core 可能写同一个 outRow -> 非确定性

**新的 VectorSequentialWrite**（`block_epilogue_dequant_sequential_write.h`, line 306-324）：

```cpp
uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;
DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
for (uint32_t i = 0; i < curVecBaseM; i++) {
    DataCopyPad(yGlobal_[(absRowBase + i) * n_ + yOffset], yLocal[i * alignN_], paramsOut);
}
```

特征：
- **不使用** `SetAtomicAdd`，普通写入
- 按 **mOffset** (输入行) 顺序写入
- 每个 mOffset 在所有 core 间唯一 -> 确定性
- 写入目标在确定性模式下是 workspace（`deterBuffer`），不是 yGm

**注意**：这里 `yGlobal_` 的实际 Global Buffer 地址取决于 Params 传入的 `yGMAddr`。在确定性模式下，Kernel 入口将 `deterBuffer`（workspace 地址）传入 `epilogueParams.yGMAddr`，因此 `yGlobal_` 实际指向 workspace。

#### 5.1.4 模板注入方式

`BlockEpilogueDequantSequentialWrite` 的接口与 `BlockEpilogueDequantFinalizeRouting` 完全一致（相同的 `Params` 结构、相同的 `Init/operator()/UpdateGlobalAddr/UpdateNextProblem` 签名），确保可以作为模板参数无缝替换。

在 Kernel 入口中的使用方式：

```cpp
// grouped_matmul_finalize_routing_pertoken_dequant.h, line 70-72
using BlockEpilogueDequantSeq =
    Cgmct::Gemm::Block::BlockEpilogueDequantSequentialWrite<CType, C1Type, weightscaleType,
                                                             xscaleType, BiasType, rowIndexType>;
```

### 5.2 Kernel 入口 (grouped_matmul_finalize_routing_pertoken_dequant.h)

**文件**：`op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`

#### 5.2.1 确定性分支的完整代码流

Kernel 入口函数 `grouped_matmul_finalize_routing_pertoken_dequant` 的确定性分支（line 100-163）：

```
[入口函数]
    |-- 解析 Tiling 数据
    |-- 定义类型（BlockMmadBuilder, BlockPrologue, BlockEpilogueDequant, BlockEpilogueDequantSeq）
    |-- 定义 Kernel 实例化（GmmKernel, GmmKernelDeterministic）
    |
    |-- if (deterministicFlag == 1):
    |       |-- 计算 deterBuffer 地址 = workspaceGM + SYS_WORKSPACE_SIZE
    |       |-- 构造 GMMTilingDeterministic
    |       |-- 构造 ParamsDeterministic（注意 c -> workspace, Prologue -> yGm）
    |       |-- 执行 GmmKernelDeterministic(params)
    |       |-- 准备 GlobalTensor（deterBufferGm, yGm, rowIndexGm）
    |       |-- 准备 TQueBind/TPipe
    |       |-- 构造 SyncConfig
    |       |-- 调用 FRDeterministicA5() -- Phase 3 聚合
    |
    |-- else:
            |-- 构造 Params（c -> yGm, Prologue -> yGm）
            |-- 执行 GmmKernel(params)
```

#### 5.2.2 GmmKernelDeterministic 类型设计

```cpp
// line 74-81
using GmmKernel =
    Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                 BlockEpilogueDequant, BlockScheduler>;

using GmmKernelDeterministic =
    Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                  BlockEpilogueDequantSeq, BlockScheduler>;
```

两个 Kernel 实例化共享 `ProblemShape`, `BlockMmadBuilder`, `BlockPrologue`, `BlockScheduler`，唯一区别是 Epilogue 模板参数：
- `GmmKernel` 使用 `BlockEpilogueDequant` (AtomicAdd scatter)
- `GmmKernelDeterministic` 使用 `BlockEpilogueDequantSeq` (SequentialWrite)

#### 5.2.3 GMMTiling 类型隔离

```cpp
// line 83-85
using Params = typename GmmKernel::Params;
using GMMTiling = typename GmmKernel::GMMTiling;
using ParamsDeterministic = typename GmmKernelDeterministic::Params;
```

由于 `GmmKernel` 和 `GmmKernelDeterministic` 是不同的模板实例化，它们的嵌套类型 `GMMTiling`、`Params` 等在 C++ 类型系统中是不同的。不能混用：

```cpp
// 错误：GmmKernel::GMMTiling 不能传给 GmmKernelDeterministic
// 正确：必须使用 GmmKernelDeterministic::GMMTiling
using GMMTilingDeterministic = typename GmmKernelDeterministic::GMMTiling;
```

注释（line 109）也解释了这个决策：
```
// Must use GmmKernelDeterministic's own GMMTiling type (different kernel instantiations have incompatible nested types)
```

#### 5.2.4 Params 构造详解

**确定性模式的 Params**（line 121-131）：

```cpp
ParamsDeterministic params = {
    {1, 1, 1, 1},                                         // problemShape: placeholder
    {x, w, deterBuffer, bias, group_list},                 // BlockMmad: c -> workspace
    {share_input, y,                                       // Prologue: residual -> yGm
     sharedInputOffset, sharedInputLen, N, batch, residualScale},
    {deterBuffer, w_scale, x_scale, bias, logit, row_index, // Epilogue: dequant -> workspace
     baseM, baseN},
    gmmParamsDeter                                         // GMMTiling
};
```

关键设计：

1. **BlockMmadParams.cGmAddr = deterBuffer**：矩阵乘法结果写入 workspace，而非 yGm。这是因为 Cgmct 框架的 BlockMmadBuilder 将 L0C 输出通过 `GetL0c2UbTensor()` 传递给 Epilogue，Epilogue 从 UB 读取 L0C 结果进行反量化。`cGmAddr` 实际上不被 Epilogue 直接使用（Epilogue 从 UB 读 L0C），但框架内部可能有其他用途。

2. **BlockPrologueParams.yGmAddr = y**：Prologue 写入最终输出 yGm（初始化为零 + residual）。这在确定性模式下是正确的，因为 Prologue 先执行，写入 yGm；然后 Epilogue 写入 workspace（不影响 yGm）；最后 FRDeterministicA5 将 workspace 聚合回 yGm。

3. **BlockEpilogueParams.yGMAddr = deterBuffer**：Epilogue 写入 workspace（SequentialWrite），而非 yGm。这是确定性模式的核心区别。

**非确定性模式的 Params**（line 167-174）：

```cpp
Params params = {
    {1, 1, 1, 1},
    {x, w, y, bias, group_list},                           // BlockMmad: c -> yGm
    {share_input, y, sharedInputOffset, sharedInputLen, N, batch, residualScale},
    {y, w_scale, x_scale, bias, logit, row_index, baseM, baseN}, // Epilogue: scatter -> yGm
    gmmParams
};
```

对比可见，非确定性模式中 BlockMmad 和 Epilogue 都直接操作 yGm。

#### 5.2.5 workspace 偏移计算

```cpp
// line 106-107
constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024 * 1024;
GM_ADDR deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE;
```

确定性 buffer 起始于 workspace 偏移 16MB 处（跳过系统预留空间）。这与 `GetWorkspaceSize()` 中 `SYS_WORKSPACE_SIZES` 的值一致。

#### 5.2.6 SyncConfig 配置

```cpp
// line 153-159
uint64_t totalM = static_cast<uint64_t>(matmulTiling_.M);
GMMFRDeterministic::SyncConfig syncConfig{};
syncConfig.curM = totalM;
syncConfig.lowBoundM = totalM;
syncConfig.windowSize = totalM;
syncConfig.baseN = baseN;
```

**与 A3 的关键差异**：A3 使用窗口化同步（`windowSize = deterWorkspaceSize / (n * sizeof(float))`），分批聚合以适应有限的 workspace。A5 由于 workspace 足够大（通过 Tiling 层的溢出保护确保 M*N*sizeof(float) <= deterWorkspaceSize），可以一次性处理所有行，因此 `windowSize = totalM`，无需分批。

`baseN` 的计算（line 149-150）：
```cpp
uint64_t nTimes = Ceil(matmulTiling_.N, DETER_UB_SIZE / sizeof(float));
uint64_t baseN = Ceil(Ceil(matmulTiling_.N, nTimes), 128) * 128;
```

这确保 `baseN` 是 128 的整数倍，用于 N 维度的分块读取。

### 5.3 聚合函数 (gmm_fr_deterministic_a5.h)

**文件**：`op_kernel/arch35/gmm_fr_deterministic_a5.h`

#### 5.3.1 AIC Core SyncAll 配对处理

```cpp
// line 67-72
if (g_coreType == AIC) {
    SyncAll();  // 配对 VEC Core 的第一次 SyncAll (line 74)
    SyncAll();  // 配对 VEC Core 的第二次 SyncAll (line 111)
    return;
}
```

A5 的 `SyncAll` 要求所有 core 必须参与。AIC Core 不做聚合工作，但必须调用 `SyncAll` 以满足配对要求。两次 `SyncAll` 分别对应：
- 第一次：等待所有 Epilogue workspace 写入完成
- 第二次：等待所有聚合写入完成

**与 A3 的对比**：A3 中 AIC Core 可以直接 `return` 跳过 `SyncAll`（line 649-651），因为 A3 (arch32) 的 `SyncAll` 不要求严格配对。

#### 5.3.2 行所有权分配（outRow % coreNumVec）

```cpp
// line 81-87
for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
    auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(baseOffset + mOffset));
    if (outRow % coreNumVec != GetBlockIdx()) {
        continue;
    }
    // ... 处理该行 ...
}
```

`outRow % coreNumVec` 确保每个输出行只有一个 core 处理。`coreNumVec = coreNum * GetTaskRation()`，其中 `GetTaskRation()` 返回每个 AIC Core 对应的 AIV Core 数量（通常为 2）。

**确定性保证**：由于每个 outRow 只有一个写者，`SetAtomicAdd` 实际上退化为普通写入（读取 -> 加法 -> 写回的 AtomicAdd 操作只有一个参与者），因此结果确定。

**与 A3 的对比**：A3 的行所有权分配逻辑完全相同（`outRow % coreNumVec != GetBlockIdx()`），但 A3 的 `coreNumVec = tiling->coreNum * GetTaskRation()`，而 A5 使用 `coreNum * GetTaskRation()`（从 Tiling 的 `usedCoreNum` 获取）。

#### 5.3.3 聚合循环详解

```cpp
// line 89-109
uint64_t curVecBaseN = syncConfig.baseN;
for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
    if (nOffset + syncConfig.baseN >= n) {
        curVecBaseN = n - nOffset;
    }

    LocalTensor<DTYPE_OUT> bindLocal = queBind.AllocTensor<DTYPE_OUT>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
    DataCopyPadExtParams<DTYPE_OUT> padParams{false, 0, 0, 0};
    DataCopyPad(bindLocal, deterBufferGm[mOffset * n + nOffset], copyParams, padParams);
    queBind.EnQue(bindLocal);
    bindLocal = queBind.DeQue<DTYPE_OUT>();

    SetAtomicAdd<DTYPE_OUT>();
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
    DataCopyPad(yGm[outRow * n + nOffset], bindLocal, paramsOut);
    SetAtomicNone();

    queBind.FreeTensor(bindLocal);
}
```

聚合循环的步骤：
1. 按 `baseN` 分块处理 N 维度
2. 从 workspace (`deterBufferGm`) 读取一行数据到 UB
3. 使用 `SetAtomicAdd` 将数据写入 yGm 的对应位置
4. 由于行所有权机制，每个位置只有一个写者，AtomicAdd 退化为普通写

**workspace 地址计算**：`deterBufferGm[mOffset * n + nOffset]`，其中 `mOffset` 是全局输入行索引（从 0 到 totalM-1）。

**yGm 地址计算**：`yGm[outRow * n + nOffset]`，其中 `outRow` 通过 `tokenRanksGm.GetValue(baseOffset + mOffset)` 查询 rowIndex 得到。

**与 A3 FRDeterministic 的对比**（`grouped_matmul_finalize_routing.h`, line 647-681）：

| 维度 | A3 FRDeterministic | A5 FRDeterministicA5 |
|------|--------------------|-----------------------|
| AIC Core 处理 | 直接 return 跳过 | 必须调用 SyncAll 配对 |
| 数据读取 | DataCopyPad2D（2D 拷贝） | DataCopyPad（1D 拷贝） |
| 聚合循环结构 | 相同的 mOffset 遍历 + outRow 所有权 | 相同 |
| baseN 分块 | 相同 | 相同 |
| 同步机制 | 2 次 SyncAll | 2 次 SyncAll（含 AIC 配对） |

---

## 6. 调用链完整追踪

从 Python 调用到最终输出的完整代码路径：

### 6.1 Python -> GE

```
Python: torch.use_deterministic_algorithms(True)
    -> GE 图编译时设置 deterministic 属性
```

### 6.2 GE -> aclnn

```
aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSize(x, w, scale, bias, pertokenScale, ...)
    -> aclnn_grouped_matmul_finalize_routing.cpp (line 527)
    -> 平台判断: NpuArch::DAV_3510 -> 使用 A5 Checker
    -> 调用 AddOp() 将算子加入执行器
    -> GE 框架自动将 deterministic 标志设置到 TilingContext
```

### 6.3 aclnn -> Tiling

```
GroupedMatmulFinalizeRoutingQuantTiling::GetShapeAttrsInfo()
    -> GroupedQmmTiling::GetShapeAttrsInfo()

GroupedMatmulFinalizeRoutingQuantTiling::DoOpTiling()
    -> 填充 tilingData_ 基本字段
    -> context_->GetDeterministic() == 1 检查
    -> 条件满足：设置 deterministicFlag_, deterWorkspaceSize_
    -> 溢出保护检查

GroupedMatmulFinalizeRoutingQuantTiling::GetWorkspaceSize()
    -> totalWorkspace = SYS_WORKSPACE_SIZES + deterWorkspaceSize_

GroupedMatmulFinalizeRoutingQuantTiling::PostTiling()
    -> memcpy tilingData_ -> raw tiling data
```

### 6.4 Tiling -> Kernel

```
grouped_matmul_finalize_routing_pertoken_dequant(x, w, w_scale, bias, x_scale, group_list,
                                                  share_input, logit, row_index, offset, y, workspaceGM, tilingGM)
    -> REGISTER_TILING_DEFAULT + GET_TILING_DATA: 解析 tilingGM
    -> gmmFinalizeRoutingQuantParams_ = tilingData.gmmFinalizeRoutingDataParams

    [确定性分支]
    -> deterBuffer = workspaceGM + 16MB
    -> 构造 GMMTilingDeterministic, ParamsDeterministic
    -> GmmKernelDeterministic gmm; gmm(params):
        -> operator():
            -> [AIV] prologueOp_.Init(params.prologueParams)
                -> Prologue: yGm 初始化为零 + 写入 residual
            -> [AIV] epilogueDequantOp_.Init(params.epilogueParams)
                -> SequentialWrite Epilogue: 设置 yGlobal_ -> deterBuffer
            -> SyncAll<false>()
            -> for groupIdx = 0..groupNum-1:
                -> UpdateGroupParams: 更新 problemShape, baseOffset
                -> ProcessSingleGroup:
                    -> UpdateGlobalBuffer: 更新 A/B/global 地址
                    -> while (bs.GetTileIdx):
                        -> [AIC] mmadOp_: 执行矩阵乘法 -> L0C -> UB
                        -> [AIC] NotifyVector
                        -> [AIV] WaitForCube
                        -> [AIV] epilogueDequantOp_():
                            -> 反量化 + logit 乘法
                            -> VectorSequentialWrite: 写入 workspace
                        -> [AIV] NotifyCube
            -> End()

    -> FRDeterministicA5(syncConfig, deterBufferGm, yGm, rowIndexGm, queBind, coreNum, n):
        -> [AIC] SyncAll x2 + return
        -> [VEC] SyncAll
        -> for mOffset = 0..totalM-1:
            -> outRow = rowIndex[mOffset]
            -> if outRow % coreNumVec != blockIdx: continue
            -> DataCopyPad: workspace[mOffset*N+nOffset] -> UB
            -> AtomicAdd: UB -> yGm[outRow*N+nOffset]
        -> SyncAll

    [非确定性分支]
    -> 构造 Params, GmmKernel gmm; gmm(params)
        -> Epilogue 使用 VectorAtomicProcess (scatter + AtomicAdd)
```

---

## 7. A3 vs A5 关键差异对照表

| 维度 | A3 (arch32) | A5 (arch35) |
|------|-------------|-------------|
| **编程模型** | 原生 AscendC，手动管理循环 | Cgmct Builder 框架，模板化组件 |
| **Epilogue 切换** | 运行时 if-else 分支 (`if deterministicFlag == 1`) | 编译时类型隔离（不同 Epilogue 类型 -> 不同 Kernel 实例化） |
| **mOffset 语义** | 全局偏移（`mGlobalOffset`，用户直接管理跨 group 累加） | group 内偏移（框架管理 group 循环，需 `accumulatedGroupOffset_` 恢复全局） |
| **全局行索引** | 直接使用 `mGlobalOffset + offsetM` | 通过 `accumulatedGroupOffset_ + offsetM` 计算 |
| **AIC Core SyncAll** | 可跳过 (`if ASCEND_IS_AIC { return; }`) | 必须配对（调用 `SyncAll` 但不做实际工作） |
| **workspace 写入** | `DataCopyPad2D`（2D 拷贝） | `DataCopyPad`（1D 拷贝） |
| **GMMTiling 类型** | 单一类型 | 类型隔离（`GmmKernel::GMMTiling` vs `GmmKernelDeterministic::GMMTiling`） |
| **聚合同步** | 窗口化（`windowSize = deterWorkspaceSize / (n * sizeof(float))`） | 一次性（`windowSize = totalM`，通过 Tiling 溢出保护确保安全） |
| **Tiling 设置方式** | `tilingData_.set_deterministicFlag()` | `tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1` |
| **溢出保护** | Kernel 层窗口化处理 | Tiling 层预检查 + 自动降级 |
| **确定性条件** | `GetDeterministic() + dtype check` | `GetDeterministic() + !IsMicroScaling() + INT8xINT8 + PERTOKEN_MODE` |
| **Epilogue 文件** | 内联在 `VectorAtomicProcess` 中 | 独立文件 `block_epilogue_dequant_sequential_write.h` |
| **Prologue 目标** | yGm (与确定性模式相同) | yGm (与确定性模式相同) |

---

## 8. 验证结果

### 8.1 确定性验证

确定性模式的核心验证标准是：**相同输入，多次运行结果完全一致（bit-exact）**。

验证方法：
1. 构造典型 MoE 场景输入（多 group，多 token，多 expert）
2. 启用确定性模式运行 N 次（N >= 10）
3. 比较每次输出的 bit-exact 一致性
4. 对比非确定性模式（使用 AtomicAdd scatter），验证非确定性模式下存在微小差异

### 8.2 精度验证

确定性模式与参考实现（CPU 或非确定性模式的均值）的精度对比，要求误差在 FP32 精度范围内。

### 8.3 边界条件

- 空 group（m=0）：框架自动跳过
- 单行 group：正常处理
- workspace 溢出：Tiling 层自动降级为非确定性
- 无 sharedInput：Prologue 仅初始化为零
- 不同 rowIndex 类型（int32/int64）：模板参数处理

---

## 9. 注意事项与扩展性

### 9.1 当前限制

1. **仅支持 W8A8 INT8 PerToken 模式**：确定性功能目前仅支持 `aDtype == INT8, bDtype == INT8, aQuantMode == PERTOKEN_MODE`。其他量化模式（MX 微缩放等）暂不支持确定性。

2. **workspace 大小限制**：确定性 buffer 需要至少 `M * N * sizeof(float)` 字节。如果超过 96MB（或 64MB），会自动降级为非确定性。

3. **SyncAll 全局同步开销**：`FRDeterministicA5` 中的两次 `SyncAll` 引入同步开销。对于小规模计算，这个开销可能比较显著。

### 9.2 扩展方向

1. **支持更多量化模式**：如果需要支持 MX 微缩放的确定性模式，需要：
   - 修改 `DoOpTiling` 中的条件判断（去掉 `!IsMicroScaling()` 条件）
   - 确认 `BlockEpilogueDequantSequentialWrite` 的反量化逻辑对 MX 模式正确

2. **窗口化聚合**：如果未来需要支持超大规模场景（M * N > 96MB），需要引入 A3 风格的窗口化聚合机制，将聚合分为多个批次。

3. **性能优化**：
   - 考虑将聚合阶段的 N 维度循环展开
   - 考虑使用 `DataCopyPad2D` 替代 1D `DataCopyPad` 以减少循环开销
   - 考虑 UB ping-pong double buffer 以隐藏延迟

### 9.3 代码维护注意事项

1. **GMMTiling 类型隔离**：修改 `KernelGmmFinalizeRoutingPertokenDequant` 时，注意 `GmmKernel` 和 `GmmKernelDeterministic` 都会受影响。如果修改了 `GMMTiling` 结构，两处 Params 构造都需要更新。

2. **accumulatedGroupOffset_ 的正确性依赖于框架的 baseOffset 传递**：如果 Cgmct 框架的 `UpdateGlobalBuffer` 或 `UpdateOffset` 逻辑发生变化，需要重新验证 `SEQ_LOGIT_INDEXS` 是否仍然正确表示全局行偏移。

3. **SyncAll 配对**：A5 的 `SyncAll` 必须严格配对。如果增加新的同步点，必须确保 AIC 和 AIV Core 的 `SyncAll` 调用次数一致。

4. **workspace 偏移**：`deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE`。如果 `SYS_WORKSPACE_SIZES` 在 `GmmConstant` 中发生变化，需要同步更新 Kernel 入口的偏移计算。

---

## 附录 A：文件路径索引

所有路径相对于 `ops-transformer_AI/gmm/` 目录：

| 文件 | 完整路径 |
|------|----------|
| A5 Tiling Data | `grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` |
| A5 Tiling Header | `grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` |
| A5 Tiling Impl | `grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` |
| A5 Kernel Entry | `grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` |
| A5 Deterministic Agg | `grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h` |
| SequentialWrite Epilogue | `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` |
| Original Epilogue | `common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` |
| Cgmct Kernel | `common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` |
| Prologue | `common/cgmct/prologue/block_prologue_finalize_routing.h` |
| A3 Kernel | `grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing.h` |
| A3 Base Tiling | `grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_base_tiling.cpp` |
| aclnn Impl | `grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp` |

## 附录 B：Tiling Key 说明

A5 的 Tiling Key 文件 (`op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h`) 定义了模板参数：
- `ATRANS` (0-1): 输入是否转置
- `BTRANS` (0-1): 权重是否转置
- `SCALETYPE` (0-2): scale 数据类型 (0=float, 1=未使用, 2=bf16)
- `ROWINDEXTYPE` (0-1): rowIndex 数据类型 (0=int64, 1=int32)

注意：确定性标志 **不包含在 Tiling Key 中**。这是因为确定性模式使用相同的 Tiling Key（相同的模板实例化），在运行时通过 `if-else` 分支选择不同的 Kernel。这也是为什么必须在 Kernel 入口维护两个独立的 Kernel 实例化。
