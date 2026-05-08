# GMMFR 确定性特性 A5 迁移 — 代码修改明细

> 版本：v1.0
> 范围：仅 W8A8（INT8×INT8）PerToken 全量化路径
> 策略：Plan A（Params 重定向 y 地址到 workspace + FRDeterministicA5 延迟聚合）
> 基于仓库：`ops-transformer_AI/gmm/grouped_matmul_finalize_routing`

---

## 修改文件总览

| # | 文件路径 | 操作 | 层级 | 优先级 |
|---|---|---|---|---|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | **修改** | Tiling 数据结构 | P0 |
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | **修改** | Tiling 声明 | P1 |
| 3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | **修改** | Tiling 计算 | P1 |
| 4 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | **新增** | Kernel 确定性聚合 | P2 |
| 5 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | **修改** | Kernel W8A8 路径 | P3 |

**不修改的文件**（明确排除）：

| 文件 | 原因 |
|---|---|
| `op_kernel/arch35/grouped_matmul_finalize_routing.h` | MX 格式路径（FP8/FP4），不在 W8A8 范围内 |
| `op_kernel/arch35/weight_quant_basic_block/*` | Weight Quant 伪量化路径（FP8×FP4），不在范围内 |
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | 确定性分支在 pertoken_dequant.h 内部处理，无需修改入口 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.*` | Weight Quant 不在范围内 |
| `op_host/grouped_matmul_finalize_routing_def.cpp` | 算子定义，确定性是运行时行为 |
| `op_api/*` | 接口层，workspace 框架自动管理 |

---

## 文件 1：Tiling 数据结构（P0）

### 文件路径
`op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`

### 修改位置
第 26-39 行，`GMMFinalizeRoutingDataParams` 结构体

### 修改原因
Kernel 执行时需要从 Tiling 数据读取确定性开关（`deterministicFlag`）和确定性 workspace 大小（`deterWorkspaceSize`）。当前结构体的 `reserved2` 字段未被使用，正好替换为这两个确定性字段。A3 原型在 `op_host/grouped_matmul_finalize_routing_tiling.h` 第 59-60 行使用 `TILING_DATA_FIELD_DEF` 宏定义了相同的字段。

### 修改前代码（第 26-40 行）

```cpp
#pragma pack(push, 8)
struct GMMFinalizeRoutingDataParams {
    uint32_t groupNum = 0;
    uint32_t batch = 0;
    uint32_t sharedInputOffset = 0;
    uint32_t sharedInputLen = 0;
    float residualScale = 0;
    uint32_t aQuantMode = 0;
    uint32_t bQuantMode = 0;
    uint32_t biasDtype = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint16_t reserved1 = 0;
    uint32_t reserved2 = 0;      // ← 替换此行
};
#pragma pack(pop)
```

### 修改后代码

```cpp
#pragma pack(push, 8)
struct GMMFinalizeRoutingDataParams {
    uint32_t groupNum = 0;
    uint32_t batch = 0;
    uint32_t sharedInputOffset = 0;
    uint32_t sharedInputLen = 0;
    float residualScale = 0;
    uint32_t aQuantMode = 0;
    uint32_t bQuantMode = 0;
    uint32_t biasDtype = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint16_t reserved1 = 0;
    // ↓↓↓ 替换 reserved2 ↓↓↓
    uint32_t deterministicFlag = 0;      // 0=非确定性, 1=确定性
    uint32_t deterWorkspaceSize = 0;     // 确定性 workspace 大小（字节）
};
#pragma pack(pop)
```

### 注意事项
- 新增 `deterministicFlag` 和 `deterWorkspaceSize` 共 8 字节，替换原来 `reserved2` 的 4 字节
- **结构体大小变化**：原结构体 `reserved2` 为 4 字节，替换为 2 个 uint32_t = 8 字节，结构体增大 4 字节
- 由于使用了 `#pragma pack(push, 8)` 对齐，原 `reserved2` 后面可能有 padding，需确认实际 sizeof 是否变化
- **建议**：如果需要保持结构体大小不变，可以将 `reserved1`（uint16_t）改为 uint8_t 以腾出空间，或者确认 Tiling 序列化机制是否容忍大小变化

---

## 文件 2：Tiling 类声明（P1）

### 文件路径
`op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h`

### 修改位置
第 94-125 行，`GroupedMatmulFinalizeRoutingQuantTiling` 类的 private 区域

### 修改原因
Tiling 类需要在 CPU 侧维护确定性相关的成员变量，用于在 `DoOpTiling()` 中计算并传递给 Kernel。A3 原型在 `op_host/grouped_matmul_finalize_routing_base_tiling.h` 第 87 行声明了相同成员。

### 修改前代码（第 117-125 行）

```cpp
    GMMFinalizeRoutingTilingData tilingData_;
    uint64_t sharedInputLen_ = 0;
    uint64_t sharedInputOffset_ = 0;
    uint64_t rowIndex_ = 0;
    float sharedInputWeight_ = 1.0;
    uint64_t outputBs_ = 0;
    int8_t scaleType_ = 0;
    int8_t rowIndexType_ = 0;
};
```

### 修改后代码

```cpp
    GMMFinalizeRoutingTilingData tilingData_;
    uint64_t sharedInputLen_ = 0;
    uint64_t sharedInputOffset_ = 0;
    uint64_t rowIndex_ = 0;
    float sharedInputWeight_ = 1.0;
    uint64_t outputBs_ = 0;
    int8_t scaleType_ = 0;
    int8_t rowIndexType_ = 0;
    // ↓↓↓ 新增：确定性相关成员 ↓↓↓
    uint32_t deterministicFlag_ = 0;       // 确定性标志：0=关闭, 1=开启
    uint32_t deterWorkspaceSize_ = 0;      // 确定性 workspace 大小
};
```

---

## 文件 3：Tiling 计算实现（P1）

### 文件路径
`op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`

### 修改位置 A：`DoOpTiling()` 函数，第 456 行之后

### 修改原因
`DoOpTiling()` 是 CPU 侧 Tiling 计算的核心函数。需要在所有参数填充完成后，读取框架的 `GetDeterministic()` 开关，计算确定性 workspace 大小并写入 Tiling 数据。A3 原型为 `DeterministicTilingProcess()` 函数（base_tiling.cpp 第 413-425 行）。

### 修改前代码（第 445-460 行）

```cpp
ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoOpTiling()
{
    tilingData_.gmmFinalizeRoutingDataParams.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmFinalizeRoutingDataParams.batch = static_cast<uint32_t>(outputBs_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset = static_cast<uint32_t>(sharedInputOffset_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen = static_cast<uint32_t>(sharedInputLen_);
    tilingData_.gmmFinalizeRoutingDataParams.residualScale = static_cast<float>(sharedInputWeight_);
    tilingData_.gmmFinalizeRoutingDataParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    tilingData_.gmmFinalizeRoutingDataParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmFinalizeRoutingDataParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias ? 1 : 0);

    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}
```

### 修改后代码

```cpp
ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::DoOpTiling()
{
    tilingData_.gmmFinalizeRoutingDataParams.groupNum = static_cast<uint32_t>(inputParams_.groupNum);
    tilingData_.gmmFinalizeRoutingDataParams.batch = static_cast<uint32_t>(outputBs_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset = static_cast<uint32_t>(sharedInputOffset_);
    tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen = static_cast<uint32_t>(sharedInputLen_);
    tilingData_.gmmFinalizeRoutingDataParams.residualScale = static_cast<float>(sharedInputWeight_);
    tilingData_.gmmFinalizeRoutingDataParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmFinalizeRoutingDataParams.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    tilingData_.gmmFinalizeRoutingDataParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmFinalizeRoutingDataParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias ? 1 : 0);

    // ↓↓↓ 新增：确定性 Tiling 处理 ↓↓↓
    // A3 原型: DeterministicTilingProcess() (base_tiling.cpp 第413-425行)
    // 仅在 PerToken 模式（非 MX）且数据类型为 INT8 时支持确定性
    if (context_->GetDeterministic() == 1 && !IsMicroScaling() &&
        inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8) {
        deterministicFlag_ = 1;
        tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t l2Size = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
        // A3 原型: l2 > 96MB 用 96MB，否则用 64MB
        constexpr uint32_t DETER_WORK_SPACE_SIZE = 96UL * 1024 * 1024;
        constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64UL * 1024 * 1024;
        deterWorkspaceSize_ = l2Size > DETER_WORK_SPACE_SIZE
                              ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
        tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
    }
    // ↑↑↑ 新增结束 ↑↑↑

    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}
```

### 修改位置 B：`PostTiling()` 函数，第 512-529 行

### 修改原因
如果结构体大小因新增字段而变化，需要确保 `PostTiling()` 中 `tilingDataSize` 和 `memcpy_s` 操作使用新的结构体大小。由于 `sizeof(GMMFinalizeRoutingTilingData)` 会自动反映结构体大小变化，此处**无需额外修改**，但需要确认 `GetRawTilingData()->GetCapacity()` 足以容纳新的 Tiling 数据。

### 修改位置 C：`PrintQuantParams()` 函数末尾（第 545-563 行）

### 修改原因
调试时需要打印确定性参数，与 A3 原型保持一致。

### 修改前代码（第 553-561 行）

```cpp
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << tilingData_.gmmFinalizeRoutingDataParams.groupNum
        << ", groupListType = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.groupListType)
        << ", batch = " << tilingData_.gmmFinalizeRoutingDataParams.batch
        << ", sharedInputOffset = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset
        << ", sharedInputLen = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen
        << ", residualScale = " << tilingData_.gmmFinalizeRoutingDataParams.residualScale
        << ", aQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.aQuantMode
        << ", bQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.bQuantMode
        << ", hasBias = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.hasBias);
    OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
```

### 修改后代码

```cpp
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << tilingData_.gmmFinalizeRoutingDataParams.groupNum
        << ", groupListType = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.groupListType)
        << ", batch = " << tilingData_.gmmFinalizeRoutingDataParams.batch
        << ", sharedInputOffset = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputOffset
        << ", sharedInputLen = " << tilingData_.gmmFinalizeRoutingDataParams.sharedInputLen
        << ", residualScale = " << tilingData_.gmmFinalizeRoutingDataParams.residualScale
        << ", aQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.aQuantMode
        << ", bQuantMode = " << tilingData_.gmmFinalizeRoutingDataParams.bQuantMode
        << ", hasBias = " << static_cast<uint32_t>(tilingData_.gmmFinalizeRoutingDataParams.hasBias)
        // ↓↓↓ 新增：打印确定性参数 ↓↓↓
        << ", deterministicFlag = " << tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag
        << ", deterWorkspaceSize = " << tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize;
        // ↑↑↑ 新增结束 ↑↑↑
    OP_LOGD(context_->GetNodeName(), "%s", oss.str().c_str());
```

---

## 文件 4：新增 A5 确定性聚合函数（P2）

### 文件路径（新增）
`op_kernel/arch35/gmm_fr_deterministic_a5.h`

### 新增原因
A5 平台的确定性延迟聚合函数，参考 A3 的 `FRDeterministic()`（`op_kernel/grouped_matmul_finalize_routing.h` 第 653-687 行）。适配 A5 的 Cgmct 框架和更严格的同步要求（SyncAll 必须 1:1 配对）。

### 完整新文件内容

```cpp
/**
 * 新增文件: op_kernel/arch35/gmm_fr_deterministic_a5.h
 *
 * 功能: A5 平台确定性延迟聚合函数
 * A3 原型: op_kernel/grouped_matmul_finalize_routing.h 第653-687行 (FRDeterministic)
 *          op_kernel/grouped_matmul_finalize_routing.h 第622-649行 (VectorSync)
 *
 * 核心原理:
 *   1. SyncAll() - 全核同步，确保所有核已完成 workspace 写入
 *   2. 按 outRow % coreNumVec 分配行归属（每个核只处理自己负责的行）
 *   3. 从 workspace 中间结果读取到 UB (queBind)
 *   4. SetAtomicAdd + DataCopyPad 写入最终 yGm（此时只有一个核写入该行，顺序确定）
 *   5. SyncAll() - 再次同步，确保所有核完成写入
 */

#ifndef GMM_FR_DETERMINISTIC_A5_H
#define GMM_FR_DETERMINISTIC_A5_H

#include "kernel_operator.h"

namespace GMMFRDeterministic {

using namespace AscendC;

// 与 A3 保持一致
constexpr uint32_t DETER_UB_SIZE = 12 * 1024;   // 确定性 UB 缓冲区大小 (12KB)
constexpr uint32_t BUFFER_NUM = 2;

// 滑动窗口同步配置（与 A3 SyncConfig 完全一致）
struct SyncConfig {
    uint64_t curM = 0;        // 当前累计行偏移
    uint64_t curGroup = 0;    // 当前 group 索引
    uint64_t curGroupM = 0;   // 当前 group 内累计行
    uint64_t lowBoundM = 0;   // 窗口下界
    uint64_t windowSize = 0;  // 窗口大小 = deterWorkspaceSize / (N * sizeof(float))
    uint64_t baseN = 0;       // N 方向分块大小，128 对齐
};

/**
 * FRDeterministicA5 - A5 版确定性延迟聚合核心函数
 *
 * A3 原型: QuantGroupMatmul<P>::FRDeterministic()
 *         (op_kernel/grouped_matmul_finalize_routing.h 第653-687行)
 *
 * 模板参数:
 *   DTYPE_OUT       - 输出数据类型 (float)
 *   ROW_INDEX_DTYPE - 行索引数据类型 (int64_t 或 int32_t)
 *
 * 执行流程:
 *   1. SyncAll() 全核同步
 *   2. 按 outRow % coreNumVec 分配行归属
 *   3. workspace → UB (queBind) → SetAtomicAdd 写入 yGm
 *   4. SyncAll() 全核同步
 */
template <typename DTYPE_OUT, typename ROW_INDEX_DTYPE>
__aicore__ inline void FRDeterministicA5(
    SyncConfig& syncConfig,
    GlobalTensor<DTYPE_OUT>& deterBufferGm,        // workspace 确定性缓冲区
    GlobalTensor<DTYPE_OUT>& yGm,                  // 最终输出
    GlobalTensor<ROW_INDEX_DTYPE>& tokenRanksGm,   // 行索引
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>& queBind,  // UB 中转 buffer
    uint32_t coreNum,                              // AIC 核数
    uint32_t n)                                    // 总列数 N
{
    if (g_coreType == AIC) {
        return;                    // Cube 核不参与
    }

    SyncAll();                     // 第一步：全核同步，确保 workspace 写入完成

    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = coreNum * GetTaskRation();  // Vector 核总数
    uint64_t baseOffset = syncConfig.lowBoundM - syncConfig.windowSize;

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        // 从 tokenRanks 获取该行对应的输出行号
        auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(baseOffset + mOffset));

        // 行归属判断：只有 outRow % coreNumVec == GetBlockIdx() 的核才处理
        if (outRow % coreNumVec != GetBlockIdx()) {
            continue;
        }

        uint64_t curVecBaseN = syncConfig.baseN;
        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            if (nOffset + syncConfig.baseN >= n) {
                curVecBaseN = n - nOffset;     // 尾块处理
            }

            // 从 workspace 读取到 UB
            LocalTensor<DTYPE_OUT> bindLocal = queBind.AllocTensor<DTYPE_OUT>();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
            DataCopyPad(bindLocal, deterBufferGm[mOffset * n + nOffset], copyParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_OUT>();

            // 原子写入最终输出（此时只有归属核写入，顺序确定）
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
            DataCopyPad(yGm[outRow * n + nOffset], bindLocal, paramsOut);
            SetAtomicNone();

            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();                     // 最后一步：全核同步，确保写入完成
}

}  // namespace GMMFRDeterministic

#endif
```

### 关键设计说明

| 设计点 | 说明 |
|---|---|
| **命名空间** | 使用 `GMMFRDeterministic` 命名空间，避免与现有代码冲突 |
| **SyncAll 配对** | A5 要求 SyncAll 必须在所有核上严格 1:1 配对，不能有条件跳过 |
| **queBind 初始化** | 在调用方（pertoken_dequant.h）中初始化，此处仅使用引用 |
| **模板参数 ROW_INDEX_DTYPE** | 支持 int64_t 和 int32_t 两种行索引类型 |
| **尾块处理** | N 方向最后一块可能不足 baseN，需要特殊处理 |
| **行归属策略** | `outRow % coreNumVec == GetBlockIdx()`，与 A3 完全一致 |

---

## 文件 5：Kernel W8A8 路径（P3，核心修改）

### 文件路径
`op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`

### 修改位置 A：文件开头 include 区域（第 18-24 行之后）

### 修改原因
需要引入新增的确定性聚合函数头文件。

### 修改前代码（第 18-24 行）

```cpp
#include "cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h"
#include "cgmct/block/block_mmad_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;
```

### 修改后代码

```cpp
#include "cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h"
#include "cgmct/block/block_mmad_builder.h"
#include "cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "grouped_matmul_finalize_routing_tiling_data.h"
#include "gmm_fr_deterministic_a5.h"                  // ← 新增
using namespace GMMFRDeterministic;                   // ← 新增

using namespace Cgmct::Gemm;
using namespace Cgmct::Gemm::Kernel;
```

---

### 修改位置 B：`grouped_matmul_finalize_routing_pertoken_dequant()` 函数体（第 73-91 行）

### 修改原因
这是 W8A8/INT8 PerToken 全量化路径的 Cgmct Kernel。当前 `BlockEpilogueDequantFinalizeRouting` 使用 `SetAtomicAdd` 直接写 `yGm`（非确定性根源）。确定性模式下需要将 Epilogue 的输出目标从 `yGm` 重定向到 workspace，Cgmct Kernel 完成后再调用 `FRDeterministicA5` 聚合。

### 修改前代码（第 73-91 行）

```cpp
    GMMTiling gmmParams{gmmFinalizeRoutingQuantParams_.groupNum,
                        gmmFinalizeRoutingQuantParams_.groupListType,
                        matmulTiling_.baseM,
                        matmulTiling_.baseN,
                        matmulTiling_.baseK,
                        gmmFinalizeRoutingQuantParams_.hasBias};

    gmmParams.matmulTiling = &matmulTiling_;
    Params params = {
        {1, 1, 1, 1},                // problem shape
        {x, w, y, bias, group_list}, // BlockMmadParams
        {share_input, y, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
         gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N, gmmFinalizeRoutingQuantParams_.batch,
         gmmFinalizeRoutingQuantParams_.residualScale},                                          // prologue params
        {y, w_scale, x_scale, bias, logit, row_index, matmulTiling_.baseM, matmulTiling_.baseN}, // epilogue params
        gmmParams};
    GmmKernel gmm;
    gmm(params);
}
```

### 修改后代码

```cpp
    GMMTiling gmmParams{gmmFinalizeRoutingQuantParams_.groupNum,
                        gmmFinalizeRoutingQuantParams_.groupListType,
                        matmulTiling_.baseM,
                        matmulTiling_.baseN,
                        matmulTiling_.baseK,
                        gmmFinalizeRoutingQuantParams_.hasBias};

    gmmParams.matmulTiling = &matmulTiling_;

    // ↓↓↓ 新增：确定性/非确定性分支 ↓↓↓
    if (gmmFinalizeRoutingQuantParams_.deterministicFlag == 1) {
        // ============ 确定性模式 ============
        // 策略：将 Cgmct Params 中的 y 地址重定向到 workspace 中的确定性缓冲区
        // Cgmct Kernel 正常执行，Epilogue 的 SetAtomicAdd 写到 workspace 而非 yGm
        // Kernel 完成后调用 FRDeterministicA5 聚合

        // 计算确定性缓冲区偏移（在已有 workspace 之后）
        // A3 原型: Init 第162-166行
        //   workspace + parallNum * baseM * baseN * sizeof(int32_t) * coreNum
        uint64_t deterBufferOffset = static_cast<uint64_t>(matmulTiling_.usedCoreNum) *
            matmulTiling_.baseM * matmulTiling_.baseN * sizeof(int32_t);
        GM_ADDR deterBuffer = workspaceGM + deterBufferOffset;

        // Params 中 y 全部替换为 deterBuffer
        Params params = {
            {1, 1, 1, 1},
            {x, w, deterBuffer, bias, group_list},              // y → workspace
            {share_input, deterBuffer,                           // prologue y → workspace
             gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N,
             gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},
            {deterBuffer, w_scale, x_scale, bias, logit, row_index,  // epilogue y → workspace
             matmulTiling_.baseM, matmulTiling_.baseN},
            gmmParams};

        // 执行 Cgmct Kernel（输出到 workspace）
        GmmKernel gmm;
        gmm(params);

        // ============ 确定性聚合：workspace → yGm ============
        // A3 原型: Process 第303-306行（SyncConfig 初始化）
        //          Process 第331-334行（最终 FRDeterministic 调用）

        // 设置 GlobalTensor 指针
        GlobalTensor<float> deterBufferGm;
        deterBufferGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deterBuffer));
        GlobalTensor<float> yGm;
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        GlobalTensor<rowIndexType> tokenRanksGm;
        tokenRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ rowIndexType*>(row_index));
        GlobalTensor<int64_t> groupTokensGm;
        groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));

        // 初始化 SyncConfig
        SyncConfig syncConfig;
        syncConfig.windowSize = gmmFinalizeRoutingQuantParams_.deterWorkspaceSize /
                                (matmulTiling_.N * sizeof(float));
        syncConfig.lowBoundM = syncConfig.windowSize;
        uint64_t nTimes = Ceil(matmulTiling_.N, DETER_UB_SIZE / sizeof(float));
        syncConfig.baseN = Ceil(Ceil(matmulTiling_.N, nTimes), 128) * 128;  // 128 对齐

        // 初始化 queBind（确定性 UB 中转 buffer）
        // A3 原型: InitUbBuffer 第186-188行
        TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
        TPipe deterPipe;
        deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);

        // 最终聚合：设置 curM 为总行数，一次性处理所有行
        syncConfig.curM = matmulTiling_.M;
        FRDeterministicA5<float, rowIndexType>(
            syncConfig, deterBufferGm, yGm, tokenRanksGm, queBind,
            matmulTiling_.usedCoreNum, matmulTiling_.N);

    } else {
    // ↑↑↑ 确定性分支结束 ↑↑↑

        // ============ 非确定性模式（完全不变）============
        Params params = {
            {1, 1, 1, 1},                // problem shape
            {x, w, y, bias, group_list}, // BlockMmadParams
            {share_input, y, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N, gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},                                          // prologue params
            {y, w_scale, x_scale, bias, logit, row_index, matmulTiling_.baseM, matmulTiling_.baseN}, // epilogue params
            gmmParams};
        GmmKernel gmm;
        gmm(params);

    // ↓↓↓ 非确定性 else 分支闭合 ↓↓↓
    }
    // ↑↑↑ 分支结束 ↑↑↑
}
```

---

## 数据流对比

### 非确定性模式（原有路径，不变）

```
Input: x(INT8), w(INT8, NZ)
  │
  └── Cgmct Kernel
        ├── BlockMmad: Cube INT8×INT8 → INT32
        └── BlockEpilogueDequantFinalizeRouting:
              INT32 → FP32 × scale → FP32
              → SetAtomicAdd(yGm[tokenRanks[row] * N + nOffset])  ← 多核竞争，非确定性
```

### 确定性模式（新增路径）

```
Input: x(INT8), w(INT8, NZ)
  │
  ├── Step 1: Cgmct Kernel (y 重定向到 workspace)
  │     ├── BlockMmad: Cube INT8×INT8 → INT32
  │     └── BlockEpilogueDequantFinalizeRouting:
  │           INT32 → FP32 × scale → FP32
  │           → SetAtomicAdd(deterBuffer[mOffset * N + nOffset])  ← 顺序写入，无竞争
  │
  └── Step 2: FRDeterministicA5 (workspace → yGm)
        ├── SyncAll()                  ← 全核同步
        ├── outRow % coreNumVec == GetBlockIdx()  ← 行归属分配
        ├── 从 workspace → UB (queBind) → SetAtomicAdd(yGm[...])  ← 单核写入，确定性
        └── SyncAll()                  ← 全核同步
```

---

## 模板实例化影响

INT8 PerToken 路径有 4 种模板组合，每种都会进入确定性分支：

| scaleType | rowIndType | 模板实例化 |
|---|---|---|
| 0 (float) | 0 (int64) | `grouped_matmul_finalize_routing_pertoken_dequant<layoutA, layoutB, 0, 0>` |
| 0 (float) | 1 (int32) | `grouped_matmul_finalize_routing_pertoken_dequant<layoutA, layoutB, 0, 1>` |
| 1 (bf16) | 0 (int64) | `grouped_matmul_finalize_routing_pertoken_dequant<layoutA, layoutB, 1, 0>` |
| 1 (bf16) | 1 (int32) | `grouped_matmul_finalize_routing_pertoken_dequant<layoutA, layoutB, 1, 1>` |

所有 4 种组合共用同一个修改点，无需分别处理。

---

## 关键风险与待确认项

| # | 风险项 | 等级 | 说明 | 建议 |
|---|---|---|---|---|
| 1 | **Tiling 结构体大小变化** | 中 | `reserved2`(4B) → `deterministicFlag`(4B) + `deterWorkspaceSize`(4B) = 增加 4B | 确认 Tiling 序列化是否兼容大小变化，必要时调整 padding |
| 2 | **A5 SyncAll 可用性** | 高 | A5 SyncAll 必须严格 1:1 配对，否则死锁 | 如果不可用，改用 `CrossCoreSetFlag/WaitFlag` 全核广播 |
| 3 | **Cgmct Params y 地址替换** | 中 | 将 Params 中所有 y 地址替换为 workspace 地址，需确认 Epilogue/Prologue/BlockMmad 三处 y 参数都能正确重定向 | 编译验证 |
| 4 | **workspace 大小是否足够** | 低 | 确定性缓冲区 = totalM × N × sizeof(float)，滑动窗口限制在 96MB 内 | A3 已验证，A5 逻辑一致 |
| 5 | **TPipe/queBind 初始化时机** | 中 | 在 Cgmct Kernel 完成后才初始化 queBind，需确认不与 Cgmct 内部 pipe 冲突 | 可能需要在函数开头初始化 |
| 6 | **deterBufferOffset 计算** | 中 | `usedCoreNum * baseM * baseN * sizeof(int32_t)` 是否为已有 workspace 大小 | 需要确认 Cgmct 框架实际使用的 workspace 大小 |

---

## 实施顺序与依赖关系

```
P0: 文件1 (tiling_data.h)
 │
 ├──→ P1: 文件2 (quant_tiling.h) + 文件3 (quant_tiling.cpp)
 │         │
 │         └──→ P3: 文件5 (pertoken_dequant.h)
 │                    │
 P2: 文件4 (gmm_fr_deterministic_a5.h) ──→ P3
                                            │
                                       编译验证
                                            │
                                       端到端测试（10次执行 bit-wise 一致）
```

- P0 完成后即可开始 P1 和 P2（可并行）
- P3 依赖 P1 + P2 全部完成
- 测试依赖 P3 完成
