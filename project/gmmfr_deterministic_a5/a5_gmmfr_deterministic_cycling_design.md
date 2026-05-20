# A5 GMMFR W8A8 确定性模式 — Workspace Cycling 实现说明

> 平台：A5 (Ascend 950, DAV3510, arch35)
> 日期：2026-05-14

---

## 1. 概述

A5 GMMFR W8A8 确定性模式统一使用 **per-group 聚合 + workspace 复用** 的方式运行。每个 group 的 matmul 结果写入 workspace 后立即聚合到 yGm，然后 workspace 从头开始写下一个 group，循环利用固定大小的 workspace 空间。

---

## 2. 整体数据流

```
grouped_matmul_finalize_routing_pertoken_dequant.h (入口)
  │
  ├─ deterministicFlag != 1 → 非确定性模式（不变）
  │
  └─ deterministicFlag == 1 → 确定性模式（统一 Cycling 路径）
       │
       1. Prologue: zeros + residual -> yGm
       2. KernelGmmFRDeterministicCycling:
            for each group:
              ProcessSingleGroup -> workspace[0..curM*N]
              End() + ResetWorkspaceOffset()
              FRDeterministicA5(workspace -> yGm, globalRowOffset)
              globalRowOffset += curM
```

---

## 3. 改动文件清单

### 3.1 修改的文件（3 个）

| 文件 | 改动点 |
|------|--------|
| `op_kernel/arch35/gmm_fr_deterministic_a5.h` | 新增 `globalRowOffset` 参数 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 确定性分支统一走 Cycling kernel |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 溢出时返回错误而非降级 |

### 3.2 新增的文件（2 个，复制自共享 Cgmct）

| 文件 | 来源 |
|------|------|
| `op_kernel/arch35/block_epilogue_dequant_sequential_write_cycling.h` | 复制自 `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` |
| `op_kernel/arch35/kernel_gmm_fr_deterministic_cycling.h` | 复制自 `common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h` |

### 3.3 不修改的文件

- `common/cgmct/` 下所有共享框架代码
- `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`（已有 `deterministicFlag` 和 `deterWorkspaceSize` 字段）

---

## 4. 逐文件改动详情

### 4.1 gmm_fr_deterministic_a5.h

**改动 1**：函数签名新增 `globalRowOffset` 参数

```cpp
// 旧
__aicore__ inline void FRDeterministicA5(
    ..., uint32_t coreNum, uint32_t n)

// 新
__aicore__ inline void FRDeterministicA5(
    ..., uint32_t coreNum, uint32_t n,
    uint64_t globalRowOffset = 0)
```

**改动 2**：聚合时 rowIndex 的读取偏移

```cpp
// 旧
uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
uint64_t baseOffset = syncConfig.lowBoundM - syncConfig.windowSize;
auto outRow = tokenRanksGm.GetValue(baseOffset + mOffset);

// 新
uint64_t totalM = syncConfig.curM;
auto outRow = tokenRanksGm.GetValue(globalRowOffset + mOffset);
```

---

### 4.2 grouped_matmul_finalize_routing_pertoken_dequant.h

**改动**：确定性分支统一走 Cycling kernel，不再区分 workspace 是否溢出

```cpp
// 旧：区分溢出/非溢出两条路径
if (deterministicFlag == 1) {
    if (requiredWorkspace <= availableWorkspace) {
        // 标准路径：GmmKernelDeterministic + 外部 FRDeterministicA5
    } else {
        // Cycling 路径：GmmKernelCycling
    }
}

// 新：统一走 Cycling 路径
if (deterministicFlag == 1) {
    GmmKernelCycling gmm;
    gmm(params);   // 内部 per-group 聚合，workspace 循环复用
}
```

同时删除了不再需要的类型别名和 include：
- `BlockEpilogueDequantSeq`
- `GmmKernelDeterministic`
- `ParamsDeterministic`
- `#include "cgmct/epilogue/block_epilogue_dequant_sequential_write.h"`

---

### 4.3 grouped_matmul_finalize_routing_quant_tiling.cpp

```cpp
// 旧：溢出时降级为非确定性
if (requiredDeterSize > allocSize) {
    deterministicFlag_ = 0;
    ...
}

// 新：溢出时返回错误
if (requiredDeterSize > allocSize) {
    OP_LOGE(..., "Deterministic buffer overflow...");
    return ge::GRAPH_FAILED;
}
```

---

### 4.4 block_epilogue_dequant_sequential_write_cycling.h（新增）

**来源**：复制 `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h`

| 改动项 | 旧 | 新 |
|--------|----|----|
| 类名 | `BlockEpilogueDequantSequentialWrite` | `BlockEpilogueDequantSequentialWriteCycling` |
| 宏前缀 | `GMM_BLOCK_EPILOGUE_DEQUANT_SEQUENTIAL_` | `_CYCLING_` |
| Include 路径 | `../utils/xxx.h` | `cgmct/utils/xxx.h` |

**唯一功能新增**：`ResetWorkspaceOffset()` 公有方法

```cpp
__aicore__ inline void ResetWorkspaceOffset() { accumulatedGroupOffset_ = 0; }
```

---

### 4.5 kernel_gmm_fr_deterministic_cycling.h（新增）

**来源**：复制 `common/cgmct/kernel/kernel_gmm_finalize_routing_pertoken_dequant.h`

| 改动项 | 旧 | 新 |
|--------|----|----|
| 类名 | `KernelGmmFinalizeRoutingPertokenDequant` | `KernelGmmFRDeterministicCycling` |
| Include 路径 | `../utils/xxx.h`, `./semaphore.h` | `cgmct/utils/xxx.h`, `cgmct/kernel/semaphore.h` |
| Epilogue | 原始 SequentialWrite | Cycling SequentialWrite |

**新增结构体**：

```cpp
struct CyclingParams {
    GM_ADDR yGmAddr;        // 最终输出地址
    GM_ADDR rowIndexAddr;   // rowIndex 全局地址
    uint32_t coreNum;       // core 数
    uint32_t n;             // N 维度
    uint64_t baseN;         // N 方向分块大小
};
```

**核心改动：`operator()` per-group 聚合循环**

```cpp
// 旧：所有 group 处理完后统一聚合
for (groupIdx = 0; groupIdx < groupNum; groupIdx++) {
    ProcessSingleGroup(params, bs, groupIdx);
}
End();

// 新：每个 group 处理完后立即聚合 + 重置 workspace
// Init 阶段（不变）
prologueOp_.Init + prologueOp_()
mmadOp_.Init(...)
InitParamsAndTensor(params)
BlockSchedulerOp bs(...)
SyncAll<false>()
epilogueDequantOp_.Init(...)

// 聚合用 UB（一次分配，无条件，AIC/AIV 都调用）
TQueBind<VECIN, VECOUT, 1> aggQueBind;
TPipe aggPipe;
aggPipe.InitBuffer(aggQueBind, 2, 12*1024);

// Group 循环
uint64_t globalRowOffset = 0;
for (groupIdx = 0; groupIdx < groupNum; groupIdx++) {
    UpdateGroupParams(params, groupIdx);
    ProcessSingleGroup(params, bs, groupIdx);

    End();
    isVecSetSyncCom_ = false;
    epilogueDequantOp_.ResetWorkspaceOffset();

    // 聚合（无条件调用，AIC/AIV 都执行，FRDeterministicA5 内部区分）
    SyncAll();
    FRDeterministicA5(..., globalRowOffset);
    globalRowOffset += curM;
    SyncAll();
}
```

---

## 5. Include 路径说明

新增文件位于 `op_kernel/arch35/`，非原始 `gmm/common/cgmct/`。构建系统将 `gmm/common/` 加入 include 搜索路径：

```cpp
// 原始（从 cgmct/kernel/ 目录）
#include "../utils/common_utils.h"

// 新文件（从 arch35/ 目录）
#include "cgmct/utils/common_utils.h"
```

---

## 6. SyncAll 时序与平衡性

### 关键约束：AIC 和 AIV 的 SyncAll 必须配对

A5 的 SyncAll 同步 AIC 和 AIV。如果 AIC 和 AIV 调用次数不同，会导致 SyncAll 错位、数据竞争。

### 每轮 group 的 SyncAll 调用

```
AIC:                          AIV:
  SyncAll()  ←──匹配──→        SyncAll()
  FRDeterministicA5() {         FRDeterministicA5() {
    g_coreType==AIC:              g_coreType==AIV:
    SyncAll()  ←──匹配──→          SyncAll()
    SyncAll()  ←──匹配──→          // 聚合 workspace -> yGm //
  }                                 SyncAll()
  SyncAll()  ←──匹配──→        SyncAll()
```

AIC 每轮 4 次 SyncAll，AIV 也是 4 次。平衡。

### 实现要点

`FRDeterministicA5` 必须**无条件调用**（不在 `if ASCEND_IS_AIV` 内），函数内部通过 `g_coreType == AIC` 判断：
- AIC：执行 2 次 SyncAll 后返回（不使用 aggQueBind）
- AIV：执行 2 次 SyncAll（中间做聚合）

`TPipe aggPipe` 和 `aggPipe.InitBuffer` 也必须无条件调用，与原始非溢出路径中外部 TPipe 分配的方式一致。

---

## 7. Workspace 使用

| 场景 | workspace 大小 | 行为 |
|------|---------------|------|
| totalM=8192, N=1024 | 32MB | per-group 聚合（workspace 充裕，每 group 聚合一次） |
| totalM=32768, N=4096 | 96MB 循环复用 | 同上，workspace 每轮复用 |

Cycling 路径下 workspace 固定大小，不随 totalM 增长。
