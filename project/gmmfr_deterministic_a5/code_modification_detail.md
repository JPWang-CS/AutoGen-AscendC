# GMMFR 确定性特性 A5 迁移 — 代码修改明细

> 版本：v3.1 (Plan B - Sequential Write Epilogue)
> 范围：仅 W8A8（INT8×INT8）PerToken 全量化路径
> 策略：Plan B（新建顺序写 Epilogue + FRDeterministicA5 延迟聚合）
> 基于仓库：`ops-transformer_AI/gmm/grouped_matmul_finalize_routing`
> 日期：2026-05-08

---

## 修改文件总览

| # | 文件路径 | 操作 | 层级 |
|---|---------|------|------|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | **修改** | Tiling 数据结构 |
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | **修改** | Tiling 声明 |
| 3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | **修改** | Tiling 计算 |
| 4 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | **新增** | Kernel 确定性聚合 |
| 5 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | **修改** | Kernel W8A8 路径 |
| 6 | `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` | **新增** | Plan B 顺序写 Epilogue |

---

## Plan B 架构说明

### 为什么不用 Plan A（重定向 y 地址）？
原始 Epilogue `VectorAtomicProcess` 使用 scatter 写入：
```cpp
SetAtomicAdd<float>();
for (i) { outRow = rowIndex[i]; DataCopyPad(yGlobal[outRow * n + yOffset], ...); }
SetAtomicNone();
```
如果简单将 y 地址重定向到 workspace，workspace 中数据按 outRow 散列存放，与 FRDeterministicA5 的顺序读取 `mOffset * n + nOffset` 地址不匹配。

### Plan B 方案
1. **新建 Epilogue** `BlockEpilogueDequantSequentialWrite`：按输入行序 `(offsetM+i) * n + yOffset` 顺序写入 workspace
2. **Cgmct Kernel 执行**：使用新 Epilogue 输出到 workspace
3. **FRDeterministicA5 延迟聚合**：顺序读取 workspace，按 outRow 分配行所有权，单写者 AtomicAdd 写入最终 y

### 数据流
```
Cgmct Kernel (Plan B Epilogue)
    → workspace[16MB + (offsetM+i)*N + nOffset]  (顺序写入，无 AtomicAdd)
    → SyncAll()
FRDeterministicA5
    → workspace → UB → yGm[outRow*N + nOffset]  (AtomicAdd，单写者)
    → SyncAll()
```

---

## 修改详情

### 1. `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`

**修改原因**：需要在 tiling data 中传递确定性标志和 workspace 大小到 kernel 侧

**修改内容**：在 `GMMFinalizeRoutingDataParams` 结构体末尾 `reserved2` 之后新增两个字段

```cpp
struct GMMFinalizeRoutingDataParams {
    // ... 原有字段保持不变 ...
    uint32_t reserved2 = 0;          // 保持不变，不删除不替换
    uint32_t deterministicFlag = 0;   // 新增：0=非确定性, 1=确定性
    uint32_t deterWorkspaceSize = 0;  // 新增：确定性 workspace 大小（字节）
};
```

---

### 2. `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h`

**修改原因**：Tiling 类需要新增成员和虚函数声明

**修改内容**：
- protected 区域：在 PostTiling 和 Reset 之间新增 `GetWorkspaceSize()` override 声明
- private 区域：在 `rowIndexType_` 之后新增 `deterministicFlag_` 和 `deterWorkspaceSize_`

---

### 3. `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`

**修改原因**：实现确定性 tiling 计算逻辑

**修改内容**（3处）：

#### 3a. DoOpTiling() 确定性条件判断（hasBias 赋值之后）
- 条件：`GetDeterministic()==1` + `!IsMicroScaling()` + `INT8×INT8` + `PERTOKEN_MODE`
- 根据 L2 大小设置 workspace 上限（96MB 或 64MB）
- 溢出保护：`requiredDeterSize > deterWorkspaceSize_` 时降级为非确定性

#### 3b. GetWorkspaceSize() 实现（新函数）
- 非确定性：workspace = SYS_WORKSPACE_SIZE (16MB)
- 确定性：workspace = 16MB + deterWorkspaceSize_

#### 3c. PrintQuantParams() 扩展
- 日志新增 `deterministicFlag` 和 `deterWorkspaceSize` 字段输出

---

### 4. `op_kernel/arch35/gmm_fr_deterministic_a5.h`（新文件）

**功能**：A5 平台确定性延迟聚合核心函数

**设计**：
- AIC 核心：`SyncAll(); SyncAll(); return;`（必须配对参与同步）
- Vector 核心：
  - `SyncAll()` 等待 workspace 写入完成
  - 按 `outRow % coreNumVec == GetBlockIdx()` 分配行所有权
  - 顺序读 workspace → UB → AtomicAdd 写入 yGm
  - `SyncAll()` 等待写入完成

---

### 5. `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`

**修改原因**：在 W8A8 PerToken 路径添加确定性分支

**修改内容**：
- 新增 `#include "block_epilogue_dequant_sequential_write.h"` 和 `#include "gmm_fr_deterministic_a5.h"`
- 新增 `using namespace GMMFRDeterministic;`
- 原有代码放入 `else` 分支（零回归影响）
- 确定性 `if` 分支：
  1. 使用 `BlockEpilogueDequantSequentialWrite` Epilogue
  2. workspace 偏移 = SYS_WORKSPACE_SIZE (16MB)
  3. 构建 Params 时 y 地址指向 workspace
  4. 执行 Cgmct Kernel
  5. 调用 FRDeterministicA5 进行延迟聚合

---

### 6. `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h`（新文件）

**功能**：Plan B 顺序写 Epilogue

**与原始 Epilogue 的关键差异**：

原始 `VectorAtomicProcess`：
```cpp
SetAtomicAdd<float>();
for (i) {
    outRow = rowIndexGlobal_.GetValue(offsetM + i);
    DataCopyPad(yGlobal_[outRow * n_ + yOffset], yLocal[i * alignN_], ...);
}
SetAtomicNone();
```

新 `VectorSequentialWrite`：
```cpp
// 无 AtomicAdd，按输入行序写入
for (i) {
    DataCopyPad(yGlobal_[(offsetM + i) * n_ + yOffset], yLocal[i * alignN_], ...);
}
```

**其余方法**：与原始 `BlockEpilogueDequantFinalizeRouting` 完全一致（Init, CopyInLogit, VFDoLogitMuls, VFDoDequant*, CopyX1/X2Scale, CopyBias 等）

---

## 回归安全保证

- 所有原有字段（包括 `reserved2`）保持不变
- `deterministicFlag` 默认值为 0，不走确定性分支
- `else` 分支代码与修改前逐行一致
- 原始 `block_epilogue_dequant_finalize_routing.h` 未修改
- 其他 kernel 入口文件、weight_quant 文件、common 工具文件均未修改
