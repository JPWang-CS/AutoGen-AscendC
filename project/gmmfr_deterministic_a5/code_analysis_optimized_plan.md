# GMMFR 确定性特性 A5 迁移 — 深度代码分析与优化计划

> 版本：v3.0（范围缩减为仅 W8A8/INT8，修正 A5 数据类型映射）

## 一、A3 确定性机制完整剖析

### 1.1 核心原理

A3 的确定性解决的问题是：**多核并行 `SetAtomicAdd` 导致浮点累加顺序不确定**。

解决方案采用 **"滑动窗口 + 延迟聚合"** 三阶段策略：

```
阶段1 (Cube+Vector 正常计算):
  Cube: MatMul 结果 → workspace (mmOutGm)
  Vector: 从 workspace 读取 → 反量化 → 结果写入 workspace (mmQuantOutGm) 而非 yGm

阶段2 (VectorSync 滑动窗口触发):
  每当累计行数达到 windowSize，触发 FRDeterministic

阶段3 (FRDeterministic 延迟聚合):
  SyncAll() → 按 outRow % coreNumVec 分配行归属 → 从 workspace 读取 → SetAtomicAdd 到 yGm → SyncAll()
```

### 1.2 A3 W8A8 确定性代码路径（唯一需要迁移的路径）

**文件**：`op_kernel/grouped_matmul_finalize_routing.h`（arch32）

| 代码位置 | 功能 |
|---|---|
| 第162-166行 `Init()` | 分配 `mmQuantOutGm`（确定性 workspace） |
| 第186-188行 `InitUbBuffer()` | 分配 `queBind`（确定性 UB buffer，12KB） |
| 第303-306行 `Process()` | 初始化 `SyncConfig`（窗口大小、baseN 128对齐） |
| 第322行 | `MNBlockIdxCompute` 简化为顺序分配（`deterministicFlag==1` 时跳过对角策略） |
| 第331-334行 | 最终调用 `FRDeterministic` 处理剩余行 |
| 第381-386行 `VectorAtomicProcess()` | **写入 workspace 而非 yGm** |
| 第622-649行 `VectorSync()` | 滑动窗口触发机制 |
| 第653-687行 `FRDeterministic()` | 延迟聚合核心：`SyncAll → 按行归属 → SetAtomicAdd → SyncAll` |

### 1.3 A3 其他路径（不需要迁移确定性）

**A3 路径B：A8W4 伪量化**（`grouped_matmul_finalize_routing_antiquant_a8w4_msd.h`）
- 使用 INT8 × INT4（整数4-bit）
- A5 上无 INT4，等价路径使用 FP8 × FP4（浮点4-bit），量化方案完全不同
- **不在本迁移范围内**

### 1.4 关键数据结构

```cpp
struct SyncConfig {
    uint64_t curM = 0;        // 当前累计行偏移
    uint64_t curGroup = 0;    // 当前 group 索引
    uint64_t curGroupM = 0;   // 当前 group 内累计行
    uint64_t lowBoundM = 0;   // 窗口下界
    uint64_t windowSize = 0;  // 窗口大小 = deterWorkspaceSize / (n * sizeof(DTYPE_OUT))
    uint64_t baseN = 0;       // N 方向分块大小，128 对齐
};

constexpr uint32_t DETER_UB_SIZE = 12 * 1024;  // 确定性 UB 缓冲区大小
constexpr uint32_t DETER_WORK_SPACE_SIZE = 96 * 1024 * 1024;  // 确定性 workspace 上限 96MB
constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64 * 1024 * 1024;  // 下限 64MB
```

---

## 二、A5 当前代码结构深度分析

### 2.1 A5 Kernel 执行架构（仅 W8A8/INT8 相关路径）

```
APT 入口 (op_kernel/grouped_matmul_finalize_routing_apt.cpp)
  │
  ├── 判断 weightQuant 场景 (行25-29)
  │     若 x=FP8, w=FP4, scale=E8M0 → V310_GMM_FR_ANTI_QUANT 路径 (不在范围内)
  │
  └── 非 weightQuant → 判断 scale 类型
        │
        ├── scale = E8M0 → MX 模式
        │     └── grouped_matmul_finalize_routing_mx()  (不在范围内)
        │
        └── scale = FLOAT/BF16 → PerToken 模式 ← **本迁移目标**
              │
              ├── tilingKey 包含 scaleType (0=float, 1=bf16)
              │              和 rowIndexType (0=int64, 1=int32)
              │
              └── grouped_matmul_finalize_routing_pertoken_dequant
                    <layoutA, layoutB, scaleType, rowIndType>()
                    │
                    └── Cgmct::KernelGmmFinalizeRoutingPertokenDequant<
                          ProblemShape, BlockMmadBuilder, BlockPrologue,
                          BlockEpilogueDequantFinalizeRouting,  ← 确定性改造目标
                          BlockScheduler>
```

### 2.2 Cgmct 框架中 W8A8 Finalize Routing 的执行位置

```
Cgmct Kernel 执行流:
  BlockScheduler::调度 → BlockPrologue(前处理) → BlockMmad(Cube计算)
    → BlockEpilogue(后处理，含 Finalize Routing)
```

`BlockEpilogueDequantFinalizeRouting` 是 INT8 PerToken 路径的 Epilogue，它：
1. 从 Cube L0C 输出读取 INT32 结果
2. INT32 → FP32 类型转换
3. FP32 × scale 反量化（per-token 或 per-channel）
4. 可选：加 bias
5. 根据 `tokenRanks` 索引找到目标输出行
6. 使用 `SetAtomicAdd` 将结果原子累加到 `yGm`  ← **非确定性根源**

### 2.3 A5 数据类型完整支持矩阵

| 路径 | x/weight 数据类型 | scale 类型 | 输出类型 | weight 格式 | 是否在范围 |
|---|---|---|---|---|---|
| **PerToken INT8** | **INT8 × INT8** | **FLOAT / BF16** | **FP32** | **FRACTAL_NZ** | **是** |
| PerToken FP8 | FP8_E4M3FN × FP8_E4M3FN | FLOAT / BF16 | FP32 | FRACTAL_NZ | 否 |
| PerToken HIFLOAT8 | HIFLOAT8 × HIFLOAT8 | FLOAT / BF16 | FP32 | FRACTAL_NZ | 否 |
| MX FP8 | FP8_E4M3FN × FP8_E4M3FN | E8M0 | FP32 | ND | 否 |
| MX FP5M2 | FP8_E5M2 × FP8_E5M2 | E8M0 | FP32 | ND | 否 |
| MX FP4 | FP4_E2M1 × FP4_E2M1 | E8M0 | FP32 | ND | 否 |
| Weight Quant | FP8_E4M3FN × FP4_E2M1 | E8M0 | FP32 | FRACTAL_NZ | 否 |

### 2.4 全仓库 A5 确定性现状

| 模块 | A5 Kernel 确定性状态 | A5 Tiling 确定性状态 |
|---|---|---|
| **GMMFR** | **无** | **无** |
| flash_attention_score_grad | 无 | 有（读取 GetDeterministic 影响模板选择） |
| moe_inplace_index_add | 有（使用特殊模板） | 有 |
| weight_quant_batch_matmul_v2 | **明确不支持**（IsCapable return false） | 有 |

---

## 三、逐文件修改分析（仅 W8A8/INT8）

### 3.1 Tiling 数据结构层（Kernel 侧，NPU 执行）

#### 文件 1：`op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`

**修改原因**：Kernel 执行时需要从 Tiling 读取 `deterministicFlag` 和 `deterWorkspaceSize` 来决定走哪条路径。当前结构体 `GMMFinalizeRoutingDataParams` 的 `reserved2` 字段可以利用。

**修改内容**：
```cpp
struct GMMFinalizeRoutingDataParams {
    // ... 现有字段 ...
    uint8_t hasBias = 0;
    uint16_t reserved1 = 0;
    uint32_t deterministicFlag = 0;      // 新增：0=非确定性, 1=确定性
    uint32_t deterWorkspaceSize = 0;     // 新增：确定性 workspace 大小
};
```

**注意**：weight_quant 相关的 Tiling 结构体（`GMMFinalizeRoutingWeightQuantTilingData`）**不需要修改**，因为 Weight Quant 路径不在范围内。

---

### 3.2 Tiling 计算层（Host 侧，CPU 执行）

#### 文件 2：`op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h`

**修改原因**：PerToken 全量化 Tiling 类需要声明确定性相关的成员和方法。

**修改内容**：
- 添加 `uint32_t deterministicFlag_ = 0;` 和 `uint32_t deterWorkspaceSize_ = 0;`
- 声明 `void DeterministicTilingProcess();`

#### 文件 3：`op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`

**修改原因**：Tiling 计算是 CPU 侧第一个读取 `GetDeterministic()` 的地方。

**修改内容**：
- 在 `DoOpTiling()` 第456行（`hasBias` 赋值之后）添加确定性处理：
  ```cpp
  if (context_->GetDeterministic() == 1) {
      tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
      auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
      uint64_t l2Size = 0;
      ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
      uint32_t deterWs = l2Size > (96 * 1024 * 1024) ? (96 * 1024 * 1024) : (64 * 1024 * 1024);
      tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWs;
  }
  ```
- **注意**：仅在 PerToken 模式（非 MX）下处理确定性，MX 模式不支持确定性
- 在 `PrintQuantParams()` 中打印确定性参数

**不需要修改的 Tiling 文件**：
- `grouped_matmul_finalize_routing_weight_quant_tiling.h/.cpp` — Weight Quant 路径不在范围内

---

### 3.3 Kernel 入口层

#### 文件 4：`op_kernel/grouped_matmul_finalize_routing_apt.cpp`

**Plan A 需要修改，Plan B 不需要修改**。

**Plan A 修改内容**：在 PerToken INT8 路径调用前，计算 workspace 中确定性缓冲区偏移地址，传递给 Kernel 函数。

**Plan B 不修改此文件**：确定性分支在 `pertoken_dequant.h` 内部处理，workspace 偏移在 Kernel 内计算。

---

### 3.4 Kernel 实现层（核心修改）

#### 文件 5（Plan A）/ 文件 4（Plan B）：`op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`

**修改原因**：W8A8/INT8 PerToken 全量化路径的 Cgmct Kernel。当前 `BlockEpilogueDequantFinalizeRouting` 使用 `SetAtomicAdd` 直接写 `yGm`。确定性模式下需要改为写入中间 workspace。

**Plan A 方案**：在现有 Epilogue 中添加确定性条件分支
**Plan B 方案**：使用新的 `BlockEpilogueDequantOnly` 替换模板参数

**INT8 特有注意事项**：
- Cube 输出为 INT32（INT8 × INT8 = INT32）
- 反量化：INT32 → FP32 × scale
- 确定性 workspace 存储的是反量化后的 FP32 值
- 4 种模板组合：scaleType(2) × rowIndType(2)，每种都需要能进入确定性分支

**不需要修改的 Kernel 文件**：
- `grouped_matmul_finalize_routing.h`（MX 路径，不在范围内）
- `weight_quant_basic_block/` 目录下所有文件（伪量化路径，不在范围内）

---

### 3.5 新增文件

#### 文件 N1（Plan B 新增）：`op_kernel/arch35/block_epilogue_dequant_only.h`

新增 Epilogue：只做 INT8 反量化（INT32→FP32×scale），不做 Finalize Routing scatter。

相比现有 `BlockEpilogueDequantFinalizeRouting`：
- 移除 tokenRanks 读取逻辑
- 移除 SetAtomicAdd 逻辑
- 输出改为顺序写入 workspace（无碰撞）
- 仅支持 INT8 数据类型（简化实现）

#### 文件 N2（Plan A 和 Plan B 均需要）：`op_kernel/arch35/gmm_fr_deterministic_a5.h`

A5 平台的确定性延迟聚合函数。参考 A3 的 `FRDeterministic()`，适配 A5 的同步机制。

```cpp
__aicore__ inline void FRDeterministicA5(
    SyncConfig& syncConfig,
    GlobalTensor<float>& workspaceGm,
    GlobalTensor<float>& yGm,
    GlobalTensor<int64_t>& tokenRanksGm,
    uint32_t coreNum, uint32_t n)
{
    SyncAll();
    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = coreNum * GetTaskRation();
    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        auto outRow = tokenRanksGm.GetValue(syncConfig.lowBoundM - syncConfig.windowSize + mOffset);
        if (outRow % coreNumVec != GetBlockIdx()) continue;
        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            // 从 workspace 读取 → UB → SetAtomicAdd 到 yGm
        }
    }
    SyncAll();
}
```

**A5 同步注意事项**：
- A5 的 `SyncAll()` 必须 1:1 配对调用，不能有条件跳过
- 如果 `SyncAll()` 在 A5 不可用，改用 `CrossCoreSetFlag/WaitFlag` 实现等效全核同步

---

### 3.6 测试层

#### 文件 6：`tests/ut/op_host/test_grouped_matmul_finalize_routing_tiling.cpp`
添加 INT8 确定性 Tiling 测试用例。

#### 文件 7：`tests/ut/op_host/op_api/test_aclnn_grouped_matmul_finalize_routing_l2.cpp`
添加 INT8 确定性模式的 L2 端到端测试，验证多次执行结果 bit-wise 一致。

**不需要修改的测试文件**：
- `test_grouped_matmul_finalize_routing_weight_quant_tiling.cpp` — Weight Quant 不在范围内

---

## 四、优化后的实施计划

### 与 v2.0 计划的关键优化

1. **范围大幅缩减**：从 3 条路径（全量化+伪量化+MX）缩减为 **1 条路径**（仅 INT8 PerToken）
2. **文件数量大幅减少**：从 15 个文件减少为 **8 个**（Plan A）或 **6 个**（Plan B）
3. **修正数据类型映射**：A5 没有 INT4，Weight Quant 路径使用 FP8×FP4（与 A3 的 INT8×INT4 完全不同）
4. **双人并行**：总工期从 ~12.5 天缩减为 **~5 天**

### Plan A 优化后阶段划分

| 阶段 | 文件 | 工作量 | 依赖 | 负责人 |
|---|---|---|---|---|
| **P0：数据结构** | 1 | 0.5天 | 无 | 人员A |
| **P1：Tiling** | 2, 3 | 1.5天 | P0 | 人员A |
| **P2：确定性聚合函数** | N2(新增) | 1.5天 | P0 | 人员B |
| **P3：Kernel INT8路径** | 4, 5 | 2天 | P1+P2 | 人员B |
| **P4：测试** | 6, 7 | 1.5天 | P3 | 人员A |
| **合计** | **8个文件** | **~5天** | | |

### Plan B 优化后阶段划分

| 阶段 | 文件 | 工作量 | 依赖 | 负责人 |
|---|---|---|---|---|
| **P0：数据结构** | 1 | 0.5天 | 无 | 人员A |
| **P1：Tiling + 新 Epilogue** | 2, 3, N1(新增) | 2天 | P0 | 人员A |
| **P2：确定性聚合函数** | N2(新增) | 1.5天 | 无 | 人员B |
| **P3：Kernel INT8路径** | 4 | 1.5天 | P1+P2 | 人员B |
| **P4：测试** | 6, 7 | 1.5天 | P3 | 人员A |
| **合计** | **7个文件** | **~4天** | | |

### 验收标准

1. [ ] `context_->GetDeterministic() == 1` 时，A5 Tiling 正确计算 `deterministicFlag=1` 和 `deterWorkspaceSize`
2. [ ] 相同 INT8 输入在 A5 上执行 10 次，输出 bit-wise 一致
3. [ ] 确定性输出 vs 非确定性输出的最大相对误差 < 1e-3
4. [ ] **仅 W8A8/INT8 PerToken 路径**支持确定性，其他路径不受影响
5. [ ] 非确定性模式零回归
6. [ ] A2/A3/A5 三平台 CI 全部通过
