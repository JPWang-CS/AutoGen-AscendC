# GMMFR 确定性特性 A5 迁移项目计划

> 更新日期：2026-05-07
> 版本：v3.0（范围缩减：仅 W8A8/INT8 确定性，修正 A5 数据类型映射，双人分工）

## 一、项目概述

| 项目 | 详情 |
|---|---|
| **项目名称** | GMMFR 确定性特性 A5 迁移（W8A8/INT8） |
| **源平台** | A3 (910C) / arch32 |
| **目标平台** | A5 (950) / arch35 |
| **算子路径** | `ops-transformer_AI/gmm/grouped_matmul_finalize_routing` |
| **迁移范围** | **仅 W8A8（INT8×INT8）PerToken 全量化路径** |
| **不在范围内** | FP8/FP4/HIFLOAT8 路径、MX 格式路径、Weight Quant（伪量化）路径 |
| **总预估** | ~5 天（2 人并行） |

## 二、A3 vs A5 数据类型对比（关键修正）

### 2.1 GMMFR 各路径数据类型映射

| 量化路径 | A3 (arch32) 数据类型 | A5 (arch35) 数据类型 | 确定性需求 |
|---|---|---|---|
| **W8A8 PerToken 全量化** | INT8 × INT8 | INT8 × INT8（相同）| **需要**（本项目） |
| W4A8 伪量化 | INT8 × **INT4** | FP8 × **FP4**（类型完全不同）| 不需要 |
| FP8 PerToken | N/A | FP8_E4M3FN × FP8_E4M3FN | 不需要 |
| HIFLOAT8 PerToken | N/A | HIFLOAT8 × HIFLOAT8 | 不需要 |
| MX 格式 | N/A | FP8×FP8 / FP8_E5M2×FP8_E5M2 / FP4×FP4 | 不需要 |
| Weight Quant (NZ) | N/A | FP8_E4M3FN × FP4_E2M1 | 不需要 |

### 2.2 关键差异说明

1. **A5 没有 INT4**：A3 的 4-bit 量化使用整数 INT4，A5 使用浮点 FP4_E2M1，量化方案完全不同
2. **A5 PerToken 路径扩展**：除 INT8 外，还支持 FP8_E4M3FN 和 HIFLOAT8
3. **A5 新增 MX 格式**：使用 FP8_E8M0 作为 scale，A3 无此格式
4. **输出类型**：A5 量化路径输出固定为 FP32
5. **Weight 格式**：A5 PerToken 要求 weight 为 FRACTAL_NZ 格式；MX 要求 ND 格式

### 2.3 本项目涉及的数据类型（仅 INT8）

```
输入:
  x (activation):  DT_INT8      (INT8)
  weight:          DT_INT8      (INT8, FRACTAL_NZ 格式)
  scale:           DT_FLOAT 或 DT_BF16
  pertoken_scale:  DT_FLOAT     (可选)
  bias:            DT_BF16      (可选)

中间:
  Cube 输出:       INT32        (INT8 × INT8 = INT32)

输出:
  y:               DT_FLOAT     (INT32 → 反量化 → FP32)
```

## 三、确定性特性原理（不变）

### 3.1 问题

GMMFR 的 Finalize Routing 阶段使用 `SetAtomicAdd` 将多核计算结果累加到全局输出。由于多核并行执行顺序不确定，浮点加法不满足结合律，导致每次运行的累加顺序不同，产生微小的数值差异。

### 3.2 A3 解决方案：滑动窗口 + 延迟聚合

```
阶段1 (Cube+Vector 正常计算):
  Cube:   MatMul 结果 → workspace (mmOutGm)
  Vector: 从 workspace 读取 → 反量化 → 结果写入 workspace (mmQuantOutGm) 而非 yGm

阶段2 (VectorSync 滑动窗口触发):
  每当累计行数达到 windowSize，触发 FRDeterministic

阶段3 (FRDeterministic 延迟聚合):
  SyncAll() → 按 outRow % coreNumVec 分配行归属 → 从 workspace 读取 → SetAtomicAdd 到 yGm → SyncAll()
```

### 3.3 A3 W8A8 确定性代码路径参考

**文件**：`op_kernel/grouped_matmul_finalize_routing.h`

| 代码位置 | 功能 |
|---|---|
| 第162-166行 `Init()` | 分配 `mmQuantOutGm`（确定性 workspace） |
| 第186-188行 `InitUbBuffer()` | 分配 `queBind`（确定性 UB buffer，12KB） |
| 第303-306行 `Process()` | 初始化 `SyncConfig`（窗口大小、baseN 128对齐） |
| 第322行 | `MNBlockIdxCompute` 简化为顺序分配 |
| 第331-334行 | 最终调用 `FRDeterministic` 处理剩余行 |
| 第381-386行 `VectorAtomicProcess()` | **写入 workspace 而非 yGm** |
| 第622-649行 `VectorSync()` | 滑动窗口触发机制 |
| 第653-687行 `FRDeterministic()` | 延迟聚合核心 |

### 3.4 关键数据结构

```cpp
struct SyncConfig {
    uint64_t curM = 0;        // 当前累计行偏移
    uint64_t curGroup = 0;    // 当前 group 索引
    uint64_t curGroupM = 0;   // 当前 group 内累计行
    uint64_t lowBoundM = 0;   // 窗口下界
    uint64_t windowSize = 0;  // 窗口大小 = deterWorkspaceSize / (n * sizeof(float))
    uint64_t baseN = 0;       // N 方向分块大小，128 对齐
};

constexpr uint32_t DETER_UB_SIZE = 12 * 1024;                   // 确定性 UB 缓冲区
constexpr uint32_t DETER_WORK_SPACE_SIZE = 96 * 1024 * 1024;    // workspace 上限 96MB
constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64 * 1024 * 1024; // 下限 64MB
```

## 四、A5 架构差异（仅与 W8A8/INT8 相关的部分）

### 4.1 编程框架差异

| 维度 | A3 (arch32) | A5 (arch35) |
|---|---|---|
| **编程框架** | 原生 AscendC API | Cgmct (C++ Gemm Template) Builder 模式 |
| **Kernel 入口** | `grouped_matmul_finalize_routing.cpp` | `grouped_matmul_finalize_routing_apt.cpp` |
| **核心计算** | `MatmulImpl<aT, bT, cT, biasT, CFG_MDL>` | `BlockMxMmAicToAivBuilder<...>` |
| **Epilogue** | 手写 `VectorAtomicProcess()` + `FRDeterministic()` | `BlockEpilogueDequantFinalizeRouting` 模板类 |
| **同步方式** | `CrossCoreSetFlag/WaitFlag` + `SyncAll` | Cgmct 内置调度器 + `CrossCoreSetFlag/WaitFlag` |
| **核间同步严格性** | 较宽松 | **严格 1:1 配对**（否则死锁） |
| **L0C 大小** | 128 KB | 256 KB |

### 4.2 A5 W8A8 Kernel 执行路径

```
APT 入口 (grouped_matmul_finalize_routing_apt.cpp)
  │
  └── 非 ANTI_QUANT / PerToken 路径 (scale ≠ E8M0 → PerToken 模式)
        │
        ├── tilingKey 包含 scaleType (0=float, 1=bf16)
        │              和 rowIndexType (0=int64, 1=int32)
        │
        └── grouped_matmul_finalize_routing_pertoken_dequant<layoutA, layoutB, scaleType, rowIndType>()
              │
              └── Cgmct::KernelGmmFinalizeRoutingPertokenDequant<
                    ProblemShape, BlockMmadBuilder, BlockPrologue,
                    BlockEpilogueDequantFinalizeRouting,  ← 确定性改造目标
                    BlockScheduler>
```

### 4.3 Cgmct Epilogue 中 W8A8 的 Finalize Routing 流程

```
BlockEpilogueDequantFinalizeRouting::Process():
  1. 从 Cube 输出 L0C 读取 INT32 结果
  2. INT32 → FP32 类型转换
  3. FP32 × scale → 反量化结果 (per-token 或 per-token+per-channel)
  4. 可选：加 bias
  5. 读取 tokenRanks[globalRow] → 确定目标输出行
  6. SetAtomicAdd(yGm[tokenRanks[globalRow] * N + nOffset])  ← 非确定性根源
```

**确定性改造核心**：将步骤 6 的 `SetAtomicAdd(yGm[...])` 改为写入 workspace，延迟聚合。

## 五、迁移方案（仅 W8A8/INT8）

### 5.1 策略：Epilogue 输出重定向

**不修改 Cgmct 框架核心**，采用两层策略：

1. **第一层**：Cgmct Kernel 执行时，Epilogue 的输出目标从 `yGm` 重定向到 workspace 特定区域（按顺序写入，不做 scatter）
2. **第二层**：Kernel 执行完毕后，调用新增的 `FRDeterministicA5()` 从 workspace 读取并确定性聚合到 `yGm`

### 5.2 技术风险

| 风险 | 等级 | 缓解措施 |
|---|---|---|
| Cgmct Epilogue 输出地址能否参数化 | 中 | Epilogue Params 中 `y` 地址可直接替换为 workspace 地址 |
| A5 `SyncAll()` 是否可用 | 高 | 如不可用，改用 `CrossCoreSetFlag/WaitFlag` 全核广播 |
| workspace 容量 | 低 | 使用滑动窗口策略（与 A3 一致），上限 96MB |
| INT8 PerToken 路径模板组合数量 | 低 | scaleType(2) × rowIndType(2) = 4 种组合 |

## 六、逐文件修改清单（W8A8/INT8 范围缩减后）

### P0：Tiling 数据结构（0.5 天）

| # | 文件 | 修改内容 | 原因 |
|---|---|---|---|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 将 `reserved2` 替换为 `deterministicFlag` + `deterWorkspaceSize` | Kernel 需从 Tiling 读取确定性开关 |

### P1：Tiling 计算（1.5 天）

| # | 文件 | 修改内容 | 原因 |
|---|---|---|---|
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | 声明 `deterministicFlag_`、`deterWorkspaceSize_`、`DeterministicTilingProcess()` | Tiling 类需确定性成员 |
| 3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | `DoOpTiling()` 中读取 `GetDeterministic()`，计算 workspace 大小，填充 tiling 数据 | CPU 侧确定性开关入口 |

### P2：确定性聚合函数（1.5 天，可与 P1 并行）

| # | 文件 | 修改内容 | 原因 |
|---|---|---|---|
| 4 | **新增** `op_kernel/arch35/gmm_fr_deterministic_a5.h` | 实现 `FRDeterministicA5()`：滑动窗口 + SyncAll → 按行归属 → SetAtomicAdd → SyncAll | A5 版延迟聚合函数，参考 A3 第653-687行 |

### P3：Kernel W8A8 路径（2 天，依赖 P1+P2）

| # | 文件 | 修改内容 | 原因 |
|---|---|---|---|
| 5 | `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | 计算 workspace 偏移，传递确定性参数给 PerToken INT8 路径 | A5 唯一 Kernel 入口 |
| 6 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 确定性模式下重定向 Epilogue 输出到 workspace，完成后调用 `FRDeterministicA5` | W8A8/INT8 核心路径 |

### P4：测试（1.5 天，依赖 P3）

| # | 文件 | 修改内容 | 原因 |
|---|---|---|---|
| 7 | `tests/ut/op_host/test_grouped_matmul_finalize_routing_tiling.cpp` | 添加 INT8 确定性 Tiling 测试用例 | 验证 Tiling 计算 |
| 8 | `tests/ut/op_host/op_api/test_aclnn_grouped_matmul_finalize_routing_l2.cpp` | 添加 INT8 确定性 L2 端到端测试 | 验证多次执行 bit-wise 一致 |

### 明确不需要修改的文件

| 文件 | 原因 |
|---|---|
| `op_host/grouped_matmul_finalize_routing_def.cpp` | 算子定义，确定性是运行时行为 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.h/.cpp` | **Weight Quant 伪量化路径不在范围内**（FP8×FP4） |
| `op_kernel/arch35/weight_quant_basic_block/*` | **Weight Quant 伪量化路径不在范围内** |
| `op_kernel/arch35/grouped_matmul_finalize_routing.h` | **MX 格式路径不在范围内**（FP8/FP4） |
| `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` 中 weight_quant 部分 | 不在范围内 |
| `op_api/grouped_matmul_finalize_routing.cpp` | l0op 接口层，workspace 框架自动管理 |
| `op_api/grouped_matmul_finalize_routing_950_checker.h` | A5 参数检查，确定性不需要额外校验 |
| `op_kernel/grouped_matmul_finalize_routing.cpp/.h` | arch32 Kernel，A5 用 `_apt.cpp` |
| `op_host/op_tiling/grouped_matmul_finalize_routing_base_tiling.cpp/.h` | arch32 Tiling |

## 七、双人分工计划

### 7.1 人员分工

| 角色 | 负责人 | 职责范围 |
|---|---|---|
| **人员 A — Host/Tiling 层 + 测试** | TBD | P0 Tiling 数据结构、P1 Tiling 计算、P4 测试 |
| **人员 B — Kernel/NPU 层** | TBD | P2 确定性聚合函数、P3 Kernel 集成 |

### 7.2 时间线与并行度

```
Day 1:
  人员A: P0 Tiling数据结构 (文件1) ← 完成后通知人员B数据结构定义
  人员B: P2 FRDeterministicA5函数开发 (文件4，可独立开始)

Day 2:
  人员A: P1 Tiling计算 (文件2,3)
  人员B: P2 FRDeterministicA5函数完成 + 测试验证

Day 3:
  人员A: P4 测试框架搭建 (文件7)
  人员B: P3 APT入口修改 (文件5)

Day 4:
  人员A: P4 测试用例完善 (文件7,8)
  人员B: P3 Kernel INT8路径修改 (文件6)

Day 5:
  两人: 联调 + 集成测试 + 问题修复
```

### 7.3 关键依赖关系

```
P0 (人员A) ──→ P1 (人员A) ──→ P3 (人员B)
                                    │
P2 (人员B) ─────────────────────→ P3 (人员B)
                                    │
                               P4 (人员A) ←── P3 完成后可执行端到端测试
```

- P0 完成后，人员B 即可获知 Tiling 数据结构定义，开始适配 Kernel 侧读取
- P2 可与 P0/P1 完全并行，因为 FRDeterministicA5 函数的接口可以提前约定
- P3 依赖 P1（Tiling 需要正确传递确定性参数）+ P2（确定性函数需就绪）
- P4 的 Tiling 测试可在 P1 完成后立即开始；L2 端到端测试需等 P3 完成

### 7.4 工作量对比

| 人员 | 文件数 | 工作量 | 可独立开发天数 |
|---|---|---|---|
| 人员 A | 5 个（1改 + 2改 + 2测试） | ~3.5 天 | 3 天（P0+P1+P4前半） |
| 人员 B | 3 个（1新增 + 1改 + 1改） | ~3.5 天 | 3 天（P2 独立 + P3 前半） |
| 联调 | - | 1 天 | - |
| **合计** | **8 个文件**（6改 + 1新增 + 1测试） | **~5 天** | - |

## 八、验收标准

1. [ ] `context_->GetDeterministic() == 1` 时，A5 Tiling 正确计算 `deterministicFlag=1` 和 `deterWorkspaceSize`
2. [ ] 相同 INT8 输入在 A5 上执行 10 次，输出 **bit-wise 完全一致**
3. [ ] 确定性输出 vs 非确定性输出的最大相对误差 < 1e-3
4. [ ] **仅 W8A8/INT8 PerToken 路径**支持确定性，FP8/HIFLOAT8/MX/WeightQuant 不受影响
5. [ ] 非确定性模式零回归（与改动前结果完全一致）
6. [ ] A2/A3/A5 三平台 CI 全部通过
