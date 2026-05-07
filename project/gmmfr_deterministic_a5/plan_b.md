# GMMFR 确定性特性 A5 迁移 — Plan B

> 范围：仅 W8A8/INT8 PerToken 全量化路径
> 策略：新建 Epilogue + 分离 Finalize Routing，不修改现有 Cgmct 代码
> 约束：不修改算子输入输出定义，可增加 workspace 大小

## 一、Plan B 核心思路

### 1.1 Plan A vs Plan B 对比（仅 W8A8/INT8 范围）

| 维度 | Plan A（修改现有 Epilogue） | Plan B（本方案） |
|---|---|---|
| **Epilogue 策略** | 修改现有 `BlockEpilogueDequantFinalizeRouting`，添加确定性分支 | **新建独立 Epilogue 类**，不碰现有代码 |
| **回归风险** | 中（修改 Cgmct 共享代码，影响非确定性路径） | **极低**（新增文件，非确定性模式完全不受影响） |
| **代码耦合** | 确定性/非确定性逻辑混合在同一个 Epilogue | **完全隔离** |
| **适用数据类型** | 需考虑 INT8/FP8/HIFLOAT8 三种类型 | **仅 INT8**，新 Epilogue 更简单 |
| **workspace 策略** | 在现有 workspace 内部分区 | **增大 workspace**，为确定性模式单独分配 |

### 1.2 Plan B 核心原理

**将 Cgmct 的"MatMul + 反量化 + Finalize Routing"三合一流程，拆分为两步：**

```
Plan A（一体化）:
  Cgmct Kernel = MatMul + Dequant + FinalizeRouting(SetAtomicAdd)
  确定性改造：在 FinalizeRouting 内部添加条件分支

Plan B（分离式）:
  Step 1: Cgmct Kernel(新 Epilogue) = MatMul + Dequant + 顺序写入 workspace（不含 FinalizeRouting）
  Step 2: FRDeterministicA5() = 从 workspace 读取 → 按行归属 → SetAtomicAdd 到 yGm
```

## 二、Cgmct 架构关键事实（W8A8/INT8 路径）

从 `grouped_matmul_finalize_routing_pertoken_dequant.h` 可以看到：

```cpp
// 第63-69行
using BlockEpilogueDequant =
    Cgmct::Gemm::Block::BlockEpilogueDequantFinalizeRouting<CType, C1Type, ...>;

using GmmKernel =
    Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<
        ProblemShape, BlockMmadBuilder, BlockPrologue,
        BlockEpilogueDequant,   // ← 可替换为新的 Epilogue！
        BlockScheduler>;
```

**Epilogue 是模板参数，可以创建新的 Epilogue 类替换，完全不修改现有代码。**

同时，Epilogue 的 Params 包含输出目标地址 `y`：

```cpp
// Params 中的 epilogue params（第87行）
{y, w_scale, x_scale, bias, logit, row_index, matmulTiling_.baseM, matmulTiling_.baseN}
```

**将 `y` 替换为 workspace 地址，Epilogue 的输出就重定向到 workspace。**

## 三、Plan B 详细设计

### 3.1 Workspace 布局（W8A8/INT8）

```
Workspace 总布局:
┌─────────────────────────────────────────────────────┐
│ BlockMmad 中间结果 (Cube MatMul 输出)                │  ← 已有，不变
│ 大小 = parallNum * baseM * baseN * sizeof(int32_t) * coreNum   │
├─────────────────────────────────────────────────────┤
│ 确定性中间缓冲区 (deterministicBuffer)               │  ← 新增
│ 大小 = windowSize * N * sizeof(float)                │
│ 用途: 新 Epilogue 顺序写入，FRDeterministicA5 读取   │
│ 上限: 96MB (DETER_WORK_SPACE_SIZE)                   │
└─────────────────────────────────────────────────────┘
```

**不需要 per-core 分区**：
- 新 Epilogue 不做 finalize routing（不做 scatter），将反量化结果按原始行顺序写入 workspace
- 每个 (group, mIdx, nIdx) 的输出在 workspace 中的位置由调度器保证唯一
- 无碰撞，无需 SetAtomicAdd

### 3.2 新增 Epilogue 类（仅处理 INT8）

**文件（新增）：`op_kernel/arch35/block_epilogue_dequant_only.h`**

功能：只做 INT8 的反量化（int32 → float × scale），不做 Finalize Routing scatter。将结果按顺序写入 workspace。

```cpp
// 伪代码 - 新 Epilogue 的核心逻辑
template <typename CType /* int32_t */, typename C1Type /* float */,
          typename ScaleType, typename BiasType>
class BlockEpilogueDequantOnly {
public:
    struct Params {
        GM_ADDR outputBuffer;    // 指向 workspace 中的确定性缓冲区
        GM_ADDR wScale;
        GM_ADDR xScale;
        GM_ADDR bias;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t N;             // 总列数，用于计算行偏移
    };

    __aicore__ inline void Process(/* tile params */) {
        // 1. 从 Cube 输出 L0C 读取 INT32 结果
        // 2. INT32 → FP32 类型转换
        // 3. FP32 × scale → 反量化 (per-token 或 per-channel)
        // 4. 可选: 加 bias
        // 5. 顺序写入 workspace（不做 scatter，不用 SetAtomicAdd）
        //    outputBuffer[tileOffset] = dequantResult
    }
};
```

**与现有 Epilogue 的区别**：

| 维度 | 现有 `BlockEpilogueDequantFinalizeRouting` | 新 `BlockEpilogueDequantOnly` |
|---|---|---|
| INT32→FP32 转换 | 有 | 有（相同） |
| 反量化 (×scale) | 有 | 有（相同） |
| 读取 tokenRanks | 有（用于 scatter） | **不需要** |
| SetAtomicAdd | 有（scatter 到 yGm） | **不需要**（顺序写入 workspace） |
| 输出目标 | yGm（最终输出） | workspace（中间缓冲区） |
| 支持数据类型 | INT8/FP8/HIFLOAT8 | **仅 INT8**（简化实现） |

### 3.3 FRDeterministicA5 函数

**文件（新增）：`op_kernel/arch35/gmm_fr_deterministic_a5.h`**

功能：从 workspace 读取中间结果，执行确定性 finalize routing。

```cpp
struct DeterministicConfig {
    uint64_t totalM;         // 总行数
    uint64_t N;              // 总列数
    uint64_t windowSize;     // 滑动窗口大小
    uint64_t baseN;          // N 方向分块（128 对齐）
    uint32_t coreNum;        // 核数
};

__aicore__ inline void FRDeterministicA5(
    GM_ADDR workspace,        // 中间结果缓冲区
    GM_ADDR yGm,             // 最终输出 (FP32)
    GM_ADDR tokenRanks,      // 行索引 (INT64)
    GM_ADDR groupTokens,     // group 行数
    uint32_t groupNum,
    const DeterministicConfig& config)
{
    // 仅 Vector 核执行
    if (g_coreType == AIC) return;

    // 滑动窗口处理（与 A3 逻辑一致）
    SyncConfig syncConfig;
    syncConfig.windowSize = config.windowSize;
    syncConfig.lowBoundM = config.windowSize;
    syncConfig.baseN = config.baseN;

    // 按 group 遍历，滑动窗口触发
    for (uint32_t groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
        // ... 累计行数，窗口满时触发 FRDeterministicWindow
    }
    // 处理剩余行
    FRDeterministicWindow(workspace, yGm, tokenRanks, syncConfig, config);
}

__aicore__ inline void FRDeterministicWindow(...) {
    // A5 同步注意事项：SyncAll 必须 1:1 配对
    SyncAll();
    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = config.coreNum * GetTaskRation();

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        auto outRow = tokenRanksGm.GetValue(baseOffset + mOffset);
        if (outRow % coreNumVec != GetBlockIdx()) continue;

        for (uint64_t nOffset = 0; nOffset < config.N; nOffset += syncConfig.baseN) {
            // 从 workspace 读取 → UB
            LocalTensor<float> bindLocal = queBind.AllocTensor<float>();
            DataCopyPad2D(bindLocal, workspaceBuffer[mOffset * config.N + nOffset], ...);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<float>();

            // 确定性写入（只有归属核写入，顺序确定）
            SetAtomicAdd<float>();
            DataCopyPad(yGm[outRow * config.N + nOffset], bindLocal, ...);
            SetAtomicNone();
            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();
}
```

### 3.4 Kernel 函数改造（仅 PerToken dequant INT8 路径）

以 `grouped_matmul_finalize_routing_pertoken_dequant.h` 为例：

```cpp
template <typename layoutA, typename layoutB, int scaleType, int rowIndType>
__aicore__ inline void grouped_matmul_finalize_routing_pertoken_dequant(...)
{
    // ... 现有 tiling 读取代码不变 ...

    if (gmmFinalizeRoutingQuantParams_.deterministicFlag == 0) {
        // ==================== 非确定性模式（完全不变）====================
        using BlockEpilogueDequant =
            BlockEpilogueDequantFinalizeRouting<CType, C1Type, ...>;
        using GmmKernel = KernelGmmFinalizeRoutingPertokenDequant<
            ProblemShape, BlockMmadBuilder, BlockPrologue,
            BlockEpilogueDequant, BlockScheduler>;

        Params params = { /* 现有参数，y 指向真实输出 */ };
        GmmKernel gmm;
        gmm(params);

    } else {
        // ==================== 确定性模式（新路径，仅 INT8）===================
        // Step 1: 使用新 Epilogue，输出到 workspace
        using BlockEpilogueDeterministic =
            BlockEpilogueDequantOnly<CType, C1Type, ScaleType, BiasType>;
        using GmmKernelDet = KernelGmmFinalizeRoutingPertokenDequant<
            ProblemShape, BlockMmadBuilder, BlockPrologue,
            BlockEpilogueDeterministic, BlockScheduler>;

        GM_ADDR deterBuffer = workspaceGM + existingWorkspaceSize;

        Params params = {
            {1, 1, 1, 1},
            {x, w, reinterpret_cast<GM_ADDR>(deterBuffer), bias, group_list},
            {share_input, reinterpret_cast<GM_ADDR>(deterBuffer), ...},
            {reinterpret_cast<GM_ADDR>(deterBuffer), w_scale, x_scale, bias,
             logit, row_index, matmulTiling_.baseM, matmulTiling_.baseN},
            gmmParams};
        GmmKernelDet gmm;
        gmm(params);

        // Step 2: 确定性 Finalize Routing
        DeterministicConfig config;
        config.N = matmulTiling_.N;
        config.windowSize = gmmFinalizeRoutingQuantParams_.deterWorkspaceSize
                            / (config.N * sizeof(float));
        config.baseN = Ceil(Ceil(config.N, Ceil(config.N, DETER_UB_SIZE / sizeof(float))), 128) * 128;
        config.coreNum = matmulTiling_.usedCoreNum;
        FRDeterministicA5(deterBuffer, y, row_index, group_list,
                          gmmFinalizeRoutingQuantParams_.groupNum, config);
    }
}
```

## 四、Plan B 文件修改清单（仅 W8A8/INT8）

### 新增文件（2 个）

| # | 文件 | 内容 |
|---|---|---|
| N1 | `op_kernel/arch35/block_epilogue_dequant_only.h` | 新 Epilogue：INT8 反量化 + 顺序写入 workspace，不含 Finalize Routing |
| N2 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | 确定性聚合函数：滑动窗口 + 按行归属 + SetAtomicAdd 到 yGm |

### 修改文件（4 个）

| # | 文件 | 修改内容 | 原因 |
|---|---|---|---|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | `reserved2` → `deterministicFlag` + `deterWorkspaceSize` | Kernel 读取确定性参数 |
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | 声明确定性成员 | Tiling 类 |
| 3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | `DoOpTiling()` 中处理 `GetDeterministic()`，增大 workspace | CPU 侧确定性开关 |
| 4 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 添加 `if/else` 分支：非确定性用现有 Epilogue，确定性用新 Epilogue + FRDeterministicA5 | INT8 核心路径 |

### 明确不需要修改的文件（相比 v2.0 计划大幅减少）

| 文件 | 原因 |
|---|---|
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | **Plan B 中无需修改**（确定性分支在 pertoken_dequant.h 中处理，workspace 偏移计算在 Kernel 内部完成） |
| `op_kernel/arch35/grouped_matmul_finalize_routing.h` | **MX 格式路径不在范围内**（FP8/FP4） |
| `op_kernel/arch35/weight_quant_basic_block/*` | **Weight Quant 伪量化路径不在范围内**（FP8×FP4） |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.h/.cpp` | **Weight Quant 不在范围内** |
| `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | **Plan B 不修改现有 Epilogue** |
| 测试文件（独立计划） | Plan B 测试与 Plan A 相同 |

### 与 v2.0 计划对比

| 维度 | v2.0 计划 | v3.0（本计划） |
|---|---|---|
| 文件修改数 | 14 修改 + 1 新增 = 15 | **4 修改 + 2 新增 = 6** |
| 涉及路径 | 全量化 + 伪量化 + MX | **仅 INT8 PerToken** |
| 数据类型 | INT8 + FP8 + FP4 + HIFLOAT8 | **仅 INT8** |
| 预估工作量 | ~10 天 | **~4 天** |

## 五、双人分工计划

### 5.1 人员分工

| 角色 | 负责文件 | 具体任务 |
|---|---|---|
| **人员 A — Host/Tiling + 新 Epilogue** | 文件 1, 2, 3, N1 | P0 Tiling 数据结构、P1 Tiling 计算、新 Epilogue 类 |
| **人员 B — 确定性函数 + Kernel 集成** | 文件 N2, 4 | P2 FRDeterministicA5、Kernel INT8 路径改造 |

### 5.2 时间线

```
Day 1:
  人员A: P0 Tiling数据结构 (文件1) + P1 Tiling声明 (文件2)
  人员B: P2 FRDeterministicA5函数开发 (文件N2)

Day 2:
  人员A: P1 Tiling计算实现 (文件3) + 新Epilogue开发 (文件N1)
  人员B: P2 FRDeterministicA5函数完成 + 单元验证

Day 3:
  人员A: 新Epilogue完成 + 测试
  人员B: Kernel INT8路径改造 (文件4)

Day 4:
  两人: 联调 + 集成测试 + 问题修复
```

### 5.3 关键依赖

```
P0+P1 (人员A) ──────────────→ P3 Kernel (人员B)
                                      │
P2 FRDeterministicA5 (人员B) ──────→ P3 Kernel (人员B)
                                      │
新Epilogue (人员A) ────────────────→ P3 Kernel (人员B，需要include新Epilogue头文件)
```

- 人员 A 需要先完成新 Epilogue 类 (N1)，人员 B 才能在 Kernel 中引用
- 人员 B 的 FRDeterministicA5 (N2) 可完全独立开发
- 最关键路径：人员 A 完成文件 1,2,3,N1 → 人员 B 完成文件 4

## 六、Plan B 优势分析

### 6.1 对比 Plan A（仅 W8A8/INT8 场景）

| 维度 | Plan A | Plan B |
|---|---|---|
| 修改现有 Cgmct 代码 | **需要**（修改 BlockEpilogueDequantFinalizeRouting） | **不需要** |
| 非确定性路径回归风险 | 中（修改了共享 Epilogue） | **零**（非确定性路径代码完全不动） |
| 新增代码复杂度 | 中（在现有 Epilogue 中添加分支） | **低**（新 Epilogue 只做 INT8 反量化，比现有更简单） |
| 文件修改数 | 8（6改+1新+1测试） | **6**（4改+2新） |
| 调试隔离性 | 差（确定性/非确定性混在一起） | **好**（完全分离） |

### 6.2 Workspace 开销

```
Plan B workspace (滑动窗口):
  已有 workspace + windowSize * N * sizeof(float)
  windowSize = min(totalM, 96MB / (N * sizeof(float)))
  例: N=4096, 96MB / (4096 * 4) = 6144 行
```

## 七、验收标准

1. [ ] `context_->GetDeterministic() == 1` 时，A5 Tiling 正确增大 workspace 并设置确定性标志
2. [ ] 相同 INT8 输入在 A5 上执行 10 次，输出 **bit-wise 完全一致**
3. [ ] 确定性输出 vs 非确定性输出的最大相对误差 < 1e-3
4. [ ] **仅 W8A8/INT8 PerToken 路径**支持确定性
5. [ ] 非确定性模式零回归（与改动前结果完全一致）
6. [ ] A2/A3/A5 三平台 CI 全部通过
