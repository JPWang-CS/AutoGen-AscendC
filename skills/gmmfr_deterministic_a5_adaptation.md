# GMMFR 确定性 A5 适配 Skill 文档

## 1. 概述

### 1.1 任务类型

A5 (arch35/950) 平台确定性 (deterministic) 特性适配，针对 GMM 系列算子中需要可复现浮点累加结果的场景。

### 1.2 适用场景

- 使用 Cgmct Builder 框架的 GMM 系列算子（grouped_matmul_finalize_routing 等），在 scatter + AtomicAdd 模式下需要确定性输出
- 从 A3 (arch32/910B) 向 A5 (arch35/950) 移植确定性特性
- 任何涉及多核并行浮点累加且需要结果可复现的算子开发

### 1.3 源码参考路径

| 组件 | 路径 |
|------|------|
| A5 确定性 Epilogue | `gmm/common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` |
| A5 确定性聚合函数 | `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/gmm_fr_deterministic_a5.h` |
| A5 Kernel 入口 | `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` |
| 原始 Epilogue (非确定性) | `gmm/common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` |
| A3 原型 Kernel | `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing.h` |

---

## 2. 核心知识点

### 2.1 确定性编程原则

#### 2.1.1 浮点累加顺序不确定性的原理

浮点加法不满足结合律：`(a + b) + c != a + (b + c)` 在有限精度下。当多个核并行对同一地址执行 `AtomicAdd` 时，累加顺序取决于核的执行时序，而核调度顺序每次运行可能不同，导致最终结果有微小差异。

#### 2.1.2 AtomicAdd 不确定性的根源

```cpp
// 非确定性代码模式：
SetAtomicAdd<float>();
for (uint32_t i = 0; i < curVecBaseM; i++) {
    auto outRow = rowIndex.GetValue(offsetM + i);  // 不同核可能得到相同 outRow
    DataCopyPad(yGm[outRow * n + yOffset], yLocal[i * alignN], ...);  // 多核竞写同一地址
}
SetAtomicNone();
```

问题：多个核可能对同一个 `outRow` 执行 AtomicAdd，由于浮点加法不满足结合律，累加顺序不确定。

#### 2.1.3 确定性保证的核心原则

**每个地址只有一个写入者 (single-writer-per-address)**

确定性模式通过三阶段数据流实现：

```
Phase 1: Prologue -> yGm (初始化为零 + 写入 residual)
Phase 2: Epilogue (SequentialWrite) -> workspace (仅写 dequant 结果，无 AtomicAdd)
Phase 3: Aggregation -> yGm (AtomicAdd，但每个 outRow 只由一个核处理)
```

- Phase 2：workspace 中每个地址 `(mOffset * N + nOffset)` 由唯一的核写入，因为 Cgmct 调度保证每个 tile 只分配给一个核
- Phase 3：每个 `outRow` 由唯一的核处理（`outRow % coreNumVec == GetBlockIdx()`），AtomicAdd 只在单核内执行，不涉及多核竞写

---

### 2.2 A5 Cgmct Epilogue 模式对比

#### 2.2.1 BlockEpilogueDequantFinalizeRouting（非确定性，scatter + AtomicAdd）

```cpp
// 文件: block_epilogue_dequant_finalize_routing.h
// 关键函数: VectorAtomicProcess

void VectorAtomicProcess(uint32_t curBaseN, uint32_t curVecBaseM,
                         uint64_t offsetM, uint64_t yOffset,
                         LocalTensor<DataTypeOut> &yLocal)
{
    SetAtomicAdd<float>();  // 原子模式
    for (uint32_t i = 0; i < curVecBaseM; i++) {
        auto outRow = rowIndexGlobal_.GetValue(offsetM + i);  // scatter 查表
        DataCopyPad(yGlobal_[outRow * n_ + yOffset], yLocal[i * alignN_], ...);
        // 多核可能写入同一 outRow -> 不确定性
    }
    SetAtomicNone();
}
```

特点：
- 直接写最终输出 `yGm`（scatter by output row）
- 需要 AtomicAdd 处理多行映射到同一输出行的情况
- 多核竞写同一地址，累加顺序不确定

#### 2.2.2 BlockEpilogueDequantSequentialWrite（确定性，sequential write）

```cpp
// 文件: block_epilogue_dequant_sequential_write.h
// 关键函数: VectorSequentialWrite

void VectorSequentialWrite(uint32_t curBaseN, uint32_t curVecBaseM,
                           uint64_t offsetM, uint64_t yOffset,
                           LocalTensor<DataTypeOut> &yLocal)
{
    uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
    for (uint32_t i = 0; i < curVecBaseM; i++) {
        // 按输入行顺序写入 workspace，NOT scatter
        // 每个 (absRowBase+i, yOffset) 地址全局唯一
        DataCopyPad(yGlobal_[(absRowBase + i) * n_ + yOffset], yLocal[i * alignN_], paramsOut);
    }
    // 无 AtomicAdd！直接写入
}
```

关键差异：

| 特性 | FinalizeRouting (非确定性) | SequentialWrite (确定性) |
|------|--------------------------|-------------------------|
| 写入目标 | yGm（最终输出） | workspace（中间缓冲区） |
| 寻址方式 | scatter by outRow | sequential by mOffset |
| 是否 AtomicAdd | 是 | 否 |
| 多核竞写 | 有（同一 outRow） | 无（每个 mOffset 唯一） |
| 需要额外聚合 | 否 | 是（FRDeterministicA5） |

#### 2.2.3 accumulatedGroupOffset_ 机制

Cgmct 框架的 `offsetM` 是 group 内相对偏移，不是全局偏移。如果不做修正，不同 group 会写到 workspace 的重叠地址。

```cpp
// 在 UpdateGlobalAddr 中捕获跨 group 的累积偏移
void UpdateGlobalAddr(const BlockCoord &baseOffset) {
    accumulatedGroupOffset_ = static_cast<uint64_t>(Get<SEQ_LOGIT_INDEXS>(baseOffset));
    // SEQ_LOGIT_INDEXS 是 kernel 通过 UpdateOffset() 累积计算的值
    // ...
}

// 在 VectorSequentialWrite 中使用
uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;
// absRowBase 是全局唯一的行索引
```

---

### 2.3 A5 平台特殊性

#### 2.3.1 SyncAll 严格 1:1 配对要求

A5 平台的 `SyncAll()` 要求所有核（包括 AIC 和 AIV）必须参与，且每个核调用 SyncAll 的次数必须完全一致（严格 1:1 配对）。否则会导致死锁。

```cpp
// gmm_fr_deterministic_a5.h 中的处理
void FRDeterministicA5(...) {
    if (g_coreType == AIC) {
        // AIC 核只参与同步，不处理数据
        SyncAll();  // 配对 1
        SyncAll();  // 配对 2
        return;
    }
    SyncAll();  // 配对 1: 确保 workspace 写入完成
    // ... 数据处理 ...
    SyncAll();  // 配对 2: 确保聚合写入完成
}
```

SyncAll 调用统计：

| 模式 | Cgmct Kernel 内部 | FRDeterministicA5 | 总计 |
|------|-------------------|-------------------|------|
| 非确定性 | 1 | 0 | 1 |
| 确定性 | 1 | 2 | 3 |

注意：Cgmct Kernel 内部的 SyncAll 由框架统一处理，所有核（包括确定性/非确定性分支）都会调用。确定性分支额外多 2 个 SyncAll（在 FRDeterministicA5 中），由 `deterministicFlag` 控制所有核走同一分支，保证配对正确。

#### 2.3.2 Cgmct 框架的 workspace 机制

```
workspace 布局:
  [0, SYS_WORKSPACE_SIZE)           -- 系统工作空间 (16MB)，AscendC 底层使用
  [SYS_WORKSPACE_SIZE, ...)         -- 确定性 workspace (deterBuffer)
```

```cpp
constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024 * 1024;  // 16MB
GM_ADDR deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE;
```

workspace 大小在 Tiling 中根据 L2 大小决定：

```cpp
// op_host 中
uint64_t l2_size;
ascendcPlatform.GetCoreMemSize(CoreMemType::L2, l2_size);
deterWorkspaceSize_ = l2_size > (96 * 1024 * 1024) ? (96 * 1024 * 1024) : (64 * 1024 * 1024);
// 通常 64MB 或 96MB
```

#### 2.3.3 A5 与 A3 确定性实现的差异

| 方面 | A3 (arch32) | A5 (arch35) |
|------|-------------|-------------|
| 编程框架 | 原生 AscendC API (MatmulImpl) | Cgmct Builder 模式 |
| Cube 输出 | 写入 workspace (GM) | 写入 UB (l0cOutUb) |
| Epilogue 位置 | 内联在 Kernel 类中 | 独立 Block 类 |
| 滑动窗口 | 有 (VectorSync 控制) | 无 (简化为一次全量聚合) |
| 中间缓冲区 | mmQuantOutGm + workspace | deterBuffer (workspace 偏移) |
| 同步方式 | AIC/AIV 独立，CrossCore + SyncAll | SyncAll 统一，所有核必须配对 |

A3 的滑动窗口机制：当总行数超过 workspace 容量时，分段处理。A5 简化处理：workspace 通常是 64-96MB，足以容纳所有中间结果，因此无需滑动窗口，一次性聚合。

---

### 2.4 GMMTiling 类型系统

#### 2.4.1 核心问题

GMMTiling 是 Kernel 模板内部的嵌套类型。不同的 Epilogue 模板参数会导致 Kernel 实例化出不同的类型，从而产生**不兼容的 GMMTiling 类型**。

```cpp
// 非确定性 Kernel 使用 FinalizeRouting Epilogue
using BlockEpilogueDequant =
    Block::BlockEpilogueDequantFinalizeRouting<CType, C1Type, weightscaleType, xscaleType, BiasType, rowIndexType>;

using GmmKernel = Kernel::KernelGmmFinalizeRoutingPertokenDequant<
    ProblemShape, BlockMmadBuilder, BlockPrologue, BlockEpilogueDequant, BlockScheduler>;

using GMMTiling = typename GmmKernel::GMMTiling;  // 类型 A


// 确定性 Kernel 使用 SequentialWrite Epilogue
using BlockEpilogueDequantSeq =
    Block::BlockEpilogueDequantSequentialWrite<CType, C1Type, weightscaleType, xscaleType, BiasType, rowIndexType>;

using GmmKernelDeterministic = Kernel::KernelGmmFinalizeRoutingPertokenDequant<
    ProblemShape, BlockMmadBuilder, BlockPrologue, BlockEpilogueDequantSeq, BlockScheduler>;

using GMMTilingDeterministic = typename GmmKernelDeterministic::GMMTiling;  // 类型 B
// GMMTiling 和 GMMTilingDeterministic 是不同的 C++ 类型，不能互换！
```

#### 2.4.2 实践影响

确定性分支必须：
1. 使用独立的 `GmmKernelDeterministic` 实例
2. 使用独立的 `GMMTilingDeterministic` 类型的参数对象
3. 使用独立的 `ParamsDeterministic` 类型的参数结构体

```cpp
// 正确做法：为确定性分支创建独立的参数
GMMTilingDeterministic gmmParamsDeter{...};
gmmParamsDeter.matmulTiling = &matmulTiling_;

ParamsDeterministic params = {
    {1, 1, 1, 1},
    {x, w, deterBuffer, bias, group_list},      // BlockMmad: c -> workspace
    {share_input, y, ...},                        // Prologue: y -> yGm
    {deterBuffer, w_scale, x_scale, ...},         // Epilogue: y -> workspace
    gmmParamsDeter                                // 独立的 GMMTiling 对象
};
```

错误做法会导致编译错误：
```
error: no viable conversion from 'GMMTiling' to 'GMMTilingDeterministic'
```

---

## 3. 调试经验

### 3.1 症状到原因映射

#### 3.1.1 hash 每次运行不同

**症状**：精度测试中，多次运行相同输入，输出的 hash 值不一致。

**根因**：多核并行累加顺序不确定。Epilogue 中使用 scatter + AtomicAdd 写入 yGm，不同核的执行时序影响累加结果。

**排查方法**：
1. 检查 Epilogue 是否使用了 `SetAtomicAdd` + scatter 写入
2. 确认是否所有核都会对同一地址执行 AtomicAdd
3. 验证是否走入了确定性分支（`deterministicFlag == 1`）

#### 3.1.2 输出值系统性偏小

**症状**：精度测试中 `actual` 值系统性小于 `expected`，error ratio 约 0.5x ~ 0.97x。

**根因**：聚合阶段的 `totalM` 计算错误，导致累加范围不足。常见错误是使用 `batch`（输出行数）而非 `M`（输入总行数）。

```cpp
// 错误：batch 是输出行数（去重后的行数），不是输入总行数
uint64_t totalM = static_cast<uint64_t>(gmmFinalizeRoutingQuantParams_.batch);

// 正确：M 是输入总行数（= batch * topK），对应 workspace 中实际写入的行数
uint64_t totalM = static_cast<uint64_t>(matmulTiling_.M);
```

**排查方法**：
1. 打印 `totalM`、`batch`、`M` 的值，确认含义
2. 对照 workspace 写入时的行范围，确保聚合读取范围匹配
3. 如果 error ratio 约为 `1/topK`，说明少累加了一个 topK 维度

#### 3.1.3 编译错误 GMMTiling 类型不兼容

**症状**：编译时出现 `no viable conversion` 或 `template argument deduction failed`。

**根因**：非确定性和确定性 Kernel 的 GMMTiling 是不同类型（参见 2.4 节）。

**排查方法**：
1. 检查是否混用了 `GmmKernel::GMMTiling` 和 `GmmKernelDeterministic::GMMTiling`
2. 确保确定性分支使用独立的 `GMMTilingDeterministic` 类型
3. 确保确定性分支使用独立的 `ParamsDeterministic` 类型

#### 3.1.4 输出值严重错误（~97.5% error rate）

**症状**：精度测试中 error rate 极高，输出值与期望值差异巨大。

**根因**：Prologue 和 Epilogue 写入了同一块 workspace 缓冲区。Prologue 写入 residual 数据，Epilogue 覆盖了相同地址的 dequant 结果，导致 residual 完全丢失。

**排查方法**：
1. 检查 Prologue 的 `yGmAddr` 参数指向哪里
2. 检查 Epilogue 的 `yGMAddr` 参数指向哪里
3. 确保两者不指向相同的 buffer

### 3.2 调试方法论

#### 3.2.1 从精度测试输出推断根因

精度测试通常输出以下信息：
- `hash`：输出的 hash 值（多次运行比较判断确定性）
- `error ratio`（actual / expected）：判断精度方向
- `max abs error`：最大绝对误差

| hash 稳定? | error ratio | 推断 |
|------------|-------------|------|
| 不稳定 | 接近 1.0 | 基本正确但有微小不确定性 -> AtomicAdd 多核竞写 |
| 稳定 | 系统性偏小 | 累加范围不足 -> totalM 用错 |
| 稳定 | 严重偏离 | 数据丢失 -> Prologue/Epilogue 冲突 |
| N/A | N/A | 编译失败 -> GMMTiling 类型不兼容 |

#### 3.2.2 对比 A3 原型定位 A5 差异

A3 原型文件（`grouped_matmul_finalize_routing.h`）是验证 A5 实现正确性的重要参照。对比要点：

1. **数据流**：A3 的 `VectorAtomicProcess`（确定性模式）写 `mmQuantOutGm`，A5 的 `VectorSequentialWrite` 写 `deterBuffer`
2. **聚合逻辑**：A3 的 `FRDeterministic` 和 A5 的 `FRDeterministicA5` 结构应一致
3. **SyncAll 次数**：A3 中 AIC 核不参与 SyncAll（`if ASCEND_IS_AIC return`），A5 中 AIC 核必须参与

#### 3.2.3 验证数据流的每个阶段

按阶段验证：

1. **Phase 1 (Prologue)**：验证 yGm 是否正确初始化（零 + residual）
   - 可以临时跳过 Phase 2/3，直接读 yGm 验证

2. **Phase 2 (Epilogue -> workspace)**：验证 workspace 中的值是否正确
   - 检查 workspace 中的行数是否等于 `M`（输入总行数）
   - 验证 `(absRowBase + i) * n_ + yOffset` 地址计算是否正确

3. **Phase 3 (Aggregation)**：验证聚合结果
   - 检查 `totalM` 是否等于 workspace 中的行数
   - 检查 `outRow % coreNumVec == GetBlockIdx()` 分配是否正确
   - 确认 AtomicAdd 是在 residual 之上累加，而非覆盖

---

## 4. 常见陷阱

### 4.1 totalM 使用 batch 而非 M

```cpp
// 错误
uint64_t totalM = static_cast<uint64_t>(gmmFinalizeRoutingQuantParams_.batch);
// batch = 输出行数（去重后），比如 [0,1,2] 3行

// 正确
uint64_t totalM = static_cast<uint64_t>(matmulTiling_.M);
// M = 输入总行数 = batch * topK，比如 topK=8 时 M=24
```

后果：聚合只处理部分 workspace 行，导致输出值系统性偏小。

### 4.2 Prologue 和 Epilogue 指向同一 workspace

```cpp
// 错误：Prologue 和 Epilogue 都指向 deterBuffer
ParamsDeterministic params = {
    ...
    {share_input, deterBuffer, ...},   // Prologue -> workspace (错误!)
    {deterBuffer, w_scale, ...},        // Epilogue -> workspace
    ...
};

// 正确：Prologue 指向最终输出 y，Epilogue 指向 workspace
ParamsDeterministic params = {
    ...
    {share_input, y, ...},              // Prologue -> yGm (正确)
    {deterBuffer, w_scale, ...},        // Epilogue -> workspace
    ...
};
```

后果：Prologue 写入的 residual 被 Epilogue 的 dequant 结果覆盖，导致 residual 丢失。

### 4.3 修改共享 Epilogue 文件影响非确定性路径

`BlockEpilogueDequantSequentialWrite` 是确定性专用的 Epilogue，与 `BlockEpilogueDequantFinalizeRouting` 是独立的文件。但如果修改了 `BlockEpilogueDequantFinalizeRouting`（非确定性路径共享的文件），可能意外影响非确定性路径。

**预防措施**：
- 确定性适配应创建新的 Epilogue 文件，不要修改原始文件
- 如果必须修改共享文件，添加条件判断（如 `deterministicFlag`），避免影响非确定性路径

### 4.4 workspace 大小计算遗漏

A5 workspace 布局：`SYS_WORKSPACE_SIZE(16MB) + deterWorkspaceSize(64-96MB)`

容易遗漏的要点：
- `deterBuffer` 的起始地址必须是 `workspaceGM + SYS_WORKSPACE_SIZE`，不能从 0 开始
- `deterWorkspaceSize` 必须在 Tiling 中计算并传递到 Kernel
- workspace 大小需满足 `M * N * sizeof(float) <= deterWorkspaceSize`

### 4.5 Cgmct offsetM 是 group 内相对偏移

Cgmct 框架为每个 group 独立调度 tile，`offsetM` 是当前 group 内的行偏移，不是全局偏移。

```cpp
// 必须使用 accumulatedGroupOffset_ 修正为全局偏移
uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;
// 如果不用 accumulatedGroupOffset_，不同 group 会写到 workspace 重叠地址
```

### 4.6 A5 SyncAll 必须所有核参与

A3 中 AIC 核可以跳过 SyncAll（`if ASCEND_IS_AIC return`），但 A5 中 AIC 核也必须调用 SyncAll。

```cpp
// A5 正确做法
if (g_coreType == AIC) {
    SyncAll();  // 必须参与
    SyncAll();  // 必须参与
    return;
}
```

如果 AIC 核不参与 SyncAll，会导致死锁。

---

## 5. 修改模板

### 5.1 通用适配步骤

以下步骤适用于将确定性特性从 A3 移植到 A5 的 Cgmct 框架：

#### 步骤 1：创建确定性 Epilogue

基于原始 Epilogue 文件，创建新的确定性变体：

```bash
# 从原始 Epilogue 复制
cp block_epilogue_dequant_finalize_routing.h block_epilogue_dequant_sequential_write.h
```

需要修改的关键函数：

```cpp
// 1. 将 VectorAtomicProcess 替换为 VectorSequentialWrite
// 2. 添加 accumulatedGroupOffset_ 成员变量
// 3. 在 UpdateGlobalAddr 中捕获累积偏移

// 替换 VectorAtomicProcess:
//   旧: SetAtomicAdd + scatter by outRow
//   新: direct write by absRowBase (no AtomicAdd)
void VectorSequentialWrite(uint32_t curBaseN, uint32_t curVecBaseM,
                           uint64_t offsetM, uint64_t yOffset,
                           LocalTensor<DataTypeOut> &yLocal)
{
    uint64_t absRowBase = accumulatedGroupOffset_ + offsetM;
    DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curBaseN * sizeof(DataTypeOut)), 0, 0, 0};
    for (uint32_t i = 0; i < curVecBaseM; i++) {
        DataCopyPad(yGlobal_[(absRowBase + i) * n_ + yOffset], yLocal[i * alignN_], paramsOut);
    }
}

// 添加成员:
uint64_t accumulatedGroupOffset_ = 0;

// 在 UpdateGlobalAddr 中:
accumulatedGroupOffset_ = static_cast<uint64_t>(Get<SEQ_LOGIT_INDEXS>(baseOffset));
```

#### 步骤 2：创建确定性聚合函数

```cpp
// 文件: gmm_fr_deterministic_a5.h
// 模板:
template <typename DTYPE_OUT, typename ROW_INDEX_DTYPE>
__aicore__ inline void FRDeterministicA5(
    SyncConfig& syncConfig,
    GlobalTensor<DTYPE_OUT>& deterBufferGm,
    GlobalTensor<DTYPE_OUT>& yGm,
    GlobalTensor<ROW_INDEX_DTYPE>& tokenRanksGm,
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>& queBind,
    uint32_t coreNum,
    uint32_t n)
{
    // AIC 核只同步不处理
    if (g_coreType == AIC) {
        SyncAll();
        SyncAll();
        return;
    }

    SyncAll();  // 确保所有 workspace 写入完成

    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = coreNum * GetTaskRation();
    uint64_t baseOffset = syncConfig.lowBoundM - syncConfig.windowSize;

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(baseOffset + mOffset));
        if (outRow % coreNumVec != GetBlockIdx()) continue;

        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            // workspace -> UB
            LocalTensor<DTYPE_OUT> bindLocal = queBind.AllocTensor<DTYPE_OUT>();
            DataCopyPad(bindLocal, deterBufferGm[mOffset * n + nOffset], ...);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_OUT>();

            // UB -> yGm (AtomicAdd, single writer per outRow)
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyPad(yGm[outRow * n + nOffset], bindLocal, ...);
            SetAtomicNone();

            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();  // 确保所有聚合写入完成
}
```

#### 步骤 3：修改 Kernel 入口文件

```cpp
// 在 Kernel 入口文件中添加确定性分支:

// 1. 定义确定性 Kernel 类型（注意：GMMTiling 是不同类型!）
using BlockEpilogueDequantSeq =
    Block::BlockEpilogueDequantSequentialWrite<CType, C1Type, weightscaleType,
                                                xscaleType, BiasType, rowIndexType>;

using GmmKernelDeterministic =
    Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder,
        BlockPrologue, BlockEpilogueDequantSeq, BlockScheduler>;

using GMMTilingDeterministic = typename GmmKernelDeterministic::GMMTiling;
using ParamsDeterministic = typename GmmKernelDeterministic::Params;

// 2. 在主函数中分支处理
if (deterministicFlag == 1) {
    // 确定性模式
    constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024 * 1024;
    GM_ADDR deterBuffer = workspaceGM + SYS_WORKSPACE_SIZE;

    // 使用独立的 GMMTiling 类型
    GMMTilingDeterministic gmmParamsDeter{...};
    gmmParamsDeter.matmulTiling = &matmulTiling_;

    // 注意: Prologue -> yGm (NOT workspace!), Epilogue -> workspace
    ParamsDeterministic params = {
        {1, 1, 1, 1},
        {x, w, deterBuffer, bias, group_list},   // Cube 输出到 workspace
        {share_input, y, ...},                     // Prologue -> yGm (关键!)
        {deterBuffer, w_scale, x_scale, ...},      // Epilogue -> workspace
        gmmParamsDeter
    };

    GmmKernelDeterministic gmm;
    gmm(params);

    // 聚合
    // ... 设置 GlobalTensor, TQueBind 等 ...
    GMMFRDeterministic::FRDeterministicA5<float, rowIndexType>(...);

} else {
    // 非确定性模式（原有逻辑不变）
    Params params = {...};
    GmmKernel gmm;
    gmm(params);
}
```

#### 步骤 4：修改 Tiling

```cpp
// 在 Tiling 处理函数中添加:
void DeterministicTilingProcess() {
    if (context_->GetDeterministic() == 0) {
        deterministicFlag_ = 0;
        return;
    }
    deterministicFlag_ = 1;
    uint64_t l2_size;
    ascendcPlatform.GetCoreMemSize(CoreMemType::L2, l2_size);
    deterWorkspaceSize_ = l2_size > (96 * 1024 * 1024) ? (96 * 1024 * 1024) : (64 * 1024 * 1024);
    workspaceSize_ += deterWorkspaceSize_;
}
```

### 5.2 参数配置检查清单

在实现确定性适配后，逐项检查：

- [ ] **GMMTiling 类型**：确定性分支使用 `GMMTilingDeterministic`，非确定性使用 `GMMTiling`
- [ ] **Params 类型**：确定性分支使用 `ParamsDeterministic`
- [ ] **Prologue 目标**：`yGmAddr` 指向 `y`（最终输出），不指向 `deterBuffer`
- [ ] **Epilogue 目标**：`yGMAddr` 指向 `deterBuffer`（workspace）
- [ ] **Cube 输出目标**：`cGMAddr` 指向 `deterBuffer`（workspace）
- [ ] **deterBuffer 偏移**：`workspaceGM + SYS_WORKSPACE_SIZE`
- [ ] **totalM 计算**：使用 `matmulTiling_.M`（输入总行数），不使用 `batch`
- [ ] **SyncAll 配对**：AIC 核也参与 SyncAll，总次数与非 AIC 核一致
- [ ] **accumulatedGroupOffset_**：在 `UpdateGlobalAddr` 中正确捕获跨 group 累积偏移
- [ ] **workspace 大小**：满足 `M * N * sizeof(float) <= deterWorkspaceSize`

### 5.3 文件修改清单模板

| 文件 | 操作 | 说明 |
|------|------|------|
| `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` | 新建 | 确定性 Epilogue，基于原始 Epilogue 修改 |
| `op_kernel/arch35/gmm_fr_deterministic_a5.h` | 新建 | 确定性聚合函数 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 修改 | 添加确定性分支、include 新文件 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 修改 | 添加 `deterministicFlag`、`deterWorkspaceSize` |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 修改 | 添加 Tiling 数据字段 |
