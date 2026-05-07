# GMMFR 确定性特性分析报告

## 一、算子概述

**算子名**: grouped_matmul_finalize_routing (GMMFR)
**路径**: `ops-transformer_AI/gmm/grouped_matmul_finalize_routing`
**功能**: 分组矩阵乘法 + 最终路由归约（Finalize Routing），用于 MoE 模型中 Expert 计算后的结果聚合
**类型**: Cube+Vector 混合算子 (`KERNEL_TYPE_MIX_AIC_1_2`)

## 二、确定性特性原理

### 2.1 为什么需要确定性

GMMFR 的 Finalize Routing 阶段使用 `SetAtomicAdd` 将多核计算结果累加到全局输出。由于多核并行执行顺序不确定，浮点加法不满足结合律，每次运行的累加顺序不同会导致微小的数值差异。

确定性模式通过以下方式保证结果可复现：
1. 使用 workspace 作为中间缓冲区，避免直接原子写 GM
2. 通过 `SyncAll()` 控制核间执行顺序
3. 在所有核完成一段计算后，由指定核统一读取中间结果并写入最终输出
4. 使用 `SetAtomicAdd` 仅在最后阶段、按固定顺序累加

### 2.2 Tiling 侧实现

**文件**: `op_host/grouped_matmul_finalize_routing_base_tiling.cpp`

```cpp
void DeterministicTilingProcess() {
    if (context_->GetDeterministic() == 0) {
        deterministicFlag_ = 0;   // 非确定性：正常模式
        return;
    }
    deterministicFlag_ = 1;        // 确定性模式
    // 根据平台 L2 大小分配 workspace
    uint64_t l2_size;
    ascendcPlatform.GetCoreMemSize(CoreMemType::L2, l2_size);
    deterWorkspaceSize_ = l2_size > 96MB ? 96MB : 64MB;
    workspaceSize_ += deterWorkspaceSize_;
}
```

Tiling 数据字段：
- `deterministicFlag` (uint32_t): 0=非确定性, 1=确定性
- `deterWorkspaceSize` (uint32_t): 确定性 workspace 大小

### 2.3 Kernel 侧实现（arch32）

**文件**: `op_kernel/grouped_matmul_finalize_routing.h`

#### 初始化阶段

```cpp
void Init(...) {
    if (tiling->deterministicFlag == 1) {
        // 分配中间输出缓冲区（在 workspace 中）
        mmQuantOutGm.SetGlobalBuffer(
            workspace + parallNum * baseM * baseN * sizeof(int32_t) * coreNum);
        // 分配 UB 中的绑定队列
        pipe->InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);
    }
}
```

#### VectorSync（滑动窗口同步）

```cpp
void VectorSync(MNConfig& mnConfig, SyncConfig& syncConfig) {
    if (tiling->deterministicFlag == 0) return;

    // 按滑动窗口大小同步
    while (mnConfig.curBlockM > syncConfig.lowBoundM) {
        // 计算当前窗口内的 group 行数
        while (syncConfig.curGroup < tiling->groupNum) { ... }
        // 执行确定性归约
        FRDeterministic(syncConfig);
        // 滑动窗口前进
        syncConfig.lowBoundM = syncConfig.curM + syncConfig.windowSize;
    }
}
```

#### FRDeterministic（核心确定性函数）

```cpp
void FRDeterministic(SyncConfig& syncConfig) {
    SyncAll();  // 所有核同步

    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = tiling->coreNum * GetTaskRation();
    uint64_t n = tiling->n;

    // 每个核只处理自己负责的输出行
    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        auto outRow = tokenRanksGm.GetValue(baseMOffset + mOffset);
        if (outRow % coreNumVec != GetBlockIdx()) continue;

        // 按 baseN 分块处理，避免 UB 溢出
        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            // 从 workspace 中间结果读取
            LocalTensor<DTYPE_OUT> bindLocal = queBind.AllocTensor<DTYPE_OUT>();
            DataCopyPad2D(bindLocal, mmQuantOutGm[mOffset * n + nOffset], ...);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_OUT>();

            // 原子加到最终输出
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyPad(yGm[outRow * n + nOffset], bindLocal, ...);
            SetAtomicNone();

            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();  // 再次同步，确保所有核完成写入
}
```

#### MNBlockIdxCompute 调整

当 `deterministicFlag == 1` 时，块索引计算简化为顺序分配（不跳过行）：
```cpp
void MNBlockIdxCompute(MNConfig& mnConfig, ..., const uint32_t deterministicFlag) {
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1 || deterministicFlag == 1) {
        // 顺序分配：blockIdx % blockDimM
    }
}
```

#### VectorAtomicProcess 调整

确定性模式下，Vector 核将结果写入 workspace 而非直接写 GM：
```cpp
void VectorAtomicProcess(...) {
    if (tiling->deterministicFlag == 1) {
        // 写入 workspace 中间缓冲区
        DataCopyPad2D(mmQuantOutGm[...], yLocal, dimParams, tiling->n);
        return;
    }
    // 非确定性：直接原子写 GM
    SetAtomicAdd<float>();
    DataCopyPad(yGm[outRow * tiling->n + ...], yLocal, ...);
    SetAtomicNone();
}
```

### 2.4 SyncConfig 数据结构

```cpp
struct SyncConfig {
    uint64_t windowSize = 0;      // 窗口大小（workspace 可容纳的行数）
    uint64_t lowBoundM = 0;       // 当前窗口下界
    uint64_t curM = 0;            // 当前行偏移
    uint64_t curGroup = 0;        // 当前 group 索引
    uint64_t curGroupM = 0;       // 当前 group 内行偏移
    uint64_t baseN = 0;           // N 方向分块大小
};
```

## 三、A5 (arch35) 当前状态

**A5 完全没有确定性实现**：
- `op_kernel/arch35/` 目录下无任何 `deterministic` 相关代码
- `op_host/op_tiling/arch35/` 的 Tiling 代码未包含 `deterministicFlag` 处理
- A5 使用 Cgmct 框架，其 Prologue/Epilogue 机制与 arch32 的直接 API 调用完全不同

## 四、迁移关键挑战

1. **编程框架差异**：arch32 用原生 AscendC API，arch35 用 Cgmct Builder 模式
2. **同步机制差异**：A5 要求更严格的核间同步配对
3. **中间缓冲区管理**：A5 的 workspace 分配方式不同
4. **Epilogue 改造**：需在 Cgmct 的 BlockEpilogue 中插入确定性逻辑
5. **Tiling 适配**：需在 A5 Tiling 中添加 deterministicFlag 和 deterWorkspaceSize
