/**
 * A3 原型: 工具函数、常量和数据结构
 *
 * 源文件: op_kernel/grouped_matmul_finalize_routing_utils.h
 *         op_kernel/grouped_matmul_finalize_routing.h (Init/InitUbBuffer/Process)
 *
 * 关键常量和数据结构:
 */

// ============ 确定性相关常量 ============
// 源: op_kernel/grouped_matmul_finalize_routing_utils.h 第30行
constexpr uint32_t DETER_UB_SIZE = 12 * 1024;  // 确定性 UB 缓冲区大小 (12KB)


// ============ SyncConfig 数据结构 ============
// 源: op_kernel/grouped_matmul_finalize_routing_utils.h 第49-56行
struct SyncConfig {
    uint64_t curM = 0;        // 当前累计行偏移
    uint64_t curGroup = 0;    // 当前 group 索引
    uint64_t curGroupM = 0;   // 当前 group 内累计行
    uint64_t lowBoundM = 0;   // 窗口下界
    uint64_t windowSize = 0;  // 窗口大小 = deterWorkspaceSize / (n * sizeof(DTYPE_OUT))
    uint64_t baseN = 0;       // N 方向分块大小，128 对齐
};


// ============ MNBlockIdxCompute 确定性调整 ============
// 源: op_kernel/grouped_matmul_finalize_routing_utils.h 第140-161行
// 当 deterministicFlag == 1 时，使用简单的顺序分配策略（不使用对角策略）
__aicore__ inline void MNBlockIdxCompute(MNConfig& mnConfig, const uint32_t curBlock,
                                         const uint32_t count, const uint32_t thresholdMDimN,
                                         const uint32_t deterministicFlag)
{
    // 确定性模式：顺序分配（简单取模），保证每个核处理固定顺序的块
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1 || deterministicFlag == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        // 非确定性模式：对角策略（提高缓存命中率，但分配顺序不确定）
        // ... 复杂的对角分配逻辑 ...
    }
}


// ============ Init 中确定性 workspace 初始化 ============
// 源: op_kernel/grouped_matmul_finalize_routing.h 第162-166行
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::Init(...)
{
    // ... 其他初始化 ...

    // 确定性模式：分配 mmQuantOutGm 指向 workspace 中的确定性缓冲区
    if (tiling->deterministicFlag == 1) {
        mmQuantOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_OUT *>(
            initParams.workspace +
            tiling->parallNum * tiling->matmulTiling.baseM * tiling->matmulTiling.baseN *
            sizeof(int32_t) * tiling->coreNum));
    }
    // ... 其他初始化 ...
}


// ============ InitUbBuffer 中确定性 UB buffer 初始化 ============
// 源: op_kernel/grouped_matmul_finalize_routing.h 第186-188行
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::InitUbBuffer()
{
    if ASCEND_IS_AIC { return; }

    // ... 其他 buffer 初始化 ...

    // 确定性模式：分配 queBind 用于 FRDeterministic 中的中转 buffer
    if (tiling->deterministicFlag == 1) {
        pipe->InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);
    }
    // ... 其他 buffer 初始化 ...
}


// ============ Process 中 SyncConfig 初始化 ============
// 源: op_kernel/grouped_matmul_finalize_routing.h 第290-334行
template <class P>
__aicore__ inline void QuantGroupMatmul<P>::Process()
{
    // ... PreProcess ...

    SyncConfig syncConfig;
    syncConfig.windowSize = tiling->deterWorkspaceSize / (tiling->n * sizeof(DTYPE_OUT));
    syncConfig.lowBoundM = syncConfig.windowSize;
    uint64_t nTimes = Ceil(tiling->n, DETER_UB_SIZE / sizeof(DTYPE_OUT));
    syncConfig.baseN = Ceil(Ceil(tiling->n, nTimes), 128) * 128;  // 128 对齐

    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        // ... 遍历 group ...
        while (curBlock < curCount) {
            MNBlockIdxCompute(mnConfig, curBlock, preCount, thresholdMDimN, tiling->deterministicFlag);
            MMCompute(groupIdx, mnConfig);
            VectorSync(mnConfig, syncConfig);       // ← 滑动窗口触发
            VectorCompute(groupIdx, mnConfig, syncConfig);
            curBlock += tiling->coreNum;
        }
        // ...
    }

    // 处理剩余行
    if (tiling->deterministicFlag == 1) {
        syncConfig.curM = mnConfig.offsetM;
        FRDeterministic(syncConfig);    // ← 最终确定性聚合
    }
}
