/**
 * 新增文件: op_kernel/arch35/gmm_fr_deterministic_a5.h
 *
 * 新增原因: A5 平台的确定性延迟聚合函数。参考 A3 的 FRDeterministic()
 *           (op_kernel/grouped_matmul_finalize_routing.h 第653-687行)，
 *           但适配 A5 的 Cgmct 框架和更严格的同步要求。
 *
 * A3 原型: FRDeterministic() 使用 SyncAll + 按行归属 + SetAtomicAdd
 * A5 适配:
 *   - SyncAll 必须在所有核上严格配对（A5 要求）
 *   - DETER_UB_SIZE = 12KB，用于 queBind 中转 buffer
 *   - 使用 A3 的 DataCopyPad2D 辅助函数（需要在 A5 中实现等效版本）
 */

#ifndef GMM_FR_DETERMINISTIC_A5_H
#define GMM_FR_DETERMINISTIC_A5_H

#include "kernel_operator.h"

namespace GMMFRDeterministic {

using namespace AscendC;

// 与 A3 保持一致
constexpr uint32_t DETER_UB_SIZE = 12 * 1024;
constexpr uint32_t BUFFER_NUM = 2;

// 滑动窗口同步配置（与 A3 SyncConfig 完全一致）
struct SyncConfig {
    uint64_t curM = 0;
    uint64_t curGroup = 0;
    uint64_t curGroupM = 0;
    uint64_t lowBoundM = 0;
    uint64_t windowSize = 0;
    uint64_t baseN = 0;
};

// 确定性聚合所需的全局参数
struct DeterministicParams {
    __gm__ float* deterBuffer;       // workspace 中确定性缓冲区地址
    __gm__ float* yGm;              // 最终输出地址
    __gm__ int64_t* tokenRanksGm;   // 行索引
    __gm__ int64_t* groupTokensGm;  // group 行数
    uint32_t n;                      // 总列数
    uint32_t groupNum;               // group 数量
    uint32_t coreNum;                // AIC 核数
    uint32_t groupListType;          // 0=累计型, 1=差值型
    uint32_t deterministicFlag;      // 确定性标志
    uint32_t deterWorkspaceSize;     // 确定性 workspace 大小
};

/**
 * DataCopyPad2D 辅助函数 - A5 版本
 * A3 原型: op_kernel/grouped_matmul_finalize_routing.h 第50-62行
 */
template <typename T>
__aicore__ inline void DataCopyPad2DLocalToGlobal(
    const GlobalTensor<T> dst, const LocalTensor<T> src,
    uint32_t dim0, uint32_t dim1, uint32_t srcDim0, uint32_t dstDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

/**
 * DataCopyPad2D 辅助函数 - GM to Local
 * A3 原型: op_kernel/grouped_matmul_finalize_routing.h 第50-62行
 */
template <typename T>
__aicore__ inline void DataCopyPad2DGlobalToLocal(
    const LocalTensor<T> dst, const GlobalTensor<T> src,
    uint32_t dim0, uint32_t dim1, uint32_t srcDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    params.dstStride = Ceil(dim0 * sizeof(T), 32) % 2;
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

/**
 * FRDeterministicA5 - A5 版确定性延迟聚合核心函数
 *
 * A3 原型: QuantGroupMatmul<P>::FRDeterministic()
 *         (op_kernel/grouped_matmul_finalize_routing.h 第653-687行)
 *
 * 执行流程（与 A3 完全一致）:
 *   1. SyncAll() - 全核同步
 *   2. 按 outRow % coreNumVec 分配行归属
 *   3. 从 workspace 读取 → UB (queBind)
 *   4. SetAtomicAdd + DataCopyPad 写入 yGm
 *   5. SyncAll() - 全核同步
 */
template <typename DTYPE_OUT, typename ROW_INDEX_DTYPE>
__aicore__ inline void FRDeterministicA5(
    SyncConfig& syncConfig,
    GlobalTensor<DTYPE_OUT>& deterBufferGm,
    GlobalTensor<DTYPE_OUT>& yGm,
    GlobalTensor<ROW_INDEX_DTYPE>& tokenRanksGm,
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>& queBind,
    uint32_t coreNum, uint32_t n)
{
    if (g_coreType == AIC) {
        return;                    // Cube 核不参与
    }

    SyncAll();                     // 第一步：全核同步

    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = coreNum * GetTaskRation();
    uint64_t baseOffset = syncConfig.lowBoundM - syncConfig.windowSize;

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        auto outRow = static_cast<uint64_t>(tokenRanksGm.GetValue(baseOffset + mOffset));

        // 行归属判断（与 A3 完全一致）
        if (outRow % coreNumVec != GetBlockIdx()) {
            continue;
        }

        uint64_t curVecBaseN = syncConfig.baseN;
        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            if (nOffset + syncConfig.baseN >= n) {
                curVecBaseN = n - nOffset;     // 尾块
            }

            // 从 workspace 读取到 UB
            LocalTensor<DTYPE_OUT> bindLocal = queBind.AllocTensor<DTYPE_OUT>();
            DataCopyPad2DGlobalToLocal(bindLocal, deterBufferGm[mOffset * n + nOffset],
                                       static_cast<uint32_t>(curVecBaseN), 1,
                                       static_cast<uint32_t>(curVecBaseN));
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_OUT>();

            // 原子写入最终输出
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(DTYPE_OUT)), 0, 0, 0};
            DataCopyPad(yGm[outRow * n + nOffset], bindLocal, paramsOut);
            SetAtomicNone();

            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();                     // 最后一步：全核同步
}

/**
 * VectorSyncA5 - A5 版滑动窗口触发
 *
 * A3 原型: QuantGroupMatmul<P>::VectorSync()
 *         (op_kernel/grouped_matmul_finalize_routing.h 第622-649行)
 *
 * 逻辑：每当累计行数达到 windowSize，触发一次 FRDeterministicA5
 */
template <typename DTYPE_OUT, typename ROW_INDEX_DTYPE>
__aicore__ inline void VectorSyncA5(
    uint64_t curBlockM,
    SyncConfig& syncConfig,
    GlobalTensor<DTYPE_OUT>& deterBufferGm,
    GlobalTensor<DTYPE_OUT>& yGm,
    GlobalTensor<ROW_INDEX_DTYPE>& tokenRanksGm,
    GlobalTensor<int64_t>& groupTokensGm,
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>& queBind,
    uint32_t groupNum, uint32_t coreNum, uint32_t n,
    uint32_t singleM, uint32_t groupListType)
{
    if (g_coreType == AIC) return;

    while (curBlockM > syncConfig.lowBoundM) {
        while (syncConfig.curGroup < groupNum) {
            uint32_t mi = static_cast<uint32_t>(groupTokensGm.GetValue(syncConfig.curGroup));
            if (groupListType == 0 && syncConfig.curGroup > 0) {
                mi -= static_cast<uint32_t>(groupTokensGm.GetValue(syncConfig.curGroup - 1));
            }
            if (syncConfig.curGroupM + mi <= syncConfig.lowBoundM) {
                syncConfig.curGroupM += mi;
                syncConfig.curM = syncConfig.curGroupM;
                syncConfig.curGroup++;
            } else {
                syncConfig.curM += (syncConfig.lowBoundM - syncConfig.curM) / singleM * singleM;
                break;
            }
        }
        FRDeterministicA5<DTYPE_OUT, ROW_INDEX_DTYPE>(
            syncConfig, deterBufferGm, yGm, tokenRanksGm, queBind, coreNum, n);
        syncConfig.lowBoundM = syncConfig.curM + syncConfig.windowSize;
    }
}

}  // namespace GMMFRDeterministic

#endif
