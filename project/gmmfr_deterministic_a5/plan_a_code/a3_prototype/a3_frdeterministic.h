/**
 * A3 原型: FRDeterministic 延迟聚合核心函数
 *
 * 源文件: op_kernel/grouped_matmul_finalize_routing.h 第653-687行
 *
 * 这是确定性机制的核心。执行流程:
 *   1. SyncAll() - 全核同步，确保所有核已完成 workspace 写入
 *   2. 遍历当前窗口内的所有行
 *   3. 按 outRow % coreNumVec 分配行归属（每个核只处理自己负责的行）
 *   4. 从 workspace 中间结果读取到 UB
 *   5. SetAtomicAdd + DataCopyPad 写入最终 yGm（此时只有一个核写入该行，顺序确定）
 *   6. SyncAll() - 再次同步，确保所有核完成写入
 *
 * A5 迁移要点:
 *   - A5 的 SyncAll() 必须在所有核上严格配对
 *   - queBind 需要在 InitUbBuffer 中初始化（大小 DETER_UB_SIZE = 12KB）
 *   - DataCopyPad2D 是 A3 的自定义辅助函数，A5 需要适配
 */

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::FRDeterministic(SyncConfig& syncConfig)
{
    if ASCEND_IS_AIC {
        return;                    // Cube 核不参与
    }
    SyncAll();                     // 第一步：全核同步

    uint64_t totalM = syncConfig.curM - (syncConfig.lowBoundM - syncConfig.windowSize);
    uint64_t coreNumVec = tiling->coreNum * GetTaskRation();  // Vector 核总数
    uint64_t n = tiling->n;

    for (uint64_t mOffset = 0; mOffset < totalM; mOffset++) {
        // 从 tokenRanks 获取该行对应的输出行号
        auto outRow = static_cast<uint64_t>(
            tokenRanksGm.GetValue((syncConfig.lowBoundM - syncConfig.windowSize) + mOffset));

        // 行归属判断：只有 outRow % coreNumVec == GetBlockIdx() 的核才处理
        if (outRow % coreNumVec != GetBlockIdx()) {
            continue;
        }

        uint64_t curVecBaseN = syncConfig.baseN;
        for (uint64_t nOffset = 0; nOffset < n; nOffset += syncConfig.baseN) {
            if (nOffset + syncConfig.baseN >= n) {
                curVecBaseN = n - nOffset;     // 尾块处理
            }
            DataCopyExtParams paramsOut{1, static_cast<uint32_t>(curVecBaseN * sizeof(float)), 0, 0, 0};
            DataCopy2DDimParams copyDimParams{static_cast<uint32_t>(1),
                                              static_cast<uint32_t>(curVecBaseN),
                                              static_cast<uint32_t>(curVecBaseN)};

            // 从 workspace 中间结果读取到 UB
            LocalTensor<DTYPE_OUT> bindLocal = queBind.AllocTensor<DTYPE_OUT>();
            DataCopyPad2D(bindLocal, mmQuantOutGm[mOffset * n + nOffset], copyDimParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<DTYPE_OUT>();

            // 原子写入最终输出（此时只有归属核写入，顺序确定）
            SetAtomicAdd<DTYPE_OUT>();
            DataCopyPad(yGm[outRow * tiling->n + nOffset], bindLocal, paramsOut);
            SetAtomicNone();

            queBind.FreeTensor(bindLocal);
        }
    }
    SyncAll();                     // 最后一步：全核同步
}
