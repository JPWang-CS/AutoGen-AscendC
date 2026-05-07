/**
 * A3 原型: VectorAtomicProcess 确定性分支
 *
 * 源文件: op_kernel/grouped_matmul_finalize_routing.h 第377-402行
 *
 * 原理:
 *   非确定性模式: SetAtomicAdd + DataCopyPad 直接写入 yGm（多核竞争，顺序不确定）
 *   确定性模式:   DataCopyPad2D 写入 workspace（mmQuantOutGm），后续由 FRDeterministic 统一聚合
 *
 * 关键区别:
 *   确定性写 mmQuantOutGm[mOffset * n + nOffset]，其中 mOffset 是相对于窗口起始的偏移
 *   非确定性写 yGm[outRow * tiling->n + nOffset]，直接 scatter 到目标行
 *
 * A5 迁移要点:
 *   在 A5 的 VCV Basic Block 或 Cgmct Epilogue 中需要相同的分支逻辑
 */

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::VectorAtomicProcess(
    const VectorAtomicParams& vecAParams, const SyncConfig& syncConfig)
{
    LocalTensor<DTYPE_OUT> yLocal = vecOutQueue.DeQue<DTYPE_OUT>();

    if constexpr (P::combine) {
        // ===== combine 模式（带 finalize routing scatter）=====

        if (tiling->deterministicFlag == 1) {
            // 确定性：写入 workspace 中间缓冲区（不 scatter，按原始顺序）
            DataCopy2DDimParams dimParams{vecAParams.curVecBaseM, vecAParams.curVecBaseN, vecAParams.alignBaseN};
            DataCopyPad2D(mmQuantOutGm[vecAParams.yGmOffset1 - (syncConfig.lowBoundM - syncConfig.windowSize) * tiling->n],
                yLocal, dimParams, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
            return;
        }

        // 非确定性：直接 scatter + AtomicAdd 到 yGm
        SetAtomicAdd<float>();
        DataCopyExtParams paramsOut{1, static_cast<uint32_t>(vecAParams.curVecBaseN * sizeof(float)), 1, 1, 0};
        for (uint32_t i = 0; i < vecAParams.curVecBaseM; i++) {
            auto outRow = static_cast<uint64_t>(
                tokenRanksGm.GetValue(vecAParams.mGlobalOffset + vecAParams.offsetM + i));
            DataCopyPad(yGm[outRow * tiling->n + vecAParams.yGmOffset0],
                        yLocal[i * vecAParams.alignBaseN], paramsOut);
        }
        SetAtomicNone();

    } else {
        // ===== 非 combine 模式（无 scatter，直接写入）=====
        DataCopy2DDimParams dimParams{vecAParams.curVecBaseM, vecAParams.curVecBaseN, vecAParams.alignBaseN};
        DataCopyPad2D(yGm[vecAParams.yGmOffset1], yLocal, dimParams, tiling->n);
    }
    vecOutQueue.FreeTensor(yLocal);
}
