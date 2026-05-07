/**
 * A3 原型: VectorSync 滑动窗口触发机制
 *
 * 源文件: op_kernel/grouped_matmul_finalize_routing.h 第622-649行
 *
 * 原理:
 *   workspace 大小有限（96MB），不能存放所有行的中间结果。
 *   因此使用滑动窗口：每当累计行数达到 windowSize，触发一次 FRDeterministic。
 *   FRDeterministic 完成后，workspace 中的数据已写入 yGm，可以复用。
 *
 * A5 迁移要点:
 *   - 滑动窗口逻辑与平台无关，可直接复用
 *   - groupTokensGm.GetValue() 读取 group 行数，这部分 GM 读取在 A5 上相同
 *   - 需要适配 A5 的 SyncAll 行为
 */

template <class P>
__aicore__ inline void QuantGroupMatmul<P>::VectorSync(MNConfig& mnConfig, SyncConfig& syncConfig)
{
    if ASCEND_IS_AIC {
        return;                    // Cube 核不参与
    }
    if (tiling->deterministicFlag == 0) {
        return;                    // 非确定性模式不触发
    }

    // 滑动窗口：当累计行数超过窗口下界时触发
    while (mnConfig.curBlockM > syncConfig.lowBoundM) {
        // 按 group 累计行数
        while (syncConfig.curGroup < tiling->groupNum) {
            uint32_t mi = static_cast<uint32_t>(groupTokensGm.GetValue(syncConfig.curGroup));
            if constexpr (P::groupListType) {
                if (syncConfig.curGroup > 0) {
                    mi -= static_cast<uint32_t>(groupTokensGm.GetValue(syncConfig.curGroup - 1));
                }
            }
            if (syncConfig.curGroupM + mi <= syncConfig.lowBoundM) {
                syncConfig.curGroupM += mi;
                syncConfig.curM = syncConfig.curGroupM;
                syncConfig.curGroup++;
            } else {
                syncConfig.curM += (syncConfig.lowBoundM - syncConfig.curM) / mnConfig.singleM * mnConfig.singleM;
                break;
            }
        }
        // 触发确定性聚合
        FRDeterministic(syncConfig);
        // 窗口滑动
        syncConfig.lowBoundM = syncConfig.curM + syncConfig.windowSize;
    }
}
