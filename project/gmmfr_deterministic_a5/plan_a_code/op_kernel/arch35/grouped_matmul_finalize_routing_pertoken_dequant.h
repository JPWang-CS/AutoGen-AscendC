/**
 * 修改文件 9: op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h
 *
 * 修改原因: 全量化路径 (K-C/T-C) 的 Cgmct Kernel。
 *           方案A：不修改 Epilogue，通过 Params 重定向 y 地址到 workspace，
 *           然后在 Cgmct Kernel 完成后调用 FRDeterministicA5 聚合到真实 yGm。
 *
 * A3 原型:
 *   - Init 第162-166行: mmQuantOutGm 指向 workspace 偏移
 *   - InitUbBuffer 第186-188行: queBind 分配 DETER_UB_SIZE
 *   - Process 第303-306行: SyncConfig 初始化
 *   - Process 第331-334行: 最终 FRDeterministic 调用
 *
 * 修改内容:
 *   1. 读取 deterministicFlag
 *   2. 确定性模式下：Params 中 y → workspace 偏移，Kernel 完成后调用 FRDeterministicA5
 *   3. 非确定性模式下：完全不变
 *
 * 注意：以下为 diff 模式，仅展示新增/修改部分，其余代码保持原样
 */

// === 在文件开头添加 include ===
// <<< ADD BEGIN
#include "gmm_fr_deterministic_a5.h"
using namespace GMMFRDeterministic;
// <<< ADD END


template <typename layoutA, typename layoutB, int scaleType, int rowIndType>
__aicore__ inline void grouped_matmul_finalize_routing_pertoken_dequant(
    GM_ADDR x, GM_ADDR w, GM_ADDR w_scale, GM_ADDR bias, GM_ADDR x_scale, GM_ADDR group_list, GM_ADDR share_input,
    GM_ADDR logit, GM_ADDR row_index, GM_ADDR offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    auto gmmFinalizeRoutingQuantParams_ = tilingData.gmmFinalizeRoutingDataParams;
    auto matmulTiling_ = tilingData.matmulTiling;

    using L1TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;
    using L0TileShape = AscendC::Shape<Cgmct::Gemm::_0, Cgmct::Gemm::_0, Cgmct::Gemm::_0>;

    using AType = DTYPE_X;
    using BType = DTYPE_W;
    using CType = DTYPE_Y;
    using LayoutA = layoutA;
    using LayoutB = layoutB;
    using LayoutC = layout::RowMajorAlign;
    using weightscaleType = std::conditional_t<scaleType == 2, bfloat16_t, float>;
    using BiasType = bfloat16_t;
    using LayoutBias = layout::RowMajor;
    using C1Type = std::conditional_t<std::is_same_v<AType, int8_t>, int32_t, float>;
    using xscaleType = float;
    using rowIndexType = std::conditional_t<rowIndType == 1, int32_t, int64_t>;
    using ProblemShape = Cgmct::Gemm::MatmulShape;
    using BlockScheduler = Cgmct::Gemm::GroupedMatmulAswtWithTailSplitScheduler;
    using BlockMmadBuilder =
        Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, C1Type, LayoutC, BiasType, LayoutBias, L1TileShape,
                                L0TileShape, BlockScheduler, MatmulMultiBlock<>,
                                Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;

    using BlockPrologue = Cgmct::Gemm::Block::BlockPrologueFinalizeRouting<CType, BiasType>;

    using BlockEpilogueDequant =
        Cgmct::Gemm::Block::BlockEpilogueDequantFinalizeRouting<CType, C1Type, weightscaleType, xscaleType, BiasType,
                                                                rowIndexType>;

    using GmmKernel =
        Cgmct::Gemm::Kernel::KernelGmmFinalizeRoutingPertokenDequant<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                     BlockEpilogueDequant, BlockScheduler>;
    using Params = typename GmmKernel::Params;
    using GMMTiling = typename GmmKernel::GMMTiling;

    GMMTiling gmmParams{gmmFinalizeRoutingQuantParams_.groupNum,
                        gmmFinalizeRoutingQuantParams_.groupListType,
                        matmulTiling_.baseM,
                        matmulTiling_.baseN,
                        matmulTiling_.baseK,
                        gmmFinalizeRoutingQuantParams_.hasBias};

    gmmParams.matmulTiling = &matmulTiling_;

    // <<< ADD BEGIN: 确定性分支
    if (gmmFinalizeRoutingQuantParams_.deterministicFlag == 1) {
        // ============ 确定性模式 ============
        // 方案A: 将 Params 中的 y 地址重定向到 workspace 中的确定性缓冲区
        // Cgmct Kernel 正常执行，Epilogue 的 SetAtomicAdd 写到 workspace 而非 yGm
        // Kernel 完成后调用 FRDeterministicA5 聚合

        // 计算确定性缓冲区偏移（在已有 workspace 之后）
        // 参考 A3 Init 第162-166行: workspace + parallNum * baseM * baseN * sizeof(int32_t) * coreNum
        uint64_t deterBufferOffset = static_cast<uint64_t>(matmulTiling_.usedCoreNum) *
            matmulTiling_.baseM * matmulTiling_.baseN * sizeof(int32_t);
        GM_ADDR deterBuffer = workspaceGM + deterBufferOffset;

        Params params = {
            {1, 1, 1, 1},
            {x, w, deterBuffer, bias, group_list},             // y → workspace 确定性缓冲区
            {share_input, deterBuffer,                         // prologue y → workspace
             gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N, gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},
            {deterBuffer, w_scale, x_scale, bias, logit, row_index,  // epilogue y → workspace
             matmulTiling_.baseM, matmulTiling_.baseN},
            gmmParams};

        // 执行 Cgmct Kernel（输出到 workspace）
        GmmKernel gmm;
        gmm(params);

        // 执行确定性聚合：workspace → yGm
        // 参考 A3 Process 第303-306行（SyncConfig 初始化）和第331-334行（最终调用）
        GlobalTensor<float> deterBufferGm;
        deterBufferGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deterBuffer));
        GlobalTensor<float> yGm;
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        GlobalTensor<rowIndexType> tokenRanksGm;
        tokenRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ rowIndexType*>(row_index));
        GlobalTensor<int64_t> groupTokensGm;
        groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));

        SyncConfig syncConfig;
        syncConfig.windowSize = gmmFinalizeRoutingQuantParams_.deterWorkspaceSize /
                                (matmulTiling_.N * sizeof(float));
        syncConfig.lowBoundM = syncConfig.windowSize;
        uint64_t nTimes = Ceil(matmulTiling_.N, DETER_UB_SIZE / sizeof(float));
        syncConfig.baseN = Ceil(Ceil(matmulTiling_.N, nTimes), 128) * 128;

        // 初始化 queBind（确定性 UB 中转 buffer）
        // 参考 A3 InitUbBuffer 第186-188行
        TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
        TPipe deterPipe;
        deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);

        // 最终聚合
        syncConfig.curM = matmulTiling_.M;  // 总行数
        FRDeterministicA5<float, rowIndexType>(
            syncConfig, deterBufferGm, yGm, tokenRanksGm, queBind,
            matmulTiling_.usedCoreNum, matmulTiling_.N);

    } else {
    // <<< ADD END

        // ============ 非确定性模式（完全不变）============
        Params params = {
            {1, 1, 1, 1},
            {x, w, y, bias, group_list},
            {share_input, y, gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N, gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},
            {y, w_scale, x_scale, bias, logit, row_index, matmulTiling_.baseM, matmulTiling_.baseN},
            gmmParams};
        GmmKernel gmm;
        gmm(params);

    // <<< ADD BEGIN
    }
    // <<< ADD END
}
