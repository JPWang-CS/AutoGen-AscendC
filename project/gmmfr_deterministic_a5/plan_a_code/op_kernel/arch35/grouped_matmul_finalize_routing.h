/**
 * 修改文件 10: op_kernel/arch35/grouped_matmul_finalize_routing.h
 *
 * 修改原因: MX 格式路径，与 pertoken_dequant 完全相同的修改逻辑。
 *           方案A：Params 中 y 重定向到 workspace + FRDeterministicA5 聚合。
 *
 * A3 原型: 同修改文件9（A3 MX 路径使用相同的 FRDeterministic）
 *
 * 修改内容与 pertoken_dequant.h 完全对称
 */

// === 在文件开头添加 include ===
// <<< ADD BEGIN
#include "gmm_fr_deterministic_a5.h"
using namespace GMMFRDeterministic;
// <<< ADD END


template <typename layoutA, typename layoutB>
__aicore__ inline void grouped_matmul_finalize_routing_mx(GM_ADDR x, GM_ADDR w, GM_ADDR w_scale, GM_ADDR bias,
                                                       GM_ADDR x_scale, GM_ADDR group_list, GM_ADDR share_input,
                                                       GM_ADDR logit, GM_ADDR row_index, GM_ADDR offset, GM_ADDR y,
                                                       GM_ADDR workspaceGM, GM_ADDR tilingGM)
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
    using weightscaleType = AscendC::fp8_e8m0_t;
    using BiasType = bfloat16_t;

    using ProblemShape = Cgmct::Gemm::MatmulShape;
    using BlockScheduler = Cgmct::Gemm::GroupedMatmulAswtWithTailSplitScheduler;
    using BlockMmadPolicy = Cgmct::Gemm::GMMPerTile<>;
    using BlockMmadBuilder =
        Block::BlockMxMmAicToAivBuilder<AType, LayoutA, BType, LayoutB, BiasType, CType, LayoutC, L1TileShape,
                                        L0TileShape, BlockScheduler, QuantMatmulWithTileMultiBlock<>,
                                        Tile::TileCopy<Arch::DAV3510, Tile::CopyInAndCopyOutSplitMWithParams>>;

    using BlockPrologue = Cgmct::Gemm::Block::BlockPrologueFinalizeRouting<CType, BiasType>;
    using BlockEpilogue = Cgmct::Gemm::Block::BlockEpilogueFinalizeRouting<CType>;

    using GmmKernel = Cgmct::Gemm::Kernel::KernelGmmFinalizeRouting<ProblemShape, BlockMmadBuilder, BlockPrologue,
                                                                  BlockEpilogue, BlockScheduler>;
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
        uint64_t deterBufferOffset = static_cast<uint64_t>(matmulTiling_.usedCoreNum) *
            matmulTiling_.baseM * matmulTiling_.baseN * sizeof(int32_t);
        GM_ADDR deterBuffer = workspaceGM + deterBufferOffset;

        Params params = {
            {1, 1, 1, 1},
            {x, w, w_scale, x_scale, deterBuffer, group_list, bias},  // y → workspace
            {share_input, deterBuffer,
             gmmFinalizeRoutingQuantParams_.sharedInputOffset,
             gmmFinalizeRoutingQuantParams_.sharedInputLen, matmulTiling_.N,
             gmmFinalizeRoutingQuantParams_.batch,
             gmmFinalizeRoutingQuantParams_.residualScale},
            {deterBuffer, w_scale, x_scale, bias, logit, row_index,   // y → workspace
             matmulTiling_.baseM, matmulTiling_.baseN},
            gmmParams};

        GmmKernel gmm;
        gmm(params);

        // 确定性聚合
        GlobalTensor<float> deterBufferGm;
        deterBufferGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(deterBuffer));
        GlobalTensor<float> yGm;
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        GlobalTensor<int64_t> tokenRanksGm;
        tokenRanksGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(row_index));
        GlobalTensor<int64_t> groupTokensGm;
        groupTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));

        SyncConfig syncConfig;
        syncConfig.windowSize = gmmFinalizeRoutingQuantParams_.deterWorkspaceSize /
                                (matmulTiling_.N * sizeof(float));
        syncConfig.lowBoundM = syncConfig.windowSize;
        uint64_t nTimes = Ceil(matmulTiling_.N, DETER_UB_SIZE / sizeof(float));
        syncConfig.baseN = Ceil(Ceil(matmulTiling_.N, nTimes), 128) * 128;

        TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
        TPipe deterPipe;
        deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);

        syncConfig.curM = matmulTiling_.M;
        FRDeterministicA5<float, int64_t>(
            syncConfig, deterBufferGm, yGm, tokenRanksGm, queBind,
            matmulTiling_.usedCoreNum, matmulTiling_.N);

    } else {
    // <<< ADD END

        // ============ 非确定性模式（完全不变）============
        Params params = {
            {1, 1, 1, 1},
            {x, w, w_scale, x_scale, y, group_list, bias},
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
