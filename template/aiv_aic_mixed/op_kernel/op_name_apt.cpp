/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * AIV+AIC 混合算子 - arch35 Kernel 入口 (A5/950)
 *
 * A5 使用 Cgmct 框架，Builder 模式组织 Cube+Vector 协同计算
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

// === A5 混合算子模板 ===
// A5 平台使用 Cgmct (C++ Gemm Template) 框架
// 参考 GMMFR 的 arch35 实现模式

template <int ATRANS, int BTRANS, int SCALETYPE>
__global__ __aicore__ void op_name(
    GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // A5 平台：使用 Cgmct Builder 模式
    // 1. 定义 MatMul 类型
    // using AType = DTYPE_X;
    // using BType = DTYPE_W;
    // using CType = DTYPE_Y;
    // using LayoutA = Cgmct::Gemm::layout::RowMajor;
    // using LayoutB = Cgmct::Gemm::layout::Nz;

    // 2. 定义 Builder
    // using BlockMmadBuilder = Cgmct::Gemm::Block::BlockMxMmAicToAivBuilder<
    //     AType, LayoutA, BType, LayoutB, BiasType, CType, LayoutC,
    //     L1TileShape, L0TileShape, BlockScheduler, ...>;

    // 3. 定义 Prologue/Epilogue
    // using BlockPrologue = Cgmct::Gemm::Block::BlockPrologueMyOp<...>;
    // using BlockEpilogue = Cgmct::Gemm::Block::BlockEpilogueMyOp<...>;

    // 4. 定义 Kernel 并执行
    // using GmmKernel = Cgmct::Gemm::Kernel::KernelMyOp<
    //     ProblemShape, BlockMmadBuilder, BlockPrologue, BlockEpilogue, BlockScheduler>;
    // GmmKernel gmm;
    // gmm(params);

    // TODO: 替换为实际实现
}

#endif
