/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 纯 AIC (Cube Only) 算子 Kernel 实现 - arch32
 */

#ifndef __OP_KERNEL_OP_NAME_CUBE_H__
#define __OP_KERNEL_OP_NAME_CUBE_H__

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

// MatMul 类型定义 - 根据实际数据类型修改
using aT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using bT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using biasT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MT = MatmulImpl<aT, bT, cT, biasT, CFG_MDL>;

class OpNameCubeKernel {
public:
    __aicore__ inline OpNameCubeKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR bias,
                                const TCubeTiling* matmulTiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    MT mm_;
    GlobalTensor<half> xGm_;
    GlobalTensor<half> wGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> biasGm_;
    const TCubeTiling* tiling_;
    uint32_t coreIdx_;
    uint32_t singleM_;
    uint32_t singleN_;
    uint32_t singleK_;
    uint32_t totalM_;
    uint32_t totalN_;
    uint32_t totalK_;
};

__aicore__ inline void OpNameCubeKernel::Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR bias,
                                               const TCubeTiling* matmulTiling, TPipe* pipe)
{
    pipe_ = pipe;
    tiling_ = matmulTiling;
    coreIdx_ = GetBlockIdx();

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));

    totalM_ = tiling_->singleCoreM;
    totalN_ = tiling_->singleCoreN;
    totalK_ = tiling_->singleCoreK;
    singleM_ = tiling_->baseM;
    singleN_ = tiling_->baseN;
    singleK_ = tiling_->baseK;

    mm_.Init(tiling_, pipe_);
}

__aicore__ inline void OpNameCubeKernel::Process()
{
    // 设置整体矩阵形状
    mm_.SetOrgShape(totalM_, totalN_, totalK_);

    // 按 singleM x singleN 分块迭代计算
    uint32_t mIterations = (totalM_ + singleM_ - 1) / singleM_;
    uint32_t nIterations = (totalN_ + singleN_ - 1) / singleN_;

    for (uint32_t mIdx = 0; mIdx < mIterations; mIdx++) {
        for (uint32_t nIdx = 0; nIdx < nIterations; nIdx++) {
            uint32_t curM = std::min(singleM_, totalM_ - mIdx * singleM_);
            uint32_t curN = std::min(singleN_, totalN_ - nIdx * singleN_);

            mm_.SetSingleShape(curM, curN, totalK_);

            uint64_t aOffset = static_cast<uint64_t>(mIdx) * singleM_ * totalK_;
            uint64_t bOffset = static_cast<uint64_t>(nIdx) * singleN_ * totalK_;
            uint64_t cOffset = static_cast<uint64_t>(mIdx) * singleM_ * totalN_
                             + static_cast<uint64_t>(nIdx) * singleN_;

            mm_.SetTensorA(xGm_[aOffset]);
            mm_.SetTensorB(wGm_[bOffset]);

            while (mm_.Iterate()) {
                mm_.GetTensorC(yGm_[cOffset], 0, true);
            }
        }
    }
}

#endif
