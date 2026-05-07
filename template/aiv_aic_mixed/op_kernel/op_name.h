/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * AIV+AIC 混合算子 Kernel 实现 - arch32
 *
 * 典型场景：Cube 做 MatMul，Vector 做后处理（反量化、归约、激活函数等）
 * 核间通过 CrossCoreSetFlag/CrossCoreWaitFlag 同步
 */

#ifndef __OP_KERNEL_OP_NAME_MIXED_H__
#define __OP_KERNEL_OP_NAME_MIXED_H__

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

// 软件同步标记
static constexpr uint8_t SYNC_AIC_TO_AIV = 0x01;
static constexpr uint8_t SYNC_AIV_TO_AIC = 0x02;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t CV_PARALL_NUM = 1;  // Cube-Vector 并行度

// MatMul 类型 - 根据实际场景修改
using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using bT = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using MT = MatmulImpl<aT, bT, cT, biasT, CFG_MDL>;

template <typename TILING_TYPE>
class OpNameMixedKernel {
public:
    __aicore__ inline OpNameMixedKernel(MT& mm) : mm_(mm) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace,
                                GM_ADDR scale, const TILING_TYPE* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CubeCompute(uint32_t groupIdx);
    __aicore__ inline void VectorCompute(uint32_t groupIdx);
    __aicore__ inline void InitUbBuffer();

    MT& mm_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> vecOutQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<int8_t> wGm_;
    GlobalTensor<int32_t> mmOutGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> yGm_;
    const TILING_TYPE* tiling_;
    uint32_t coreIdx_;
    uint32_t cubeCount_;
};

template <typename TILING_TYPE>
__aicore__ inline void OpNameMixedKernel<TILING_TYPE>::Init(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR scale, const TILING_TYPE* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    tiling_ = tilingData;
    coreIdx_ = GetBlockIdx();
    cubeCount_ = 0;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x));
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(w));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    mmOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));

    if ASCEND_IS_AIC {
        mm_.Init(&tiling_->matmulTiling, pipe_);
    }
    InitUbBuffer();
}

template <typename TILING_TYPE>
__aicore__ inline void OpNameMixedKernel<TILING_TYPE>::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    // Vector 核初始化 UB 缓冲区
    pipe_->InitBuffer(scaleInQueue_, BUFFER_NUM, tiling_->matmulTiling.baseN * sizeof(float));
    pipe_->InitBuffer(vecOutQueue_, BUFFER_NUM, tiling_->ubCalSize * sizeof(float));
    pipe_->InitBuffer(tmpBuf_, tiling_->ubRestBytes);
}

template <typename TILING_TYPE>
__aicore__ inline void OpNameMixedKernel<TILING_TYPE>::Process()
{
    for (uint32_t groupIdx = 0; groupIdx < tiling_->groupNum; ++groupIdx) {
        CubeCompute(groupIdx);
        VectorCompute(groupIdx);
    }
}

template <typename TILING_TYPE>
__aicore__ inline void OpNameMixedKernel<TILING_TYPE>::CubeCompute(uint32_t groupIdx)
{
    if ASCEND_IS_AIC {
        // Cube 核：执行矩阵乘法
        uint32_t baseM = tiling_->matmulTiling.baseM;
        uint32_t baseN = tiling_->matmulTiling.baseN;
        uint32_t baseK = tiling_->matmulTiling.baseK;

        uint64_t aOffset = static_cast<uint64_t>(groupIdx) * baseM * baseK;
        uint64_t bOffset = static_cast<uint64_t>(groupIdx) * baseN * baseK;
        uint64_t cOffset = static_cast<uint64_t>(cubeCount_) * baseM * baseN;

        mm_.SetOrgShape(baseM, baseN, baseK);
        mm_.SetSingleShape(baseM, baseN, baseK);
        mm_.SetTensorA(xGm_[aOffset]);
        mm_.SetTensorB(wGm_[bOffset]);

        while (mm_.Iterate()) {
            mm_.GetTensorC(mmOutGm_[cOffset], 0, true);
            // 通知 Vector 核数据已就绪
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
            cOffset += baseM * baseN;
        }
    }
    cubeCount_++;
}

template <typename TILING_TYPE>
__aicore__ inline void OpNameMixedKernel<TILING_TYPE>::VectorCompute(uint32_t groupIdx)
{
    if ASCEND_IS_AIV {
        // Vector 核：等待 Cube 核完成
        if (cubeCount_ > 0) {
            CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        }

        // 从 workspace 读取 Cube 输出，执行后处理
        uint32_t baseM = tiling_->matmulTiling.baseM;
        uint32_t baseN = tiling_->matmulTiling.baseN;

        LocalTensor<float> scaleLocal = scaleInQueue_.AllocTensor<float>();
        DataCopy(scaleLocal, scaleGm_[groupIdx * baseN], baseN);
        scaleInQueue_.EnQue(scaleLocal);

        LocalTensor<float> yLocal = vecOutQueue_.AllocTensor<float>();
        // === 后处理计算（反量化、激活等）===
        // 示例：从 int32 结果反量化为 float
        // Cast(yLocal, mmOutLocal, RoundMode::CAST_NONE, baseM * baseN);
        // Mul(yLocal, yLocal, scaleLocal, baseM * baseN);
        vecOutQueue_.EnQue<float>(yLocal);
        scaleInQueue_.FreeTensor(scaleLocal);

        yLocal = vecOutQueue_.DeQue<float>();
        DataCopy(yGm_[groupIdx * baseM * baseN], yLocal, baseM * baseN);
        vecOutQueue_.FreeTensor(yLocal);

        // 通知 Cube 核 Vector 已完成
        CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIV_TO_AIC);
    }
}

#endif
