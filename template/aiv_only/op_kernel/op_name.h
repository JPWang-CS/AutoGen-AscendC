/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#ifndef __OP_KERNEL_OP_NAME_H__
#define __OP_KERNEL_OP_NAME_H__

#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;

class OpNameKernel {
public:
    __aicore__ inline OpNameKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t tileLength, uint32_t tileNum,
                                uint32_t lastTileLength, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerTile(uint32_t length);
    __aicore__ inline void CopyIn(uint32_t length);
    __aicore__ inline void Compute(uint32_t length);
    __aicore__ inline void CopyOut(uint32_t length);

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    uint32_t totalLength_;
    uint32_t tileLength_;
    uint32_t tileNum_;
    uint32_t lastTileLength_;
    uint32_t coreOffset_;
};

__aicore__ inline void OpNameKernel::Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                           uint32_t tileLength, uint32_t tileNum,
                                           uint32_t lastTileLength, TPipe* pipe)
{
    pipe_ = pipe;
    totalLength_ = totalLength;
    tileLength_ = tileLength;
    tileNum_ = tileNum;
    lastTileLength_ = lastTileLength;
    coreOffset_ = GetBlockIdx() * tileNum_ * tileLength_;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    pipe_->InitBuffer(xInQueue_, BUFFER_NUM, tileLength_ * sizeof(float));
    pipe_->InitBuffer(yOutQueue_, BUFFER_NUM, tileLength_ * sizeof(float));
}

__aicore__ inline void OpNameKernel::Process()
{
    uint32_t loopCount = tileNum_ - 1;
    for (uint32_t i = 0; i < loopCount; i++) {
        CopyIn(tileLength_);
        Compute(tileLength_);
        CopyOut(tileLength_);
    }
    // 尾块处理
    if (lastTileLength_ > 0) {
        CopyIn(lastTileLength_);
        Compute(lastTileLength_);
        CopyOut(lastTileLength_);
    }
}

__aicore__ inline void OpNameKernel::CopyIn(uint32_t length)
{
    LocalTensor<float> xLocal = xInQueue_.AllocTensor<float>();
    DataCopy(xLocal, xGm_[coreOffset_ + length], length);
    xInQueue_.EnQue(xLocal);
}

__aicore__ inline void OpNameKernel::Compute(uint32_t length)
{
    LocalTensor<float> xLocal = xInQueue_.DeQue<float>();
    LocalTensor<float> yLocal = yOutQueue_.AllocTensor<float>();
    // === 在此处替换为实际计算逻辑 ===
    // 示例: 逐元素操作
    // Add(yLocal, xLocal, xLocal, length);
    // Mul(yLocal, xLocal, scalarValue, length);
    // Cast(yLocal, xLocal, RoundMode::CAST_NONE, length);
    yOutQueue_.EnQue<float>(yLocal);
    xInQueue_.FreeTensor(xLocal);
}

__aicore__ inline void OpNameKernel::CopyOut(uint32_t length)
{
    LocalTensor<float> yLocal = yOutQueue_.DeQue<float>();
    DataCopy(yGm_[coreOffset_ + length], yLocal, length);
    yOutQueue_.FreeTensor(yLocal);
}

#endif
