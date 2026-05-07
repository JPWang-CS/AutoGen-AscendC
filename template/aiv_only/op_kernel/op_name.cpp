/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 纯 AIV 算子 - arch32 Kernel 入口 (A2/A3)
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "kernel_operator.h"
#include "op_name.h"

extern "C" __global__ __aicore__ void op_name(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM)
{
    // 纯 Vector 算子：声明为 AIV_ONLY
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // Cube 核直接返回，不参与计算
    if (g_coreType == AIC) { return; }

    // 获取 Tiling 数据
    GET_TILING_DATA(tilingData, tilingGM);

    // 创建算子实例并执行
    TPipe pipe;
    OpNameKernel op;
    op.Init(x, y,
            tilingData.totalLength,
            tilingData.tileLength,
            tilingData.tileNum,
            tilingData.lastTileLength,
            &pipe);
    op.Process();
}
#endif
