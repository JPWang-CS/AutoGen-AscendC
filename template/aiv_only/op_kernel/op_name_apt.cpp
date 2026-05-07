/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 纯 AIV 算子 - arch35 Kernel 入口 (A5/950)
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "kernel_operator.h"
// A5 平台可能使用不同的头文件
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#endif

extern "C" __global__ __aicore__ void op_name(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (g_coreType == AIC) { return; }

    // A5 平台获取 Tiling 数据
    REGISTER_TILING_DEFAULT(OpNameTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

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
