/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 纯 AIC 算子 - arch32 Kernel 入口 (A2/A3)
 * 注意：纯 Cube 算子极少单独使用，通常 Cube 和 Vector 配合
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_name.h"

using namespace AscendC;
using namespace matmul;

extern "C" __global__ __aicore__ void op_name(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                               GM_ADDR bias, GM_ADDR tilingGM)
{
    // 混合模式但仅 Cube 部分执行计算
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA(tilingData, tilingGM);

    if ASCEND_IS_AIC {
        TPipe pipe;
        // 注意：纯 Cube 算子通常不需要 Vector 核参与
        // 如果使用了 MIX_AIC_1_2，Vector 核需要空转或做辅助操作
        OpNameCubeKernel op;
        op.Init(x, w, y, bias, &tilingData.matmulTiling, &pipe);
        op.Process();
    }
    // Vector 核空转 - 不做任何操作
}
#endif
