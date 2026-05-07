/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * AIV+AIC 混合算子 - arch32 Kernel 入口 (A2/A3)
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_name.h"

using namespace AscendC;
using namespace matmul;

// 宏：简化实例化逻辑
#define OP_MIXED_IMPL(RowIndexDtype, ScaleType)                              \
    do {                                                                       \
        TPipe pipe;                                                            \
        MT mm;                                                                 \
        if ASCEND_IS_AIC {                                                     \
            mm.SetSubBlockIdx(0);                                              \
            mm.Init(&tilingData.matmulTiling, &pipe);                          \
        }                                                                      \
        OpNameMixedKernel<decltype(tilingData)> op(mm);                        \
        op.Init(x, w, y, workspaceGM, scale, &tilingData, &pipe);             \
        op.Process();                                                          \
    } while (0)

extern "C" __global__ __aicore__ void op_name(
    GM_ADDR x, GM_ADDR w, GM_ADDR scale, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 混合模式: 1 Cube + 2 Vector
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA(tilingData, tilingGM);

    // 按 TilingKey 分发不同实现
    if (TILING_KEY_IS(1)) {
        OP_MIXED_IMPL(int64_t, float);
    } else if (TILING_KEY_IS(2)) {
        OP_MIXED_IMPL(int32_t, bfloat16_t);
    }
}
#endif
