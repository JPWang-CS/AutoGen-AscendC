/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include "register/op_impl_registry.h"

namespace optiling {

static ge::graphStatus OpNameInferShape(gert::InferShapeContext* context)
{
    // 输出 shape 与输入相同
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    gert::StorageShape* outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(OpNameTemplate).InferShape(OpNameInferShape);

}
