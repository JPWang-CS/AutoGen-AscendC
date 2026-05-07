/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 纯 AIC 算子 Tiling 实现
 */

#include "op_name_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

namespace optiling {

static ge::graphStatus OpNameCubeTilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // Cube 算子使用 AIC 核数
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();

    // 获取输入 Shape
    auto xShape = context->GetInputShape(0)->GetStorageShape();
    auto wShape = context->GetInputShape(1)->GetStorageShape();
    uint32_t m = xShape.GetStorageShape().GetShapeSize(0);
    uint32_t k = xShape.GetStorageShape().GetShapeSize(1);
    uint32_t n = wShape.GetStorageShape().GetShapeSize(1);

    // 计算 MatMul Tiling
    auto tilingData = context->GetTilingData<OpNameCubeTilingData>();
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(aicNum);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                           matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(m, n, k);
    cubeTiling.SetBufferSpace(512 * 1024, 128 * 1024);  // L1, L0C

    if (cubeTiling.GetTilingData().IsAbilityAvailable() != SUCCESS) {
        return ge::GRAPH_PARAM_INVALID;
    }
    tilingData->set_matmulTiling(cubeTiling.GetTilingData());

    tilingData->set_coreNum(aicNum);
    tilingData->set_batch(1);
    tilingData->set_m(m);
    tilingData->set_n(n);
    tilingData->set_k(k);

    context->SetBlockDim(aicNum);
    context->SetTilingKey(1);

    size_t workspaceSize = 0;
    context->GetTilingData()->SetWorkspaceSize(workspaceSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(OpNameCubeTemplate)
    .Tiling(OpNameCubeTilingFunc)
    .TilingParse<OpNameCubeCompileInfo>([](gert::TilingParseContext* context) {
        auto compileInfo = context->GetCompiledInfo<OpNameCubeCompileInfo>();
        auto platformInfo = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->aicNum = ascendcPlatform.GetCoreNumAic();
        compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo->l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0C, compileInfo->l0CSize);
        compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    });

}
