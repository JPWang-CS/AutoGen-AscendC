/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include "op_name_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"

namespace optiling {

static ge::graphStatus OpNameTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    // 2. 获取输入 Shape
    auto inputShape = context->GetInputShape(0)->GetStorageShape();
    uint32_t totalLength = inputShape.GetStorageShape().GetShapeSize(0);

    // 3. 计算 Tiling 参数
    uint32_t tileLength = 256;  // 每次处理的元素数，根据 UB 大小调整
    uint32_t alignTileLen = (tileLength + 31) / 32 * 32;  // 32 bytes 对齐
    uint32_t totalTiles = (totalLength + alignTileLen - 1) / alignTileLen;
    uint32_t tilesPerCore = (totalTiles + coreNum - 1) / coreNum;
    uint32_t usedCoreNum = std::min(coreNum, totalTiles);

    // 4. 设置 Tiling 数据
    auto tilingData = context->GetTilingData<OpNameTilingData>();
    tilingData->set_totalLength(totalLength);
    tilingData->set_tileLength(alignTileLen);
    tilingData->set_tileNum(tilesPerCore);
    tilingData->set_lastTileLength(totalLength - alignTileLen * (totalTiles - 1));
    tilingData->set_coreNum(usedCoreNum);

    // 5. 设置 BlockDim 和 TilingKey
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1);

    // 6. 计算 Workspace
    size_t workspaceSize = 0;
    context->GetTilingData()->SetWorkspaceSize(workspaceSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(OpNameTemplate)
    .Tiling(OpNameTilingFunc)
    .TilingParse<OpNameCompileInfo>([](gert::TilingParseContext* context) {
        // 解析 Compile Info
        auto compileInfo = context->GetCompiledInfo<OpNameCompileInfo>();
        auto platformInfo = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
        compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    });

}
