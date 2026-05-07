/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#ifndef __OP_HOST_OP_NAME_TILING_H__
#define __OP_HOST_OP_NAME_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"

namespace optiling {

struct OpNameCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
};

// 纯 Vector 算子 Tiling 数据结构
BEGIN_TILING_DATA_DEF(OpNameTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);    // 总数据长度
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);      // 每次处理的数据长度
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);          // 主块数量
  TILING_DATA_FIELD_DEF(uint32_t, lastTileLength);   // 尾块长度
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);          // 使用的核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OpNameTemplate, OpNameTilingData)
}
#endif
