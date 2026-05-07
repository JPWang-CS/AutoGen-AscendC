/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 纯 AIC 算子 Tiling 数据结构
 */

#ifndef __OP_HOST_OP_NAME_CUBE_TILING_H__
#define __OP_HOST_OP_NAME_CUBE_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"

namespace optiling {

struct OpNameCubeCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t l1Size{0UL};
    uint64_t l0CSize{0UL};
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
};

BEGIN_TILING_DATA_DEF(OpNameCubeTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, k);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OpNameCubeTemplate, OpNameCubeTilingData)
}
#endif
