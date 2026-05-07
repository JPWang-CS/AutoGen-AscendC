/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * AIV+AIC 混合算子 Tiling 数据结构
 */

#ifndef __OP_HOST_OP_NAME_MIXED_TILING_H__
#define __OP_HOST_OP_NAME_MIXED_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"

namespace optiling {

struct OpNameMixedCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l0CSize{0UL};
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
};

BEGIN_TILING_DATA_DEF(OpNameMixedTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, vBaseM);          // Vector 每次处理的行数
  TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);        // Vector 每次计算的元素数
  TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);      // UB 剩余字节数（给 TBuf）
  TILING_DATA_FIELD_DEF(uint32_t, parallNum);        // Cube-Vector 并行度
  // === 确定性相关字段（可选）===
  TILING_DATA_FIELD_DEF(uint32_t, deterministicFlag);  // 0=非确定性, 1=确定性
  TILING_DATA_FIELD_DEF(uint32_t, deterWorkspaceSize); // 确定性 workspace 大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OpNameMixedTemplate, OpNameMixedTilingData)
}
#endif
