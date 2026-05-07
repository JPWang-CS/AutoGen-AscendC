/**
 * A3 原型: Tiling 数据结构中的确定性字段
 *
 * 源文件: op_host/grouped_matmul_finalize_routing_tiling.h 第40-64行
 *
 * 原理:
 *   deterministicFlag (uint32_t): 0=非确定性, 1=确定性
 *   deterWorkspaceSize (uint32_t): 确定性中间缓冲区大小 (96M 或 64M)
 *
 * A5 迁移要点:
 *   A5 使用 GMMFinalizeRoutingDataParams (在 tiling_data.h 中) 而非 BEGIN_TILING_DATA_DEF 宏
 *   需要将 reserved2 字段替换为确定性字段
 */

// ============ 原型: arch32 Tiling 数据结构 ============
// 源: op_host/grouped_matmul_finalize_routing_tiling.h 第40-64行
namespace optiling {

BEGIN_TILING_DATA_DEF(GroupMatmulFRTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, totalInGroup);
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, vBaseM);
  TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
  TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
  TILING_DATA_FIELD_DEF(uint32_t, parallNum);
  TILING_DATA_FIELD_DEF(uint32_t, sharedInputOffset);
  TILING_DATA_FIELD_DEF(uint32_t, sharedInputLen);
  TILING_DATA_FIELD_DEF(float, residualScale);
  TILING_DATA_FIELD_DEF(uint32_t, quantGroupNum);
  TILING_DATA_FIELD_DEF(uint32_t, withOffset);
  TILING_DATA_FIELD_DEF(uint32_t, hasPertokenScale);
  TILING_DATA_FIELD_DEF(uint32_t, hasBias);
  TILING_DATA_FIELD_DEF(uint32_t, deterministicFlag);       // ← 确定性标志
  TILING_DATA_FIELD_DEF(uint32_t, deterWorkspaceSize);      // ← 确定性 workspace 大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulFinalizeRouting, GroupMatmulFRTilingData)
REGISTER_TILING_DATA_CLASS(GroupMatmulFRTilingDataOp, GroupMatmulFRTilingData)
}
