/**
 * 修改文件 1 (补充): op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h
 *
 * 修改原因: 伪量化路径有独立的 Tiling 数据结构，同样需要携带确定性参数。
 *
 * A3 原型: 与修改文件1相同 (A3 A8W4 路径使用同一个 GroupMatmulFRTilingData)
 * A5 差异: 伪量化路径使用独立的 GMMFinalizeRoutingWeightQuantTilingData
 *
 * 修改内容: 在对应结构体中添加 deterministicFlag 和 deterWorkspaceSize
 *
 * 注意: 此文件的具体结构取决于 gmm_fr_weight_quant_tiling_data.h 的实际内容。
 *       以下是通用修改模板，需根据实际结构体字段调整。
 */

// 在 GMMFinalizeRoutingWeightQuantTilingData 或类似结构体中添加:

// <<< ADD BEGIN: 确定性字段（与修改文件1相同）
uint32_t deterministicFlag = 0;      // 0=非确定性, 1=确定性
uint32_t deterWorkspaceSize = 0;     // 确定性 workspace 大小
// <<< ADD END
