/**
 * 修改文件 5: op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.h
 *
 * 修改原因: 伪量化 Tiling 类需要声明确定性相关成员。
 *
 * A3 原型: 与修改文件3相同 (A3 使用同一个 BaseTiling 类)
 * A5 差异: A5 伪量化使用独立的 WeightQuantTiling 类
 *
 * 修改内容: 在 private 区域添加确定性成员（与修改文件3完全相同）
 */

// === 在类的 private 区域添加 ===

// <<< ADD BEGIN
uint32_t deterministicFlag_ = 0;
uint32_t deterWorkspaceSize_ = 0;
// <<< ADD END
