/**
 * 修改文件 3: op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h
 *
 * 修改原因: 全量化 Tiling 类需要声明确定性相关的成员变量。
 *
 * A3 原型: op_host/grouped_matmul_finalize_routing_base_tiling.h
 *   第58行: void DeterministicTilingProcess();
 *   第87行: uint32_t deterministicFlag_;
 *         uint32_t deterWorkspaceSize_;
 *
 * 修改内容: 在 private 区域添加确定性成员
 */

// === 在类的 private 区域（约第117行 tilingData_ 之后）添加 ===

// <<< ADD BEGIN: 确定性相关成员
// A3 原型: base_tiling.h 第87行
uint32_t deterministicFlag_ = 0;
uint32_t deterWorkspaceSize_ = 0;
// <<< ADD END
