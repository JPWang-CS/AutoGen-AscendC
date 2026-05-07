/**
 * 修改文件 6: op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.cpp
 *
 * 修改原因: 伪量化路径的 Tiling 计算同样需要处理 GetDeterministic()。
 *
 * A3 原型: 与修改文件4相同 (A3 使用同一个 DeterministicTilingProcess)
 *
 * 修改内容: 与修改文件4完全相同的确定性处理逻辑
 */

// === 在 DoOpTiling() 末尾添加（与修改文件4完全相同）===

// <<< ADD BEGIN
if (context_->GetDeterministic() == 1) {
    tilingData_.gmmFinalizeRoutingWeightQuantDataParams.deterministicFlag = 1;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l2Size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    uint32_t deterWs = l2Size > (96UL * 1024 * 1024) ? (96UL * 1024 * 1024) : (64UL * 1024 * 1024);
    tilingData_.gmmFinalizeRoutingWeightQuantDataParams.deterWorkspaceSize = deterWs;
}
// <<< ADD END


// === 在 Print 函数末尾添加 ===

// <<< ADD BEGIN
oss << ", deterministicFlag = " << tilingData_.xxxParams.deterministicFlag
    << ", deterWorkspaceSize = " << tilingData_.xxxParams.deterWorkspaceSize;
// <<< ADD END
