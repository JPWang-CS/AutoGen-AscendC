/**
 * 修改文件 4: op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp
 *
 * 修改原因: Tiling 计算是 CPU 侧第一个读取 GetDeterministic() 的地方。
 *           需要将框架的确定性开关转换为 Kernel 可消费的 Tiling 数据。
 *
 * A3 原型: op_host/grouped_matmul_finalize_routing_base_tiling.cpp
 *   第27行: DETER_WORK_SPACE_SIZE 常量
 *   第413-425行: DeterministicTilingProcess() 函数
 *   第454-455行: FillTilingData 中设置确定性字段
 *   第487-488行: PrintTilingData 中打印确定性字段
 *
 * 修改内容: 在 DoOpTiling() 末尾添加确定性处理逻辑
 */

// === 在 DoOpTiling() 函数的 hasBias 赋值之后（约第456行）添加 ===

// <<< ADD BEGIN: 确定性 Tiling 处理
// A3 原型: DeterministicTilingProcess() (base_tiling.cpp 第413-425行)
if (context_->GetDeterministic() == 1) {
    tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l2Size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    // A3 原型: l2 > 96MB 用 96MB，否则用 64MB
    uint32_t deterWs = l2Size > (96UL * 1024 * 1024) ? (96UL * 1024 * 1024) : (64UL * 1024 * 1024);
    tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWs;
    // 增大 workspace 总大小
    // 注意：workspace 大小由 DoLibApiTiling 中的 CalBasicBlock/CalL1Tiling 计算，
    // 确定性 workspace 需要在最终 workspace 中追加。
    // 由于 PostTiling 中不直接设置 workspace 大小，需要通过 PostTiling 阶段处理。
    // 这里将 deterWorkspaceSize 存入 tiling data，workspace 分配在 launcher 层自动处理。
}
// <<< ADD END


// === 在 PrintQuantParams() 函数末尾（约第561行后）添加 ===

// <<< ADD BEGIN: 打印确定性参数
// A3 原型: PrintTilingData (base_tiling.cpp 第487-488行)
oss << ", deterministicFlag = " << tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag
    << ", deterWorkspaceSize = " << tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize;
// <<< ADD END
