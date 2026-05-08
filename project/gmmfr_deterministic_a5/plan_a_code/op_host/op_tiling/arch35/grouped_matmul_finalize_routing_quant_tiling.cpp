/**
 * 修改文件 3: op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp
 *
 * 修改原因: Tiling 计算是 CPU 侧第一个读取 GetDeterministic() 的地方。
 *           仅 W8A8 (INT8×INT8) PerToken 模式支持确定性。
 *
 * A3 原型: op_host/grouped_matmul_finalize_routing_base_tiling.cpp
 *   第27行: DETER_WORK_SPACE_SIZE 常量
 *   第413-425行: DeterministicTilingProcess() 函数
 *   第454-455行: FillTilingData 中设置确定性字段
 *   第487-488行: PrintTilingData 中打印确定性字段
 */

// ================================================================
// 修改位置 A: DoOpTiling() 函数，hasBias 赋值之后
// ================================================================

// <<< ADD BEGIN: 确定性 Tiling 处理
    // 仅 W8A8 (INT8×INT8) PerToken 模式支持确定性
    if (context_->GetDeterministic() == 1 && !IsMicroScaling() &&
        inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8) {
        deterministicFlag_ = 1;
        tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag = 1;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint64_t l2Size = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
        constexpr uint32_t DETER_WORK_SPACE_SIZE = 96UL * 1024 * 1024;
        constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64UL * 1024 * 1024;
        deterWorkspaceSize_ = l2Size > DETER_WORK_SPACE_SIZE
                              ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
        tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize = deterWorkspaceSize_;
    }
// <<< ADD END


// ================================================================
// 修改位置 B: PrintQuantParams() 函数，hasBias 打印之后
// ================================================================

// <<< ADD BEGIN: 打印确定性参数
        << ", deterministicFlag = " << tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag
        << ", deterWorkspaceSize = " << tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize;
// <<< ADD END
