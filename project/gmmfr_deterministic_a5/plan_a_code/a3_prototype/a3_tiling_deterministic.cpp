/**
 * A3 原型: Tiling 侧确定性处理
 *
 * 源文件: op_host/grouped_matmul_finalize_routing_base_tiling.cpp 第27行, 第413-425行, 第454行
 *
 * 原理:
 *   1. 读取 context_->GetDeterministic() 判断是否开启确定性
 *   2. 根据 L2 大小分配 workspace (96M 或 64M)
 *   3. 将 deterministicFlag 和 deterWorkspaceSize 写入 Tiling 数据
 *
 * A5 迁移要点:
 *   - A5 Tiling 使用不同的框架 (GroupedMatmulFinalizeRoutingQuantTiling 而非 BaseTiling)
 *   - workspace 大小计算逻辑相同
 *   - 需要填充到 A5 的 GMMFinalizeRoutingDataParams 而非 GroupMatmulFRTilingData
 */

// ============ 原型1: workspace 大小常量 ============
// 源: op_host/grouped_matmul_finalize_routing_base_tiling.cpp 第27行
constexpr uint32_t DETER_WORK_SPACE_SIZE = 96 * 1024 * 1024;       // 确定性 workspace 上限 96MB
constexpr uint32_t DETER_WORK_SPACE_LOWER_SIZE = 64 * 1024 * 1024; // 确定性 workspace 下限 64MB


// ============ 原型2: DeterministicTilingProcess 函数 ============
// 源: op_host/grouped_matmul_finalize_routing_base_tiling.cpp 第413-425行
void GroupedMatmulFinalizeRoutingBaseTiling::DeterministicTilingProcess()
{
    if (context_->GetDeterministic() == 0) {
        deterministicFlag_ = 0;   // 非确定性
        return;
    }
    deterministicFlag_ = 1;        // 确定性模式开启
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t l2_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2_size);
    // L2 > 96MB 用 96MB，否则用 64MB
    deterWorkspaceSize_ = l2_size > DETER_WORK_SPACE_SIZE ? DETER_WORK_SPACE_SIZE : DETER_WORK_SPACE_LOWER_SIZE;
    workspaceSize_ += deterWorkspaceSize_;  // 追加到总 workspace
}


// ============ 原型3: 填充 Tiling 数据 ============
// 源: op_host/grouped_matmul_finalize_routing_base_tiling.cpp 第454-455行
void GroupedMatmulFinalizeRoutingBaseTiling::FillTilingData()
{
    // ... 其他字段 ...
    tilingData_.set_deterministicFlag(deterministicFlag_);
    tilingData_.set_deterWorkspaceSize(deterWorkspaceSize_);
}


// ============ 原型4: 打印 Tiling 数据 ============
// 源: op_host/grouped_matmul_finalize_routing_base_tiling.cpp 第487-488行
void GroupedMatmulFinalizeRoutingBaseTiling::PrintTilingData()
{
    // ... 其他字段 ...
    OP_LOGD(context_->GetNodeName(), "deterministicFlag: [%u]", tilingData_.get_deterministicFlag());
    OP_LOGD(context_->GetNodeName(), "deterWorkspaceSize: [%u]", tilingData_.get_deterWorkspaceSize());
}
