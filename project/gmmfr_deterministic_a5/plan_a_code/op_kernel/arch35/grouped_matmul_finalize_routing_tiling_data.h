/**
 * 修改文件 1: op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h
 *
 * 修改原因: Kernel 执行时需要从 Tiling 读取 deterministicFlag 和 deterWorkspaceSize。
 *           当前 GMMFinalizeRoutingDataParams 的 reserved2 字段未被使用，替换为确定性字段。
 *
 * A3 原型: op_host/grouped_matmul_finalize_routing_tiling.h 第59-60行
 */

// ========== 修改: 将 reserved2 替换为确定性字段 ==========

#pragma pack(push, 8)
struct GMMFinalizeRoutingDataParams {
    uint32_t groupNum = 0;
    uint32_t batch = 0;
    uint32_t sharedInputOffset = 0;
    uint32_t sharedInputLen = 0;
    float residualScale = 0;
    uint32_t aQuantMode = 0;
    uint32_t bQuantMode = 0;
    uint32_t biasDtype = 0;
    uint8_t groupListType = 0;
    uint8_t hasBias = 0;
    uint16_t reserved1 = 0;
    // <<< MODIFY: reserved2 替换为确定性字段
    uint32_t deterministicFlag = 0;      // 0=非确定性, 1=确定性
    uint32_t deterWorkspaceSize = 0;     // 确定性 workspace 大小（字节）
};
#pragma pack(pop)
