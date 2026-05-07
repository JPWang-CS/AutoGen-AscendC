/**
 * 修改文件 10: op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h
 *
 * 修改原因: 伪量化路径的控制器。Process() 中调用 VCV Basic Block 的 ComputeBasicBlock
 *           和 End()，其中包含 finalize routing (SetAtomicAdd)。
 *           确定性模式下需要:
 *           a) 将 workspace 地址作为 y 传入 VCV Basic Block（方案A 核心思路）
 *           b) 所有 basic block 处理完成后调用 FRDeterministicA5
 *
 * A3 原型:
 *   - Process() 第307-334行: 遍历 group，每个 block 执行 MMCompute → VectorSync → VectorCompute
 *   - VectorSync 第622-649行: 滑动窗口触发 FRDeterministic
 *   - Process 最后第331-334行: 最终 FRDeterministic 调用
 *
 * 方案A 策略:
 *   不修改 VCV Basic Block 内部的 RoutingYToGm，
 *   而是在 resplit_controller 层面将 y 地址替换为 workspace 地址。
 *   Cgmct 的 VCV Basic Block 的 SetAtomicAdd 写入 workspace（无碰撞，因为 workspace 够大），
 *   然后 FRDeterministicA5 从 workspace 聚合到真实 yGm。
 *
 * 注意: 需要包含 gmm_fr_deterministic_a5.h
 */

// === 在文件开头添加 include ===
// <<< ADD BEGIN
#include "gmm_fr_deterministic_a5.h"
using namespace GMMFRDeterministic;
// <<< ADD END


// === 在类的成员变量中添加（private 区域）===

// <<< ADD BEGIN
// 确定性相关成员
uint32_t deterministicFlag_ = 0;
__gm__ yType *realYGm_ = nullptr;              // 真实输出地址（保存，用于 FRDeterministicA5）
__gm__ yType *deterBufferGm_ = nullptr;        // 确定性 workspace 缓冲区
uint32_t deterWorkspaceSize_ = 0;
// <<< ADD END


// === 修改 Init() 函数，接收确定性参数 ===

// <<< MODIFY BEGIN: 在 Init 函数末尾添加确定性初始化
// 从 tiling 数据读取确定性参数
deterministicFlag_ = baseTiling->deterministicFlag;  // 需要确保 tiling 数据中有此字段
deterWorkspaceSize_ = baseTiling->deterWorkspaceSize;

if (deterministicFlag_ == 1) {
    // A3 原型: Init 第162-166行
    // 保存真实 y 地址，将 yGm_ 替换为 workspace 确定性缓冲区
    realYGm_ = yGm_;
    // 计算 workspace 中确定性缓冲区偏移
    // 偏移量 = 已有 workspace 占用空间（Cube MatMul 中间结果等）
    uint64_t deterBufferOffset = /* 已有 workspace 大小 */;
    deterBufferGm_ = reinterpret_cast<__gm__ yType*>(
        reinterpret_cast<uint8_t*>(workspaceAddr) + deterBufferOffset);
    // 将 yGm_ 重定向到 workspace
    yGm_ = deterBufferGm_;
}
// <<< MODIFY END


// === 修改 Process() 函数，在末尾添加确定性聚合 ===

// <<< ADD BEGIN: 在 Process() 函数的所有 basic block 处理完成后添加
// A3 原型: Process 第331-334行

if (deterministicFlag_ == 1) {
    // 所有 basic block 的 finalize routing 已写入 workspace
    // 执行确定性聚合: workspace → 真实 yGm
    // A3 原型: FRDeterministic (frdeterministic.h 第653-687行)

    GlobalTensor<yType> deterBufferTensor;
    deterBufferTensor.SetGlobalBuffer(deterBufferGm_);
    GlobalTensor<yType> yTensor;
    yTensor.SetGlobalBuffer(realYGm_);
    GlobalTensor<rowIndexType> tokenRanksTensor;
    tokenRanksTensor.SetGlobalBuffer(rowIndexAddr_);
    GlobalTensor<int64_t> groupTokensTensor;
    groupTokensTensor.SetGlobalBuffer(groupListAddr_);

    // A3 原型: Process 第303-306行（SyncConfig 初始化）
    SyncConfig syncConfig;
    uint32_t totalN = /* 总列数 */;
    syncConfig.windowSize = deterWorkspaceSize_ / (totalN * sizeof(yType));
    syncConfig.lowBoundM = syncConfig.windowSize;
    uint64_t nTimes = Ceil(totalN, DETER_UB_SIZE / sizeof(yType));
    syncConfig.baseN = Ceil(Ceil(totalN, nTimes), 128) * 128;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queBind;
    TPipe deterPipe;
    deterPipe.InitBuffer(queBind, BUFFER_NUM, DETER_UB_SIZE);

    // 最终确定性聚合
    syncConfig.curM = /* 总行数 */;
    FRDeterministicA5<yType, rowIndexType>(
        syncConfig, deterBufferTensor, yTensor, tokenRanksTensor, queBind,
        coreNum_, totalN);
}
// <<< ADD END
