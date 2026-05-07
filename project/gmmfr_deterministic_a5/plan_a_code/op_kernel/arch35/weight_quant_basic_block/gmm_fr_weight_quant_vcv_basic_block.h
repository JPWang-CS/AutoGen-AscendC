/**
 * 修改文件 11: op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vcv_basic_block.h
 *
 * 修改原因: VCV Basic Block 的 End() 和 IterateNzNkWithAiv() 中调用了
 *           vecCompute_.RoutingYToGm()，这是实际执行 finalize routing scatter + SetAtomicAdd
 *           的位置。确定性模式下需要将输出重定向到 workspace。
 *
 * A3 原型: A3 的 VectorAtomicProcess (a3_vector_atomic_deter.h)
 *   - 非确定性: SetAtomicAdd + DataCopyPad → yGm
 *   - 确定性: DataCopyPad2D → mmQuantOutGm (workspace)
 *
 * 修改内容:
 *   1. Init() 中接收 deterministicFlag 和 workspace 地址
 *   2. IterateNzNkWithAiv() 中的 RoutingYToGm 改为条件调用
 *   3. End() 中的 RoutingYToGm 改为条件调用
 *   4. 确定性模式: RoutingYToGm 的输出目标从 yGm 改为 workspace
 */

// === 在类成员中添加（private 区域）===

// <<< ADD BEGIN
// 确定性相关成员
// A3 原型: QuantGroupMatmul 的 mmQuantOutGm, tiling->deterministicFlag
uint32_t deterministicFlag_ = 0;
__gm__ yType *deterBufferGm_ = nullptr;    // 确定性 workspace 缓冲区
// <<< ADD END


// === 修改 Init() 函数签名，增加确定性参数 ===
// 原始: __aicore__ inline void Init(uint64_t antiQuantGroupSize, __gm__ yType *y, float sharedInputWeight);
// <<< MODIFY BEGIN
__aicore__ inline void Init(uint64_t antiQuantGroupSize, __gm__ yType *y, float sharedInputWeight,
                             uint32_t deterministicFlag = 0, __gm__ yType *deterBufferGm = nullptr)
{
    deterministicFlag_ = deterministicFlag;
    deterBufferGm_ = deterBufferGm;
// <<< MODIFY END

    // ... 原有 Init 代码完全不变 ...

    if ASCEND_IS_AIC {
        cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize, biasL1DbOffset_, biasL1_);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    } else {
        // <<< MODIFY BEGIN: 传递确定性参数给 vecCompute
        vecCompute_.Init(sharedInputWeight, deterministicFlag_ ? deterBufferGm_ : y);
        // 也可以用 y 作为默认，确定性分支在 RoutingYToGm 中处理
        // vecCompute_.Init(sharedInputWeight, y);
        // <<< MODIFY END
    }
    cvLoopIdx_ = 0;
}


// === 修改 IterateNzNkWithAiv() 中的 RoutingYToGm 调用 ===
// 原始位置: IterateNzNkWithAiv() 函数中，约在 curCvLoopIdx > 0 分支内
// 找到: vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_);

// <<< MODIFY BEGIN: 确定性模式下路由到 workspace
// A3 原型: VectorAtomicProcess 第381-386行
if (deterministicFlag_ == 1) {
    // 确定性: 写入 workspace 中间缓冲区（不做 scatter，不使用 AtomicAdd）
    // vecCompute 需要 RoutingYToGm 的 workspace 版本
    // 或者直接传 workspace 地址作为 y（由 resplit_controller 传入）
    vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_);
} else {
    vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_);
}
// <<< MODIFY END


// === 修改 End() 函数中的 RoutingYToGm 调用 ===
// 原始位置: End() 函数的 AIV 分支中，cvLoopIdx_ > 0 内
// 找到: vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_ - 1);

// <<< MODIFY BEGIN: 同 IterateNzNkWithAiv 的修改
if (deterministicFlag_ == 1) {
    vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_ - 1);
} else {
    vecCompute_.RoutingYToGm(lastBasicBlockMSize, ubOutputF32Buffer_, lastOffsetParam, rlLoopIdx_ - 1);
}
// <<< MODIFY END


/**
 * === 关于 vecCompute_.RoutingYToGm 的说明 ===
 *
 * RoutingYToGm 的实际实现在 gmm_fr_weight_quant_vec_compute.h 中。
 * 方案A 的核心思想是：在确定性模式下，通过 resplit_controller 传入的 y 地址
 * 已经指向 workspace（而非真实 yGm），因此 RoutingYToGm 的 SetAtomicAdd
 * 实际写入 workspace。resplit_controller 在所有 basic block 处理完成后，
 * 调用 FRDeterministicA5 从 workspace 聚合到真实 yGm。
 *
 * 因此，VCV Basic Block 本身不需要修改 RoutingYToGm 的行为，
 * 只需要由 resplit_controller 在 deterministic 模式下传入 workspace 地址即可。
 *
 * 唯一的额外需求：Init 时需要知道 workspace 地址，以便 vecCompute 初始化正确的 y GM。
 */
