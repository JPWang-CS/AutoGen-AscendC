# GMMFR 确定性特性 A5 迁移 — 服务器验证清单

> 版本：v3.1 (Plan B - Sequential Write Epilogue)
> 范围：仅 W8A8（INT8×INT8）PerToken 全量化路径
> 日期：2026-05-08
> 代码路径：`ops-transformer_AI/gmm/grouped_matmul_finalize_routing`

---

## 一、修改文件总览

| # | 文件路径 | 操作 | 关键变更 |
|---|---------|------|---------|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 修改 | 在 `reserved2` 之后新增 `deterministicFlag` + `deterWorkspaceSize` |
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | 修改 | 新增 `GetWorkspaceSize()` 声明 + 两个私有成员 |
| 3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | 修改 | DoOpTiling 确定性逻辑 + GetWorkspaceSize 实现 + PrintQuantParams 扩展 |
| 4 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | **新增** | FRDeterministicA5 延迟聚合函数 |
| 5 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 修改 | 新增 if/else 确定性分支（Plan B Epilogue） |
| 6 | `common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h` | **新增** | Plan B 顺序写 Epilogue（无 scatter，无 AtomicAdd） |

### 未修改的文件（零影响保证）

| 文件路径 | 状态 |
|---------|------|
| `common/cgmct/epilogue/block_epilogue_dequant_finalize_routing.h` | 未修改，已验证逐行一致 |
| `op_kernel/arch35/grouped_matmul_finalize_routing.h` | 未修改 |
| `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_key.h` | 未修改 |
| `op_kernel/arch35/weight_quant_basic_block/*` | 未修改 |
| `op_kernel/arch35/common/*` | 未修改 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` 中原有逻辑 | 未修改（仅新增代码块） |

---

## 二、逐文件 Diff 详情

### 文件 1：`op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h`

**变更类型**：仅新增字段，不修改已有字段

```diff
  uint8_t hasBias = 0;
  uint16_t reserved1 = 0;
  uint32_t reserved2 = 0;          // 保持不变
+ uint32_t deterministicFlag = 0;   // 新增：0=非确定性, 1=确定性
+ uint32_t deterWorkspaceSize = 0;  // 新增：确定性 workspace 大小（字节）
```

**验证要点**：
- [ ] `reserved2` 字段未被删除或替换
- [ ] 新字段在结构体末尾，`#pragma pack(push, 8)` 对齐无影响
- [ ] `sizeof(GMMFinalizeRoutingDataParams)` 增加了 8 字节

---

### 文件 2：`op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h`

**变更类型**：仅新增声明，不修改已有声明

```diff
  // protected 区域：
+ ge::graphStatus GetWorkspaceSize() override;   // 新增（在 PostTiling 和 Reset 之间）

  // private 区域末尾：
  int8_t rowIndexType_ = 0;
+ uint32_t deterministicFlag_ = 0;      // 新增
+ uint32_t deterWorkspaceSize_ = 0;     // 新增
```

**验证要点**：
- [ ] 原有 `Reset()` 声明位置不变
- [ ] `GetWorkspaceSize()` 是 override，父类需有对应虚函数

---

### 文件 3：`op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp`

**变更类型**：三处新增代码块

#### 变更 3a：DoOpTiling() 确定性 Tiling（在 hasBias 赋值之后、PrintQuantParams 之前）

```cpp
// 新增确定性条件判断块（约 25 行）
if (context_->GetDeterministic() == 1 && !IsMicroScaling() &&
    inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8 &&
    inputParams_.aQuantMode == optiling::QuantMode::PERTOKEN_MODE) {
    // 设置标志、计算 workspace 大小、溢出保护
}
```

**验证要点**：
- [ ] 条件：`GetDeterministic()==1` + `!IsMicroScaling()` + `INT8×INT8` + `PERTOKEN_MODE`
- [ ] 溢出保护：`requiredDeterSize > deterWorkspaceSize_` 时降级为非确定性
- [ ] `deterministicFlag_==0` 时原有行为完全不变

#### 变更 3b：GetWorkspaceSize() 实现（新函数）

```cpp
ge::graphStatus GroupedMatmulFinalizeRoutingQuantTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t totalWorkspace = SYS_WORKSPACE_SIZE;  // 16MB
    if (deterministicFlag_ == 1 && deterWorkspaceSize_ > 0) {
        totalWorkspace += static_cast<size_t>(deterWorkspaceSize_);
    }
    workspaces[0] = totalWorkspace;
    return ge::GRAPH_SUCCESS;
}
```

**验证要点**：
- [ ] 非确定性模式下 workspace 大小不变（仅 16MB）
- [ ] 确定性模式下 workspace = 16MB + deterWorkspaceSize_

#### 变更 3c：PrintQuantParams() 扩展

```diff
  << ", hasBias = " << ...
+ << ", deterministicFlag = " << tilingData_.gmmFinalizeRoutingDataParams.deterministicFlag
+ << ", deterWorkspaceSize = " << tilingData_.gmmFinalizeRoutingDataParams.deterWorkspaceSize;
```

---

### 文件 4：`op_kernel/arch35/gmm_fr_deterministic_a5.h`（新文件）

**功能**：A5 平台确定性延迟聚合核心函数

**关键设计**：
1. AIC 核心：`SyncAll(); SyncAll(); return;`（必须配对参与同步）
2. Vector 核心：
   - `SyncAll()` - 等待所有 Epilogue workspace 写入完成
   - 按 `outRow % coreNumVec == GetBlockIdx()` 分配行所有权
   - 顺序读取 workspace → UB → SetAtomicAdd 写入最终输出（单写者保证）
   - `SyncAll()` - 等待所有写入完成

**验证要点**：
- [ ] AIC 核心确实调用了两次 SyncAll（A5 要求严格配对）
- [ ] `tokenRanksGm.GetValue(baseOffset + mOffset)` 读取 row index 正确
- [ ] workspace 读取地址 `mOffset * n + nOffset` 与 Epilogue 顺序写地址一致
- [ ] 最终输出地址 `outRow * n + nOffset` 使用 AtomicAdd（多 token 可能汇聚到同一 outRow）

---

### 文件 5：`op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h`

**变更类型**：if/else 分支，else 分支完全保留原有代码

```cpp
if (gmmFinalizeRoutingQuantParams_.deterministicFlag == 1) {
    // ===== 确定性分支（Plan B） =====
    // 1. 使用 BlockEpilogueDequantSequentialWrite（顺序写 Epilogue）
    // 2. 输出到 workspace（offset = SYS_WORKSPACE_SIZE = 16MB）
    // 3. 执行 Cgmct Kernel
    // 4. 调用 FRDeterministicA5 进行延迟聚合
} else {
    // ===== 原有分支（完全不变） =====
    // 原始 Params 构造和 gmm(params) 调用
}
```

**验证要点**：
- [ ] `else` 分支代码与修改前完全一致
- [ ] 确定性分支使用 `SYS_WORKSPACE_SIZE` (16MB) 作为 workspace 偏移
- [ ] 确定性分支的 Params 中 y 地址指向 `deterBuffer` 而非原始 `y`
- [ ] Prologue 的 y 地址也指向 `deterBuffer`
- [ ] `TPipe` 和 `TQueBind` 在确定性分支内正确初始化

---

### 文件 6：`common/cgmct/epilogue/block_epilogue_dequant_sequential_write.h`（新文件）

**功能**：Plan B 顺序写 Epilogue，替代原始 scatter + AtomicAdd 模式

**与原始 `block_epilogue_dequant_finalize_routing.h` 的唯一差异**：

| 方法 | 原始（非确定性） | 新（确定性） |
|------|-----------------|-------------|
| `VectorAtomicProcess` | `SetAtomicAdd → outRow * n_ → DataCopyPad → SetAtomicNone` | 不存在 |
| `VectorSequentialWrite` | 不存在 | 直接 `DataCopyPad((offsetM+i) * n_, ...)` 无 AtomicAdd |
| `operator()` | 调用 `VectorAtomicProcess` | 调用 `VectorSequentialWrite` |

**验证要点**：
- [ ] 所有常量使用 `SEQ_` 前缀，避免与原始文件的匿名命名空间冲突
- [ ] `VectorSequentialWrite` 地址计算：`(offsetM + i) * n_ + yOffset`（按输入行序）
- [ ] 无 `SetAtomicAdd` / `SetAtomicNone` 调用
- [ ] 其余方法（Init, CopyInLogit, VFDoLogitMuls, VFDoDequant* 等）与原始完全一致

---

## 三、回归安全验证

### 非确定性路径（默认路径）影响分析

| 检查项 | 结果 |
|--------|------|
| Tiling struct 新增字段不影响原有字段偏移 | 通过（新增在末尾，pack(8)） |
| `deterministicFlag` 默认值为 0 | 通过 |
| `DoOpTiling()` 原有赋值逻辑不变 | 通过（新增代码块在原有赋值之后） |
| `GetWorkspaceSize()` 非确定性时仅返回 SYS_WORKSPACE_SIZE | 通过 |
| `pertoken_dequant.h` else 分支代码与修改前逐行一致 | 通过（已对比验证） |
| 原始 `block_epilogue_dequant_finalize_routing.h` 未被修改 | 通过（已对比 562 行完全一致） |
| 其他 kernel 入口文件未修改 | 通过 |

---

## 四、服务器验证步骤

### 步骤 1：编译验证
```bash
# 编译 ops-transformer_AI 仓
# 确认无编译错误，特别是：
#   - tiling_data.h 结构体新增字段对齐
#   - gmm_fr_deterministic_a5.h 的 AscendC API 调用
#   - block_epilogue_dequant_sequential_write.h 的 MicroAPI 调用
```

### 步骤 2：非确定性回归测试
```bash
# 运行原有 W8A8 PerToken 测试用例
# 预期：deterministicFlag = 0, 结果与修改前完全一致
# 重点验证：
#   - 输出数值一致性
#   - workspace 大小不变（仅 16MB）
#   - 无额外性能退化
```

### 步骤 3：确定性功能测试
```bash
# 设置 deterministic=1，运行 W8A8 PerToken 测试用例
# 预期：deterministicFlag = 1
# 重点验证：
#   - workspace 大小 = 16MB + deterWorkspaceSize_
#   - 输出数值正确性（与 CPU 参考结果对比）
#   - 连续 10 次运行结果 bit-wise 一致
```

### 步骤 4：确定性边界测试
```bash
# 测试大 shape（M*N*4 > 96MB）→ 预期降级为非确定性，打印 WARNING 日志
# 测试非 INT8 输入 → 预期 deterministicFlag = 0
# 测试非 PerToken 模式 → 预期 deterministicFlag = 0
# 测试 MX 量化模式 → 预期 deterministicFlag = 0
```

---

## 五、风险提示

1. **SyncAll 配对**：A5 平台要求所有核心（AIC+Vector）严格 1:1 SyncAll 配对。若 Cgmct 框架内部有其他 SyncAll 调用，需要确保总配对数一致。

2. **workspace 大小**：确定性 workspace 上限 96MB（L2 > 96MB 时）或 64MB。若 M*N*4 > 上限，自动降级为非确定性并打印 WARNING。

3. **性能影响**：确定性路径增加了一次完整的数据拷贝（workspace → yGm），预计有 5-15% 的性能开销。

4. **仅 W8A8 PerToken**：确定性特性仅在 INT8×INT8 + PerToken 量化路径下启用。其他路径（FP8、MX、WeightQuant 等）不受影响。
