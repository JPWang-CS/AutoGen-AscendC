# AscendC 编程模型知识库

## 一、核心架构

### 1.1 AI Core 内部结构

AscendC 编程模型基于华为昇腾 AI Core 架构，包含两类计算核心：

- **AIV (AI Vector)**：向量计算核，负责逐元素运算、归约、数据搬运等
- **AIC (AI Cube)**：矩阵计算核，负责矩阵乘法（MatMul）运算

每个 AI Core 内部存储层级：
```
GM (Global Memory) → L2 Cache → L1 (Local Memory) → L0A/L0B → L0C → UB (Unified Buffer)
```

### 1.2 SPMD 编程范式

AscendC 采用 SPMD（Single Program Multiple Data）编程模型：
- 多个 AI Core 执行相同代码，处理不同数据
- 通过 `GetBlockIdx()` 获取当前核编号
- 通过 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 区分核心类型

### 1.3 三种 Kernel 类型

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);   // 纯向量算子
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // 1 Cube + 2 Vector 混合
```

## 二、核心 API

### 2.1 Tensor 类型

```cpp
// 全局张量（GM 空间）
GlobalTensor<T> xGm;
xGm.SetGlobalBuffer((__gm__ T*)addr);

// 本地张量（UB/L0C 等空间）
LocalTensor<T> xLocal = queBuf.AllocTensor<T>();
LocalTensor<T> xLocal = tBuf.Get<T>();
```

### 2.2 数据搬运

```cpp
// GM → UB (VecIn)
DataCopy(localTensor, globalTensor, copyParams);
DataCopyPad(localTensor, globalTensor, extParams, padParams);

// UB → GM (VecOut)
DataCopy(globalTensor, localTensor, copyParams);
DataCopyPad(globalTensor, localTensor, extParams);

// 2D 搬运参数
DataCopyExtParams{blockCount, blockLen, srcStride, dstStride, ...};
DataCopy2DDimParams{dim0, dim1, srcDim0};
```

### 2.3 向量计算 API

```cpp
// 逐元素运算
Add(localDst, src1, src2, count);
Sub(localDst, src1, src2, count);
Mul(localDst, src1, src2, count);
Div(localDst, src1, src2, count);
Exp(localDst, src, count);

// 类型转换
Cast(localDst, src, RoundMode::CAST_NONE, count);

// 归约
ReduceSum(localDst, src, tmpBuf, count);

// 广播
Brcb(localDst, src, count, {1, 8}); // 沿行广播

// 填充
Duplicate(localDst, value, count);
```

### 2.4 矩阵计算 API (MatMul)

```cpp
#include "lib/matmul_intf.h"

// MatMul 类型定义
using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using bT = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using biasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using MT = matmul::MatmulImpl<aT, bT, cT, biasT, CFG_MDL>;

// 初始化和使用
MT mm;
mm.Init(&tilingData, &pipe);
mm.SetOrgShape(M, N, K);
mm.SetSingleShape(m, n, k);
mm.SetTensorA(xGm[offset]);
mm.SetTensorB(wGm[offset]);
while (mm.Iterate()) {
    mm.GetTensorC(outGm[offset], 0, true);
}
```

### 2.5 缓冲区管理

```cpp
TPipe pipe;

// Queue: 生产者-消费者模式
TQue<QuePosition::VECIN, 1> inQueue;   // 输入队列
TQue<QuePosition::VECOUT, 1> outQueue;  // 输出队列
pipe.InitBuffer(inQueue, BUFFER_NUM, bufferSize);

auto local = inQueue.AllocTensor<T>();
inQueue.EnQue(local);
local = inQueue.DeQue<T>();
inQueue.FreeTensor(local);

// Buffer: 直接分配
TBuf<TPosition::VECCALC> tmpBuf;
pipe.InitBuffer(tmpBuf, bufferSize);
auto local = tmpBuf.Get<T>();
auto local2 = tmpBuf.GetWithOffset<T>(count, offset);
```

### 2.6 同步机制

```cpp
// 核内同步
PIPE_BAR(PIPE_V);           // Vector 流水线屏障
PIPE_BAR(PIPE_M);           // MatMul 流水线屏障
pipe_barrier(PIPE_V);       // 同上

// Cube-Vector 核间同步
CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // Cube 通知 Vector
CrossCoreWaitFlag(SYNC_AIC_TO_AIV);               // Vector 等待 Cube

// 全核同步（所有核同步）
SyncAll();

// 软件同步标记值
static constexpr uint8_t SYNC_AIC_TO_AIV = 0x01;
static constexpr uint8_t SYNC_AIV_TO_AIC = 0x02;
```

### 2.7 原子操作

```cpp
SetAtomicAdd<float>();       // 开启 float 原子加
DataCopy(...);               // 该 DataCopy 将执行原子加
SetAtomicNone();             // 关闭原子操作

SetAtomicAdd<half>();
SetAtomicMax<float>();
```

## 三、Kernel 入口模板

```cpp
extern "C" __global__ __aicore__ void op_name(GM_ADDR x, ..., GM_ADDR tiling)
{
    // 1. 获取 Tiling 数据
    GET_TILING_DATA(tilingData, tiling);

    // 2. 设置 Kernel 类型
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // 3. 按 Tiling Key 分发
    if (TILING_KEY_IS(key_value)) {
        // 初始化 Pipe, MatMul 等
        TPipe pipe;
        // 创建算子实例并执行
        MyOp<Params> op;
        op.Init(...);
        op.Process();
    }
}
```

## 四、Tiling 框架

### 4.1 Tiling 数据结构

```cpp
BEGIN_TILING_DATA_DEF(MyOpTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  // ...
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MyOp, MyOpTilingData)
```

### 4.2 Tiling 基类流程

```cpp
class MyTiling : public TilingBaseClass {
    ge::graphStatus DoTiling() override {
        GetShapeAttrsInfo();     // 1. 获取输入 Shape/属性
        GetPlatformInfo();       // 2. 获取平台信息
        IsCapable();             // 3. 能力检查
        DoOpTiling();            // 4. 计算切分参数
        DoLibApiTiling();        // 5. 高阶 API Tiling
        GetWorkspaceSize();      // 6. 计算 Workspace
        PostTiling();            // 7. 保存 Tiling 数据
        context_->SetTilingKey(GetTilingKey()); // 8. 设置 Key
        return SUCCESS;
    }
};
```
