# AscendC 官方 API 学习知识文档

> 整理时间: 2026-05-06  
> 来源: 昇腾社区官方文档 + ops-transformer 代码仓

---

## 一、核函数 (Kernel Function) 编程模型

### 1.1 核函数定义规则

核函数是 AscendC 算子设备侧实现的入口，需遵循以下规则:

```cpp
// 核函数必须使用 __global__ 和 __aicore__ 限定符
extern "C" __global__ __aicore__ void kernel_name(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    KernelOp op;
    op.Init(x, y, z);
    op.Process();
}

// GM_ADDR 宏定义
#define GM_ADDR __gm__ uint8_t*
```

### 1.2 核函数调用方式

```cpp
// 使用 <<<blockDim, l2ctrl, stream>>> 内核调用符
kernel_name<<<blockDim, nullptr, stream>>>(x, y, z);

// 异步调用，需同步等待
aclrtSynchronizeStream(stream);
```

### 1.3 模板核函数

```cpp
template<int a, typename T>
__global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    // 使用模板参数
}

// 调用
add_custom<20, float><<<blockDim, nullptr, stream>>>(x, y, z);
```

**限制**: 仅支持 <<<>>> 调用方式，暂不支持自定义数据类型作为模板参数。

---

## 二、AIV / AIC 编程模型与架构差异

### 2.1 耦合架构 vs 分离架构

| 特性 | 耦合架构 (A2/910B) | 分离架构 (A3/910C, A5/950) |
|------|---------------------|---------------------------|
| Vector/Cube 关系 | 集成在一起 | 分离独立 |
| blockDim 含义 | 设置 AICore 核实例数 | 按 AIV/AIC/AIV+AIC 组合设置 |
| 核数获取 | GetCoreNumAiv / GetCoreNumAic | GetCoreNumAic / GetCoreNumAiv |

### 2.2 分离架构的 blockDim 设置规则

- **仅 Vector 算子**: blockDim 设为 Vector 核数 (如 40)
- **仅 Cube 算子**: blockDim 设为 Cube 核数 (如 20)
- **Vector/Cube 融合算子**: 按 AIV+AIC 组合设置 (如 2AIV+1AIC=1 组合，设为 20)

### 2.3 平台宏定义

从 ops-transformer 代码仓中发现的平台宏:

```cpp
// A2 (910B) 平台 - C220 架构
#ifdef __DAV_C220_VEC__   // Vector Core 编译时定义
#ifdef __DAV_C220_CUBE__  // Cube Core 编译时定义

// A3 (910C) 平台 - C310 架构
#if defined(__DAV_C310_CUBE__)  // A3 Cube Core

// 部分代码中的组合判断
#if defined (__DAV_C220_CUBE__) || defined (__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
```

### 2.4 编程模型中的关键区别

**Vector Core (__DAV_C220_VEC__) 编程**:
- 可访问 UB (Unified Buffer) - TPosition::VECIN
- 使用 LocalTensor 操作向量计算 API (Add, Sub, Mul, Div 等)

**Cube Core (__DAV_C220_CUBE__) 编程**:
- 可访问 L1/CB (TPosition::A1), L0A (TPosition::A2), L0B (TPosition::B2), L0C (TPosition::CO1)
- 使用 Matmul 高阶 API 进行矩阵计算

### 2.5 实际代码中的平台分支模式 (来自 mem.h)

```cpp
#ifdef __DAV_C220_VEC__
    // Vector Core: 只初始化 UB
    tensor[ASCEND_UB] = LocalTensor<uint8_t>(TPosition::VECIN, 0, ubSize);
#elif __DAV_C220_CUBE__
    // Cube Core: 初始化 L1, L0A, L0B, L0C
    tensor[ASCEND_CB]  = LocalTensor<uint8_t>(TPosition::A1, 0, l1Size);
    tensor[ASCEND_L0A] = LocalTensor<uint8_t>(TPosition::A2, 0, l0ASize);
    tensor[ASCEND_L0B] = LocalTensor<uint8_t>(TPosition::B2, 0, l0BSize);
    tensor[ASCEND_L0C] = LocalTensor<uint8_t>(TPosition::CO1, 0, l0CSize);
#else
    // 耦合架构: 初始化所有 Buffer
    // UB + L1 + L0A + L0B + L0C
#endif
```

---

## 三、内存层次与 Buffer 管理

### 3.1 硬件规格 (来自 hardware.h)

| 内存层级 | 大小 | TPosition |
|----------|------|-----------|
| L2 Cache | 192 MB | - |
| L1 (CB) | 512 KB | A1 |
| L0A | 64 KB | A2 |
| L0B | 64 KB | B2 |
| L0C | 128 KB | CO1 |
| UB | 192 KB | VECIN |
| Fractal Size | 512 | - |

### 3.2 BufferType 枚举

```cpp
enum class BufferType { ASCEND_UB, ASCEND_CB, ASCEND_L0A, ASCEND_L0B, ASCEND_L0C, ASCEND_MAX };

// BufferType 到 TPosition 的映射
ASCEND_UB  -> TPosition::VECIN
ASCEND_CB  -> TPosition::A1
ASCEND_L0A -> TPosition::A2
ASCEND_L0B -> TPosition::B2
ASCEND_L0C -> TPosition::CO1
```

### 3.3 TPipe / TQue 内存管理

```cpp
// 初始化
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;

// 分配 Buffer (参数: queue, buffer数, 大小字节)
pipe.InitBuffer(inQueueX, 1, bufferSize * sizeof(dataType));

// 使用流程
LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();  // 分配
DataCopy(xLocal, xGm[offset], length);               // 搬入
inQueueX.EnQue(xLocal);                               // 入队
LocalTensor<T> xLocal = inQueueX.DeQue<T>();          // 出队
// ... 计算 ...
inQueueX.FreeTensor(xLocal);                          // 释放
```

---

## 四、DataCopy 数据搬运 API

### 4.1 基本数据搬运 (连续)

```cpp
// GM -> UB (GlobalTensor -> LocalTensor)
DataCopy(LocalTensor<T>& dst, GlobalTensor<T>& src, const uint32_t count);

// UB -> GM (LocalTensor -> GlobalTensor)
DataCopy(GlobalTensor<T>& dst, LocalTensor<T>& src, const uint32_t count);
```

### 4.2 切片数据搬运

```cpp
// GM -> Local
template <typename T>
void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
              const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[],
              const uint32_t dimValue = 1);

// Local -> GM
template <typename T>
void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src,
              const SliceInfo dstSliceInfo[], const SliceInfo srcSliceInfo[],
              const uint32_t dimValue = 1);
```

### 4.3 SliceInfo 结构体

```cpp
struct SliceInfo {
    uint32_t startIndex;  // 切片起始元素位置
    uint32_t endIndex;    // 切片终止元素位置
    uint32_t stride;      // 切片间隔元素个数
    uint32_t burstLen;    // 每片数据长度，单位: datablock (32B)
    uint32_t shapeValue;  // 当前维度原始长度，单位: 元素个数
};
```

### 4.4 支持的数据通路

| 产品 | GM -> UB 数据类型 |
|------|-------------------|
| Atlas A2 | int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, half, bfloat16_t, float |
| Atlas A3 | int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, half, bfloat16_t, float |
| Atlas 推理 AI Core | int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, half, float |

### 4.5 对齐要求

- datablock 大小 = 32 字节
- 切片搬运的 burstLen 需手动计算: `横向切片元素数 * sizeof(T) / 32`
- 切片元素数 * sizeof(T) 必须是 32 字节的倍数
- SliceInfo 数组大小不超过 8

---

## 五、向量计算 API

### 5.1 基本运算 (逐元素)

```cpp
// 二元运算
Add(dst, src0, src1, count);    // dst = src0 + src1
Sub(dst, src0, src1, count);    // dst = src0 - src1
Mul(dst, src0, src1, count);    // dst = src0 * src1
Div(dst, src0, src1, count);    // dst = src0 / src1

// 标量运算
Adds(dst, src, scalar, count);  // dst = src + scalar

// 运算符重载 (整个 Tensor 参与)
// dst = src0 + src1 (运算符方式)
```

### 5.2 带重复参数的高级运算

```cpp
// 使用 BinaryRepeatParams 控制跨越和步长
Add<T, false>(dst, src0, src1,
    (uint64_t)0,  // mask
    repeat,       // 重复次数
    BinaryRepeatParams(
        dstBlockStride, src0BlockStride, src1BlockStride,
        dstRepeatStride, src0RepeatStride, src1RepeatStride));
```

### 5.3 精度转换

```cpp
// Cast 接口用于数据类型转换
Cast(dst, src, RoundMode, count);

// ToFloat 接口: 将 bfloat16_t 等转为 float
ToFloat(dst, src, count);
```

---

## 六、Matmul 矩阵计算 API (Cube 高阶 API)

### 6.1 Matmul 对象创建

```cpp
#include "lib/matmul_intf.h"

// 使用 MatmulType 定义输入输出类型
typedef MatmulType<TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL> aType;
typedef MatmulType<TPosition::GM, CubeFormat::ND, half, true, LayoutMode::NORMAL> bType;
typedef MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL> cType;
typedef MatmulType<TPosition::GM, CubeFormat::ND, float> biasType;

// 配置 MatmulConfig
constexpr MatmulConfig MM_CFG = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1);

// 创建 Matmul 对象
Matmul<aType, bType, cType, biasType, MM_CFG> mm;
```

### 6.2 Matmul 使用流程

```cpp
// 1. 注册 Matmul 对象
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

// 2. 设置输入矩阵
mm.SetTensorA(gm_a);     // 左矩阵 A
mm.SetTensorB(gm_b);     // 右矩阵 B
mm.SetBias(gm_bias);     // Bias (可选)

// 3. 执行矩阵乘
mm.Iterate(gm_c);        // 单次计算
// 或
mm.IterateBatch(gm_c, batchA, batchB, false);  // 批量计算

// 4. 结束
mm.End();
```

### 6.3 Tiling 配置

```cpp
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);
tiling.SetDim(1);
tiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
tiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
tiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
tiling.SetShape(M, N, K);
tiling.SetOrgShape(M, N, K);
tiling.EnableBias(true);
tiling.SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```

### 6.4 Batch Matmul

支持 4 种 Layout 类型:
- **BSNGD**: Batch, Spatial, N, Group, D
- **SBNGD**: Spatial, Batch, N, Group, D
- **BNGS1S2**: Batch, N, Group, S1, S2
- **NORMAL**: BMNK 标准排布

```cpp
// 设置 Batch 信息 (NORMAL 模式)
tiling.SetBatchInfoForNormal(BATCH_NUM_A, BATCH_NUM_B, M, N, K);

// Kernel 端批量计算
mm.IterateBatch(gm_c, batchA, batchB, false);
```

**约束**:
- 只支持 Norm 模板
- BSNGD/SBNGD/BNGS1S2 要求多 Batch 数据总和 < L1 Buffer 大小
- 不支持量化/反量化
- 异步模式不支持搬运到 UB
- 不支持 enableMixDualMaster (双主模式)

### 6.5 GEMV 模式

当 M=1 时，可开启 GEMV 模式提升效率:
- 在 Tiling 侧和 Kernel 侧配置 A 矩阵数据格式为 VECTOR
- 未开启时，M 方向按非对齐场景处理

---

## 七、Tiling 策略

### 7.1 Tiling 基本概念

由于硬件内部存储 (UB/L1) 无法容纳完整数据，需将数据切分为小块处理。

### 7.2 Tiling 结构体定义

```cpp
struct AddCustomTilingData {
    uint32_t blockLength;      // 每个核计算的数据长度
    uint32_t tileNum;          // 每个核上主块数据块个数
    uint32_t tileLength;       // 每个核上主块数据块长度
    uint32_t lastTileLength;   // 每个核上尾块长度
};
```

### 7.3 Tiling 计算公式

```cpp
constexpr uint32_t BLOCK_SIZE = 32;  // datablock = 32 字节
constexpr uint32_t NUM_BLOCKS = 8;   // 使用核数
constexpr uint32_t UB_BLOCK_NUM = 100;  // UB 可用 block 数量

// 1. 计算对齐后的总长度
uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
totalLengthAligned = (totalLength % alignNum == 0) ?
    totalLength : ((totalLength + alignNum - 1) / alignNum) * alignNum;

// 2. 计算每核数据长度
blockLength = totalLengthAligned / NUM_BLOCKS;

// 3. 计算 tileNum
tileNum = blockLength / (alignNum * UB_BLOCK_NUM);

// 4. 计算主块和尾块
if (tileNum == 0) {
    // 仅尾块场景
    tileLength = 0;
    lastTileLength = ((blockLength + alignNum - 1) / alignNum) * alignNum;
} else if ((blockLength / alignNum) % UB_BLOCK_NUM == 0) {
    // 仅主块场景
    tileLength = UB_BLOCK_NUM * alignNum;
    lastTileLength = 0;
} else {
    // 主块 + 尾块
    tileLength = UB_BLOCK_NUM * alignNum;
    lastTileLength = blockLength - tileNum * tileLength;
}
```

### 7.4 Tiling 优化策略

**多核切分**:
- `context->SetBlockDim(BLOCK_DIM)` 设置核数
- 耦合架构: blockDim 设为 AICore 核数
- 分离架构: 根据 AIV/AIC 分别设置

**L2Cache 切分**:
- 当输入数据超过 L2Cache (192MB) 时使能
- 将数据均分切分，使每次计算能命中 L2Cache
- 带宽差异: L2Cache ~7TB/s vs HBM ~1.6TB/s

**核间负载均衡**:
- 避免计算拖尾 (部分核空闲)
- 调整拖尾核位置实现全局负载最优

### 7.5 尾块处理 Kernel 模式

```cpp
// Init 中取最大值分配内存
uint32_t initBufferLength = max(tileLength, lastTileLength);
pipe.InitBuffer(inQueueX, 1, initBufferLength * sizeof(dataType));

// Process 中分别处理主块和尾块
for (uint32_t i = 0; i < tileNum; i++) {
    CopyIn(i, tileLength);
    Compute(i, tileLength);
    CopyOut(i, tileLength);
}
if (lastTileLength > 0) {
    CopyIn(tileNum, lastTileLength);
    Compute(tileNum, lastTileLength);
    CopyOut(tileNum, lastTileLength);
}
```

---

## 八、确定性计算

### 8.1 问题背景

默认不开启确定性计算。相同硬件和输入下，多次执行结果可能不同。
**原因**: 异步多线程执行导致浮点数累加顺序变化，以及原子操作 (如 AtomicAdd)。

### 8.2 开启方式

**ATC 工具参数**:
```bash
--deterministic=1   # 开启确定性计算
--deterministic=0   # 默认，不开启
```

### 8.3 代码中的确定性处理模式

```cpp
// ops-transformer 中的实现模式
static constexpr bool deterministic = Deterministic;  // 编译期常量

// 条件类型选择
using scatterAddGmType = typename std::conditional<deterministic, GlobalTensor<T>, int8_t>::type;

// 运行时分支
if constexpr (deterministic) {
    // 确定性模式: 使用非原子操作，逐核串行处理
} else {
    // 非确定性模式: 可使用原子操作，并行处理
}

// 核数调整: 确定性模式可能使用更少的核
int64_t actualUsedCoreNum = deterministic ? coreNum : min(totalSize, (int64_t)coreNum);
```

### 8.4 注意事项

- 开启确定性计算往往导致算子执行变慢
- 适用于调试、精度调优场景
- ops-transformer 中部分算子 (如 sparse_lightning_indexer_grad_kl_loss) 支持 deterministic 属性

---

## 九、平台宏速查表

| 宏定义 | 平台 | 含义 |
|--------|------|------|
| `__DAV_C220_VEC__` | A2 (910B) | Vector Core 编译 |
| `__DAV_C220_CUBE__` | A2 (910B) | Cube Core 编译 |
| `__DAV_C310_CUBE__` | A3 (910C) | Cube Core 编译 |
| `__DAV_310R6_CUBE__` | A3 变体 | Cube Core 编译 |

**判断当前是 Vector 还是 Cube Core**: 同一架构下，`__DAV_C220_VEC__` 和 `__DAV_C220_CUBE__` 互斥。

---

## 十、常用辅助函数

```cpp
// 向上对齐
template <typename T>
inline T RoundUp(const T val, const T align) {
    return (val + align - 1) / align * align;
}

// 向上取整除法
template <typename T>
inline T CeilDiv(const T dividend, const T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// 最小值
template <typename T>
inline T Min(const T lhs, const T rhs) {
    return lhs < rhs ? lhs : rhs;
}
```

---

## 参考文档链接

- [核函数编程模型 (CANN 8.0)](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0014.html)
- [Batch Matmul (CANN 8.2)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0041.html)
- [Tiling 优化技巧](https://www.hiascend.com/developer/techArticles/20240920-1)
- [尾块 Tiling (CANN 9.0)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00009.html)
- [DataCopy 切片搬运 API (CANN 8.5)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0105.html)
- [deterministic 参数说明 (ATC)](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devaids/auxiliarydevtool/atlasatc_16_0125.html)
- [AscendC API 接口入门](https://zhuanlan.zhihu.com/p/1791441008)
- [Matmul 高阶 API 深入讲解](https://hwcomputing.csdn.net/6943b281836da3214485c2fc.html)
