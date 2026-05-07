# AscendC 平台差异知识库

## 一、平台架构映射

| SOC 版本 | 架构目录 | 芯片系列 | Kernel 宏 |
|---|---|---|---|
| `ascend910b` | `arch32` | Atlas A2 (910B) | `__CCE_AICORE__ == 220` |
| `ascend910_93` | `arch32` | Atlas A3 (910C) | `__CCE_AICORE__ == 220` |
| `ascend950` | `arch35` | Ascend 950 (A5) | `__CCE_AICORE__ == 310` |
| `mc62cm12a` | `arch38` | MC62CM12A | - |
| `kirinx90` | `arch32` | Kirin X90 | 共享 arch32 |

**关键规则**：A2/A3 共享 arch32 代码，A5 使用 arch35，代码不可混用。

## 二、硬件参数差异

| 参数 | A2/A3 (arch32) | A5 (arch35) |
|---|---|---|
| UB 大小 | 192 KB | 192 KB+ |
| L1 大小 | 512 KB | 512 KB |
| L0A/L0B 大小 | 64 KB | 64 KB |
| L0C 大小 | 128 KB | 256 KB |
| L2 Cache | 192 MB | 更大 |
| Cube Freq | ~1.5 GHz | 更高 |

## 三、编程范式差异

### 3.1 arch32 (A2/A3) 编程方式

使用原生 AscendC API：
```cpp
// 直接使用 MatmulImpl
using MT = matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL>;
MT mm;
mm.Init(&tiling, &pipe);

// Cube/Vector 通过宏区分
#ifdef __DAV_C220_CUBE__
    // Cube 核路径
#else
    // Vector 核路径
#endif

// 核间同步
CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);
CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
```

### 3.2 arch35 (A5) 编程方式

使用 Cgmct (C++ Gemm Template) 框架：
```cpp
// 使用 Builder 模式
using BlockMmadBuilder = Block::BlockMxMmAicToAivBuilder<
    AType, LayoutA, BType, LayoutB, BiasType, CType, LayoutC,
    L1TileShape, L0TileShape, BlockScheduler, ...>;

using GmmKernel = Kernel::KernelGmmFinalizeRouting<
    ProblemShape, BlockMmadBuilder, BlockPrologue, BlockEpilogue, BlockScheduler>;

GmmKernel gmm;
gmm(params);

// Cube/Vector 使用 std::conditional 选择 Dummy/Real 实现
using CubeBlockType = std::conditional_t<g_coreType == AIC,
    RealCubeBlock, DummyCubeBlock>;
using VecBlockType = std::conditional_t<g_coreType == AIC,
    DummyVecBlock, RealVecBlock>;
```

### 3.3 Kernel 入口文件命名

| 平台 | 入口文件 | 说明 |
|---|---|---|
| arch32 | `op_name.cpp` | `__CCE_AICORE__ == 220` |
| arch35 | `op_name_apt.cpp` | `__CCE_AICORE__ == 310` |

### 3.4 A5 独有特性

1. **CCU 1.0 通信加速器**：替代 A2 的 AICPU 通信方式
2. **SIMT 线程级并行**：支持线程级编程模型
3. **RegBase (MicroAPI)**：新的低级编程范式
4. **Cube-Vector 直连通路**：UB2L1, L0C2UB
5. **更严格的核间同步**：CrossCoreSetFlag/WaitFlag 必须严格配对

## 四、A5 迁移关键注意事项

### 4.1 核间同步

A5 要求 `CrossCoreSetFlag` 和 `CrossCoreWaitFlag` 严格 1:1 配对：
- A2 上的冗余 SetFlag 在 A5 上可能导致死锁
- 需要仔细审查所有同步点

### 4.2 Tiling 策略

- A5 的 L0C 容量翻倍（128KB → 256KB）
- 沿用 A2 的 tiling 参数可能导致性能浪费
- 需要根据 A5 硬件参数重新调优

### 4.3 数据类型

- A5 支持 FP8 (E4M3/E5M2)、FP4 (E2M1/E1M2)、FP8_E8M0 等新数据类型
- A5 不支持 `__NPU_ARCH__ == 3003` 或 `3113`（这些是不支持 BF16 的老平台）

### 4.4 算子定义差异

A5 的 OpAICoreConfig 需要独立配置：
```cpp
OpAICoreConfig config950;
config950.ExtendCfgInfo("prebuildPattern.value", "Opaque");
config950.ExtendCfgInfo("coreType.value", "AiCore");
config950.ExtendCfgInfo("opFile.value", "op_name_apt");
this->AICore().AddConfig("ascend950", config950);
```

### 4.5 A5 Kernel 编译条件

```cpp
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    // A5 kernel code
#endif
```

## 五、CMake 构建差异

```cmake
# 平台判断
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    # A5 特殊配置
    set(ARCH_DIRECTORY "arch35")
else()
    set(ARCH_DIRECTORY "arch32")
endif()

# A5 编译选项
target_compile_definitions(op_kernel PRIVATE
    -DENABLE_CV_COMM_VIA_SSBUF=true  # A5 Cube-Vector 通信
)
```
