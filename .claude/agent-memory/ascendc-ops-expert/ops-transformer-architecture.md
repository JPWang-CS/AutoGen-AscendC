---
name: ops-transformer 架构概览
description: ops-transformer 仓库的整体架构、目录结构、构建系统和平台适配机制
type: reference
---

# ops-transformer 架构概览

## 版本与依赖
- 版本: 9.0.0
- CANN 依赖: >=8.5 (runtime, opbase, hcomm, ge-executor, metadef, ge-compiler, asc-devkit, bisheng-compiler, asc-tools)
- 运行时依赖: 额外需要 ops-nn >=8.5, hccl >=8.5

## 顶层目录结构

```
ops-transformer_AI/
├── attention/         # Attention 类算子（FlashAttention, PromptFlashAttention, Sparse, NSA 等）
├── ffn/               # FFN 类算子
├── gmm/               # Grouped MatMul 类算子（GMM, GMM+SwiGLU, GMM FinalizeRouting, QuantGMM 等）
├── moe/               # MoE 类算子（InitRouting, TokenPermute, TopK, FinalizeRouting 等）
├── mc2/               # 通算融合类算子（MatMul+AllReduce, MatMul+ReduceScatter, AllGather+MatMul, MegaMoE 等）
├── mhc/               # MHC (Multi-Head Cross-attention) 类算子
├── posembedding/      # 位置编码类算子（RoPE, ApplyRotaryPosEmb 等）
├── common/            # 公共框架代码
├── experimental/      # 实验性算子（用户可在此贡献自定义算子）
├── torch_extension/   # PyTorch 扩展（npu_ops_transformer Python 包）
├── cmake/             # 构建系统 CMake 脚本
├── scripts/           # Kernel 构建脚本
├── tests/             # 测试框架和配置
├── examples/          # 端到端示例
├── docs/              # 文档
├── 3rdparty/          # 第三方依赖（catlass）
├── build.sh           # 主构建脚本（72KB，功能丰富）
└── CMakeLists.txt     # 顶层 CMake 入口
```

## 单算子标准目录结构

```
${op_class}/${op_name}/
├── CMakeLists.txt
├── README.md
├── docs/              # 算子文档（aclnn 接口说明）
├── examples/          # 调用示例（aclnn/GEIR）
├── op_host/           # Host 侧实现
│   ├── config/        # 平台配置（ascend910b/, ascend910_93/）
│   │   └── ${soc}/
│   │       ├── ${op}_binary.json
│   │       └── ${op}_simplified_key.ini
│   ├── op_tiling/     # 平台特定 Tiling
│   │   └── arch35/    # A5/950 平台 Tiling
│   ├── ${op}_def.cpp          # 算子定义（输入输出、数据类型）
│   ├── ${op}_infershape.cpp   # 形状推导
│   ├── ${op}_tiling.cpp/.h    # Tiling 实现
│   └── ${op}_base_tiling.cpp/.h # 基础 Tiling
├── op_kernel/         # Kernel 侧实现
│   ├── arch32/        # A2(910B)/A3(910C) 平台 Kernel
│   ├── arch35/        # A5(950) 平台 Kernel
│   ├── ${op}.cpp      # arch32 Kernel 入口（#if __CCE_AICORE__ == 220）
│   └── ${op}_apt.cpp  # arch35 Kernel 入口（#if __CCE_AICORE__ == 310）
├── op_api/            # aclnn 接口层
├── op_graph/          # 图融合实现
└── tests/             # 测试用例
```

## 平台适配机制

### SOC 与架构目录映射
| SOC | 架构目录 | 对应芯片 |
|-----|---------|---------|
| ascend310p | arch20 | Ascend 310P |
| ascend910b | arch32 (also arch22) | Atlas A2 (910B) |
| ascend910_93 | arch32 (also arch22) | Atlas A3 (910C) |
| ascend950 | arch35 | Ascend 950 (A5) |
| mc62cm12a | arch38 | MC62CM12A |
| kirinx90 | arch32 | Kirin X90 |
| kirin9030 | arch32 | Kirin 9030 |

### Kernel 代码分发
- `__CCE_AICORE__ == 220` -> arch32 代码路径（A2/A3）
- `__CCE_AICORE__ == 310` -> arch35 代码路径（A5）
- 入口文件分离：`*.cpp` 为 arch32 入口，`*_apt.cpp` 为 arch35 入口

### Tiling 代码分发
- 根目录 tiling 文件适用于 arch32
- `op_tiling/arch35/` 下的文件专用于 A5 平台
- 编译系统根据 `ASCEND_COMPUTE_UNIT` 变量自动选择

## 构建系统关键点

### build.sh 主要选项
```bash
./build.sh --pkg                     # 构建含 kernel bin 的运行包
./build.sh --jit                     # 构建 JIT 包（无 kernel bin）
./build.sh --soc=ascend910b          # 指定目标 SOC
./build.sh --soc=ascend910b,ascend950  # 多 SOC 编译
```

### 支持的 SOC 列表
ascend910b, ascend910_93, ascend950, ascend310p, kirinx90, kirin9030, mc62cm12a

### 关键 CMake 变量
- `ASCEND_COMPUTE_UNIT`: 目标 SOC（默认 ascend910b）
- `ASCEND_OP_NAME`: 要编译的算子（默认 ALL）
- `BUILD_OPEN_PROJECT`: 开源项目构建模式
- `ENABLE_EXPERIMENTAL`: 启用实验性算子
- `VENDOR_NAME`: 厂商名（默认 custom）

## Tiling 框架
- 基类: `Ops::Transformer::OpTiling::TilingBaseClass` (common/include/tiling_base/tiling_base.h)
- DoTiling 流程: GetShapeAttrsInfo -> GetPlatformInfo -> IsCapable -> DoOpTiling -> DoLibApiTiling -> GetWorkspaceSize -> PostTiling
- TilingKey 机制: 通过 uint64_t key 标识不同 Tiling 策略

## 公共基础设施
- `common/include/op_kernel/`: Kernel 侧公共头文件（mem.h, simd.h, pse.h, iterator.h 等）
- `common/include/tiling_base/`: Tiling 基类和工具
- `common/include/common/`: 通用工具
- `common/include/external/`: 外部接口封装
- `common/include/framework/`: ONNX 插件框架

## 算子分类统计
- Attention: ~50+ 子算子（flash_attention, prompt_flash_attention, sparse, NSA, MLA 等）
- GMM: 9 子算子（grouped_matmul 及变体）
- MoE: ~30+ 子算子（init_routing, token_permute, finalize_routing, topk 等）
- MC2: ~40+ 子算子（通算融合、通信+计算组合）
- PosEmbedding: ~13 子算子（RoPE 变体）
- FFN: ~7 子算子
- MHC: ~8 子算子
