# Plan A 修改代码说明

> 范围：仅 W8A8（INT8×INT8）PerToken 全量化路径
> 策略：Plan A（Params 重定向 y 地址到 workspace + FRDeterministicA5 延迟聚合）

本目录包含 GMMFR 确定性特性 A5 迁移 (Plan A) 的所有修改代码。
代码已同步到 `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/` 对应路径。

## 目录结构

```
plan_a_code/
├── README.md                                    ← 本文件
├── a3_prototype/                                ← A3 确定性原型代码参考
│   ├── a3_tiling_deterministic.cpp              ← A3 Tiling 侧确定性处理原型
│   ├── a3_tiling_data.h                         ← A3 Tiling 数据结构原型
│   ├── a3_frdeterministic.h                     ← A3 FRDeterministic 核心实现原型
│   ├── a3_vector_sync.h                         ← A3 VectorSync 滑动窗口原型
│   ├── a3_vector_atomic_deter.h                 ← A3 VectorAtomicProcess 确定性分支原型
│   └── a3_utils.h                               ← A3 工具函数和数据结构原型
│
├── op_kernel/arch35/                            ← Kernel 侧修改
│   ├── grouped_matmul_finalize_routing_tiling_data.h    ← 修改1: reserved2 → 确定性字段
│   ├── gmm_fr_deterministic_a5.h                        ← 新增: A5 确定性聚合函数
│   └── grouped_matmul_finalize_routing_pertoken_dequant.h ← 修改: 添加确定性分支
│
└── op_host/op_tiling/arch35/                   ← Tiling 侧修改
    ├── grouped_matmul_finalize_routing_quant_tiling.h     ← 修改: 声明确定性成员
    └── grouped_matmul_finalize_routing_quant_tiling.cpp   ← 修改: 实现确定性 Tiling
```

## 修改文件列表（W8A8 INT8 范围）

| # | 文件路径 | 操作 | 状态 |
|---|---|---|---|
| 1 | `op_kernel/arch35/grouped_matmul_finalize_routing_tiling_data.h` | 修改 reserved2 → deterministicFlag + deterWorkspaceSize | 已同步 |
| 2 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.h` | 声明确定性成员 | 已同步 |
| 3 | `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_quant_tiling.cpp` | DoOpTiling() + PrintQuantParams() | 已同步 |
| 4 | `op_kernel/arch35/gmm_fr_deterministic_a5.h` | **新增** A5 确定性聚合函数 | 已同步 |
| 5 | `op_kernel/arch35/grouped_matmul_finalize_routing_pertoken_dequant.h` | 添加 if/else 确定性分支 | 已同步 |

## 明确不修改的文件（W8A8 INT8 范围外）

| 文件 | 原因 |
|---|---|
| `op_kernel/arch35/grouped_matmul_finalize_routing.h` | **MX 格式路径**（FP8/FP4），不在 W8A8 范围内 |
| `op_kernel/arch35/weight_quant_basic_block/*` | **Weight Quant 伪量化路径**（FP8×FP4），不在范围内 |
| `op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.*` | **Weight Quant**，不在范围内 |
| `op_kernel/grouped_matmul_finalize_routing_apt.cpp` | 确定性分支在 pertoken_dequant.h 内部处理 |
| `op_host/grouped_matmul_finalize_routing_def.cpp` | 算子定义，确定性是运行时行为 |
| `op_kernel/grouped_matmul_finalize_routing.cpp/.h` | arch32 Kernel，A5 用 `_apt.cpp` |
| `op_api/*` | 接口层，workspace 框架自动管理 |

## 使用说明

1. 先阅读 `a3_prototype/` 下的 A3 原型代码，理解确定性机制
2. 每个修改文件顶部都有注释说明修改原因
3. 修改代码已同步到 ops-transformer_AI 对应路径，plan_a_code 为参考副本
4. 详细修改明细见 `code_modification_detail.md`
