# Plan A 修改代码说明

本目录包含 GMMFR 确定性特性 A5 迁移 (Plan A) 的所有修改代码。

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
│   ├── grouped_matmul_finalize_routing_tiling_data.h    ← 修改1: 添加确定性字段
│   ├── gmm_fr_deterministic_a5.h                        ← 新增: A5 确定性聚合函数
│   ├── grouped_matmul_finalize_routing_pertoken_dequant.h ← 修改: 添加确定性分支
│   ├── grouped_matmul_finalize_routing.h                ← 修改: 添加确定性分支
│   └── weight_quant_basic_block/
│       ├── gmm_fr_weight_quant_tiling_data.h            ← 修改2: 添加确定性字段
│       ├── gmm_fr_weight_quant_resplit_controller.h     ← 修改: 添加确定性分支
│       └── gmm_fr_weight_quant_vcv_basic_block.h        ← 修改: 确定性输出重定向
│
└── op_host/op_tiling/arch35/                   ← Tiling 侧修改
    ├── grouped_matmul_finalize_routing_quant_tiling.h     ← 修改: 声明确定性成员
    ├── grouped_matmul_finalize_routing_quant_tiling.cpp   ← 修改: 实现确定性 Tiling
    ├── grouped_matmul_finalize_routing_weight_quant_tiling.h ← 修改: 声明确定性成员
    └── grouped_matmul_finalize_routing_weight_quant_tiling.cpp ← 修改: 实现确定性 Tiling
```

## 使用说明

1. 先阅读 `a3_prototype/` 下的 A3 原型代码，理解确定性机制
2. 每个修改文件顶部都有 `/* MODIFICATION REASON */` 注释说明修改原因
3. 修改代码中用 `// <<< ADD BEGIN` 和 `// <<< ADD END` 标记新增部分
4. 用 `// <<< MODIFY BEGIN` 和 `// <<< MODIFY END` 标记修改部分
5. 将修改文件复制到 ops-transformer_AI 对应路径即可

## 未修改文件说明

以下文件明确不需要修改：

| 文件 | 原因 |
|---|---|
| op_host/grouped_matmul_finalize_routing_def.cpp | 算子定义，确定性是运行时行为，不影响输入/输出/属性 |
| op_host/grouped_matmul_finalize_routing_tiling.h | arch32 Tiling，A5 不使用 |
| op_host/grouped_matmul_finalize_routing_base_tiling.cpp/.h | arch32 Tiling 逻辑 |
| op_kernel/grouped_matmul_finalize_routing.cpp/.h | arch32 Kernel，A5 用 _apt.cpp |
| op_api/* | 接口层，workspace 框架自动管理 |
