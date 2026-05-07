# AscendC 标准算子目录结构

## 一、顶层目录

```
op_name/
├── CMakeLists.txt          # 算子构建配置
├── README.md               # 算子说明文档
├── docs/                   # API 文档
│   └── aclnnOpName.md
├── examples/               # 调用示例
│   ├── test_aclnn_op_name.cpp          # arch32 示例
│   └── arch35/                          # A5 示例
│       └── test_aclnn_op_name.cpp
├── op_host/                # Host 侧代码（CPU 执行）
│   ├── CMakeLists.txt
│   ├── op_name_def.cpp                  # 算子定义（输入/输出/属性/数据类型）
│   ├── op_name_infershape.cpp           # 输出 Shape 推导
│   ├── op_name_tiling.cpp               # Tiling 入口
│   ├── op_name_tiling.h                 # Tiling 数据结构定义
│   ├── op_name_base_tiling.cpp/.h       # Tiling 基类实现
│   └── op_tiling/                       # 平台特定 Tiling
│       ├── arch32/                      # A2/A3 Tiling
│       └── arch35/                      # A5 Tiling
├── op_kernel/              # Kernel 代码（NPU 执行）
│   ├── op_name.cpp                      # arch32 Kernel 入口
│   ├── op_name_apt.cpp                  # arch35 (A5) Kernel 入口
│   ├── op_name.h                        # arch32 Kernel 主实现
│   ├── op_name_utils.h                  # 工具函数
│   ├── arch32/                          # A2/A3 Kernel 细节
│   └── arch35/                          # A5 Kernel 细节
│       ├── op_name.h
│       ├── op_name_tiling_data.h
│       ├── op_name_tiling_key.h
│       └── common/                      # A5 公共模块
├── op_api/                 # API 层
│   ├── op_name.h/.cpp                   # l0op 接口
│   ├── aclnn_op_name.h/.cpp             # aclnn 接口（V1/V2/V3...）
│   └── op_name_950_checker.h            # A5 参数检查
├── op_graph/               # 图算子 Proto
│   └── op_name_proto.h
├── framework/              # ONNX 插件（可选）
└── tests/                  # 测试
    └── ut/
        └── op_host/
            ├── op_api/                  # L2 API 测试
            ├── test_op_name_tiling.cpp
            └── op_kernel/
                └── test_op_name.cpp
```

## 二、文件职责说明

### 2.1 op_host/op_name_def.cpp
- 定义算子类，继承 `OpDef`
- 声明输入/输出/属性的数据类型和格式
- 为不同平台配置 `OpAICoreConfig`
- 使用 `OP_ADD()` 注册

### 2.2 op_host/op_name_tiling.h
- 使用 `BEGIN_TILING_DATA_DEF` / `END_TILING_DATA_DEF` 定义 Tiling 数据结构
- 定义 CompileInfo 结构（平台参数）
- 使用 `REGISTER_TILING_DATA_CLASS` 注册

### 2.3 op_host/op_name_tiling.cpp
- Tiling 计算逻辑
- 读取平台信息、Shape、属性
- 计算切分参数、BlockDim、TilingKey
- 计算 Workspace 大小

### 2.4 op_kernel/op_name.cpp (arch32 入口)
- `__CCE_AICORE__ == 220` 条件编译
- 获取 Tiling 数据
- 设置 Kernel 类型
- 按 TilingKey 分发到不同实现

### 2.5 op_kernel/op_name_apt.cpp (arch35 入口)
- `__CCE_AICORE__ == 310` 条件编译
- 使用 Cgmct 框架
- 按 TilingKey 分发

## 三、CMakeLists.txt 模板

```cmake
# 算子顶层 CMakeLists.txt
file(GLOB OP_HOST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp")
file(GLOB OP_HOST_TILING_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling/*.cpp")

# 注册算子
op_host_aclnn(op_name ${OP_HOST_SRC} ${OP_HOST_TILING_SRC})

# Kernel 编译
add_bin_compile_target(op_name_kernel op_kernel/op_name.cpp ${ASCEND_COMPUTE_UNIT})
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_bin_compile_target(op_name_kernel_apt op_kernel/op_name_apt.cpp ${ASCEND_COMPUTE_UNIT})
endif()
```

## 四、命名规范

| 类型 | 风格 | 示例 |
|---|---|---|
| 目录名 | snake_case | `grouped_matmul_finalize_routing` |
| 算子类名(def) | PascalCase | `GroupedMatmulFinalizeRouting` |
| Kernel 类名 | PascalCase | `QuantGroupMatmul` |
| Tiling 数据 | PascalCase + TilingData | `GroupMatmulFRTilingData` |
| 文件名 | snake_case | `op_name_tiling.cpp` |
| 常量 | UPPER_SNAKE_CASE | `BUFFER_NUM`, `SYNC_AIC_TO_AIV` |
| 成员变量 | 末尾下划线 | `pipe_`, `xGm_`, `tilingData_` |
| 宏 | UPPER_SNAKE_CASE | `GMMFR_A8W8_IMPL` |

## 五、版权头

每个源文件必须包含 CANN Open Software License 版权声明：
```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0 (the "License").
 * ...
 */
```
