# 纯 AIV 算子 CMakeLists.txt

# Host 侧源文件
file(GLOB OP_HOST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp")
file(GLOB OP_HOST_TILING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling/arch32/*.cpp"
)

# 注册算子到 Host 侧
op_host_aclnn(op_name ${OP_HOST_SRC} ${OP_HOST_TILING_SRC})

# Kernel 编译 - arch32 (A2/A3)
add_bin_compile_target(op_name_kernel
    op_kernel/op_name.cpp
    ${ASCEND_COMPUTE_UNIT}
)

# Kernel 编译 - arch35 (A5) - 仅当目标包含 ascend950 时编译
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_bin_compile_target(op_name_kernel_apt
        op_kernel/op_name_apt.cpp
        ${ASCEND_COMPUTE_UNIT}
    )
endif()
