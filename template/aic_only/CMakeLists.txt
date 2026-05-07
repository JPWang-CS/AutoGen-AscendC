# 纯 AIC (Cube) 算子 CMakeLists.txt

file(GLOB OP_HOST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/op_host/*.cpp")
file(GLOB OP_HOST_TILING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_host/op_tiling/arch32/*.cpp"
)

op_host_aclnn(op_name ${OP_HOST_SRC} ${OP_HOST_TILING_SRC})

add_bin_compile_target(op_name_kernel
    op_kernel/op_name.cpp
    ${ASCEND_COMPUTE_UNIT}
)

if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_bin_compile_target(op_name_kernel_apt
        op_kernel/op_name_apt.cpp
        ${ASCEND_COMPUTE_UNIT}
    )
endif()
