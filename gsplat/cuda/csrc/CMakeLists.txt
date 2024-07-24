cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 编译器和架构
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES "70;75;86")

# 设置 CUDA 编译标志
set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# 设置 Caffe2 使用 CUDNN 和 CUSPARSELT
set(CAFFE2_USE_CUDNN 1)
set(CAFFE2_USE_CUSPARSELT 1)

# 设置 Torch 的路径
set(CMAKE_PREFIX_PATH "/home/cvgluser/Downloads/libtorch-cxx11-abi-shared-with-deps-2.3.1+cu121/libtorch")
#set(CMAKE_PREFIX_PATH /home/cvgluser/.conda/envs/gsplat/lib/python3.9/site-packages/torch/share/cmake;
#)

project(gsplat LANGUAGES CXX CUDA)
set(LIB_NAME gsplat-1.1.1)
# 查找 Torch 库
find_package(Torch REQUIRED)

# 设置 C++ 编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 搜索源文件
file(GLOB source
    "*.cu"
    "*.cuh"
    "*.cpp"
    "*.h"
)
list(REMOVE_ITEM ${source} "main.cpp")

# 由于v1.1.1改成了模版编程，此条命令可能会出现问题
add_library(${LIB_NAME} ${source})

# 设置属性
set_target_properties(${LIB_NAME} PROPERTIES CUDA_ARCHITECTURES "70;75;86")

# 包含目录
target_include_directories(${LIB_NAME} PUBLIC
    /usr/include/python3.8
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 链接 Torch 库
target_link_libraries(${LIB_NAME} ${TORCH_LIBRARIES})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}  ${LIB_NAME})

