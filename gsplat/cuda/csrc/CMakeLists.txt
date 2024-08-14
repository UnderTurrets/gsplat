cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(gsplat LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置 CUDA 编译器和架构
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES "70;75;86")

# 设置 Caffe2 使用 CUDNN 和 CUSPARSELT
set(CAFFE2_USE_CUDNN 1)
set(CAFFE2_USE_CUSPARSELT 1)

# 设置 Torch 的路径
set(CMAKE_PREFIX_PATH "/home/cvgluser/Downloads/libtorch-cxx11-abi-shared-with-deps-2.3.1+cu121/libtorch")

# 查找 Torch 库
find_package(Torch REQUIRED)
find_package(glm REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

# 搜索源文件
file(GLOB source "*.h" "*.cu" "*.cuh" "*.cpp")
list(REMOVE_ITEM ${source} "ext.cpp")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 创建可执行文件
add_library(${PROJECT_NAME} ${source})

# 设置 CUDA 架构
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70;75;86")

# 链接必要的库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${TORCH_LIBRARIES}
    glm
    cusolver
    cudart
    cublas
    Python3::Python
)

# 设置 PyTorch 包的头文件路径
target_include_directories(${PROJECT_NAME} PUBLIC
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${Python3_INCLUDE_DIRS}
)

add_executable(testSparse_solver test/testSparse_solver.cpp)
target_link_directories(testSparse_solver PUBLIC ${PROJECT_NAME})
target_link_libraries(testSparse_solver ${PROJECT_NAME})