cmake_minimum_required(VERSION 3.20)  # You can adjust the minimum required version
  # Ti 2080 uses 75. V100 uses 70. RTX 4090 uses 89.
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_FLAGS_DEBUG -g)
set(CMAKE_CUDA_FLAGS_MINSIZEREL -O1 -DNDEBUG)
set(CMAKE_CUDA_FLAGS_RELEASE -O3 -DNDEBUG)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO -O2 -g -DNDEBUG)

project(gsplat CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_PREFIX_PATH /home/cvgluser/Downloads/libtorch-cxx11-abi-shared-with-deps-2.3.1+cu118/libtorch;
)
find_package(Torch REQUIRED)
include_directories(../../cuda/csrc/third_party/glm/)

# our library library
add_library(${PROJECT_NAME}
        forward.cuh
        forward.cu
        backward.cuh
        backward.cu
        bindings.h
        backward.cuh
        helpers.cuh
        config.h
        ext.cpp
        sh.cuh
)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_include_directories(gsplat PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/glm
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70;75;86")
