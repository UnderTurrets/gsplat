cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_FLAGS_DEBUG -g)
set(CMAKE_CUDA_FLAGS_MINSIZEREL -O1 -DNDEBUG)
set(CMAKE_CUDA_FLAGS_RELEASE -O3 -DNDEBUG)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO -O2 -g -DNDEBUG)



project(gsplat LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_PREFIX_PATH /home/cvgluser/Downloads/libtorch-cxx11-abi-shared-with-deps-2.3.1+cu118/libtorch;
)

file(GLOB source
        "*.cu"
        "*.cuh"
        "*.cpp"
        )


find_package(Torch REQUIRED)
add_library(${PROJECT_NAME} SHARED
        ${source}
)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "70;75;86")
target_include_directories(${PROJECT_NAME} PUBLIC /usr/include/python3.8)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
