//
// Created by cvgluser on 24-8-9.
//

#include "helpers.cuh"
#include <eigen3/Eigen/Core>
#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include "bindings.h"

template <class T>
__global__ void parallelize_sparse_matrix_kernel(
                                                const T* __restrict__ A,
                                                const T* __restrict__ b,
                                                const uint32_t dim,
                                                const uint32_t block_size,
                                                // output
                                                T* __restrict__ solve) {
    uint32_t idx = cooperative_groups::this_grid().thread_rank();
    uint32_t equation_num = (dim-1) / block_size +1;
    if (idx >= equation_num) {
        return;
    }else if (idx == equation_num - 1) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>A_sub(block_size, block_size);
        const uint32_t start_idx = (dim-block_size) * dim + (dim-block_size);
        for (uint32_t i = 0; i < block_size; i++) {
            for (uint32_t j = 0; j < block_size; j++) {
                A_sub(i,j) = A[start_idx + i * dim + j];
            }
        }
        Eigen::Matrix<T, Eigen::Dynamic, 1> b_sub(block_size,1);
        for (uint32_t i = 0; i < block_size; i++) {
            b_sub(i) = b[(dim-block_size) + i];
        }
        // 使用LU分解
        Eigen::Matrix<T, Eigen::Dynamic, 1> solve_sub = A_sub.lu().solve(b_sub);
        for (uint32_t i = 0; i < block_size; i++) {
            solve[(dim-block_size) + i] = solve_sub(i);
        }
    }else {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>A_sub(block_size,block_size);
        const uint32_t start_idx = idx * block_size * dim + idx * block_size;
        for (uint32_t i = 0; i < block_size; i++) {
            for (uint32_t j = 0; j < block_size; j++) {
                A_sub(i,j) = A[start_idx + i * dim + j];
            }
        }
        Eigen::Matrix<T, Eigen::Dynamic, 1> b_sub(block_size,1);
        for (uint32_t i = 0; i < block_size; i++) {
            b_sub(i) = b[idx * block_size + i];
        }
        // 使用LU分解
        Eigen::Matrix<T, Eigen::Dynamic, 1> solve_sub = A_sub.lu().solve(b_sub);
        for (uint32_t i = 0; i < block_size; i++) {
            solve[idx * block_size + i] = solve_sub(i);
        }
    }
}

namespace cg = cooperative_groups;
torch::Tensor parallelize_sparse_matrix(const torch::Tensor& A, const torch::Tensor& b, const uint32_t block_size) {
    DEVICE_GUARD(A);
    CHECK_INPUT(A);
    CHECK_INPUT(b);
    uint32_t dim = b.size(0);
    torch::Tensor solve = torch::empty_like(b,b.options());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t threads = (dim-1) / block_size + 1;
    parallelize_sparse_matrix_kernel<float><<<(threads - 1) / N_THREADS +1, N_THREADS, 0, stream>>>(
            A.data_ptr<float>(),
            b.data_ptr<float>(),
            dim,
            block_size,
            solve.data_ptr<float>()
    );
    return solve;
}

