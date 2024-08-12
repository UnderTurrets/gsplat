//
// Created by cvgluser on 24-8-9.
//

#include "helpers.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cusolverDn.h>
#include <torch/extension.h>
#include "bindings.h"
namespace cg = cooperative_groups;

void solve_block_with_cusolver(cusolverDnHandle_t cusolverH, float* A_sub, float* b_sub, int block_size) {
    int* d_info;
    int* d_pivots;
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_pivots, block_size * sizeof(int));

    float* d_work;
    int lwork;
    cusolverDnSgetrf_bufferSize(cusolverH, block_size, block_size, A_sub, block_size, &lwork);
    cudaMalloc(&d_work, lwork * sizeof(float));

    cusolverDnSgetrf(cusolverH, block_size, block_size, A_sub, block_size, d_work, d_pivots, d_info);
    cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, block_size, 1, A_sub, block_size, d_pivots, b_sub, block_size, d_info);

    cudaFree(d_info);
    cudaFree(d_pivots);
    cudaFree(d_work);
}

torch::Tensor parallelize_sparse_matrix(const torch::Tensor& A, const torch::Tensor& b, const uint32_t block_size) {
    DEVICE_GUARD(A);
    CHECK_INPUT(A);
    CHECK_INPUT(b);
    uint32_t dim = b.size(0);
    torch::Tensor solve = torch::empty_like(b,b.options());
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    uint32_t current_idx = 0;
    uint32_t block_num = (dim-1) / block_size + 1;
    for(int i =0;i<block_num;i++) {
        uint32_t end_idx = 0;
        if (current_idx + block_size > dim) {
            end_idx = dim;
            current_idx = end_idx - block_size;
        }else {
            end_idx = current_idx + block_size;
        }
        torch::Tensor A_sub = A.index({
            torch::indexing::Slice(current_idx, end_idx),
            torch::indexing::Slice(current_idx, end_idx)});
        torch::Tensor b_sub = b.index({torch::indexing::Slice(current_idx,end_idx)});
        solve_block_with_cusolver(
            cusolverH,
            A_sub.data_ptr<float>(),
            b_sub.data_ptr<float>(),
            block_size
        );
        solve.index({torch::indexing::Slice(current_idx, end_idx)}) = b_sub;
        current_idx += block_size;
    }
    cusolverDnDestroy(cusolverH);
    return solve;
}

