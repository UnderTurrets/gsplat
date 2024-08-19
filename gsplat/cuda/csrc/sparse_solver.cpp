//
// Created by cvgluser on 24-8-9.
//
#include "bindings.h"
#include <cusolverDn.h>
#include <cuda_runtime.h>

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

torch::Tensor sparse_coo_slice_to_dense(const torch::Tensor& coo_tensor,
                                        const std::pair<int64_t, int64_t> row_range,
                                        const std::pair<int64_t, int64_t> col_range) {
    // 提取COO张量的indices和values
    auto indices = coo_tensor._indices();
    auto values = coo_tensor._values();

    // 提取切片的行和列范围
    int64_t row_start = row_range.first;
    int64_t row_end = row_range.second;
    int64_t col_start = col_range.first;
    int64_t col_end = col_range.second;

    // 创建掩码来筛选符合行和列范围的元素
    auto row_mask = (indices[0] >= row_start) & (indices[0] < row_end);
    auto col_mask = (indices[1] >= col_start) & (indices[1] < col_end);
    auto mask = row_mask & col_mask;

    // 根据掩码筛选出符合条件的索引和值
    auto new_indices = indices.index({torch::indexing::Ellipsis, mask});
    auto new_values = values.index({mask});

    // 更新索引以反映新的坐标系
    new_indices[0] -= row_start;
    new_indices[1] -= col_start;

    // 切片后的矩阵形状
    auto new_shape = std::vector<int64_t>{row_end - row_start, col_end - col_start};

    // 创建新的COO稀疏张量
    auto sliced_tensor = torch::sparse_coo_tensor(new_indices, new_values, new_shape, coo_tensor.options());

    return sliced_tensor.to_dense();
}

torch::Tensor
parallelize_sparse_matrix(const torch::Tensor& A, const torch::Tensor& b, const uint32_t block_size) {
    DEVICE_GUARD(A);
    CHECK_CUDA(A);
    CHECK_CUDA(b);
    uint32_t dim = b.size(0);
    torch::Tensor solve = torch::empty_like(b,b.options());

    // cusolverDnHandle_t cusolverH;
    // cusolverDnCreate(&cusolverH);

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
        torch::Tensor A_sub;
        if (A.layout() == torch::kSparse) {
            A_sub = sparse_coo_slice_to_dense(A, {current_idx, end_idx}, {current_idx, end_idx});
        }else if (A.layout() == torch::kStrided) {
            A_sub = A.index({
                torch::indexing::Slice(current_idx, end_idx),
                torch::indexing::Slice(current_idx, end_idx)});
        }
        torch::Tensor b_sub = b.index({torch::indexing::Slice(current_idx,end_idx)});

        // method1: use torch
        torch::Tensor solve_sub = torch::linalg::solve(A_sub, b_sub,true);

        // method2: use cusolver
        // solve_block_with_cusolver(
        //     cusolverH,
        //     A_sub.data_ptr<float>(),
        //     b_sub.data_ptr<float>(),
        //     block_size
        // );

        solve.index({torch::indexing::Slice(current_idx, end_idx)}) = solve_sub.flatten();
        current_idx += block_size;
    }
    // cusolverDnDestroy(cusolverH);
    return solve;
}

