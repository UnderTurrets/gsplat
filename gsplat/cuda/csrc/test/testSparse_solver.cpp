#include "bindings.h"
#include <chrono>

int main() {
    int dim = 5000;

    // 创建一个随机的对称正定矩阵 A (dim x dim)
    torch::Tensor A = torch::randn({dim, dim}, torch::kCUDA);
    A = torch::mm(A, A.t()) + torch::eye(dim, torch::kCUDA) * dim;  // 生成对称正定矩阵

    torch::Tensor b = torch::randn(dim, torch::kCUDA);

    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        torch::Tensor factor_solve = torch::linalg::solve(A, b,true);
        // torch::Tensor solve = parallelize_sparse_matrix(A, b, block_size);
    }
    auto sparse_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start2).count();

    std::cout << "Sparse time: " << sparse_time << " ms" << std::endl;

    // at::Scalar distance = torch::norm(solve - factor_solve, 2).item();
    // std::cout << distance <<std::endl;
    //
    // std::cout << factor_solve<<std::endl;
    // std::cout << solve<<std::endl;
    return 0;
}
