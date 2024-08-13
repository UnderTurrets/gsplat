#include <torch/extension.h>
#include "bindings.h"
// 定义一个检查函数，用于比较实际结果和预期结果是否接近
bool tensors_are_close(const torch::Tensor& a, const torch::Tensor& b, float tol = 1e-4) {
    return torch::allclose(a, b, tol);
}

int main() {
    int dim = 1000;  // 使用1000阶的矩阵
    int block_size = 100;  // 设置块大小

    // 创建一个随机的对称正定矩阵 A (dim x dim)
    torch::Tensor A = torch::randn({dim, dim}, torch::kCUDA);
    A = torch::mm(A, A.t()) + torch::eye(dim, torch::kCUDA) * dim;  // 生成对称正定矩阵

    // 创建一个随机的向量 b (dim)
    torch::Tensor b = torch::randn(dim, torch::kCUDA);
    torch::Tensor factor_solve = torch::linalg::solve(A, b.view({dim, 1}),true);

    // 调用parallelize_sparse_matrix来求解Ax=b
    torch::Tensor solve = parallelize_sparse_matrix(A, b, block_size);

    at::Scalar distance = torch::norm(solve - factor_solve, 2).item();
    std::cout << distance <<std::endl;
    return 0;
}
