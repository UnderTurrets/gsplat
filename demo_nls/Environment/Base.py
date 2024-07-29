import numpy as np
import random
import copy

# x:        list     [1, x_dim]           - 参数列表
# obs:      np array [obs_dim, measurement_dim]  - 观测值数组
# residual: np array [obs_dim, residual_dim]     - 残差数组
# jacobian: np array [obs_dim * residual_dim, x_dim]  - 雅可比矩阵
# error:    scalar                              - 误差标量
# gradient: np array [x_dim, 1]                 - 梯度数组
class CostFactor:
    def __init__(self, numerical_J, x_gt, x, obs, residual_dim):
        self.numerical_J = numerical_J  # 是否使用数值雅可比矩阵


        self.x_gt = x_gt  # 真实参数值
        self.x = x  # 初始化参数
        self.obs = obs  # 观测数据
        self.obs_dim = self.obs.shape[0]  # 观测数据的数量
        self.x_dim = len(self.x)  # 参数的维度
        self.residual_dim = residual_dim  # 残差的维度
        self._jacobian = np.zeros(shape=(self.obs_dim * self.residual_dim, self.x_dim))
        self.update()  # 更新残差和雅可比矩阵


    # 重载的方法，用于计算雅可比矩阵
    def jacobian_factor(self):
        pass

    # 重载的方法，用于计算残差
    def residual_factor(self):
        pass

    # 计算加权残差
    def residual(self, weights_sqrt=None):
        if weights_sqrt is None:
            weights_sqrt = np.ones([self.obs_dim, 1])
        return self._residual.reshape(self.obs_dim, self.residual_dim) * weights_sqrt.reshape(self.obs_dim, 1)

    # 计算加权残差的平方和
    def residuals(self, weights_sqrt=None):
        if weights_sqrt is None:
            weights_sqrt = np.ones([self.obs_dim, 1])

        weights = weights_sqrt.reshape([self.obs_dim, 1])
        if self.residual_dim == 1:
            residuals = np.square(self._residual.reshape([self.obs_dim, 1]) * weights)
        else:
            residuals = np.sum(np.square(self._residual.reshape([self.obs_dim, self.residual_dim]) * weights), axis=1)
        return residuals.reshape([self.obs_dim, 1])

    # 计算加权雅可比矩阵
    def jacobian(self, weights_sqrt=None):
        if weights_sqrt is None:
            weights_sqrt = np.ones([self.obs_dim, 1])
        weights_sqrt = np.repeat(weights_sqrt, self.residual_dim)
        return self._jacobian * weights_sqrt.reshape(self.obs_dim * self.residual_dim, 1)

    # 计算误差
    def error(self, weights=None):
        if weights is None:
            weights = np.ones([self.obs_dim, 1])
        weights = np.repeat(weights, self.residual_dim)
        return 0.5 * np.sum(np.square(self._residual.reshape(-1) * weights))

    # 计算梯度
    def gradient(self, weights=None):
        if weights is None:
            weights = np.ones([self.obs_dim, 1])
        weights = np.repeat(weights, self.residual_dim)
        return self._jacobian.T @ np.diag(weights.reshape(-1)) @ self._residual.reshape(-1, 1)

    # 计算近似Hessian矩阵
    def hessian(self, weights=None):
        if weights is None:
            weights = np.ones([self.obs_dim, 1])
        weights = np.repeat(weights, self.residual_dim)
        return self._jacobian.T @ np.diag(weights.reshape(-1)) @ self._jacobian

    # 更新残差和雅可比矩阵
    def update(self):
        self._residual = self.residual_factor()
        # 使用数值化雅可比矩阵
        if self.numerical_J:
            self._jacobian = np.zeros((self.residual_dim * self.obs_dim, self.x_dim))
            for i in range(self.x_dim):
                self.x[i] -= 1e-5
                residual_1 = self.residual_factor().reshape(-1)
                self.x[i] += 2e-5
                residual_2 = self.residual_factor().reshape(-1)
                self.x[i] -= 1e-5
                self._jacobian[:, i] = (residual_2 - residual_1) / 2e-5
        else:
            self._jacobian = self.jacobian_factor()

    # 更新参数
    def step(self, update):
        self.x += update.reshape(-1)
        self.update()
        return self.x

    # 计算参数与真实值之间的距离
    def gt_distance(self):
        return np.sum(np.square(np.array(self.x) - np.array(self.x_gt)))


class SolverFactor:
    def __init__(self, optimizer_type=None, max_iter=2000, tolX=1e-5, tolOpt=1e-6, tolFun=1e-5):
        self.optimizer_type = optimizer_type  # 优化器类型
        self.max_iter = max_iter  # 最大迭代次数
        self.tolX = tolX  # 参数变化的容忍度
        self.tolOpt = tolOpt  # 梯度的容忍度
        self.tolFun = tolFun  # 误差变化的容忍度

    def solve(self, cost_factor: CostFactor, weights):
        pass

