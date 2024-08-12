import matplotlib.pyplot as plt
import numpy
import numpy as np
import copy
from tqdm import tqdm
from .Base import CostFactor, SolverFactor
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import Optional
import time

# 非线性最小二乘法（NLS）求解器类
class Classical_NLS_Solver(SolverFactor):
    def __init__(self, optimizer_type="LM", max_iter=2000, tolX=1e-4, tolOpt=1e-6, tolFun=1e-4):
        super().__init__(optimizer_type, max_iter=max_iter, tolX=tolX, tolOpt=tolOpt, tolFun=tolFun)
        self.tou = 1e-0
        self.epsilon = 1e-8
        self.iteration = 0  # 初始化迭代次数

    def solve(self, cost_factor: CostFactor, block_size: Optional[int] = -1, weights=None, show_process: bool = False, show_result: bool = False):
        # 如果权重未指定，则将其设置为全1数组
        if weights is None:
            weights = np.ones(cost_factor.obs_dim)

        hessian = cost_factor.hessian(weights)  # 计算Hessian矩阵
        gradient = cost_factor.gradient(weights)  # 计算梯度
        nu = 2  # 初始增量因子
        miu = self.tou * np.max(hessian.diagonal())  # 计算初始的调整参数
        loss_history = []
        miu_history = []
        nu_history = []
        varrho_history = []
        iteration_speed_history = []
        solve_equation_time_history = []
        # for 1DGS
        psnr_history = []
        def draw():
            fig = plt.figure(figsize=(6, 10))
            plt.subplot(2, 1, 1)
            plt.title(f'LM')
            plt.plot(miu_history, label='miu')
            plt.plot(nu_history, label='nu')
            plt.plot(varrho_history, label='varrho')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(loss_history, label='loss')
            plt.title(f"tolX={self.tolX:.2e}, tolOpt={self.tolOpt:.2e}, \n"
                      f"tolFun={self.tolFun:.2e}")
            plt.legend()
            plt.xlabel('iteration')
            plt.show()

        def draw_1DGS():
            fig = plt.figure(figsize=(6, 20))
            plt.subplot(4, 1, 1)
            plt.plot(miu_history, label='miu')
            plt.plot(nu_history, label='nu')
            plt.plot(varrho_history, label='varrho')
            plt.title(f'LM,gaussian_num={cost_factor.gaussian_num:}\n'
                      f'block size={block_size}')
            plt.legend()
            plt.subplot(4, 1, 2)
            plt.plot(loss_history, label='loss')
            plt.title(f"tolX={self.tolX:.2e}, tolOpt={self.tolOpt:.2e}, \n"
                      f"tolFun={self.tolFun:.2e}")
            plt.legend()
            # for 1DGS
            plt.subplot(4, 1, 3)
            plt.plot(psnr_history, label='psnr')
            plt.legend()
            plt.xlabel('iteration')
            plt.subplot(4, 1, 4)
            plt.plot(cost_factor.obs[:, 0], cost_factor.obs[:, 1], label='origin')
            plt.plot(cost_factor.obs[:, 0], cost_factor.reconstructed_signal(), label='reconstructed')
            plt.xlabel('x')
            plt.legend()
            fig.tight_layout()
            plt.show()

        if show_process:
            plt.figure(num="process_LM")
            plt.ion()
            plt.show()
        pbar = tqdm(range(self.max_iter))
        for iterations in pbar:
            iteration_time = time.time()
            loss_history.append(cost_factor.error(weights))
            miu_history.append(miu)
            nu_history.append(nu)
            # for 1DGS
            psnr_history.append(cost_factor.calculate_psnr())
            self.iteration = iterations  # 记录当前迭代次数
            A = hessian + miu * numpy.eye(cost_factor.x_dim)

            solve_equation_time = time.time()
            ## use sparse matrix to speed up processing of inverse
            if block_size > 0:
                update = np.zeros(shape=(cost_factor.x_dim, 1))
                current_idx = 0
                if show_process:
                    plt.figure(num="step")
                    plt.ion()
                    plt.show()
                    cost_factor_test = copy.deepcopy(cost_factor)
                for i in range((cost_factor.x_dim-1)//block_size + 1):
                    if current_idx + block_size > cost_factor.x_dim:
                        end_idx = cost_factor.x_dim
                        current_idx = end_idx - block_size
                    else:
                        end_idx = current_idx + block_size
                    A_sub = A[current_idx:end_idx, current_idx:end_idx]
                    gradient_sub = gradient[current_idx:end_idx, 0]
                    sub_update = - np.linalg.inv(A_sub) @ gradient_sub
                    # 将解的分块赋值给相应的 update 部分
                    update[current_idx:end_idx, 0] = sub_update
                    # if show_process:
                    #     test_update = np.zeros(shape=(cost_factor.x_dim, 1))
                    #     test_update[current_idx:end_idx, 0] = sub_update
                    #     cost_factor_test.step(test_update)
                    #     plt.figure(num="step")
                    #     plt.clf()
                    #     plt.plot(cost_factor_test.obs[:, 0], cost_factor_test.obs[:, 1], label='origin')
                    #     plt.plot(cost_factor_test.obs[:, 0], cost_factor_test.reconstructed_signal(), label='reconstructed')
                    #     plt.xlabel('x')
                    #     plt.legend()
                    #     plt.pause(0.01)

                    current_idx += block_size
            ## solve the equation directly
            else:
                update = - np.linalg.inv(A) @ gradient
            solve_equation_time = time.time() - solve_equation_time
            solve_equation_time_history.append(solve_equation_time)

            if np.linalg.norm(update) < self.epsilon * (np.linalg.norm(cost_factor.x) + self.epsilon):
                print('update is low so stop')
                break

            # 创建参数更新后的假设因子
            cost_factor_hypothesis = copy.deepcopy(cost_factor)
            cost_factor_hypothesis.step(update)
            ## 计算误差比率
            # varrho = (
            #         cost_factor.error(weights) - cost_factor_hypothesis.error(weights) /
            #         (0.5 * update.T @ (miu * update - gradient))
            # ).item()
            ## If use this method, other check shoule be added or iterations will stop slowly.
            varrho = ((cost_factor_hypothesis.error(weights) - cost_factor.error(weights)) /
                      (update.T @ cost_factor_hypothesis.gradient(weights))).item()
            # other check
            if np.linalg.norm(gradient, ord=np.inf) < self.tolOpt:
                print('grad is low so stop')
                break
            if np.linalg.norm(update) < self.tolX * max(1, np.linalg.norm(cost_factor.x)):
                print('update is low so stop')
                break
            if abs(cost_factor_hypothesis.error(weights) - cost_factor.error(weights)) < self.tolFun * max(1, abs(cost_factor.error(weights))):
                print('error is low so stop')
                break

            varrho_history.append(varrho)
            if show_process:
                plt.figure(num="process_LM")
                plt.clf()
                plt.subplot(4, 1, 1)
                plt.title(f'LM')
                plt.plot(cost_factor.obs[:, 0], cost_factor.obs[:, 1], label='origin')
                plt.plot(cost_factor.obs[:, 0], cost_factor.reconstructed_signal(), label='reconstructed')
                plt.xlabel('x')
                plt.legend()
                plt.subplot(4, 1, 2)
                for y_data in cost_factor.reconstructed_disperse():
                    plt.plot(cost_factor.obs[:, 0], y_data, linewidth=0.5)
                plt.subplot(4, 1, 3)
                plt.plot(miu_history, label="miu")
                plt.plot(nu_history, label="nu")
                plt.plot(varrho_history, label="varrho")
                plt.legend()
                plt.subplot(4, 1, 4)
                plt.plot(loss_history, label="loss")
                plt.legend()
                plt.xlabel("iteration")
                plt.pause(0.25)
            # 根据误差比率调整参数
            if varrho > 0:
                cost_factor = copy.deepcopy(cost_factor_hypothesis)
                hessian = cost_factor.hessian(weights)
                gradient = cost_factor.gradient(weights)
                if (numpy.linalg.norm(cost_factor.gradient(weights), ord=np.inf) <= self.epsilon): break
                miu = miu * max(1 / 3, 1 - (2 * varrho - 1) ** 3)
                nu = 2
            else:
                miu = miu * nu
                nu = nu * 2
            desc = f"LM | iteration:{iterations}"
            pbar.set_description(desc)
            iteration_time = time.time() - iteration_time
            iteration_speed_history.append(1/iteration_time)
        if show_process:
            plt.close("process_LM")
            plt.close("step")
        if show_result: draw_1DGS()
        return cost_factor, numpy.array(iteration_speed_history).mean(), numpy.array(solve_equation_time_history).mean()

# 梯度下降（GD）求解器类
class Classical_GD_Solver(SolverFactor):
    def __init__(self, optimizer_type="GD", max_iter=2000, tolX=1e-5, tolOpt=1e-6, tolFun=1e-5):
        super().__init__(optimizer_type, max_iter, tolX, tolOpt, tolFun)  # 调用父类的初始化方法，设置优化器类型
        self.lr = 1e-2  # 设置学习率
        self.iteration = 0  # 初始化迭代次数

    def solve(self, cost_factor: CostFactor, weights=None, show_process: bool = False, show_result: bool = False):
        # 如果权重未指定，则将其设置为全1数组
        if weights is None:
            weights = np.ones(cost_factor.obs_dim)

        loss_history = []
        iteration_speed_history = []
        # for 1DGS
        psnr_history = []

        def draw():
            plt.figure()
            plt.title(f'GD,tolX={self.tolX:.2e}, tolOpt={self.tolOpt:.2e}, tolFun={self.tolFun:.2e}')
            plt.plot(loss_history, label='loss')
            plt.xlabel('iteration')
            plt.show()

        def draw_1DGS():
            fig = plt.figure(figsize=(6, 12))
            plt.subplot(3, 1, 1)
            plt.plot(loss_history, label='loss')
            plt.legend()
            plt.title(f'GD,gaussian_num={cost_factor.gaussian_num:},\n'
                      f'tolX={self.tolX:.2e},'
                      f'tolOpt={self.tolOpt:.2e},'
                      f'tolFun={self.tolFun:.2e}')
            plt.subplot(3, 1, 2)
            plt.plot(psnr_history, label='psnr')
            plt.legend()
            plt.xlabel('iteration')
            plt.subplot(3, 1, 3)
            plt.plot(cost_factor.obs[:, 0], cost_factor.obs[:, 1], label='origin')
            plt.plot(cost_factor.obs[:, 0], cost_factor.reconstructed_signal(), label='reconstructed')
            plt.xlabel('x')
            plt.legend()
            fig.tight_layout()
            plt.show()

        if show_process:
            plt.figure(num="process_GD")
            plt.ion()
            plt.show()
        # 迭代求解
        pbar = tqdm(range(self.max_iter))
        for iterations in pbar:
            iteration_time = time.time()
            if show_process:
                plt.figure(num="process_GD")
                plt.cla()
                plt.title(f'GD')
                plt.plot(cost_factor.obs[:, 0], cost_factor.obs[:, 1], label='origin')
                plt.plot(cost_factor.obs[:, 0], cost_factor.reconstructed_signal(), label='reconstructed')
                plt.xlabel('x')
                plt.legend()
                plt.pause(0.001)

            self.iteration = iterations  # 记录当前迭代次数
            loss_history.append(cost_factor.error(weights))
            # for 1DGS
            psnr_history.append(cost_factor.calculate_psnr())

            g = cost_factor.gradient(weights)  # 计算梯度

            # 计算更新步长
            update = - self.lr * g
            cost_factor_hypothesis = copy.deepcopy(cost_factor)
            cost_factor_hypothesis.step(update)

            # 检查停止条件
            if np.linalg.norm(g, ord=np.inf) < self.tolOpt:
                print("gradient is low so stop")
                break
            if np.linalg.norm(update) < self.tolX * max(1, np.linalg.norm(cost_factor.x)):
                print("update is low so stop")
                break
            if abs(cost_factor_hypothesis.error(weights) - cost_factor.error(weights)) < self.tolFun * max(1,
                                                                                                           abs(cost_factor.error(
                                                                                                                   weights))):
                print("error is low so stop")
                break
            cost_factor = copy.deepcopy(cost_factor_hypothesis)

            desc = f"GD | iteration:{iterations}"
            pbar.set_description(desc)
            iteration_time = time.time() - iteration_time
            iteration_speed_history.append(1/iteration_time)
        if show_process: plt.close("process_GD")
        if show_result: draw_1DGS()
        average_iteration_speed = float('nan')
        if len(iteration_speed_history) != 0:
            average_iteration_speed = numpy.array(iteration_speed_history).mean()
        return cost_factor, average_iteration_speed


from Environment.CostFactor import CostFactor_Env1, CostFactor_Env2, CostFactor_Env3, CostFactor_Env4

# 主函数，测试NLS和GD求解器
if __name__ == "__main__":
    import sysconfig

    print(sysconfig.get_paths()['include'])

    optimizer = Classical_NLS_Solver()
    print("NLS solver example1")
    cost_factor = CostFactor_Env3()
    optimizer.solve(cost_factor)

    optimizer = Classical_GD_Solver()
    print("GD solver example1")
    cost_factor = CostFactor_Env3()
    optimizer.solve(cost_factor)
