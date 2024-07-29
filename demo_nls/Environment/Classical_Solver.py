import matplotlib.pyplot as plt
import numpy
import numpy as np
import copy
from tqdm import tqdm
from Environment.Base import CostFactor, SolverFactor

# 非线性最小二乘法（NLS）求解器类
class Classical_NLS_Solver(SolverFactor):
    def __init__(self, optimizer_type="LM", max_iter=2000, tolX=1e-3, tolOpt=1e-5, tolFun=1e-3):
        super().__init__(optimizer_type, max_iter=max_iter, tolX=tolX, tolOpt=tolOpt, tolFun=tolFun)
        self.tou = 1e-2
        self.epsilon = 1e-8
        self.iteration = 0  # 初始化迭代次数

    def solve(self, cost_factor: CostFactor, weights=None, show_process: bool = False):
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
            plt.plot(varrho_history,label='varrho')
            plt.title(f'LM,gaussian_num={cost_factor.gaussian_num:}')
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

        if show_process:
            plt.figure()
            plt.ion()
            plt.show()
        pbar = tqdm(range(self.max_iter))
        for iterations in pbar:
            loss_history.append(cost_factor.error(weights))
            miu_history.append(miu)
            nu_history.append(nu)
            # for 1DGS
            psnr_history.append(cost_factor.calculate_psnr())

            ## ============================================test on torch.autograd.grad======================================================
            # from .LMoptimizer import LevenbergMarquardt
            # import torch
            # x = torch.tensor(cost_factor.x, requires_grad=True, dtype=torch.float32)
            # obs = torch.tensor(cost_factor.obs, requires_grad=True, dtype=torch.float32)
            # residual = obs[:, 2:4] - (obs[:, 0:2] @ x.reshape(2, 3)[:, :2].t() + x.reshape(2, 3)[:, 2].t())
            # _weights = torch.ones(size=(cost_factor.obs_dim, 1)).expand(-1, cost_factor.residual_dim)
            # loss = 0.5 * torch.sum(torch.square(residual * _weights))
            # loss.backward(create_graph=True)
            ## they equals
            # print(x.grad)
            # print(g)
            # print(torch.autograd.grad(loss, x))

            ## they equals
            # print(cost_factor._jacobian)
            # def cost1(x):
            #     residual = obs[:, 2:4] - (obs[:, 0:2] @ x[:4].reshape(2, 2).t() + x[4:].t())
            #     # weights = torch.ones(size=(cost_factor.obs_dim, 1)).expand(-1, cost_factor.residual_dim)
            #     # loss = 0.5 * torch.sum(torch.square(residual * weights))
            #     return residual
            # print(torch.autograd.functional.jacobian(cost1, x))

            ## Hessian.
            ## they equals
            # print(A)
            # def cost2(x):
            #     residual = obs[:, 2:4] - (obs[:, 0:2] @ x[:4].reshape(2, 2).t() + x[4:].t())
            #     weights = torch.ones(size=(cost_factor.obs_dim, 1)).expand(-1, cost_factor.residual_dim)
            #     loss = 0.5 * torch.sum(torch.square(residual * weights))
            #     return loss
            # print(torch.autograd.functional.hessian(cost2,x))

            ## test LMoptimizer
            # a11, a12, b1 = cost_factor.x[0], cost_factor.x[1], cost_factor.x[2]
            # a21, a22, b2 = cost_factor.x[3], cost_factor.x[4], cost_factor.x[5]
            # param1 = torch.tensor(data=[[a11, a12, b1]], requires_grad=True, dtype=torch.float32)
            # param2 = torch.tensor(data=[[a21, a22, b2]], requires_grad=True, dtype=torch.float32)
            # import torch_optimizer.adahessian as adahessian
            # opt = LMoptimizer.LevenbergMarquardt([param1, param2],is_estimate=False)
            # def closure():
            #     opt.zero_grad()
            #     obs = torch.tensor(cost_factor.obs, requires_grad=True, dtype=torch.float32)
            #     A = torch.cat(tensors=[param1, param2], dim=0)[:, 0:2]
            #     B = torch.cat(tensors=[param1, param2], dim=0)[:, 2]
            #     residual = obs[:, 2:4] - (obs[:, 0:2] @ A.t() + B.t())
            #     _weights = torch.ones(size=(cost_factor.obs_dim, 1)).expand(-1, cost_factor.residual_dim)
            #     loss = 0.5 * torch.sum(torch.square(residual * _weights))
            #     loss.backward(create_graph=True)
            #     return loss
            # opt.step(closure)
            ## ============================================test======================================================

            self.iteration = iterations  # 记录当前迭代次数

            # 计算更新步长
            update = - np.linalg.inv(hessian + miu * numpy.eye(cost_factor.x_dim)) @ gradient
            if np.linalg.norm(update) < self.epsilon * (np.linalg.norm(cost_factor.x) + self.epsilon):
                print('update is low so stop')
                break

            # 创建参数更新后的假设因子
            cost_factor_hypothesis = copy.deepcopy(cost_factor)
            cost_factor_hypothesis.step(update)

            ## 计算误差比率
            varrho = (
                     (cost_factor.error(weights)) - cost_factor_hypothesis.error(weights)  /
                     (0.5 * np.array(update).T @ (miu * np.array(update)-cost_factor.gradient(weights)))
            )

            ## If use this method, other check shoule be added or iterations will stop slowly.
            # varrho = (cost_factor_hypothesis.error(weights) - cost_factor.error(weights)) / (
            #         np.array(update).T @ cost_factor_hypothesis.gradient(weights))
            ## other check
            # if np.linalg.norm(gradient, ord=np.inf) < self.tolOpt:
            #     print('grad is low so stop')
            #     break
            # if np.linalg.norm(update) < self.tolX * max(1, np.linalg.norm(cost_factor.x)):
            #     print('update is low so stop')
            #     break
            # if abs(cost_factor_hypothesis.error(weights) - cost_factor.error(weights)) < self.tolFun * max(1, abs(cost_factor.error(weights))):
            #     print('error is low so stop')
            #     break

            varrho_history.append(varrho)
            if show_process:
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
                plt.plot(loss_history,label="loss")
                plt.legend()
                plt.xlabel("iteration")
                plt.pause(0.25)

            # 根据误差比率调整参数
            if varrho > 0:
                cost_factor = copy.deepcopy(cost_factor_hypothesis)
                hessian = cost_factor.hessian(weights)
                gradient = cost_factor.gradient(weights)
                if (numpy.linalg.norm(cost_factor.gradient(weights), ord=np.inf) <= self.epsilon):break
                miu = miu * max(1 / 3, 1 - (2 * varrho - 1) ** 3)
                nu = 2
            else:
                miu = miu * nu
                nu = nu * 2
            desc = f"LM | iteration:{iterations}"
            pbar.set_description(desc)
            if pbar.format_dict["rate"] is not None:
                iteration_speed_history.append(pbar.format_dict["rate"])

        draw_1DGS()
        if len(iteration_speed_history)!=0:
            return cost_factor, numpy.array(iteration_speed_history).mean()
        else:
            return cost_factor, []


# 梯度下降（GD）求解器类
class Classical_GD_Solver(SolverFactor):
    def __init__(self, optimizer_type="GD", max_iter=2000, tolX=1e-5, tolOpt=1e-6, tolFun=1e-5):
        super().__init__(optimizer_type, max_iter, tolX, tolOpt, tolFun)  # 调用父类的初始化方法，设置优化器类型
        self.lr = 1e-2  # 设置学习率
        self.iteration = 0  # 初始化迭代次数

    def solve(self, cost_factor: CostFactor, weights=None, show_process: bool = False):
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
            plt.figure()
            plt.ion()
            plt.show()
        # 迭代求解
        pbar = tqdm(range(self.max_iter))
        for iterations in pbar:
            if show_process:
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
            if abs(cost_factor_hypothesis.error(weights) - cost_factor.error(weights)) < self.tolFun * max(1,abs(cost_factor.error(weights))):
                print("error is low so stop")
                break
            cost_factor = copy.deepcopy(cost_factor_hypothesis)

            desc = f"GD | iteration:{iterations}"
            pbar.set_description(desc)
            if pbar.format_dict["rate"] is not None:
                iteration_speed_history.append(pbar.format_dict["rate"])

        draw_1DGS()
        if len(iteration_speed_history) != 0:
            return cost_factor, numpy.array(iteration_speed_history).mean()
        else:
            return cost_factor, []


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
