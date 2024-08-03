# 从环境模块中导入成本因子类和求解器类
from Environment.Base import CostFactor
from Environment.CostFactor import CostFactor_1DGS
from Environment.Classical_Solver import Classical_NLS_Solver, Classical_GD_Solver
import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
from torch import tensor
from tqdm import tqdm
from typing import List, Optional, Union
import math

Params = List[tensor]

# 设定实验的轮数
epoch = 10
data = []

def costFunc3_adahessian_optimize(params: Params, observes: tensor, lr: Optional[float] = None,
                                  max_iterations: int = 2000, ):
    import torch_optimizer
    if lr is not None:
        opt = torch_optimizer.Adahessian(params, lr=lr)
    else:
        opt = torch_optimizer.Adahessian(params)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.01 ** (1 / max_iterations))
    loss_history = []
    for i in tqdm(range(max_iterations)):
        def closure() -> float:
            opt.zero_grad()
            A = torch.cat(tensors=[params[0], params[1]], dim=0)[:, 0:2]
            B = torch.cat(tensors=[params[0], params[1]], dim=0)[:, 2]
            residual = observes[:, 2:4] - (observes[:, 0:2] @ A.t() + B.t())
            _weights = torch.ones(size=(cost_factor.obs_dim,)).repeat(cost_factor.residual_dim)
            loss = 0.5 * torch.sum(torch.square(residual * _weights))
            loss.backward(create_graph=True)
            return loss.item()

        loss = opt.step(closure)
        scheduler.step()
        loss_history.append(loss)

    plt.title('Adahessian')
    plt.xlabel('Iteration')
    plt.plot(loss_history, label='loss')
    plt.legend()
    plt.show()
    return

def costFunc3_adam_optimize(params: Params, observes: tensor, lr: Optional[float] = None,
                            max_iterations: int = 2000, ):
    if lr is not None:
        opt = torch.optim.Adam(params, lr=lr)
    else:
        opt = torch.optim.Adam(params)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.01 ** (1 / max_iterations))
    loss_history = []
    for i in tqdm(range(max_iterations)):
        opt.zero_grad()
        A = torch.cat(tensors=[params[0], params[1]], dim=0)[:, 0:2]
        B = torch.cat(tensors=[params[0], params[1]], dim=0)[:, 2]
        residual = observes[:, 2:4] - (observes[:, 0:2] @ A.t() + B.t())
        _weights = torch.ones(size=(cost_factor.obs_dim,)).repeat(cost_factor.residual_dim)
        loss = 0.5 * torch.sum(torch.square(residual * _weights))
        loss.backward()
        opt.step()
        scheduler.step()
        loss_history.append(loss.item())

    plt.title(fr'Adam,lr={lr}')
    plt.xlabel('Iteration')
    plt.plot(loss_history, label='loss')
    plt.legend()
    plt.show()
    return

def costFunc1DGS_adam_optimize(costF: CostFactor_1DGS, lr: Optional[float] = None, max_iterations: int = 2000, show_process:bool=False):
    observes = torch.tensor(data=costF.obs, requires_grad=True, dtype=torch.float32)

    #shape: (gaussian_num,)
    means, variances, opacity = costF.get_parameters()
    means = torch.tensor(data=means, requires_grad=True, dtype=torch.float32)
    variances = torch.tensor(data=variances, requires_grad=True, dtype=torch.float32)
    opacity = torch.tensor(data=opacity, requires_grad=True, dtype=torch.float32)

    if lr is not None:
        opt = torch.optim.Adam([means, variances, opacity], lr=lr)
    else:
        opt = torch.optim.Adam([means, variances, opacity])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.01 ** (1 / max_iterations))
    loss_history = []
    panr_history = []
    iteration_speed_history = []
    y_data = torch.zeros_like(input=observes[:, 1])
    gaussian_num = means.shape[0]

    if show_process:
        plt.figure()
        plt.ion()
        plt.show()
    pbar = tqdm(range(max_iterations))
    for i in pbar:
        opt.zero_grad(set_to_none=True)
        # compute residual
        y_data_list = []
        for i in range(gaussian_num):
            y_data_list.append(opacity[i] *torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]))
        y_data_stack = torch.stack(y_data_list, dim=0)
        y_data = torch.sum(y_data_stack, dim=0)
        residual = observes[:, 1] - y_data

        ##=========================test autograd========================
        # start_time = time.time()
        # _jacobian = costF.jacobian()
        # jacobian_auto = torch.empty_like(torch.from_numpy(_jacobian))
        # for i in range(residual.shape[0]):
        #     jacobian_auto[i, :] = torch.autograd.grad(residual[i], params, retain_graph=True)[0]
        # autograd_time = time.time() - start_time
        # print(autograd_time) # 14s
        #
        # def get_residual(param):
        #     means = param.reshape(-1, 3)[:, 0]
        #     variances_log = param.reshape(-1, 3)[:, 1]
        #     opacity_logic = param.reshape(-1, 3)[:, 2]
        #     variances = torch.exp(variances_log)
        #     opacity = torch.sigmoid(opacity_logic)
        #     # compute residual
        #     y_data = torch.empty_like(input=observes[:, 1])
        #     for i in range(gaussian_num):
        #         y_data += opacity[i] * torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i])
        #     return observes[:, 1] - y_data
        # start_time = time.time()
        # jacobian_auto2 = torch.autograd.functional.jacobian(get_residual, params)
        # autograd2_time = time.time() - start_time
        # print(autograd2_time) # 14s
        ##=========================test autograd========================

        if show_process:
            plt.clf()
            plt.subplot(3, 1, 1)
            plt.plot(observes[:, 0].detach().numpy(), observes[:, 1].detach().numpy(), label='origin')
            plt.plot(observes[:, 0].detach().numpy(), y_data.detach().numpy(), label='reconstructed')
            plt.title(f'Adam,lr={lr}')
            plt.xlabel('x')
            plt.legend()
            plt.subplot(3, 1, 2)
            for single_y_data in y_data_list:
                plt.plot(observes[:, 0].detach().numpy(), single_y_data.detach().numpy(), linewidth=0.5)
            plt.subplot(3, 1, 3)
            plt.plot(panr_history, label='psnr')
            plt.legend()
            plt.xlabel('iteration')
            plt.pause(0.01)

        _weights = torch.ones(size=(costF.obs_dim,)).repeat(costF.residual_dim)
        loss = 0.5 * torch.sum(torch.square(residual * _weights))
        loss.backward()
        mse = torch.mean((observes[:, 1] - y_data) ** 2)
        psnr = 10 * torch.log10(torch.max(observes[:, 1]).item()**2 / mse)
        loss_history.append(loss.item())
        panr_history.append(psnr.item())
        desc = f"adam"
        pbar.set_description(desc)
        if pbar.format_dict["rate"] is not None:
            iteration_speed_history.append(pbar.format_dict["rate"])

        opt.step()
        scheduler.step()

    average_iteration_speed = None
    if len(iteration_speed_history) != 0:
        average_iteration_speed = numpy.array(iteration_speed_history).mean()
    return average_iteration_speed

if __name__ == '__main__':
    # 进行多次实验
    for e_i in range(epoch):
        cost_factor = CostFactor_1DGS(gaussian_num=50, is_great_init=False, parameter_space=0)
        optimizer_nls = Classical_NLS_Solver()
        optimizer_nls_sparse = Classical_NLS_Solver()
        optimizer_gd = Classical_GD_Solver(max_iter=100, tolX=1e-6, tolOpt=1e-8, tolFun=1e-6)

        ## adam
        lr = 5e-3
        costFunc1DGS_adam_optimize(cost_factor, max_iterations=1000, lr=lr, show_process=True)

        ## LM
        # start_time = time.time()
        # _, _, LM_solveTime= optimizer_nls.solve(cost_factor, show_process=True, show_result=True)
        # end_time = time.time()
        # data.append({'Method': 'naive nls', 'Iteration': optimizer_nls.iteration, 'Time': end_time - start_time,
        #              'Epoch': f'Epoch {e_i}'})

        ## sparse LM
        # start_time = time.time()
        # _, _, SparseLM_solveTime = optimizer_nls_sparse.solve(cost_factor, block_size=49, show_process=True, show_result=True)
        # end_time = time.time()
        # data.append({'Method': 'naive nls sparse', 'Iteration': optimizer_nls_sparse.iteration, 'Time': end_time - start_time,
        #              'Epoch': f'Epoch {e_i}'})

        ## GD
        # start_time = time.time()
        # optimizer_gd.solve(cost_factor, show_process=True)
        # end_time = time.time()
        # data.append({'Method': 'naive gd', 'Iteration': optimizer_gd.iteration, 'Time': end_time - start_time,
        #              'Epoch': f'Epoch {e_i}'})

    df = pd.DataFrame(data)

    # 绘制不同方法的迭代次数比较图
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Method', y='Iteration', hue='Method', data=df)
    plt.title('Iterations Comparison Across Different Methods')
    plt.xlabel('Method')
    plt.ylabel('Iterations')
    plt.show()

    # 绘制不同方法的时间比较图
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Method', y='Time', hue='Method', data=df)
    plt.title('Time Comparison Across Different Methods')
    plt.xlabel('Method')
    plt.ylabel('Time')
    plt.show()
