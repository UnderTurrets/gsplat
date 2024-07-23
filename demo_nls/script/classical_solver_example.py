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

# 初始化非线性最小二乘求解器和梯度下降求解器
optimizer_nls = Classical_NLS_Solver(max_iter=2000, tolX=1e-4, tolOpt=1e-6, tolFun=1e-4)
optimizer_gd = Classical_GD_Solver(max_iter=2000, tolX=1e-5, tolOpt=1e-6, tolFun=1e-5)

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

    plt.title('Adam')
    plt.xlabel('Iteration')
    plt.plot(loss_history, label='loss')
    plt.legend()
    plt.show()


def costFunc1DGS_adam_optimize(costFactor: CostFactor, lr: Optional[float] = None, max_iterations: int = 2000):
    observes = torch.tensor(data=cost_factor.obs, requires_grad=True, dtype=torch.float32)
    params = torch.tensor(data=cost_factor.x, requires_grad=True, dtype=torch.float32)

    if lr is not None:
        opt = torch.optim.Adam([params], lr=lr)
    else:
        opt = torch.optim.Adam([params])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.01 ** (1 / max_iterations))
    loss_history = []
    panr_history = []
    y_data = torch.zeros_like(input=observes[:, 1])
    gaussian_num = params.reshape(-1, 3)[:, 0].size(0)
    for i in tqdm(range(max_iterations)):
        opt.zero_grad()

        # shape:(N,)
        means = params.reshape(-1, 3)[:, 0]
        variances = params.reshape(-1, 3)[:, 1]
        weight = params.reshape(-1, 3)[:, 2]

        # compute residual
        y_data = torch.zeros_like(input=observes[:, 1])
        for i in range(gaussian_num):
            y_data += (weight[i] /
                       (variances[i] * math.sqrt(2 * math.pi)) *
                       torch.exp(-((observes[:, 0] - means[i]) ** 2) /
                                 (2 * (variances[i]) ** 2)
                                 )
                       )
        residual = observes[:, 1] - y_data

        _weights = torch.ones(size=(cost_factor.obs_dim,)).repeat(cost_factor.residual_dim)
        loss = 0.5 * torch.sum(torch.square(residual * _weights))
        loss.backward()
        mse = torch.mean((observes[:, 1] - y_data) ** 2)
        psnr = 20 * torch.log10(max(torch.max(observes[:, 1]).item(), torch.max(y_data).item()) / torch.sqrt(mse))
        loss_history.append(loss.item())
        panr_history.append(psnr.item())

        opt.step()
        scheduler.step()

    fig = plt.figure(figsize=(6, 18), dpi=200)

    plt.subplot(3, 1, 1)
    plt.title('Adam,gaussian_num={:}'.format(gaussian_num))
    plt.plot(loss_history, label='loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(panr_history, label='psnr')
    plt.legend()
    plt.xlabel('iteration')

    plt.subplot(3, 1, 3)
    plt.plot(cost_factor.obs[:, 0], observes[:, 1].detach().numpy(), label='original')
    plt.plot(cost_factor.obs[:, 0], y_data.detach().numpy(), label='reconstructed')
    plt.xlabel("x")
    plt.legend()

    plt.show()
    fig.tight_layout()


# 进行多次实验
for e_i in range(epoch):
    cost_factor = CostFactor_1DGS(gaussian_num=100, is_great_init=False)

    ## ==================================test on CostFactor_Env3==================================
    # a11, a12, b1 = cost_factor.x[0], cost_factor.x[1], cost_factor.x[2]
    # a21, a22, b2 = cost_factor.x[3], cost_factor.x[4], cost_factor.x[5]
    # obs = torch.tensor(cost_factor.obs, requires_grad=True, dtype=torch.float32)
    # param1 = torch.tensor(data=[[a11, a12, b1]], requires_grad=True, dtype=torch.float32)
    # param2 = torch.tensor(data=[[a21, a22, b2]], requires_grad=True, dtype=torch.float32)
    #
    # param1_adam = param1.clone().detach().requires_grad_(True)
    # param2_adam = param2.clone().detach().requires_grad_(True)
    # lr = 5e-3
    # adahessian_optimize(params=[param1, param2], observes = obs, max_iterations=200)
    # adam_optimize(params=[param1_adam, param2_adam], observes = obs, max_iterations=500)
    ## ==================================test on CostFactor_Env3==================================

    lr = 1e-1
    costFunc1DGS_adam_optimize(cost_factor, max_iterations=2000, lr=lr)

    start_time = time.time()
    optimizer_nls.solve(cost_factor)
    end_time = time.time()
    data.append({'Method': 'naive nls', 'Iteration': optimizer_nls.iteration, 'Time': end_time - start_time,
                 'Epoch': f'Epoch {e_i}'})

    start_time = time.time()
    optimizer_gd.solve(cost_factor)
    end_time = time.time()
    data.append({'Method': 'naive gd', 'Iteration': optimizer_gd.iteration, 'Time': end_time - start_time,
                 'Epoch': f'Epoch {e_i}'})

# 将结果数据转换为 Pandas 数据框
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
