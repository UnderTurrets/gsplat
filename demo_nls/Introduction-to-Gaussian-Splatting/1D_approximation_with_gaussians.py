"""
Created by Han Xu
email:736946693@qq.com
"""

import torch
from torch import tensor
import torch.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
from tqdm import tqdm
import torch_optimizer

#Load weather data
temperatures = pd.read_csv(r"DailyDelhiClimateTest.csv")['meantemp'].values


#plot temperatures
plt.plot(temperatures)
plt.title('Daily temperatures in Delhi')
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.show()

#intialize K gaussian kernels and their weights, with means and variances in the range of len(temperatures)
def initialize_gaussian_kernels(K, data_length):
    means = torch.tensor(np.random.randint(0, data_length, K).astype(float), requires_grad=True)
    variances = torch.tensor(np.random.randint(1, data_length, K).astype(float), requires_grad=True)
    weights = torch.tensor(np.random.rand(K))
    weights = weights / weights.sum()
    weights.requires_grad = True

    return means, variances, weights


def great_initialize_gaussian_kernels(means: tensor, variances: tensor, weights: tensor
                                      ) -> (torch.Tensor, torch.Tensor, torch.Tensor):

    great_means = means.clone().detach().requires_grad_(True)
    great_means = great_means + torch.randn_like(great_means) * great_means/3

    great_variances = variances.clone().detach().requires_grad_(True)
    great_variances = great_variances + torch.randn_like(great_variances) * great_variances/3
    great_variances.data = torch.clamp(great_variances.data, 0.1, len(temperatures))

    great_weights = weights.clone().detach().requires_grad_(True)
    great_weights = great_weights + torch.randn_like(great_weights) * great_weights/3
    return great_means, great_variances, great_weights


def reconstructed_signal(means, variances, weights):
    x = torch.linspace(0, len(temperatures), len(temperatures))
    pdf = torch.zeros(len(temperatures))
    K = len(means)
    for i in range(K):
        pdf += weights[i] * torch.exp(dist.Normal(means[i], variances[i]).log_prob(x))
    return pdf


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)
    return 20 * torch.log10(torch.max(original) / torch.sqrt(mse))


def optimize_gaussians(means, variances, weights, temperatures, lr=0.1, max_iterations=1000):
    # optimizer = torch.optim.Adam([means, variances, weights], lr=lr)
    # plt.title('Adam')
    optimizer = torch_optimizer.Adahessian([means, variances, weights],lr=lr)
    plt.title('Adahessian')
    import sys
    last_loss = sys.float_info.max
    last_psnr = sys.float_info.max
    psnr_list = []
    for i in tqdm(range(max_iterations), desc=f'Optimizing Gaussians for k={len(means)}'):
        def closure()->float:
            optimizer.zero_grad()
            pdf = reconstructed_signal(means, variances, weights)
            loss = torch.sum((pdf - torch.tensor(temperatures)) ** 2) / len(temperatures)
            loss.backward(create_graph=True)
            return loss.item()

        loss = optimizer.step(closure)

        # optimizer.zero_grad()
        # pdf = reconstructed_signal(means, variances, weights)
        # loss = torch.sum((pdf - torch.tensor(temperatures)) ** 2) / len(temperatures)
        # loss.backward(create_graph=True)
        # optimizer.step()

        #clip variances to be positive
        variances.data = torch.clamp(variances.data, 0.1, len(temperatures))

        # calculate PSNR
        pdf = reconstructed_signal(means, variances, weights)
        psnr = calculate_psnr(torch.tensor(temperatures), pdf).detach().item()
        psnr_list.append(psnr)

        # judge convergence
        # if (torch.linalg.norm(means.grad, ord=torch.inf) < 1e-5 or
        #         torch.linalg.norm(variances.grad, ord=torch.inf) < 1e-5 or
        #         torch.linalg.norm(weights.grad, ord=torch.inf) < 1e-5
        # ):
        #     print("grad is too small so stop optimizing")
        #     break
        #
        # loss_distance = abs(loss - last_loss)
        # if (loss_distance < 5e-6 * (last_loss)):
        #     print("loss is converge so stop optimizing")
        #     break
        #

        # if (abs(psnr-last_psnr) < 1e-4 * last_psnr):
        #     print("psnr is converge so stop optimizing")
        #     break
        #
        # last_loss = loss
        # last_psnr = psnr

    plt.xlabel('Iteration')
    plt.plot(psnr_list, label='PSNR')
    plt.legend()
    plt.show()

    print(f'Final loss: {last_loss}')
    print(f'Final PSNR: {last_psnr}')
    return means, variances, weights


#initialize K gaussian kernels
K = 100
data_length = len(temperatures)
means, variances, weights = initialize_gaussian_kernels(K, data_length)

#plot all the pdfs in one plot
# x = torch.linspace(0, len(temperatures), len(temperatures))
# pdfs = []
# for i in range(K):
#     pdfs.append(weights[i] * torch.exp(dist.Normal(means[i], variances[i]).log_prob(x)))
#     plt.plot(x, pdfs[i].detach().numpy())
# plt.show()

#plot the sum of all pdfs
# pdf = torch.zeros(len(temperatures))
# for i in range(K):
#     pdf += weights[i] * torch.exp(dist.Normal(means[i], variances[i]).log_prob(x))
# plt.plot(x, pdf.detach().numpy())
# plt.show()

means, variances, weights = optimize_gaussians(means, variances, weights, temperatures)
pdf = reconstructed_signal(means, variances, weights)

#plot the original signal and the reconstructed signal
plt.cla()
plt.plot(temperatures)
plt.plot(pdf.detach().numpy())
plt.title('Temperature in Delhi')
plt.xlabel('Day')
plt.ylabel('Temperature')
plt.legend(['Ground Truth', 'Reconstructed'])
plt.show()
print(f'PSNR: {calculate_psnr(torch.tensor(temperatures), pdf).detach().item()}')

# great_means, great_variances, great_weights = great_initialize_gaussian_kernels(means, variances, weights)
# great_means, great_variances, great_weights = optimize_gaussians(great_means, great_variances, great_weights,
#                                                                  temperatures)

