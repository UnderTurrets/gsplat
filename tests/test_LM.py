'''
Created by Han Xu
email:736946693@qq.com
'''
import matplotlib.pyplot as plt
from demo.CostFactor import CostFactor_1DGS
from tqdm import tqdm
from gsplat.LMoptimizer import LevenbergMarquardt
import torch
import numpy

if __name__ == '__main__':
    def costFunc1DGS_LM_optimize(costF: CostFactor_1DGS, max_iterations: int = 2000, show_process: bool = False, device='cuda'):
        observes = torch.tensor(data=costF.obs, requires_grad=True, dtype=torch.float32 ,device=device)
        p1 = torch.tensor(data=costF.x.reshape(-1, 3)[:, 0], requires_grad=True, dtype=torch.float32, device=device)
        p2 = torch.tensor(data=costF.x.reshape(-1, 3)[:, 1], requires_grad=True, dtype=torch.float32, device=device)
        p3 = torch.tensor(data=costF.x.reshape(-1, 3)[:, 2], requires_grad=True, dtype=torch.float32, device=device)
        obs_dim = observes.shape[0]
        gaussian_num = p1.shape[0]

        def get_parameters():
            variances = torch.empty_like(p2)
            opacity = torch.empty_like(p3)
            if costF.parameter_space == 0:
                variances = torch.square(p2)
                opacity = p3
            elif costF.parameter_space == 1:
                variances = torch.exp(p2)
                opacity = 1 / (1 + torch.exp(-p3))
            return p1, variances, opacity

        def get_jacobain():
            J = torch.zeros(size=(obs_dim, 3 * gaussian_num), device=observes.device)
            means, variances, opacity = get_parameters()
            for i in range(gaussian_num):
                if (
                        ## variance is too low or too high
                        variances[i] <= 0
                        or variances[i] >= len(observes[:, 0]) / 2
                        ## opacity is too low or too high
                        or opacity[i] <= 0
                        or opacity[i] >= max(observes[:, 1])
                ): continue
                J[:, i] = -(
                        opacity[i] * (observes[:, 0] - means[i]) / variances[i] *
                        torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i])
                )
                if costF.parameter_space == 0:
                    J[:, gaussian_num + i] = -(
                            opacity[i] * ((observes[:, 0] - means[i]) ** 2) / (p2[i] ** 3) *
                            torch.exp(-0.5 * ((observes[:, 0] - means[i]) ** 2) / p2[i] ** 2)
                    )
                    J[:, 2*gaussian_num + i] = -torch.exp(-0.5 * ((observes[:, 0] - means[i]) ** 2) / p2[i] ** 2)
                elif costF.parameter_space == 1:
                    J[:, gaussian_num + i] = -(
                            opacity[i] * ((observes[:, 0] - means[i]) ** 2) / (2 * variances[i] ** 2) *
                            torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]) * torch.exp(
                        p2[i])
                    )
                    J[:, 2*gaussian_num + i] = -(
                            torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]) *
                            torch.exp(-p3[i]) /
                            ((1 + torch.exp(-p3[i])) ** 2)
                    )
            return J

        opt = LevenbergMarquardt(params=[p1, p2, p3], tou=1e-1, block_size=69, strategy='0')
        loss_history = []
        panr_history = []
        iteration_speed_history = []
        gaussian_num = p1.shape[0]

        if show_process:
            plt.figure()
            plt.ion()
            plt.show()
        pbar = tqdm(range(max_iterations))
        for i in pbar:
            means, variances, opacity = get_parameters()
            # compute residual
            y_data_list = []
            for i in range(gaussian_num):
                y_data_list.append(opacity[i] * torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]))
            y_data_stack = torch.stack(y_data_list, dim=0)
            y_data = torch.sum(y_data_stack, dim=0)
            if show_process:
                plt.clf()
                plt.subplot(3, 1, 1)
                plt.plot(observes[:, 0].cpu().detach().numpy(), observes[:, 1].cpu().detach().numpy(), label='origin')
                plt.plot(observes[:, 0].cpu().detach().numpy(), y_data.cpu().detach().numpy(), label='reconstructed')
                plt.title(f'LM')
                plt.xlabel('x')
                plt.legend()
                plt.subplot(3, 1, 2)
                for single_y_data in y_data_list:
                    plt.plot(observes[:, 0].cpu().detach().numpy(), single_y_data.cpu().detach().numpy(), linewidth=0.5)
                plt.subplot(3, 1, 3)
                plt.plot(panr_history, label='psnr')
                plt.legend()
                plt.xlabel('iteration')
                plt.pause(0.01)

            opt.zero_grad(set_to_none=True)
            means, variances, opacity = get_parameters()
            ## compute residual
            y_data_list = []
            for i in range(gaussian_num):
                y_data_list.append(opacity[i] * torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]))
            y_data_stack = torch.stack(y_data_list, dim=0)
            y_data = torch.sum(y_data_stack, dim=0)
            residual = observes[:, 1] - y_data
            _weights = torch.ones(size=(costF.obs_dim,), device=device).repeat(costF.residual_dim)
            residual = residual * _weights

            def closure():
                opt.zero_grad(set_to_none=True)
                means, variances, opacity = get_parameters()
                # compute residual
                y_data_list = []
                for i in range(gaussian_num):
                    y_data_list.append(opacity[i] * torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]))
                y_data_stack = torch.stack(y_data_list, dim=0)
                y_data = torch.sum(y_data_stack, dim=0)
                residual = observes[:, 1] - y_data
                _weights = torch.ones(size=(costF.obs_dim,),device=device).repeat(costF.residual_dim)
                residual = residual * _weights
                return get_jacobain(), residual
            # loss = opt.step(Jacobians=[get_jacobain()],residual=residual)
            loss = opt.step(closure=closure)
            opt.zero_grad(set_to_none=True)
            mse = torch.mean((observes[:, 1] - y_data) ** 2)
            psnr = 10 * torch.log10(torch.max(observes[:, 1]).item() ** 2 / mse)
            loss_history.append(loss)
            panr_history.append(psnr.item())

            desc = f"LM"
            pbar.set_description(desc)
            if pbar.format_dict["rate"] is not None:
                iteration_speed_history.append(pbar.format_dict["rate"])

        average_iteration_speed = None
        if len(iteration_speed_history) != 0:
            average_iteration_speed = numpy.array(iteration_speed_history).mean()
        return average_iteration_speed


    cost_factor = CostFactor_1DGS(gaussian_num=50, is_great_init=False, parameter_space=0)
    costFunc1DGS_LM_optimize(costF=cost_factor, max_iterations=100, show_process=True, device='cuda')