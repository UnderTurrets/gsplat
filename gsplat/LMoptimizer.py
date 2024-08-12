'''
Created by Han Xu
email:736946693@qq.com
'''
import numpy
import torch
from typing import Any, Dict, Iterable, Optional, Union, Callable, List, overload
from torch import Tensor
from torch.optim.optimizer import (Optimizer, ParamsT, _use_grad_for_differentiable, _get_value,
                                   _stack_if_compiling, _dispatch_sqrt, _default_to_fused_or_foreach,
                                   _get_scalar_dtype, _capturable_doc, _differentiable_doc, _foreach_doc,
                                   _fused_doc, _maximize_doc, _view_as_real)
from cuda._wrapper import parallelize_sparse_matrix
Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
State = Dict[str, Any]
LossClosure = Callable[[], float]


class LevenbergMarquardt(Optimizer):
    def __init__(self,
                 params: Params,
                 tou: float = 1e-1,
                 epsilon: float = 1e-8,
                 tolX: float = 1e-3,
                 tolOpt: float = 1e-6,
                 tolFun: float = 1e-4,
                 block_size: int = 50, ) -> None:
        assert tou > 0, "invalid eps value"
        assert epsilon > 0, "invalid nu value"
        assert tolX > 0, "invalid tolX value"
        assert tolOpt > 0, "invalid tolOpt value"
        assert tolFun > 0, "invalid tolFun value"

        defaults = dict(tou=tou, epsilon=epsilon,
                        tolX=tolX, tolOpt=tolOpt, tolFun=tolFun,
                        block_size=block_size, loss=None)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('tou', self.defaults['tou'])
            group.setdefault('epsilon', self.defaults['epsilon'])
            group.setdefault('tolX', self.defaults['tolX'])
            group.setdefault('tolOpt', self.defaults['tolOpt'])
            group.setdefault('tolFun', self.defaults['tolFun'])
            group.setdefault('block_size', self.defaults['block_size'])

    def collect_grad(self, params: List[Tensor]) -> Tensor:
        gradient = []
        for p in params:
            gradient.append(p.grad)
        gradient = torch.cat(gradient).flatten().contiguous()
        return gradient

    def solve_sparse_matrix(self, A: Tensor, b: Tensor, block_size: int) -> Tensor:
        assert A.dim() == 2, "A must be a 2-dimensional matrix"
        assert b.dim() == 1, "b must be a vector"
        assert A.shape[0] == A.shape[1], 'A must be a square matrix'
        assert A.shape[0] == b.shape[0], "A.shape[0] must be equal to b.shape[0]"
        current_idx = 0
        param_dim = b.shape[0]
        solve = torch.zeros_like(b)
        for i in range((param_dim - 1) // block_size + 1):
            if current_idx + block_size > param_dim:
                end_idx = param_dim
                current_idx = end_idx - block_size
            else:
                end_idx = current_idx + block_size
            A_sub = A[current_idx:end_idx, current_idx:end_idx]
            b_sub = b[current_idx:end_idx]
            sub_update = -torch.inverse(A_sub) @ b_sub.view(-1, 1)
            # 将解的分块赋值给相应的 update 部分
            solve[current_idx:end_idx] = sub_update.flatten()
            current_idx += block_size
        return solve

    def update(self, params: List[Tensor], delta_x: Tensor) -> None:
        assert delta_x.dim() == 1, "delta_x must be a vector"
        params_dim = 0
        for p in params:
            params_dim += p.numel()
        assert params_dim == delta_x.shape[0], "params_dim must be equal to delta_x.shape[0]"
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.add_(delta_x[offset:offset + numel].view_as(p))
            offset += numel
        return

    def step(self, closure: LossClosure, Jacobians: Iterable[Tensor]) -> float:
        for group, jacobian in zip(self.param_groups, Jacobians):
            epsilon = group['epsilon']
            block_size = group['block_size']

            with torch.enable_grad():
                loss = closure()
            self.defaults['loss'] = loss
            gradient = self.collect_grad(group['params'])

            param_dim = gradient.shape[0]
            assert jacobian.dim() == 2, "Jacobian must be 2-dimensional"
            assert jacobian.shape[1] == param_dim, \
                "Jacobian.T and gradient must have the same number of columns"
            hessian = jacobian.permute(1, 0) @ jacobian

            state = self.state[group['params'][0]]
            # Initialize
            if len(state) == 0:
                state['step'] = 0
                state['nu'] = 2
                state['miu'] = group['tou'] * torch.max(torch.diag(hessian)).item()
            state['step'] += 1

            # Compute the update step for the entire parameter space
            I = torch.eye(hessian.shape[0], device=hessian.device)
            A = hessian + state['miu'] * I
            if block_size <= 0:
                update = (-torch.inverse(A) @ gradient.view(-1, 1)).flatten()
            else:
                try:
                    parallelize_sparse_matrix(A, gradient, block_size)
                except Exception as error:
                    print(f"\033[91m {error} \033[0m")
                    update = self.solve_sparse_matrix(A, gradient, block_size)

            original_params = {p: p.clone() for p in group['params']}
            params_vector = []
            for p in group['params']:
                params_vector.append(p.flatten())
            params_vector = torch.cat(params_vector)
            params_length = torch.linalg.vector_norm(params_vector).item()
            if torch.linalg.vector_norm(update) < epsilon * (params_length + epsilon):
                continue

            self.update(group['params'], update)
            with torch.enable_grad():
                new_loss = closure()
            new_gradient = self.collect_grad(group['params'])

            ## strateg1
            varrho = ((loss - new_loss) /
                      (0.5 * update @ (state['miu'] * update - gradient).view(-1, 1))
                      ).item()
            ## strateg2
            # TODO: This lead to unconvergence
            # varrho = ((new_loss - loss) /
            #           (update @ new_gradient.view(-1, 1))
            #           ).item()

            ## other check
            if torch.linalg.vector_norm(gradient) < group['tolOpt']:
                continue
            if torch.linalg.vector_norm(update) < group['tolX'] * max(1.0, params_length):
                continue
            if abs(new_loss - loss) < group['tolFun'] * max(1.0, abs(loss)):
                continue

            if varrho > 0:
                if (torch.linalg.vector_norm(new_gradient, ord=numpy.inf) <= epsilon):
                    continue
                state['miu'] *= max(1 / 3, 1 - (2 * varrho - 1) ** 3)
                state['nu'] = 2
                self.defaults['loss'] = new_loss
            else:
                for p, original in original_params.items():
                    p.data.copy_(original)
                state['miu'] *= state['nu']
                state['nu'] *= 2
        return self.defaults['loss']

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from demo.Environment.CostFactor import CostFactor_1DGS
    from tqdm import tqdm
    def costFunc1DGS_LM_optimize(costF: CostFactor_1DGS, max_iterations: int = 2000, show_process: bool = False):
        observes = torch.tensor(data=costF.obs, requires_grad=True, dtype=torch.float32)
        p1 = torch.tensor(data=costF.x.reshape(-1, 3)[:, 0], requires_grad=True, dtype=torch.float32)
        p2 = torch.tensor(data=costF.x.reshape(-1, 3)[:, 1], requires_grad=True, dtype=torch.float32)
        p3 = torch.tensor(data=costF.x.reshape(-1, 3)[:, 2], requires_grad=True, dtype=torch.float32)
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
            J = torch.zeros(size=(obs_dim, 3 * gaussian_num))
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
                J[:, 3 * i] = -(
                        opacity[i] * (observes[:, 0] - means[i]) / variances[i] *
                        torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i])
                )
                if costF.parameter_space == 0:
                    J[:, 3 * i + 1] = -(
                            opacity[i] * ((observes[:, 0] - means[i]) ** 2) / (p2[i] ** 3) *
                            torch.exp(-0.5 * ((observes[:, 0] - means[i]) ** 2) / p2[i] ** 2)
                    )
                    J[:, 3 * i + 2] = -torch.exp(-0.5 * ((observes[:, 0] - means[i]) ** 2) / p2[i] ** 2)
                elif costF.parameter_space == 1:
                    J[:, 3 * i + 1] = -(
                            opacity[i] * ((observes[:, 0] - means[i]) ** 2) / (2 * variances[i] ** 2) *
                            torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]) * torch.exp(
                        p2[i])
                    )
                    J[:, 3 * i + 2] = -(
                            torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]) *
                            torch.exp(-p3[i]) /
                            ((1 + torch.exp(-p3[i])) ** 2)
                    )
            return J

        opt = LevenbergMarquardt(params=[p1, p2, p3], tou=1e-0)
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
                plt.plot(observes[:, 0].detach().numpy(), observes[:, 1].detach().numpy(), label='origin')
                plt.plot(observes[:, 0].detach().numpy(), y_data.detach().numpy(), label='reconstructed')
                plt.title(f'LM')
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

            def closure() -> float:
                opt.zero_grad(set_to_none=True)
                means, variances, opacity = get_parameters()
                # compute residual
                y_data_list = []
                for i in range(gaussian_num):
                    y_data_list.append(opacity[i] * torch.exp(-0.5 * (observes[:, 0] - means[i]) ** 2 / variances[i]))
                y_data_stack = torch.stack(y_data_list, dim=0)
                y_data = torch.sum(y_data_stack, dim=0)
                residual = observes[:, 1] - y_data
                _weights = torch.ones(size=(costF.obs_dim,)).repeat(costF.residual_dim)
                loss = 0.5 * torch.sum(torch.square(residual * _weights))
                loss.backward()
                return loss.item()

            jacobian = get_jacobain()
            loss = opt.step(closure=closure, Jacobians=[jacobian])
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
