'''
Created by Han Xu
email:736946693@qq.com
'''
import torch
import torch.optim as Optimizer
from typing import Any, Dict, Iterable, Optional, Union, Callable
from torch import Tensor

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
State = Dict[str, Any]
LossClosure = Callable[[], float]

class LevenbergMarquardt(Optimizer.Optimizer):
    def __init__(self,
                 params: Params,
                 is_estimate: bool = True,
                 eps: float = 1e-2,
                 nu: float = 2,
                 max_iter: int = 2000,
                 tolX: float = 1e-6,
                 tolOpt: float = 1e-10,
                 tolFun: float = 1e-6) -> None:
        assert eps > 0, "invalid eps value"
        assert nu > 0, "invalid nu value"
        assert max_iter > 0, "invalid max_iter value"
        assert tolX > 0, "invalid tolX value"
        assert tolOpt > 0, "invalid tolOpt value"
        assert tolFun > 0, "invalid tolFun value"

        defaults = dict(is_estimate=is_estimate, eps=eps, nu=nu, max_iter=max_iter,
                        tolX=tolX, tolOpt=tolOpt, tolFun=tolFun)
        super(LevenbergMarquardt, self).__init__(params, defaults)

        # Initialize global state variables
        self.global_state = {
            'step': 0,
            'nu': nu,
            'miu': eps
        }

    def compute_jacobian(self) -> Tensor:
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
        return torch.cat(grads)

    def compute_hessian(self) -> Tensor:
        if self.defaults["is_estimate"]:
            return self.estimate_hessian()
        else:
            return self.true_hessian()

    def estimate_hessian(self) -> Tensor:
        # Placeholder for estimating Hessian
        pass

    def true_hessian(self) -> Tensor:
        # Calculate the true Hessian matrix using second-order gradients
        pass

    def step(self, closure: LossClosure = None) -> Optional[float]:
        assert closure is not None, "closure is None"
        with torch.enable_grad():
            initial_loss = closure()

        # Compute Jacobian and Hessian for the entire parameter space
        jacobian = self.compute_jacobian()
        print(jacobian)
        hessian = self.compute_hessian()
        print(hessian)

        # Initialize miu based on Hessian
        if self.global_state['step'] == 0:
            self.global_state['miu'] = self.defaults['eps'] * torch.max(hessian.diag()).item()

        # Store original parameters for potential rollback
        original_params = {p: p.clone() for group in self.param_groups for p in group['params']}

        # Compute the update step for the entire parameter space
        I = torch.eye(hessian.shape[0], device=hessian.device)
        step = -torch.inverse(hessian + self.global_state['miu'] * I) @ jacobian

        # Apply the tentative update to all parameters
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.add_(step[offset:offset + numel].view_as(p))
                offset += numel

        # Compute new loss after tentative updates
        with torch.enable_grad():
            new_loss = closure()

        # Check stopping criteria
        if torch.norm(jacobian, p=float('inf')) < self.defaults['tolOpt']:
            return new_loss

        if torch.norm(step) < self.defaults['tolX'] * max(1, torch.norm(step).item()):
            return new_loss

        if abs(new_loss - initial_loss) < self.defaults['tolFun'] * max(1, initial_loss):
            return new_loss

        # Calculate the error ratio
        gradient_at_new_params = self.compute_jacobian()
        rho = (new_loss - initial_loss) / ((step @ gradient_at_new_params).item())

        if rho > 0:
            # Update global state and accept the step
            self.global_state['step'] += 1
            self.global_state['miu'] *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            self.global_state['nu'] = 2
        else:
            # Reject the step and restore original parameters
            for p, original in original_params.items():
                p.data.copy_(original)

            # Increase miu and nu
            self.global_state['miu'] *= self.global_state['nu']
            self.global_state['nu'] *= 2

        return new_loss