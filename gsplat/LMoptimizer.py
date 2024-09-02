"""
Created by Han Xu
email:736946693@qq.com
"""
import numpy
import torch
from typing import Any, Dict, Iterable, Union, Callable, List, Tuple, overload, Optional
from torch import Tensor
from torch.optim.optimizer import Optimizer
from .cuda._wrapper import parallelize_sparse_matrix
import warnings

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
State = Dict[str, Any]
Closure = Callable[[], Tuple[Tensor, Tensor]]

class LevenbergMarquardt(Optimizer):
    def __init__(self,
                 params: Params,
                 tou: float = 1e-1,
                 epsilon: float = 1e-8,
                 tolX: float = 1e-3,
                 tolOpt: float = 1e-6,
                 tolFun: float = 1e-6,
                 block_size: int = -1,
                 strategy: str = '0', ) -> None:
        assert tou > 0, "invalid eps value"
        assert epsilon > 0, "invalid nu value"
        assert tolX > 0, "invalid tolX value"
        assert tolOpt > 0, "invalid tolOpt value"
        assert tolFun > 0, "invalid tolFun value"
        assert strategy in ['0', '1', ], "invalid strategy value"

        defaults = dict(tou=tou, epsilon=epsilon,
                        tolX=tolX, tolOpt=tolOpt, tolFun=tolFun,
                        block_size=block_size, strategy=strategy)
        super().__init__(params, defaults)
        print('\033[93m'
              + '''
              The order of parameters and residual must correspond to jacobian!
              Assume that there 3 kinds of parameters and 4 samples.
              example1:
              \033[0m
                  p1 = torch.tensor(data=[a1, a2, a3, a4])
                  p2 = torch.tensor(data=[b1, b2, b3, b4])
                  p3 = torch.tensor(data=[c1, c2, c3, c4])
                  opt = LevenbergMarquardt(params=[p1, p2, p3])
              \033[93m
              so, jacobian matrix should seem as followed:
                             a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4
                  residual1  .  .  .  .  .  .  .  .  .  .  .  .
                  residual2
                      .
                      .
                      .
                  residualn
              example2:
              \033[0m
                  p1 = torch.tensor(data=[a1, b1, c1])
                  p2 = torch.tensor(data=[a2, b2, c2])
                  p3 = torch.tensor(data=[a3, b3, c3])
                  p4 = torch.tensor(data=[a4, b4, c4])
                  opt = LevenbergMarquardt(params=[p1, p2, p3, p4])
              \033[93m
              jacobian:
                             a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4
                  residual1  .  .  .  .  .  .  .  .  .  .  .  .
                  residual2
                      .
                      .
                      .
                  residualn
              '''
              + '\033[0m')

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('tou', self.defaults['tou'])
            group.setdefault('epsilon', self.defaults['epsilon'])
            group.setdefault('tolX', self.defaults['tolX'])
            group.setdefault('tolOpt', self.defaults['tolOpt'])
            group.setdefault('tolFun', self.defaults['tolFun'])
            group.setdefault('block_size', self.defaults['block_size'])
            group.setdefault('strategy', self.defaults['strategy'])

    def store_params(self, group: Dict[str, Any]):
        params_vector = []
        for p in group['params']:
            params_vector.append(p.flatten())
        params_vector = torch.cat(params_vector)
        self.state[group['params'][0]]['params_vector'] = params_vector

    @staticmethod
    def update(group: Dict[str, Any], delta_x: Tensor) -> None:
        assert (delta_x.dim() == 1) or (delta_x.dim() == 2 and delta_x.size(-1) == 1), "delta_x must be a vector"
        if delta_x.dim() == 2 and delta_x.size(-1) == 1:
            delta_x = delta_x.reshape(-1)
        params = group['params']

        params_dim = 0
        for p in params:
            params_dim += p.numel()
        assert params_dim == delta_x.size(0), "params_dim must be equal to delta_x.size(0)"

        offset = 0
        for p in params:
            numel = p.numel()
            p.data.add_(delta_x[offset:offset + numel].view_as(p))
            offset += numel
        return

    def recover(self, group: Dict[str, Any]):
        params_vector = self.state[group['params'][0]]['params_vector']
        assert params_vector is not None, "params_vector must be not None"
        assert params_vector.dim() == 1, "params_vector must be a vector"
        params = group['params']

        params_dim = 0
        for p in params:
            params_dim += p.numel()
        assert params_dim == params_vector.size(0), "params_dim must be equal to params_vector.size(0)"

        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(params_vector[offset:offset + numel].view_as(p))
            offset += numel
        return

    @staticmethod
    def sparse_coo_slice(A: torch.Tensor, row_range: tuple, col_range: tuple) -> torch.Tensor:
        """
        对一个 PyTorch COO 格式的稀疏张量进行切片操作。

        参数:
        A (torch.Tensor): COO 格式的稀疏张量。
        row_range (tuple): 要切片的行范围，格式为 (row_start, row_end)。
        col_range (tuple): 要切片的列范围，格式为 (col_start, col_end)。

        返回:
        torch.Tensor: 切片后的dense张量
        """
        assert A.layout == torch.sparse_coo,A.layout
        # 提取 COO 张量的 indices 和 values
        indices = A._indices()
        values = A._values()

        # 提取切片的行和列范围
        row_start, row_end = row_range
        col_start, col_end = col_range

        # 创建掩码来筛选符合行和列范围的元素
        row_mask = (indices[0, :] >= row_start) & (indices[0, :] < row_end)
        col_mask = (indices[1, :] >= col_start) & (indices[1, :] < col_end)
        mask = row_mask & col_mask

        # 根据掩码筛选出符合条件的索引和值
        new_indices = indices[:, mask]
        new_values = values[mask]

        # 更新索引以反映新的坐标系
        new_indices[0, :] -= row_start
        new_indices[1, :] -= col_start

        # 切片后的矩阵形状
        new_shape = (row_end - row_start, col_end - col_start)

        # 创建新的 COO 稀疏张量
        sliced_A = torch.sparse_coo_tensor(new_indices, new_values, size=new_shape, device=A.device)

        return sliced_A.to_dense()

    @staticmethod
    def check_J_r(jacobian:Tensor, residual:Tensor):
        assert (jacobian.layout == torch.sparse_coo or
                jacobian.layout == torch.strided), jacobian.layout
        assert jacobian.dim() == 2, "Jacobian must be 2-dimensional"
        assert (residual.dim() == 1 or
                residual.dim() == 2 and residual.size(-1) == 1), \
            "residual must be a vector."
        if residual.dim() == 2 and residual.size(-1) == 1:
            residual = residual.reshape(-1)
        assert jacobian.size(0) == residual.size(0), \
            "Jacobian's row must correspond to residual's elements."

    def compute_update(self, block_size: int, A: Tensor, b: Tensor):
        if block_size <= 0 and A.layout == torch.strided:
            update = torch.linalg.solve(A, b)
        else:
            try:
                update = parallelize_sparse_matrix(A, b, block_size)
            except Exception as error:
                print(f"\033[91m {error} \033[0m")
                current_idx = 0
                param_dim = b.size(0)
                update = torch.zeros_like(b)
                for i in range((param_dim - 1) // block_size + 1):
                    if current_idx + block_size > param_dim:
                        end_idx = param_dim
                        current_idx = end_idx - block_size
                    else:
                        end_idx = current_idx + block_size
                    if A.layout == torch.sparse_coo:
                        A_sub = self.sparse_coo_slice(A, (current_idx, end_idx), (current_idx, end_idx))
                    else:
                        A_sub = A[current_idx:end_idx, current_idx:end_idx]
                    b_sub = b[current_idx:end_idx]
                    sub_update = torch.linalg.solve(A_sub, b_sub)
                    # 将解的分块赋值给相应的 update 部分
                    update[current_idx:end_idx] = sub_update
                    current_idx += block_size
        return -update

    def step_with_closure(self, closure: Closure) -> float:
        if self.param_groups[0]['params'][0].grad is not None:
            warnings.warn('You needn\'t input grad '
                          'because gradient will be compute automatically by jacobian and residual.',
                          DeprecationWarning)
        for group in self.param_groups:
            state = self.state[group['params'][0]]
            if 'found' in state:
                if state['found'] == True:
                    continue
            with torch.no_grad():
                jacobian, residual = closure()
            self.check_J_r(jacobian, residual)
            epsilon = group['epsilon']
            gradient = jacobian.t() @ residual
            hessian = jacobian.t() @ jacobian

            # Initialize
            if len(state) == 0:
                state['found'] = False
                state['step'] = 0
                state['nu'] = 2
                if jacobian.layout == torch.sparse_coo:
                    diag_mask = (hessian._indices()[0] == hessian._indices()[1])
                    max_diagElement = torch.max(hessian._values()[diag_mask]).item()
                    state['miu'] = group['tou'] * max_diagElement
                elif jacobian.layout == torch.strided:
                    state['miu'] = group['tou'] * torch.max(torch.diag(hessian)).item()
                state['loss'] = torch.linalg.vector_norm(residual).item()
            state['step'] += 1

            # Compute the update step for the entire parameter space
            diag_values = state['miu'] * torch.ones(hessian.shape[0], device=hessian.device)
            indices = torch.arange(0, hessian.shape[0], device=hessian.device).repeat(2, 1)
            diag_matrix = torch.sparse_coo_tensor(indices, diag_values, hessian.shape, device=hessian.device)
            A = hessian + diag_matrix
            update = self.compute_update(group['block_size'], A, gradient)
            self.store_params(group)
            params_length = torch.linalg.vector_norm(state['params_vector']).item()
            if torch.linalg.vector_norm(update) < epsilon * (params_length + epsilon):
                state['found'] = True
                continue

            self.update(group, update)
            with torch.no_grad():
                jacobian, residual = closure()
            self.check_J_r(jacobian, residual)

            last_loss = state['loss']
            new_loss = torch.linalg.vector_norm(residual).item()
            new_gradient = jacobian.t() @ residual

            varrho = None
            if group['strategy'] == '0':
                varrho = ((last_loss - new_loss) /
                          (0.5 * update @ (state['miu'] * update - gradient))
                          ).item()
            elif group['strategy'] == '1':
                varrho = ((new_loss - last_loss) / (update @ new_gradient)
                          ).item()

            ## other check
            if torch.linalg.vector_norm(gradient) < group['tolOpt']:
                state['found'] = True
                continue
            if torch.linalg.vector_norm(update) < group['tolX'] * max(1.0, params_length):
                state['found'] = True
                continue
            if abs(new_loss - last_loss) < group['tolFun'] * max(1.0, abs(last_loss)):
                state['found'] = True
                continue

            if varrho > 0:
                if torch.linalg.vector_norm(new_gradient, ord=numpy.inf) <= epsilon:
                    state['found'] = True
                    continue
                state['miu'] *= max(1 / 3, 1 - (2 * varrho - 1) ** 3)
                state['nu'] = 2
                state['loss'] = new_loss
            else:
                self.recover(group)
                state['miu'] *= state['nu']
                state['nu'] *= 2
        return self.state[self.param_groups[-1]['params'][0]]['loss']

    def step_with_jacobians(self, Jacobians: Iterable[Tensor], residual: Tensor) -> float:
        if self.param_groups[0]['params'][0].grad is not None:
            warnings.warn('You needn\'t input grad '
                          'because gradient will be compute automatically by jacobian and residual.',
                          DeprecationWarning)
        for group, jacobian in zip(self.param_groups, Jacobians):
            self.check_J_r(jacobian, residual)
            state = self.state[group['params'][0]]
            epsilon = group['epsilon']
            new_gradient = jacobian.t() @ residual
            if torch.linalg.vector_norm(new_gradient, ord=numpy.inf) <= epsilon:
                state['found'] = True
            if 'found' in state:
                if state['found']:
                    if 'gradient' in state: del state['gradient']
                    if 'last_update' in state is not None: del state['last_update']
                    continue

            hessian = jacobian.t() @ jacobian
            # first pass
            if len(state) == 0:
                state['found'] = False
                state['step'] = 0
                state['nu'] = 2
                if jacobian.layout == torch.sparse_coo:
                    diag_mask = (hessian._indices()[0] == hessian._indices()[1])
                    max_diagElement = torch.max(hessian._values()[diag_mask]).item()
                    state['miu'] = group['tou'] * max_diagElement
                elif jacobian.layout == torch.strided:
                    state['miu'] = group['tou'] * torch.max(torch.diag(hessian)).item()
                state['loss'] = torch.linalg.vector_norm(residual).item()
                self.store_params(group)
            else:
                last_update = state['last_update']
                last_gradient = state['gradient']
                last_loss = state['loss']
                new_loss = torch.linalg.vector_norm(residual).item()
                params_length = torch.linalg.vector_norm(state['params_vector']).item()

                varrho = None
                if group['strategy'] == '0':
                    varrho = ((last_loss - new_loss) /
                              (0.5 * last_update @ (state['miu'] * last_update - new_gradient))
                              ).item()
                elif group['strategy'] == '1':
                    varrho = ((new_loss - last_loss) / (last_update @ new_gradient)
                              ).item()

                ## other check
                if torch.linalg.vector_norm(last_gradient) < group['tolOpt']:
                    state['found'] = True
                    continue
                if torch.linalg.vector_norm(last_update) < group['tolX'] * max(1.0, params_length):
                    state['found'] = True
                    continue
                if abs(new_loss - last_loss) < group['tolFun'] * max(1.0, abs(last_loss)):
                    state['found'] = True
                    continue

                # judge last update
                if varrho <= 0:  # reject
                    self.recover(group)
                    state['miu'] *= state['nu']
                    state['nu'] *= 2
                else:  # accept
                    self.store_params(group)
                    state['miu'] *= max(1 / 3, 1 - (2 * varrho - 1) ** 3)
                    state['nu'] = 2
                    state['loss'] = new_loss

            # update temporarily
            diag_values = state['miu'] * torch.ones(hessian.shape[0], device=hessian.device)
            indices = torch.arange(0, hessian.shape[0], device=hessian.device).repeat(2, 1)
            diag_matrix = torch.sparse_coo_tensor(indices, diag_values, hessian.shape, device=hessian.device)
            A = hessian + diag_matrix
            update = self.compute_update(group['block_size'], A, new_gradient)
            params_length = torch.linalg.vector_norm(state['params_vector']).item()
            if torch.linalg.vector_norm(update) < epsilon * (params_length + epsilon):
                state['found'] = True
                continue
            self.update(group, update)
            state['gradient'] = new_gradient
            state['last_update'] = update
            state['step'] += 1
        return self.state[self.param_groups[-1]['params'][0]]['loss']

    def step(self, closure: Optional[Closure] = None,
             Jacobians: Optional[Iterable[Tensor]] = None, residual: Optional[Tensor] = None) -> float:
        if closure is not None:
            # 使用 closure 参数实现的 step 逻辑
            return self.step_with_closure(closure)
        elif Jacobians is not None and residual is not None:
            # 使用 Jacobians 和 residual 参数实现的 step 逻辑
            return self.step_with_jacobians(Jacobians, residual)
        else:
            raise ValueError("Invalid arguments for step method.")
