'''
Created by Han Xu
email:736946693@qq.com
'''
from demo.CostFactor import CostFactor_1DGS
from demo.Classical_Solver import Classical_NLS_Solver, Classical_GD_Solver
from demo.classical_solver_example import costFunc1DGS_adam_optimize
import matplotlib.pyplot as plt
if __name__ == '__main__':
    epoch = 20
    adamSpeed_history = []
    LMspeed_history = []
    SparseLMspeed_history = []
    GDspeed_history = []
    gaussian_num_history = []
    for e_i in range(epoch):
        gaussian_num = 800+e_i*10
        cost_factor = CostFactor_1DGS(gaussian_num=gaussian_num, is_great_init=False)
        # test iteration speed, so max_iteration is small
        optimizer_nls = Classical_NLS_Solver(max_iter=50)
        optimizer_nls_sparse = Classical_NLS_Solver(max_iter=50)
        optimizer_gd = Classical_GD_Solver(max_iter=50)

        lr = 5e-2
        adamSpeed = costFunc1DGS_adam_optimize(cost_factor, max_iterations=100, lr=lr)
        adamSpeed_history.append(adamSpeed)

        _, LMspeed, _ = optimizer_nls.solve(cost_factor)
        LMspeed_history.append(LMspeed)

        _, SparseLMspeed, _ = optimizer_nls_sparse.solve(cost_factor, block_size = 50)
        SparseLMspeed_history.append(SparseLMspeed)

        _, GDspeed = optimizer_gd.solve(cost_factor)
        GDspeed_history.append(GDspeed)

        gaussian_num_history.append(gaussian_num)

    plt.figure("speed")
    plt.plot(gaussian_num_history, adamSpeed_history, label='adam')
    plt.plot(gaussian_num_history, LMspeed_history, label='LM')
    plt.plot(gaussian_num_history, SparseLMspeed_history, label='sparseLM')
    plt.plot(gaussian_num_history, GDspeed_history, label='GD')
    plt.legend()
    plt.ylabel('iteration speed(it/s)')
    plt.xlabel('gaussian_num')
    plt.show()
