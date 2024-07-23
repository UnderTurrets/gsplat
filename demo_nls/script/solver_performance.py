'''
Created by Han Xu
email:736946693@qq.com
'''
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
epoch = 100

# test iteration speed, so max_iteration is small
optimizer_nls = Classical_NLS_Solver(max_iter=50)
optimizer_gd = Classical_GD_Solver(max_iter=50)
LMspeed_history = []
GDspeed_history = []
gaussian_num_history = []
for e_i in range(epoch):
    gaussian_num = 30+e_i*10
    cost_factor = CostFactor_1DGS(gaussian_num=gaussian_num, is_great_init=False)

    # lr = 1e-1
    # costFunc1DGS_adam_optimize(cost_factor, max_iterations=3000, lr=lr)


    LMspeed = optimizer_nls.solve(cost_factor)
    LMspeed_history.append(LMspeed)
    GDspeed = optimizer_gd.solve(cost_factor)
    GDspeed_history.append(GDspeed)
    gaussian_num_history.append(gaussian_num)

# plt.clf()
# fig = plt.figure(figsize=(6,12),dpi=300)
# plt.subplot(2,1,1)
# plt.plot(gaussian_num_history, LMspeed_history, label='LM')
# plt.legend()
# plt.ylabel('iteration speed(it/s)')
# plt.subplot(2,1,2)
# plt.plot(gaussian_num_history, GDspeed_history, label='GD')
# plt.legend()
# plt.ylabel('iteration speed(it/s)')
# plt.xlabel('gaussian_num')
# fig.tight_layout()
# plt.show()

plt.clf()
plt.plot(gaussian_num_history, LMspeed_history, label='LM')
plt.plot(gaussian_num_history, GDspeed_history, label='GD')
plt.legend()
plt.ylabel('iteration speed(it/s)')
plt.xlabel('gaussian_num')
plt.show()
