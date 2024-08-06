import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch

def plot_gaussian(mu, cov, ax, color='blue'):
    """
    绘制二维高斯分布的等高线
    :param mu: 均值向量
    :param cov: 协方差矩阵
    :param ax: Matplotlib的轴对象
    :param color: 椭圆的颜色
    """
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    ell = Ellipse(xy=(mu[0], mu[1]),
                  width=lambda_[0]*2, height=lambda_[1]*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])),
                  edgecolor=color, fc='None', lw=2)
    ax.add_patch(ell)

if __name__ == '__main__':
    import torch
    # 创建两个一维张量
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6, 8])
    # 使用torch.cat()拼接这两个张量
    result = torch.cat((tensor1, tensor2))
    result2 = result.flatten()
    result2[0]=99
    print(result)
    print(result2)
    pass

    # in a list, share the memory
    # flatten() share the memory
