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
    pass
