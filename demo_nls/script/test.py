import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
print(torch.utils.cmake_prefix_path)
print(torch.cuda.is_available())
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

mu = [0, 0]

# 定义不同的协方差矩阵
covariances = {
    '1': [[1, 0], [0, 1]],
    '2': [[1, 0.8], [0.8, 1]],
    '3': [[1, -0.8], [-0.8, 1]],
    '4': [[2, 0], [0, 0.5]]
}

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for ax, (title, cov) in zip(axes.flatten(), covariances.items()):
    plot_gaussian(mu, cov, ax)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(title)
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.grid(True)

plt.tight_layout()
plt.show()


