import os.path
import numpy
import numpy as np
import random
from Environment.Base import CostFactor
import torch
import math
from typing import List, Tuple
from numpy import ndarray
from sklearn.neighbors import NearestNeighbors


# a * exp(b * x)
class CostFactor_Env1(CostFactor):
    def __init__(self, is_initialized=False, x_gt=None, x=None, obs=None, is_numerical=False):
        # If the object is initialized, call the parent class constructor with the provided parameters.
        if is_initialized:
            super().__init__(is_numerical, x_gt, x, obs, 1)
        else:
            # Otherwise, set the number of data points and reset the environment.
            self.data_num = 100
            self.reset()
            # Call the parent class constructor with the generated data.
            super().__init__(is_numerical, self.x_gt, self.x, self.obs, 1)

    def reset(self):
        # Generate random ground truth parameters a and b.
        self.x_gt = [random.random(), random.random()]
        a, b = self.x_gt
        # Generate initial parameters by adding a small random noise to the ground truth.
        self.x = [a + 0.3 * random.random(), b + 0.3 * random.random()]
        # Create data points (x_data) uniformly spaced between 0 and 1.
        x_data = np.linspace(0, 1, self.data_num)
        # Compute the corresponding y_data with some added noise.
        y_data = a * np.exp(b * x_data) + np.random.normal(scale=0.01, size=self.data_num)
        # Stack the x_data and y_data to form observations.
        self.obs = np.stack((x_data, y_data), axis=1)

    def residual_factor(self):
        # Compute the residuals between observed data and the model's predictions.
        a, b = self.x
        return self.obs[:, 1] - (a * np.exp(b * self.obs[:, 0]))

    def jacobian_factor(self):
        # Compute the Jacobian matrix for the model.
        J = np.zeros((self.obs_dim * self.residual_dim, 2))
        a, b = self.x
        J[:, 0] = -np.exp(b * self.obs[:, 0])
        J[:, 1] = -a * self.obs[:, 0] * np.exp(b * self.obs[:, 0])
        return J


# a * exp(b * x) + c
class CostFactor_Env2(CostFactor):
    def __init__(self, is_initialized=False, x_gt=None, x=None, obs=None, is_numerical=False):
        # If the object is initialized, call the parent class constructor with the provided parameters.
        if is_initialized:
            super().__init__(is_numerical, x_gt, x, obs, 2)
        else:
            # Otherwise, set the number of data points and reset the environment.
            self.data_num = 100
            self.reset()
            # Call the parent class constructor with the generated data.
            super().__init__(is_numerical, self.x_gt, self.x, self.obs, 2)

    def reset(self):
        # Generate random ground truth parameters a, b, and c.
        self.x_gt = [5 * random.random(), 5 * random.random(), 5 * random.random()]
        a, b, c = self.x_gt
        # Generate initial parameters by adding a small random noise to the ground truth.
        self.x = [a + 0.3 * (random.random() - 0.5), b + 0.3 * (random.random() - 0.5),
                  c + 0.3 * (random.random() - 0.5)]
        # Create random data points (x_data) between 0 and 2.
        x_data = np.random.random([self.data_num, 2]) * 2
        # Compute the corresponding y_data with some added noise.
        y_data = a * np.exp(b * x_data) + c + np.random.normal(scale=0.01, size=[self.data_num, 2])
        # Initialize observations and assign x_data and y_data.
        self.obs = np.zeros([self.data_num, 4])
        self.obs[:, 0:2] = x_data
        self.obs[:, 2:4] = y_data

    def residual_factor(self):
        # Compute the residuals between observed data and the model's predictions.
        a, b, c = self.x
        return self.obs[:, 2:4] - (a * np.exp(b * self.obs[:, 0:2]) + c)

    def jacobian_factor(self):
        # Compute the Jacobian matrix for the model.
        J = np.zeros((self.obs_dim, 3 * self.residual_dim))
        a, b, c = self.x
        J[:, 0] = -np.exp(b * self.obs[:, 0])
        J[:, 1] = -a * self.obs[:, 0] * np.exp(b * self.obs[:, 0])
        J[:, 2] = -1
        J[:, 3] = -np.exp(b * self.obs[:, 1])
        J[:, 4] = -a * self.obs[:, 1] * np.exp(b * self.obs[:, 1])
        J[:, 5] = -1
        return J.reshape([self.obs_dim * self.residual_dim, 3])


# a11 * x1 + a12 * x2 + b1
# a21 * x1 + a22 * x2 + b2
class CostFactor_Env3(CostFactor):
    def __init__(self, is_initialized=False, x_gt=None, x=None, obs=None, is_numerical=False):
        # If the object is initialized, call the parent class constructor with the provided parameters.
        if is_initialized:
            super().__init__(is_numerical, x_gt, x, obs, 2)
        else:
            # Otherwise, set the number of data points and reset the environment.
            self.data_num = 100
            self.reset()
            # Call the parent class constructor with the generated data.
            super().__init__(is_numerical, self.x_gt, self.x, self.obs, 2)

    def reset(self):
        # Generate random ground truth parameters a11, a12, a21, a22, b1, and b2.
        self.x_gt = [random.random(), random.random(), random.random(), random.random(), random.random(),
                     random.random()]
        a11, a12, b1, a21, a22, b2 = self.x_gt
        # Generate initial parameters by adding a small random noise to the ground truth.
        self.x = [random.random(), random.random(), random.random(), random.random(), random.random(), random.random()]
        # Create random data points (x_data).
        x_data = np.random.random([self.data_num, 2])
        # Compute the corresponding y_data using the linear model with some added noise.
        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        y_data = x_data @ A.T + b.T + np.random.normal(scale=0.01, size=[self.data_num, 2])
        # Initialize observations and assign x_data and y_data.
        self.obs = np.zeros([self.data_num, 4])
        self.obs[:, 0:2] = x_data
        self.obs[:, 2:4] = y_data

    def residual_factor(self):
        # Compute the residuals between observed data and the model's predictions.
        a11, a12, b1, a21, a22, b2 = self.x
        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        return self.obs[:, 2:4] - (self.obs[:, 0:2] @ A.T + b.T)

    def jacobian_factor(self):
        # Compute the Jacobian matrix for the model.
        J = np.zeros((self.obs_dim * self.residual_dim, 6))
        a11, a12, b1, a21, a22, b2 = self.x
        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        for i in range(self.obs_dim):
            id = i * 2
            J[id, 0] = -self.obs[i, 0]
            J[id, 1] = -self.obs[i, 1]
            J[id, 2] = -1
            J[id + 1, 3] = -self.obs[i, 0]
            J[id + 1, 4] = -self.obs[i, 1]
            J[id + 1, 5] = -1
        return J


# a * x^3 + b * x^2 + c * x + d
class CostFactor_Env4(CostFactor):
    def __init__(self, is_initialized=False, x_gt=None, x=None, obs=None, is_numerical=False):
        # If the object is initialized, call the parent class constructor with the provided parameters.
        if is_initialized:
            super().__init__(is_numerical, x_gt, x, obs, 2)
        else:
            # Otherwise, set the number of data points and reset the environment.
            self.data_num = 100
            self.reset()
            # Call the parent class constructor with the generated data.
            super().__init__(is_numerical, self.x_gt, self.x, self.obs, 2)

    def reset(self):
        # Generate random ground truth parameters a, b, c, and d.
        self.x_gt = [random.random(), random.random(), random.random(), random.random()]
        a, b, c, d = self.x_gt
        # Generate initial parameters by adding a small random noise to the ground truth.
        self.x = [random.random(), random.random(), random.random(), random.random()]
        # Create random data points (x_data) between -1 and 1.
        x_data = 2 * (np.random.random([self.data_num, 2]) - 0.5)
        # Compute the corresponding y_data using the cubic polynomial model with some added noise.
        y_data = a * x_data ** 3 + b * x_data ** 2 + c * x_data + d + np.random.normal(scale=0.01,
                                                                                       size=[self.data_num, 2])
        # Initialize observations and assign x_data and y_data.
        self.obs = np.zeros([self.data_num, 4])
        self.obs[:, 0:2] = x_data
        self.obs[:, 2:4] = y_data

    def residual_factor(self):
        # Compute the residuals between observed data and the model's predictions.
        a, b, c, d = self.x
        x_data = self.obs[:, 0:2]
        y_data = self.obs[:, 2:4]
        y_obse = a * x_data ** 3 + b * x_data ** 2 + c * x_data + d
        return (y_data - y_obse)

    def jacobian_factor(self):
        # Compute the Jacobian matrix for the model.
        J = np.zeros((self.obs_dim, 8))
        x_data1 = self.obs[:, 0]
        x_data2 = self.obs[:, 1]
        J[:, 0] = -x_data1 ** 3
        J[:, 1] = -x_data1 ** 2
        J[:, 2] = -x_data1
        J[:, 3] = -1
        J[:, 4] = -x_data2 ** 3
        J[:, 5] = -x_data2 ** 2
        J[:, 6] = -x_data2
        J[:, 7] = -1
        return J.reshape([self.obs_dim * self.residual_dim, 4])


class CostFactor_1DGS(CostFactor):
    def __init__(self, x_gt=None, x=None, obs=None, is_numerical=False, is_great_init: bool = False,
                 gaussian_num: int = 50, parameter_space: int = 0):
        ## parameter_space:
        ## 0 :  self.x == [mean1,standard_variance1,opacity1,mean2,standard_variance2,opacity2,...],shape:(3*N,)
        ## 1 :  self.x == [mean1,log(variance1),logic(opacity1),mean2,log(variance2),logic(opacity2),...],shape:(3*N,)

        # If the object is initialized, call the parent class constructor with the provided parameters.
        self.gaussian_num = gaussian_num
        self.parameter_space = parameter_space
        if ((x_gt is not None) and
                (x is not None) and
                (obs is not None)):
            super().__init__(is_numerical, x_gt, x, obs, 1)
        else:
            self.obs_dim = random.randint(1000, 10000)
            if is_great_init:
                ## random initialization is not so good
                self.reset()
            else:
                ## ============================create target by file============================
                # import pandas as pd
                # target = pd.read_csv(
                #     rf"{os.path.dirname(__file__)}/../Introduction-to-Gaussian-Splatting/DailyDelhiClimateTest.csv")[
                #     'meantemp']
                # ## 设置移动平均窗口大小
                # window_size = 8
                # ## 计算移动平均值
                # target = target.rolling(window=window_size).mean()
                # # 用前一个有效值填充 NaN
                # target = target.bfill()
                # target = numpy.array(target.values)
                # self.obs_dim = len(target)
                # x_data = np.linspace(start=0, stop=len(target) - 1, num=self.obs_dim)
                ## ============================create target by file============================

                ## ============================create target with some usual functions============================
                x_data = np.linspace(start=0, stop=10 * np.pi, num=self.obs_dim)
                target = np.sin(x_data) + 10
                # target = np.polyval([0.01, -3, 2, 0], x=x_data) +10
                target += np.random.normal(loc=0, scale=0.005, size=len(target))  # 添加一些噪声
                ## ============================create target with some usual functions============================

                ## normalize
                target = (target - min(target)) / (max(target)-min(target))

                self.obs = np.zeros([self.obs_dim, 2])
                self.obs[:, 0] = x_data
                self.obs[:, 1] = target
                ## get the parameters
                means, variances, opacity = self.init_parameters()
                self.x_gt = None
                self.x = np.stack((means, variances, opacity), axis=0).T.reshape(-1)

            self.init_opacity = max(self.obs[:, 1]) * 0.1
            ## show data
            # import matplotlib.pyplot as plt
            # plt.plot(self.obs[:, 0], self.obs[:, 1])
            # plt.show()
            # Call the parent class constructor with the generated data.
            super().__init__(is_numerical, self.x_gt, self.x, self.obs, 1)

    def init_parameters(self) -> Tuple:
        means = np.linspace(start=min(self.obs[:, 0]), stop=max(self.obs[:, 0]), num=self.gaussian_num)
        # means = np.random.randint(low=min(self.obs[:, 0]), high=max(self.obs[:, 0]), size=(self.gaussian_num,))
        # 使用 NearestNeighbors 找到每个 mean 最近的 3 个 mean
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(means.reshape(-1, 1))
        distances, indices = nbrs.kneighbors(means.reshape(-1, 1))
        # 计算每个 mean 最近的 3 个 mean 的均方根距离的平均值，忽略自身的距离，作为方差初始值
        variances = np.sqrt(np.mean(distances[:, 1:] ** 2, axis=1))
        opacity = np.full(shape=(self.gaussian_num,), fill_value=max(self.obs[:, 1]) * 0.1)

        if self.parameter_space == 0:
            standard_variances = numpy.sqrt(variances)
            return means, standard_variances, opacity
        elif self.parameter_space == 1:
            # 在对数空间作为参数
            variances_log = numpy.log(variances)
            opacity_logic = numpy.log(opacity / (1 - opacity))
            return means, variances_log, opacity_logic

    def get_parameters(self):
        means = self.x.reshape(-1, 3)[:, 0]
        varlog_or_stdvar = self.x.reshape(-1, 3)[:, 1]
        opacity_or_opalogic = self.x.reshape(-1, 3)[:, 2]

        variances = numpy.empty_like(varlog_or_stdvar)
        opacity = numpy.empty_like(opacity_or_opalogic)
        if self.parameter_space == 0:
            variances = numpy.square(varlog_or_stdvar)
            opacity = opacity_or_opalogic
        elif self.parameter_space == 1:
            variances = numpy.exp(varlog_or_stdvar)
            opacity = 1 / (1 + numpy.exp(-opacity_or_opalogic))
        return means, variances, opacity

    def reset_single_gaussian(self, gaussian_id:int) -> None:
        # context:[mean1,var1,opacity1,mean2,var2,opacity2,...]
        self.x[3 * gaussian_id] = random.randrange(start=int(min(self.obs[:, 0])), stop=int(max(self.obs[:, 0])))
        means = self.x.reshape(-1, 3)[:, 0]
        distance_sorted = numpy.sort(numpy.square(means - self.x[3 * gaussian_id]))
        var = numpy.sqrt(distance_sorted[1:4].mean())
        if self.parameter_space == 0:
            self.x[3 * gaussian_id + 1] = numpy.sqrt(var)
            self.x[3 * gaussian_id + 2] = self.init_opacity
        elif self.parameter_space == 1:
            self.x[3 * gaussian_id + 1] = numpy.log(var)
            self.x[3 * gaussian_id + 2] = numpy.log(self.init_opacity/(1 - self.init_opacity))

    def reset(self)-> None:
        means, variances, opacity = self.init_parameters()
        self.x_gt = np.stack((means, variances, opacity), axis=0).T.reshape(-1)
        # add noise
        means_noise = means + means * np.random.randn(means.size) / 5
        variances_noise = variances + variances * np.random.randn(variances.size) / 5
        variances_noise = numpy.clip(a=variances_noise, a_min=0.01, a_max=None)
        opacity_noise = opacity + opacity * np.random.randn(opacity.size) / 5
        opacity_noise = numpy.clip(a=opacity_noise, a_min=0.01, a_max=None)
        self.x = np.stack((means_noise, variances_noise, opacity_noise), axis=0).T.reshape(-1)
        # generate data
        self.obs = np.zeros([self.obs_dim, 2])
        self.obs[:, 0] = np.linspace(start=0, stop=self.obs_dim - 1, num=self.obs_dim)
        self.obs[:, 1] = self.reconstructed_signal()

    def residual_factor(self) -> ndarray:
        # Compute the residuals between observed data and the model's predictions.
        return (self.obs[:, 1] - self.reconstructed_signal()).reshape(-1, 1)

    def jacobian_factor(self) -> ndarray:
        # Compute the Jacobian matrix for the model.
        J = np.zeros(shape=(self.obs_dim, 3 * self.gaussian_num))
        means, variances, opacity = self.get_parameters()
        varlog_or_stdvar = self.x.reshape(-1, 3)[:, 1]
        opacity_or_opalogic = self.x.reshape(-1, 3)[:, 2]
        for i in range(self.gaussian_num):
            if (
                ## variance is too low or too high
                variances[i] <= 0
                or variances[i] >= len(self.obs[:, 0])/2
                ## opacity is too low or too high
                or opacity[i] <= 0
                or opacity[i] >= max(self.obs[:, 1])
            ):continue
            J[:, 3 * i] = -(
                    opacity[i] * (self.obs[:, 0] - means[i]) / variances[i] *
                    numpy.exp(-0.5 * (self.obs[:, 0] - means[i]) ** 2 / variances[i])
            )
            if self.parameter_space == 0:
                J[:, 3 * i + 1] = -(
                        opacity[i] * ((self.obs[:, 0] - means[i]) ** 2) / (varlog_or_stdvar[i] ** 3) *
                        numpy.exp(-0.5 * ((self.obs[:, 0] - means[i]) ** 2) /  varlog_or_stdvar[i] ** 2)
                )
                J[:, 3 * i + 2] = -numpy.exp(-0.5 * ((self.obs[:, 0] - means[i]) ** 2) /  varlog_or_stdvar[i] ** 2)
            elif self.parameter_space == 1:
                J[:, 3 * i + 1] = -(
                        opacity[i] * ((self.obs[:, 0] - means[i])**2) / (2 * variances[i] ** 2) *
                        numpy.exp(-0.5 * (self.obs[:, 0] - means[i]) ** 2 / variances[i]) * numpy.exp(varlog_or_stdvar[i])
                )
                J[:, 3 * i + 2] = -(
                                    numpy.exp(-0.5 * (self.obs[:, 0] - means[i]) ** 2 / variances[i]) *
                                    numpy.exp(-opacity_or_opalogic[i])/
                                    ((1 + numpy.exp(-opacity_or_opalogic[i]))**2)
                )
        return J

    def update(self)->None:
        ## update residual
        self._residual = self.residual_factor()
        means, variances, opacity = self.get_parameters()
        # 使用数值化雅可比矩阵
        if self.numerical_J:
            self._jacobian = np.zeros(shape=(self.obs_dim, 3 * self.gaussian_num))
            for i in range(self.gaussian_num):
                if (
                        ## variance is too low or too high
                        variances[i] <= 0
                        or variances[i] >= len(self.obs[:, 0]) / 2
                        ## opacity is too low or too high
                        or opacity[i] <= 0
                        or opacity[i] >= max(self.obs[:, 1])
                ): continue
                for j in range(0, 3):
                    self.x[3 * i + j] -= 1e-5
                    residual_1 = self.residual_factor().reshape(-1)
                    self.x[3 * i + j] += 2e-5
                    residual_2 = self.residual_factor().reshape(-1)
                    self.x[3 * i + j] -= 1e-5
                    self._jacobian[:, 3 * i + j] = (residual_2 - residual_1) / 2e-5
        else:
            self._jacobian = self.jacobian_factor()

    def reconstructed_signal(self)->ndarray:
        means, variances, opacity = self.get_parameters()
        y_data = numpy.zeros(self.obs_dim)
        for i in range(self.gaussian_num):
            y_data += opacity[i] * numpy.exp(-0.5 * (self.obs[:, 0] - means[i]) ** 2 / variances[i])
        return y_data

    def reconstructed_disperse(self)->List:
        means, variances, opacity = self.get_parameters()
        y_data_list = []
        for i in range(self.gaussian_num):
            y_data_list.append(opacity[i] *numpy.exp(-0.5 * (self.obs[:, 0] - means[i]) ** 2 / variances[i]))
        return y_data_list

    def calculate_psnr(self)->float:
        mse = numpy.mean((self.obs[:, 1] - self.reconstructed_signal()) ** 2)
        return 10 * numpy.log10(numpy.max(abs(self.obs[:, 1]))**2 / mse)


if __name__ == '__main__':
    # check jacobian
    # cost_factor = CostFactor_Env1(is_numerical=True)
    # jacobian_analytical = cost_factor.jacobian_factor()
    # jacobian_numerical = cost_factor.jacobian()
    # print(np.linalg.norm(jacobian_analytical - jacobian_numerical))
    #
    # cost_factor = CostFactor_Env2(is_numerical=True)
    # jacobian_analytical = cost_factor.jacobian_factor()
    # jacobian_numerical = cost_factor.jacobian()
    # print(np.linalg.norm(jacobian_analytical - jacobian_numerical))

    cost_factor = CostFactor_1DGS(is_numerical=True, parameter_space=0)
    jacobian_analytical = cost_factor.jacobian_factor()
    jacobian_numerical = cost_factor.jacobian()
    print(np.linalg.norm(jacobian_analytical - jacobian_numerical))

    # cost_factor = CostFactor_Env4(is_numerical=True)
    # jacobian_analytical = cost_factor.jacobian_factor()
    # jacobian_numerical = cost_factor.jacobian()
    # print(np.linalg.norm(jacobian_analytical - jacobian_numerical))

