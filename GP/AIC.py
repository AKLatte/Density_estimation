from sklearn.metrics import mean_squared_error as MSE
import numpy as np
from scipy.stats import norm

import input_data

def AIC(ys_actual, ys_pred, num_feature):
    num_data = len(ys_actual)
    mse = MSE(ys_actual, ys_pred)
    return num_data * np.log(2 * np.pi) + num_data * np.log(mse) + num_data + 2 * (num_feature + 2)

dataset = input_data.dataset
ys_actual = dataset[:, 0]
ys_pred = norm.pdf(dataset[:, 1], loc=607.7, scale=169.2)

print('aic(norm): {}'.format(AIC(ys_actual, ys_pred, 2)))