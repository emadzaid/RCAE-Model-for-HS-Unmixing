import numpy as np


## First RMSE function

def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse


# ___________________________________________________

## Second RMSE function

def calculate_rmse(x, x_hat):
    P = 4
    error = np.sqrt(np.sum((x - x_hat) ** 2) / P) 
    return error
