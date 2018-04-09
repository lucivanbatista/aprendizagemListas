import math
import numpy as np

def mse(y_true, y_pred):
    n = len(y_true)
    som = np.sum((y_true - y_pred) ** 2)
    mse = som / n
    return mse

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    n = len(y_true)
    som = np.sum(abs(y_true - y_pred))
    return som / n 