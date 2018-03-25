import math
import numpy as np

def mse(y_true, y_pred):
    n = len(y_true)
    som = 0.0
    for i in range(n):
        som += (y_true[i] - y_pred[i]) ** 2
    mse = som / n
    return mse

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    n = len(y_true)
    som = 0.0
    for i in range(n):
        som += abs(y_true[i] - y_pred[i])
    mae = som / n
    return mae