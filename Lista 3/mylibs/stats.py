import math
import numpy as np

def mean(x):
    return np.sum(x) / len(x)

def stdev(x):
    m = mean(x)
    return math.sqrt(np.sum((x - m) ** 2) / len(x))

def var(y):
    m = mean(y)
    return np.sum((y - m) ** 2) / len(y)