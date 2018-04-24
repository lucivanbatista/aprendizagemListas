import numpy as np
import math

def minkowski(X, X_row, p):
    X_ = abs((X - X_row) ** p)
    return np.sum(X_, axis=1) ** (1/p)

def euclidean_distance(X, X_row):
    X_ = (X - X_row) ** 2 # Calcula a soma de cada lado (px - qx) e (py - qy)
    return np.sum(X_, axis=1) ** 0.5 # Axis 1 para somar cada coluna de sua linha

def manhattan_distance(X, X_row):
    X_ = abs(X - X_row)
    return np.sum(X_, axis=1)

def chebyshev(X, X_row):
    X_ = abs(X - X_row)
    return np.max(X_, axis=1)

def euclidean_distance_minkowski(X, X_row):
    return minkowski(X, X_row, 2)

def manhattan_distance_minkowski(X, X_row):
    return minkowski(X, X_row, 1)