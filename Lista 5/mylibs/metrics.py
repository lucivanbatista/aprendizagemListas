import numpy as np
from sklearn import metrics

def matriz_confusao(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

def accuracy(y_true, y_pred):
    cm = matriz_confusao(y_true, y_pred)
    return np.sum(np.diagonal(cm)) / np.sum(cm)

def precision(y_true, y_pred):
    cm = matriz_confusao(y_true, y_pred)
    s = np.empty(0)
    for i in range(cm.shape[0]):
        s = np.append(s, cm[i,i] / np.sum(cm[:,i]))
    return s

def accuracy1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num = np.where(y_true == y_pred)
    return num[0].shape[0]/y_pred.shape[0]

def recall(y_true, y_pred):
    cm = matriz_confusao(y_true, y_pred)
    s = np.empty(0)
    for i in range(cm.shape[0]):
        s = np.append(s, cm[i,i] / np.sum(cm[i,:]))
    return s

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r))