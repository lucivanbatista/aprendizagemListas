import numpy as np
from scipy.spatial import distance

class myKNN():
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        p = []
        
        for row in X_test: # Para cada linha em X_test
            y = self.closest(row)
            p.append(y)
        return p   
    
    def closest(self, row):
        min_dist = distance.euclidean(row, self.X_train[0]) # distancia entre a linha e o 1º elemento do X_train
        min_indice = 0 # armazenando o indice que possui a menor distancia 
        
        for i in range(1, len(self.X_train)): # Para todos os valores de 1 até o tamanho de X_train
            dist = distance.euclidean(row, self.X_train[i])
            if dist < min_dist:
                min_dist = dist
                min_indice = i
        return self.y_train[min_indice]