import numpy as np
from math import inf

def split_stratified_train_test(y, perc_train, seed=0):
    np.random.seed(seed)
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_count = np.bincount(y_indices) # quantos elementos iguais, tenho em casa classe
    class_indices = np.split(np.argsort(y_indices, kind='mergesort'), np.cumsum(class_count)[:-1])
    
    train = []
    test = []
    # n: contador de quantos elementos a classe tem, c array dos indices da classe com 'a' elementos
    for n, c in zip(class_count, class_indices): 
        number_train = (int(n * perc_train)) # qtd de elementos q vou ter no treino da classe c
        np.random.shuffle(c)
        train.extend(c[0:number_train])
        test.extend(c[number_train:])
    
    return np.array(train), np.array(test)