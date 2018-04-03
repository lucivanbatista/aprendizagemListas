import math
import numpy as np

# n_elem - número total de elementos.
# n_split - número de folds. Mínimo: 2.
# shuffle - aleatoriza a ordem dos dados (True) ou não (False).
# seed - determina uma semente para geração de números aleatórios ou não (None).
# Retorno: 2 arrays (idx_train e idx_test), cada um com n_splits elementos: 
# um com os índices de treino. Exemplo para n_splits=3, teremos idx_train[0], idx_train[1] e idx_train[2].
# um com os índices de teste. Exemplo para n_splits=3, teremos idx_test[0], idx_test[1] e idx_test[2].

def slipt_k_fold(n_elem, n_splits, shuffle, seed):
    total = [ i for i in range(n_elem)]
    
    if shuffle:
        np.random.shuffle(total)# no array dos índices vamos misturar
    if seed:
        np.random.seed(seed)
    
    fold_sizes = (n_elem // n_splits) * np.ones(n_splits, dtype=np.int)
    fold_sizes[:n_elem % n_splits] += 1
    
    idx_train = [0] * n_splits
    idx_test = [0] * n_splits
    i = 0
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        range_idx_train = np.arange(start, stop) # Usado para eliminar os elementos de idx_train
        idx_test[i] = np.array(total[start:stop])
        idx_train[i] = np.delete(total, range_idx_train)
        current = stop
        i = i + 1
        
    return idx_test, idx_train