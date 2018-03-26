import math
import numpy as np

# n_elem - número total de elementos (treino + teste).
# perc_train - percentual dos dados usados para treino.
# seed - semente para geração de números randômicos.
# saída (output): array com os índices do dados de treino e array com os índices do dados de teste.
# Exemplo de entrada: split_train_test(10, 0.7, 0)
# Saída - 2 arrays com índices dos dados de treino (idx_train) e de teste (idx_test), respectivamente: [2, 8, 4, 9, 1, 6, 7], [3, 0, 5]
def split_train_test(n_elem, perc_train, seed):
    total = [ i for i in range(n_elem)]
    indice = math.ceil(n_elem * 0.7) # 70% dos dados
    
    np.random.seed(seed)
    np.random.shuffle(total) # no array dos índices vamos misturar
    
    idx_train = total[:indice]
    idx_test = total[indice:]
    
    return idx_train, idx_test