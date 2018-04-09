from mylibs import stats as st
import numpy as np

class SimpleLinearRegression:
    
    def __isNumpyArray(self, arr):
        return type(arr) == np.ndarray
    
    #b1
    def coef(self, x, y):
        if self.__isNumpyArray(x):
            x = x[:, 0]
        
        meanX = st.mean(x)
        meanY = st.mean(y)
        return np.sum((x - meanX) * (y - meanY)) / np.sum((x - meanX) ** 2)

    #b0
    def intercept(self, x, y, b1):
        meanX = st.mean(x)
        meanY = st.mean(y)
        intercept_b0 = meanY - self.b1 * meanX
        return intercept_b0
    
    def fit(self, x, y):
        self.b1 = self.coef(x, y)
        self.b0 = self.intercept(x, y, self.b1)
        return
    
    def predict(self, x):
        self.y = np.zeros(x.size)
        for i in range(len(x)):
            self.y[i] = float(self.b0 + self.b1 * x[i])
        return self.y