from mylibs import stats as st
import numpy as np

class SimpleLinearRegression:
    
    #b1
    def coef(self, x, y):
        meanX = st.mean(x)
        meanY = st.mean(y)
        somNum = 0.0
        somDen = 0.0
        for i in range(len(x)):
            somNum += (x[i] - meanX) * (y[i] - meanY)
            somDen += (x[i] - meanX) ** 2
        return somNum / somDen

    #b0
    def intercept(self, x, y, b1):
        meanX = st.mean(x)
        meanY = st.mean(y)
        intercept_b0 = meanY - self.b1 * meanX
        return intercept_b0[0]
    
    def fit(self, x, y):
        self.b1 = self.coef(x, y)
        self.b0 = self.intercept(x, y, self.b1)
        return
    
    def predict(self, x):
        self.y = np.zeros(x.size)
        for i in range(len(x)):
            self.y[i] = float(self.b0 + self.b1 * x[i])
        return self.y