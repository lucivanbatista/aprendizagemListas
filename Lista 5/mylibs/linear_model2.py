import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate = 0.0001, epochs = 5000):
        self.beta = None
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, x_, y_):
        x = np.hstack((np.ones(x_.shape[0]).reshape(x_.shape[0], 1), x_)) # add 1 for beta_0 intercept
        y = y_.reshape(y_.shape[0], 1)
        
        if self.beta is None:
            self.beta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        
        for step in np.arange(self.epochs):
            x_beta = np.dot(x, self.beta)
            y_hat = 1 / (1 + np.exp(-x_beta))
            likelihood = np.sum(np.log(1 - y_hat)) + np.dot(y.T, x_beta)
            preds = np.round( y_hat )
            gradient = np.dot(np.transpose(x), y - y_hat)
            self.beta = self.beta + self.learning_rate * gradient
    
    def predict(self, x_):
        x = np.hstack((np.ones(x_.shape[0]).reshape(x_.shape[0], 1), x_))
        x_beta = np.dot(x, self.beta)
        y_hat = 1 / (1 + np.exp(-x_beta))
        preds = (np.round(y_hat)).reshape((y_hat.shape[0],))
        
        return preds
    
    def predict_proba(self, x_):
        x = np.hstack((np.ones(x_.shape[0]).reshape(x_.shape[0], 1), x_))
        x_beta = np.dot(x, self.beta)
        y_hat = 1 / (1 + np.exp(-x_beta))
        preds = y_hat.reshape((y_hat.shape[0],))
        
        return preds