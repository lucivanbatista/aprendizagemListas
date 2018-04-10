import numpy as np

class LogisticRegression:
    
    def fit(self, x, y):
        beta = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        x_beta = np.dot(x, beta)
        y_hat = 1 / (1 + np.exp(-x_beta))

        likelihood = np.sum(np.log(1 - y_hat)) + np.dot(y.T, x_beta)
        
        learning_rate = 0.0001
        epochs = 100
        
        for step in np.arange(epochs):
            x_beta = np.dot(x, beta)
            y_hat = 1 / (1 + np.exp(-x_beta))
            likelihood = np.sum(np.log(1 - y_hat)) + np.dot(y.T, x_beta)
            preds = np.round( y_hat )
            accuracy = np.sum(preds == y)*1.00/len(preds)
            gradient = np.dot(np.transpose(x), y - y_hat)
            beta = beta + learning_rate*gradient
            if( step % 10 == 0):
                print("After step {}, likelihood: {}; accuracy: {}".format(step+1, likelihood, accuracy))
                
        self.beta = beta
        
        
        

    def predict(self, x):
        b0 = self.beta[0]
        b1 = self.beta[1]
        b2 = self.beta[2]
        x1 = np.array(x[0])
        x2 = np.array(x[1])
        return np.round(1.0 / (1.0 + np.exp(-(b0 + b1 * x1 + b2 * x2))))