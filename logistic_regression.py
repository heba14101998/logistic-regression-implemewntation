import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters        
        self.weights = None
        self.bias = None
        
        
    def fit (self, X, y):
        
        n_samples, n_features = X.shape
        
        # initalize the weights
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            
            # 1. multiply the X matrix and the weights 
            z = np.dot(X, self.weights) + self.bias
            # 2. apply the sigmoid function
            y_pred = self._sigmoid(z) 
            # 3. compute gradients and update weights and bias
            self.weights -= (self.lr/n_samples) * np.dot(X.T, (y_pred - y))
            self.bias -= (self.lr/n_samples) * np.sum(y_pred - y)
            
            
    def predict(self, X):
        
        # 1. multiply the X matrix and the weights 
        z = np.dot(X, self.weights) + self.bias
        # 2. apply the sigmoid function
        y_pred = self._sigmoid(z) 
        y_pred_res = [1 if i >0.5 else 0 for i in y_pred]
        
        return np.array(y_pred_res)
    
    def _sigmoid(self, z):
        return 1/ (1 + np.exp(-z) )
    
        