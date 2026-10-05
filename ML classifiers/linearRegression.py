import numpy as np
from numpyClassifier import NumpyClassifier
from helperFunctions import *

# Linear regression classifier class
class NumpyLinRegClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias=bias
    
    def fit(self, X_train, t_train, lr = 0.1, epochs=10):
        """X_train is a NxM matrix, N data points, M features
        t_train is avector of length N,
        the target class values for the training data
        lr is our learning rate
        """
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        (N, M) = X_train.shape
        
        self.weights = weights = np.zeros(M)
        
        for epoch in range(epochs):
            # print("Epoch", epoch)
            weights -= lr / N *  X_train.T @ (X_train @ weights - t_train)      
    
    def predict(self, X, threshold=0.5):
        """X is a KxM matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold

def cl_tuner(
    cl: NumpyLinRegClass,
    X,
    T,
    X_val,
    t_val,
    lr_min,
    lr_max,
    max_epochs,
    step_size=0.01,
):
    bestAcc = 0
    bestlr = 0
    bestEpochs = 0
    for lr in np.arange(lr_min, lr_max, step_size):
        for epochs in range(max_epochs):
            cl.fit(X, T, lr, epochs)
            predicted = cl.predict(X_val)
            currentAcc = accuracy(predicted, t_val)
            if currentAcc > bestAcc:
                bestAcc, bestlr, bestEpochs = currentAcc, lr, epochs
    return [bestAcc, bestlr, bestEpochs]