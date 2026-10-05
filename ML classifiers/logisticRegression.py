import numpy as np
from numpyClassifier import NumpyClassifier
from helperFunctions import *

def logRegTuner(
    x,
    t,
    valSet: tuple[np.ndarray, np.ndarray],
    lr_min=0.01,
    lr_max=2,
    step_size=0.01,
):
    cl = NumpyLogReg()
    bestAcc = float("-inf")
    tuningValues = None
    for lr in np.arange(lr_min, lr_max, step_size):
        for tol in np.arange(0.01, 0.001, -0.001):
            result = cl.fit(x, t, lr, tol, valSet=valSet)
            if result is None:
                raise RuntimeError("NumpyLogReg.fit returned no validation result.")
            epochs, acc = result
            # print(f"acc|lr|tol = {acc}|{lr}|{tol}")
            if acc > bestAcc:
                bestAcc = acc
                tuningValues = [lr, tol, epochs]
    if tuningValues is None:
        raise ValueError("No parameter combinations were tested.")

    print(f"Logreg-model successfully tuned. Best accuracy: {bestAcc} achieved with lr|tol|epochs = {tuningValues[0]}|{round(tuningValues[1], 3)}|{tuningValues[2]}")
    return tuningValues

class NumpyLogReg(NumpyClassifier):
    def __init__(self, bias = -1):
        self.bias = bias
        self.n_epochs_trained = 0
        self.lastTrainingHistory = {}   

    def fit(self, X_train, t_train, eta = 0.1, tol = 0.001, epochs = 20, n_epochs_no_update = 5, valSet: tuple[np.ndarray, np.ndarray] | None = None):
        """X_train is a Nxm matrix, N data points, m features
        t_train is avector of length N,
        the targets values for the training data"""
        (N, m) = X_train.shape
        X_train = add_bias(X_train, self.bias)
        self.weights = weights = np.zeros(m+1)
        # print("weights initialized:\n", self.weights)

        if valSet is not None:
            x_val, t_val = valSet
            plotTrainLoss = []
            plotValLoss = []
            plotTrainAcc = []
            plotValAcc = []

            minLoss = float("inf")
            bestWeights = weights.copy()
            bestAcc = accuracy(self.predict(x_val), t_val)
            bestEpochs = 0
            patience = n_epochs_no_update
            epochs_trained = 0

            for epoch in range(epochs):
                weights -= eta / N * X_train.T @ (self.forward(X_train) - t_train)
                epochs_trained = epoch + 1

                trainLoss = np.mean(self.cross_entropy(X_train, t_train, False))
                trainAcc = accuracy(self.predict(X_train, bias = False), t_train)

                valLoss = np.mean(self.cross_entropy(x_val, t_val))
                valAcc = accuracy(self.predict(x_val), t_val)

                plotTrainLoss.append(trainLoss) #for plotting purposes
                plotValLoss.append(valLoss)
                plotTrainAcc.append(trainAcc)
                plotValAcc.append(valAcc)

                improvement = (minLoss - valLoss) / minLoss if np.isfinite(minLoss) else float("inf")
                if valLoss < minLoss:
                    minLoss = valLoss
                    bestWeights = weights.copy()
                    bestAcc = valAcc
                    bestEpochs = epoch + 1

                if improvement > tol:
                    patience = n_epochs_no_update
                else:
                    patience -= 1
                    if patience == 0:
                        break

            self.lastTrainingHistory = {"trainLoss" : plotTrainLoss, "valLoss" : plotValLoss, "trainAcc" : plotTrainAcc, "valAcc" : plotValAcc}
            self.weights = bestWeights
            self.n_epochs_trained = epochs_trained
            # print(f"logreg-model successfully trained. Best acc found: {bestAcc} in {bestEpochs} epochs")
            return [bestEpochs, bestAcc]
        else:
            for i in range(epochs):
                weights -= eta / N * X_train.T @ (self.forward(X_train) - t_train)
            self.n_epochs_trained = epochs

    def cross_entropy(self, x, t, bias = True): #Returns in-sample cross entropy loss
        if bias:
            x = add_bias(x, self.bias)
        y = self.forward(x)
        y = np.clip(y, 1e-12, 1 - 1e-12)  #prevents numerical overflow (log0)
        return -(t*np.log(y) + (1-t)*np.log(1-y)) #cross entropy loss calculated for whole dataset

    
    def forward(self, X):
        return sigmoid(X @ self.weights)
    
    def predict(self, x, threshold=0.5, bias = True):
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        if bias:
            z = add_bias(x, self.bias)
            return (self.forward(z) > threshold).astype('int')
        return (self.forward(x) > threshold).astype('int')
    
    def predict_probability(self, x, bias = True):
        if bias:
            z = add_bias(x, self.bias)
            return self.forward(z)
        return self.forward(x)

    def getHistory(self):
        return self.lastTrainingHistory
