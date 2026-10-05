import numpy as np
from numpyClassifier import NumpyClassifier
from helperFunctions import *
import time

def multinomial_tuner(
    x,
    t,
    valSet: tuple[np.ndarray, np.ndarray],
    eta_min=0.01,
    eta_max=2,
    step_size=0.01,
):
    print("Multinomial tuning commenced.")
    start = time.perf_counter()
    tuningValues = None

    cl = NumpyMultinomial()
    bestAcc = float("-inf")
    for lr in np.arange(eta_min, eta_max, step_size):
        for tol in np.arange(0.01, 0.001, -0.001):
            result = cl.fit(x, t, lr, tol, valSet=valSet)
            if result is None:
                raise RuntimeError("NumpyMultinomial.fit returned no validation result.")
            epochs, acc = result
            if acc > bestAcc:
                bestAcc = acc
                tuningValues = [lr, tol, epochs]

    if tuningValues is None:
        raise ValueError("No parameter combinations were tested.")

    end = time.perf_counter()
    elapsed_time = end - start
    print(f"Tuning finished. Time elapsed: {elapsed_time:.1f}s")
    print(f"Best accuracy achieved: {bestAcc} with tuning values (lr|tol|epochs): {tuningValues[0]}|{tuningValues[1]:.3}|{tuningValues[2]}\n")
    return tuningValues
    
class NumpyMultinomial(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias=bias
        self.trained = False
    
    def fit(self, x_train, t_multi_train, eta=0.01, tol=0.001, n_epochs_no_update=5, valSet: tuple[np.ndarray, np.ndarray] | None = None):
        if valSet is None:
            raise ValueError("valSet must be a pair of validation features and labels.")

        x,t = x_train, t_multi_train
        xv, tv = valSet
        (N,m) = x.shape #N datapoints, m features
        k = np.max(t) + 1 #k distinct classes

        xb = add_bias(x, self.bias)
        t_onehot = one_hot_encode(t)
        self.weights = weights = np.zeros((m+1, k)) #matrix with shape (m+1, k) | m+1 features (with bias), k classes

        minLoss = float("inf")
        bestWeights = weights.copy()
        bestAcc = accuracy(self.predict(xv), tv)
        bestEpochs = 0
        patience = n_epochs_no_update

        epochs = 0
        while patience > 0:
            y_hat = self.forward(xb, bias = False) #predicted probabilities
            gradient = xb.T @ (y_hat - t_onehot)/N
            weights -= eta*gradient #update weights w/ gradient descent

            val_loss = self.cross_entropy(xv, tv)
            valAcc = accuracy(self.predict(xv), tv)

            improvement = (minLoss - val_loss) / minLoss if np.isfinite(minLoss) else float("inf")
            if val_loss < minLoss:
                minLoss = val_loss
                bestWeights = weights.copy()
                bestAcc = valAcc
                bestEpochs = epochs + 1

            if improvement > tol:
                patience = n_epochs_no_update
            else:
                # print(f"Improvement insufficient. deltaLoss: {deltaLoss}")
                patience -= 1
            epochs += 1
        
        # print(f"Model succesfully trained. Best accuracy: {bestAcc}")
        self.trained = True
        self.weights = bestWeights
        return [bestEpochs, bestAcc]
    
    def fitSimple(self, x_train, t_multi_train, eta, epochs):
        (N,m) = x_train.shape #N datapoints, m features
        k = np.max(t_multi_train) + 1 #k classes

        xb = add_bias(x_train, self.bias)
        t_onehot = one_hot_encode(t_multi_train)
        self.weights = weights = np.zeros((m+1, k))

        for i in range(epochs):
            y_hat = self.forward(xb, bias = False) #predicted probabilities
            gradient = xb.T @ (y_hat - t_onehot)/N
            weights -= eta*gradient #update weights w/ gradient descent

        self.trained = True
        self.weights = weights
        print("simple fitting completed")

    def cross_entropy(self, x, t, bias = True): #categorical cross-entropy loss
        y = self.forward(x, bias)
        y = np.clip(y, 1e-12, 1 - 1e-12)  #prevents numerical overflow (log0)
        t_onehot = one_hot_encode(t)
        loss_per_sample = -np.sum(t_onehot * np.log(y), axis = 1)
        return float(np.mean(loss_per_sample))

    def forward(self, x, bias = True): #forwards to softmax() function to calculate y_hat (predicted values)
        if bias:
            x = add_bias(x, self.bias)
        return softmax(x @ self.weights)

    def predict(self, x, bias = True):
        y_hat = self.forward(x, bias)
        predicted_classes = np.argmax(y_hat, axis = 1)
        return predicted_classes