import numpy as np
import matplotlib.pyplot as plt

# Important functions
def add_bias(X, bias):
    """X is a NxM matrix: N datapoints, M features
    bias is a bias term, -1 or 1, or any other scalar. Use 0 for no bias
    Return a Nx(M+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1)) * bias # Make a N*1 matrix of biases
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 

def accuracy(predicted, gold):
    return np.mean(predicted == gold)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(logits):
    exp_scores = np.exp(logits) #e^kz for k in logits
    sigma = np.sum(exp_scores, axis = 1, keepdims = True)
    return exp_scores/sigma

def one_hot_encode(t, numClasses = None): #takes list t as input, returns a matrix representing every y-value in t as a one-hot-coded list
    n_classes = np.max(t) + 1 if numClasses is None else numClasses
    n_datapoints = len(t)
    one_hot = np.zeros((n_datapoints, n_classes)) 
    one_hot[np.arange(n_datapoints), t] = 1 #np vector magic. Does the same as the loop under, but more efficient
    # for i in range(n_datapoints):
    #     j = t[i]
    #     one_hot[i][j] = 1
    return one_hot
    
    # ex: t = [0,1,2,0,3]. num_classes = 4 [0, 1, 2, 3] are the possible classes
    # len(t) == 5 -> 5 datapoints, each belonging to one of the classes.
    # One-hot encoder creates a matrix containing a list per datapoint representing the t-value in one-hot-coded format
    # ex: [[1,0,0,0,0], [0,1,0,0,0], ...] for the example t, where the first list symbolizes t value 0, and the second one t value 1.

def plotLoss(trainLoss, valLoss):
    plt.figure(figsize=(8,5))
    plt.plot(trainLoss, label="Training loss")
    plt.plot(valLoss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

def plotAcc(trainAcc, valAcc):
    plt.figure(figsize=(8,5))
    plt.plot(valAcc, label="validation accuracy")
    plt.plot(trainAcc, label="training accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.title("Training and Validation accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

def print_tuning_values(values):
    print(f"best accuracy: {values[0]}")
    print(f"optimal values for lr and number of epochs: {values[1]}|{values[2]}")

def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a prediction of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    plt.figure(figsize=size) # You may adjust this

    if type(clf) == list:
        for cl in clf:
            Z = cl.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    else:    
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Classify each meshpoint.
        Z = Z.reshape(xx.shape)
        # Put the result into a color plot
        plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")
