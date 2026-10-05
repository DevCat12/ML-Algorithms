from numpyClassifier import NumpyClassifier
from logisticRegression import NumpyLogReg
import numpy as np
from logisticRegression import *
import time

def ova_tuner(x, t, valSet):
    print("Tuning commenced.")
    # print(f"Dataset snippets:\nx: {x[0:10]}, t: {t[0:10]}\n")

    start = time.perf_counter()
    tuningValues = logRegTuner(x, t, valSet)
    end = time.perf_counter()
    elapsed_time = end - start

    print(f"Tuning finished. Time elapsed: {elapsed_time:.1f}s\n")
    return tuningValues

class NumpyOneVsRest(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias=bias
        self.trained = False

    def fit(self,x_train, tm_train, x_val, tm_val):

        if self.trained == True:
            retrain = input("Model already trained. Retrain? (y/n)")
            if retrain == "f":
                return
        
        ova_datasets, fittingValues = self.tune_datasets(x_train, tm_train, x_val, tm_val)
        logregModels = []
        
        if len(ova_datasets) != len(fittingValues):
            print("lengths not matching!")
            return
         
        for i in range(len(ova_datasets)):
            model = NumpyLogReg()
            t_train_ova = ova_datasets[i][0]
            t_val_ova = ova_datasets[i][1]

            lr = fittingValues[i][0]
            tol = fittingValues[i][1]
            epochs = fittingValues[i][2]

            model.fit(x_train, t_train_ova, lr, tol, epochs)
            logregModels.append(model)
        
        print("Model successfully trained")
        
        self.logregModels = logregModels
        self.fittingValues = fittingValues
        self.ova_datasets = ova_datasets
        self.trained = True
        return self.logregModels

    def tune_datasets(self, x, tm_train, x_val, tm_val):
        """X_train is a Nxm matrix, N data points, m features
        t_train is avector of length N,
        the targets values for the training data"""
        ova_datasets = []
        fittingValues = []

        classes = np.unique(tm_train)
        for i in classes:
            t_train = (tm_train == i).astype('int')
            t_val = (tm_val == i).astype('int')
            tuningValues = ova_tuner(x, t_train, [x_val, t_val])
            fittingValues.append(tuningValues)
            ova_datasets.append((t_train, t_val))
        print("Model successfully tuned. Returning datasets and optimal fitting values")
        self.classes = classes
        return ova_datasets, fittingValues
    
    def predict(self, x):
        if self.trained == False:
            print("Model not trained. Cannot predict scores")
            return
        # Initialize a zero array for the prediction scores
        n_classes = len(self.classes)
        m_datapoints = x.shape[0]
        scores = np.zeros((n_classes, m_datapoints))

        # Run the input through each logistic regression model
        for i, model in enumerate(self.logregModels):
            # The score for a class is the predicted probability that the input belongs to that class
            scores[i] = model.predict_probability(x)
        # The prediction for each input is the class that got the highest score
        return np.argmax(scores, axis=0)