import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split,LeaveOneOut,KFold


def GetMSE(predictions,yTest):
    errors = (predictions-yTest)
    mse = np.sum(errors**2)/len(errors)
    return mse

def KFoldCVLambdaSelector(X,y,lambdas,model):
    np.random.seed(1234)
    #split the data
    kf = KFold(n_splits=len(lambdas),random_state=1234,shuffle=False)
    bestModelLambda = -1
    bestModelMSE = float("inf")
    currentIter = 0
    #for each split, apply the regression and store the value of lambda
    for trainIndex, testIndex in kf.split(X):
        #get the splits
        xTrain, xTest = X.iloc[trainIndex], X.iloc[testIndex]
        yTrain, yTest = y.iloc[trainIndex], y.iloc[testIndex]
        #set the lambda value
        currLambda = lambdas[currentIter]
        currentIter += 1
        modelFit = model(alpha=currLambda).fit(xTrain,yTrain)
        mse = GetMSE(modelFit.predict(xTest),yTest)
        if mse < bestModelMSE:
            bestModelMSE = mse
            bestModelLambda = currLambda
    
    return (bestModelLambda)

#define own bootstrap function
def bootstrap(data,bootFunc,iterations):
    n = iterations
    #empty array of 
    modelResults = []
    for i in range(0,n):
        np.random.seed(i)
        indices = np.random.choice(n, n, replace=True)
        modelResults.append(bootFunc(data,indices))
    return modelResults


def InjectFactors(df,columns):
    result = df;
    for col in columns:
        result[col] = pd.factorize(df[col])[0]
    return result