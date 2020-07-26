#!flask/bin/python

import pandas as pd
from sklearn import linear_model
from sklearn import datasets
import pickle
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import time
import os


def BuildModel():
    iris = sklearn.datasets.load_iris()
    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.90)
    # creating model
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train, labels_train)
    print('Model built successfully')
    return rf


def backupCurrentModel(modelFileName):
    
    status = 'false'
    #Get Timestamp to rename file
    T = time.asctime()
    T = T.replace(" ", "_") 
    T = T.replace(":", "_")
    # checking that file1.txt exists or not
	# if does not exist - will open myfile and read
    if os.path.exists(modelFileName):
        print(modelFileName," does exist.")
        # changing the file name
        os.rename(modelFileName, modelFileName+"__"+T)
        status = 'true'
    else:
        print(modelFileName," does not exist.")
    
    return status

if __name__ == '__main__':
    modelFileName = "saved_models/iris_Model.sav"
        
    model  = BuildModel()
    # checking that model exists or not
    backupStatus = backupCurrentModel(modelFileName)
    print('Model backup status ',backupStatus)
    pickle.dump(model, open(modelFileName, 'wb'))
    print("Model has been serialized as ",modelFileName)