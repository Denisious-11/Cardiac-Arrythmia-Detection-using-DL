###import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#collecting non-informative features [features with one-level cardinality]  
#(note : A boolean column, which only can have the values of true or false , has a cardinality of 2.)

def get_preprocess1(data):
    #creating an empty list
    non_features = []

    # store number of columns
    number_of_columns = data.shape[1] 
    print(number_of_columns)


    for i in range(number_of_columns):
        if len(data.iloc[:,i].unique()) == 1:

            #perform appending
            non_features.append(data.columns[i])

    # print(non_features)
    # print(len(non_features))

    return non_features


def select_features(x, y):

    # configure to select a subset of features
    fs = SelectKBest(score_func=f_classif, k="all")

    # learn relationship from training data
    fs.fit(x, y)

    return fs