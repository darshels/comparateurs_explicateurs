# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 23:03:26 2020

@author: chaza
"""

from sklearn.preprocessing import OneHotEncoder, Imputer
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np
import comparateurs_explicateurs.comparateur as comp

np.set_printoptions(suppress=True)
x=pd.read_csv("breast_cancer.csv", header=None)

#x = x.head(150)

data = x.iloc[:,1:32]

data = data.replace("?", -1)


imp = Imputer(missing_values=-1, strategy='mean')

imp.fit(data)
data = imp.transform(data)
data = pd.DataFrame(data)
y = x.iloc[:,0]


n_f = data.shape[1]
feature_names = list() #nom des variables à expliquer
for i in range(0,n_f):
    feature_names.append(str(i))
data = np.array(data)
y = np.array(y)
feature_names = np.array(feature_names)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)


y = np.array(y)

result = comp.comparateur("breast_cancer", data, y, feature_names, "Arbre", 5)
result = comp.comparateur("breast_cancer", data, y, feature_names, "Regression Logistique", 5)
result = comp.comparateur("breast_cancer", data, y, feature_names, "KNN", 5)
result = comp.comparateur("breast_cancer", data, y, feature_names, "Random Forest", 5)
result = comp.comparateur("breast_cancer", data, y, feature_names, "SVM", 5)
print("RESULT=", result)

