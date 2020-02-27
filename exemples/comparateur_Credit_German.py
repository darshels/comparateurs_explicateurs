#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import comparateurs_explicateurs.comparateur as comp

x=pd.read_csv("GermanData.csv", header=None)

data_temp = x.iloc[:,0:19]
data_quali = data_temp
data_quanti = data_temp
y = x.iloc[:,20]
y -= 1

onehot = OneHotEncoder()
i=0
for column in data_temp.columns:
    if (data_temp[column].dtype == type(object)):
    #if(i==0):
        data_quanti = data_quanti.drop(column, axis=1)
    else : 
        data_quali = data_quali.drop(column, axis=1)

data_quali = onehot.fit_transform(data_quali)
data_quali = data_quali.toarray()
data_quali = pd.DataFrame(data_quali)

data= pd.concat([data_quali, data_quanti], axis=1)

n_f = data.shape[1]
feature_names = list() #nom des variables à expliquer
for i in range(0,n_f):
    feature_names.append(str(i))


data = np.array(data)

result = comp.comparateur("credit_german", data, y, feature_names, "Arbre", 5)
result = comp.comparateur("credit_german", data, y, feature_names, "Regression Logistique", 5)
result = comp.comparateur("credit_german", data, y, feature_names, "KNN", 5)
result = comp.comparateur("credit_german", data, y, feature_names, "Random Forest", 5)
result = comp.comparateur("credit_german", data, y, feature_names, "SVM", 5)
print("RESULT=", result)
