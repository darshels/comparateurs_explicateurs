#!/usr/bin/env python
# coding: utf-8

#----------------------Données------------------------#
#Mettre les données sous forme data(variables à expliquer pour tous les individus), y (variable explicative
#pour tous les individus) et feature_names (nom des variables à expliquer et si elles n'ont pas de noms en créer)
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import comparateur as comp
        
np.set_printoptions(suppress=True)

x=pd.read_csv("TicTacToe.csv")
data_temp = x.iloc[:,0:9]
y = x.iloc[:,9]
data_quali = data_temp
data_quanti = data_temp

onehot = OneHotEncoder()
for column in data_temp.columns:
    if (data_temp[column].dtype == type(object)):
        data_quanti = data_quanti.drop(column, axis=1)
    else : 
        data_quali = data_quali.drop(column, axis=1)

data_quali = onehot.fit_transform(data_quali)
data_quali = data_quali.toarray()
data_quali = pd.DataFrame(data_quali)

data= pd.concat([data_quali, data_quanti], axis=1)

le =  preprocessing.LabelEncoder()
y = le.fit_transform(y)
        
# Execution sequentiel

n_f = data.shape[1]
feature_names = []
for i in range(0,n_f):
    feature_names.append(str(i))

data = np.array(data)
y  = np.array(y)
    

result = comp.comparateur("tictactoe", data, y, feature_names, "Arbre", 5)
result = comp.comparateur("tictactoe", data, y, feature_names, "Regression Logistique", 5)
result = comp.comparateur("tictactoe", data, y, feature_names, "KNN", 5)
result = comp.comparateur("tictactoe", data, y, feature_names, "Random Forest", 5)
result = comp.comparateur("tictactoe", data, y, feature_names, "SVM", 5)
print("RESULT=", result)

        
        
        
