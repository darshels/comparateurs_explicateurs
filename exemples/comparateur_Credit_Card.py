#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import comparateur as comp
np.set_printoptions(suppress=True)
x=pd.read_excel("creditcard.xls")
data = x.drop(['default payment next month'], axis=1)
feature_names = data.columns
y = x.iloc[:,23]

n_f = data.shape[1]
feature_names = list() #nom des variables à expliquer
for i in range(0,n_f):
    feature_names.append(str(i))
data = np.array(data)
y = np.array(y)
feature_names = np.array(feature_names)

result = comp.comparateur("creditcard", data, y, feature_names, "Arbre", 5)
result = comp.comparateur("creditcard", data, y, feature_names, "Regression Logistique", 5)
result = comp.comparateur("creditcard", data, y, feature_names, "KNN", 5)
result = comp.comparateur("creditcard", data, y, feature_names, "Random Forest", 5)
result = comp.comparateur("creditcard", data, y, feature_names, "SVM", 5)
print("RESULT=", result)
