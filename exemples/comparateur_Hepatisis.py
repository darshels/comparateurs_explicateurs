#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import comparateur as comp
x=pd.read_csv("Hepatisis.csv")
data = x.iloc[:,1:19]
from sklearn.impute import SimpleImputer
data = data.replace("?", -1)
imp = SimpleImputer(missing_values=-1, strategy='mean')
imp.fit(data)
data = imp.transform(data)
data = pd.DataFrame(data)
y = x.iloc[:,0]
y -= 1
n_f = data.shape[1]
feature_names = list() #nom des variables à expliquer
for i in range(0,n_f):
    feature_names.append(str(i))
data = np.array(data)
y = np.array(y)
feature_names = np.array(feature_names)

result = comp.comparateur("hepatisis", data, y, feature_names, "Arbre", 5)
result = comp.comparateur("hepatisis", data, y, feature_names, "Regression Logistique", 5)
result = comp.comparateur("hepatisis", data, y, feature_names, "KNN", 5)
result = comp.comparateur("hepatisis", data, y, feature_names, "Random Forest", 5)
result = comp.comparateur("hepatisis", data, y, feature_names, "SVM", 5)
print("RESULT=", result)