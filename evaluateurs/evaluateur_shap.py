# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#import tensorflow_datasets as tfds
import warnings
warnings.simplefilter("ignore")

def get_modele(modele, train_data, train_labels, test_data):
    # Modele et explicateur
    if (modele == "Arbre"):
        model = DecisionTreeClassifier().fit(train_data,y=train_labels)
    elif (modele == "Regression Logistique"):
        model = LogisticRegression().fit(train_data, train_labels)
    elif (modele == "KNN"):
        model = KNeighborsClassifier(n_neighbors=5).fit(train_data, train_labels)
    elif (modele == "Random Forest"):
        model = RandomForestClassifier().fit(train_data, train_labels)
    elif (modele == "SVM"):
        model = SVC(probability=True).fit(train_data, train_labels)
    elif (modele == "NN"):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=train_data[0].shape),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
        model.fit(train_data, train_labels, verbose=0)
    else: 
        return("Ce nom de modèle n'est pas pris en charge")
    
    return model

def eval_shap(data, y, feature_names, nb_tw, modele, model, shap_values, ind):
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, y, test_size=0.3, random_state=ind)
    
    nb_diff = 0 
    n_test = len(test_labels)
    
    for i in range (0, n_test) : 
      res = shap_values.shap_values(test_data[i])
      list_tuples =[(j, abs(i)) for i, j in zip(res[0],feature_names)]
      list_tuples.sort(key=lambda tup: tup[1], reverse=True)
      attributes =[x[0] for x in list_tuples]
      #On prédit la valeur de l'individu
      if modele == 'NN':
            pred_before = model.predict_classes(test_data[i].reshape(1,-1))[0,0]
      else:
            pred_before = model.predict([test_data[i,:]])

      #utw = untrustworthy
      #n : nombre de variables explicatives
      n = len(attributes)

      utw_names = attributes[nb_tw:n]

      data2 = np.copy(data)
      data2 = pd.DataFrame(data2, columns=feature_names)
      data2 = np.array(data2.drop(columns=utw_names))

      train_data2, test_data2, train_labels, test_labels = train_test_split(data2, y, test_size=0.3, random_state=ind)
      model2 = get_modele(modele, train_data2, train_labels, test_data2)
      if modele == 'NN':
            pred_after = model2.predict_classes(test_data2[i].reshape(1,-1))[0,0]
      else:
            pred_after = model2.predict([test_data2[i,:]])


      if ((pred_after-pred_before) != 0):
        nb_diff = nb_diff + 1

    return(nb_diff/n_test)