# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:33:38 2020

@author: chaza
"""

import numpy as np
import pandas as pd
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
#from ibreakdown.explainer import ClassificationExplainer
import explainer_ibd3
from explainer_ibd3 import ClassificationExplainer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#import tensorflow_datasets as tfds
from keras import optimizers
import warnings
warnings.simplefilter("ignore")


def get_modele(modele, train_data, train_labels):
    # Modele
    
    if (modele == "Arbre"):
        model = tree.DecisionTreeClassifier().fit(train_data,y=train_labels)
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

def eval_ibd3(data, y, feature_names, nb_tw, modele, model, ind):
    nb_diff = 0 
   
    train_data, test_data, train_labels, test_labels = train_test_split(data, y, test_size=0.3, random_state=ind)
        
    # Construction de l'explainer de iBreakDown
    explainer = ClassificationExplainer(model, modele)
    explainer.fit(train_data, feature_names)
    
    n_test = len(test_labels) #Nombre d'invidividus dans l'échantillon test

    for i in range (0, n_test) :
        explanation = explainer.explain(test_data[i,:])
        #Prédiction de la valeur de l'individu
        if modele == 'NN':
            pred_before = model.predict_classes(test_data[i].reshape(1,-1))[0,0]
        else:
            pred_before = model.predict([test_data[i,:]])

        #-----------Regroupement des index des variables et des valeurs absolues des contributions dans un meme dataframe------------
    
        tableau1 = pd.DataFrame(explanation)
        tableau1.columns = ['index', 'contribution_abs']
        
        #Tri des variables par contribution décroissante
        tableau1 = tableau1.sort_values(by='contribution_abs', ascending=False)
   
    #--------Récupération variables trust-worthy-----------------#
        nb_v = 0 #nombre de variables que l'on garde
        m=0
        tw_names = []
        while (nb_v < nb_tw):
            case = tableau1.iloc[m,0]
            #print("CASE", case)
            if not isinstance(case, tuple):
                if not case in tw_names:
                    tw_names.append(str(case))
                    nb_v +=1
            else: 
                n_elements = len(case)
                for j in range (n_elements):
                    if (nb_v < nb_tw):
                        if not case[j] in tw_names:
                            tw_names.append(str(case[j]))
                            nb_v+=1
            m+=1
        #--------Fin récupération variables trustworthy-----------------

        #Création du dataframe ne contenant que les variables trustworthy
        data2 = np.copy(data)
        data2 = pd.DataFrame(data2, columns=feature_names)
        data2 = data2.filter(tw_names, axis=1) 
        data2 = np.array(data2)
        train_data2, test_data2, train_labels, test_labels = train_test_split(data2, y, test_size=0.3, random_state=ind)
        model2 = get_modele(modele, train_data2, train_labels)
        
        #prédiction de la nouvelle valeur de l'individu
        if modele == 'NN':
            pred_after = model2.predict_classes(test_data2[i].reshape(1,-1))[0,0]
        else:
            pred_after = model2.predict([test_data2[i,:]])
        
        #Si les deux valeurs sont différentes, on incrémente le nombre de mauvaises explicabilités
        if ((pred_after-pred_before) != 0):
          nb_diff = nb_diff + 1
    return(nb_diff/n_test)  


