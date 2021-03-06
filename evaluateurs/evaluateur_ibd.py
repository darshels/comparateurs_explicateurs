# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ibreakdown.explainer import ClassificationExplainer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from __future__ import absolute_import, division, print_function, unicode_literals
#import tensorflow_datasets as tfds
import warnings
warnings.simplefilter("ignore")

def get_modele(modele, train_data, train_labels):
    # Modele
    
    if (modele == "Arbre"):
        model = DecisionTreeClassifier().fit(train_data,y=train_labels)
    elif (modele == "Regression Logistique"):
        model = LogisticRegression().fit(train_data, train_labels)
    elif (modele == "KNN"):
        model = KNeighborsClassifier(n_neighbors=5).fit(train_data, train_labels)
    elif (modele == "Random Forest"):
        model = RandomForestClassifier().fit(train_data, train_labels)
    elif (modele == "SVM"):
        model = SVC().fit(train_data, train_labels)
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

def eval_ibd(data, y, feature_names, nb_tw, modele, model, ind):
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, y, test_size=0.3, random_state=ind)
    # Construction de l'explainer de iBreakDown
    explainer = ClassificationExplainer(model)
    explainer.fit(train_data, feature_names)
    nb_diff = 0 
    n_test = len(test_labels) #Nombre d'invidividus dans l'échantillon test

    #print(explanation.__dict__)
    for i in range (0, n_test) :
    #Pour chaque individu dans l'échantillon test
        #Création de l'explication iBreakDown pour l'individu i
        explanation = explainer.explain(test_data[i,:])
        #Prédiction de la valeur de l'individu
        
        if modele == 'NN':
            pred_before = model.predict_classes(test_data[i].reshape(1,-1))[0,0]
        else:
            pred_before = model.predict([test_data[i,:]])

        #-----------Regroupement des index des variables et des valeurs absolues des contributions dans un meme dataframe------------
        contrib_abs = abs(np.array(explanation.contributions[:,0]))
        tableau1 = list(zip(explanation.feature_indexes, contrib_abs))    
        tableau1 = pd.DataFrame(tableau1, columns= ['index', 'contribution_abs'])
        #Tri des variables par contribution décroissante
        tableau1 = tableau1.sort_values(by='contribution_abs', ascending=False)
        n_contrib = len(tableau1.iloc[:,0])
        for k in range (n_contrib):
          if not isinstance(tableau1.iat[k,0], tuple) :
            tableau1.iloc[k,0] = explanation.columns[tableau1.iloc[k,0]]
          else : 
            n_elements = len(tableau1.iat[k,0])
            tab = []
            for j in range (n_elements):
              tab.insert(j, explanation.columns[tableau1.iat[k,0][j]])
            tab = tuple(tab)
            tableau1.iat[k,0] = tab
       #-------------------------fin regroupement-----------------------------

        #-------------Récupération variables trustworthy-----------------
        nb_v = 0 #nombre de variables que l'on garde
        m=0
        tw_names = []
        while (nb_v < nb_tw):
          case = tableau1.iloc[m,0]
          if not isinstance(case, tuple) :
              if not case in tw_names:
                  tw_names.append(case)
                  nb_v +=1
          else: 
            n_elements = len(case)
            for j in range (n_elements):
              if (nb_v < nb_tw):
                  if not case[j] in tw_names:
                      tw_names.append(case[j])
                      nb_v+=1
          m+=1
        #--------Fin récupération variables trustworthy-----------------

        #Création du dataframe ne contenant que les variables trustworthy
        data2 = np.copy(data)
        data2 = pd.DataFrame(data2, columns=feature_names)
        data2 = data2.filter(tw_names, axis=1)        
        data2 = np.array(data2)
        #entrainement du nouveau modèle n'ayant que les variables tw
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