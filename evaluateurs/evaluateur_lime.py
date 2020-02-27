# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow import keras
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import lime.lime_tabular
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
    return model


def eval_lime(data, y, feature_names, nb_tw, modele, model, ind):

    train_data, test_data, train_labels, test_labels = train_test_split(data, y, test_size=0.3, random_state=ind)
    # Construction de l'explainer de LIME
    exp = lime.lime_tabular.LimeTabularExplainer(train_data, feature_names=feature_names, class_names=[y])
    nb_diff = 0 
    n_test = len(test_labels)
    
    
    def pred_proba(data):
        return np.array([[1-x[0], x[0]] for x in model.predict(data)])
        
    
    for i in range (0, n_test) : 
        #On prédit la valeur de l'individu
        if modele == 'NN':
            explanation = exp.explain_instance(test_data[i,:], pred_proba, num_features=len(feature_names))
            pred_before = model.predict_classes(test_data[i].reshape(1,-1))[0,0]
        else:            
            explanation = exp.explain_instance(test_data[i,:], model.predict_proba, num_features=len(feature_names))
            pred_before = model.predict([test_data[i,:]])

        #------------------On trie les variables par significativité pour cette prédiction
        attributes = pd.DataFrame([explanation.domain_mapper.exp_feature_names[x[0]] for x in explanation.as_map()[1]])
        attributes = attributes[attributes[0] != "Intercept"].reset_index(drop=True)
        #-------------------

        #n : nombre de variables explicatives
        n = len(attributes)

        #utw stock les variables untrustworthy pour cette prédiction et tw les trustworthy
        utw = attributes.iloc[nb_tw:n, :]
        utw_names = utw.iloc[:,0]

        data2 = np.copy(data)
        data2 = pd.DataFrame(data2, columns=feature_names)
        data2 = np.array(data2.drop(columns=utw_names))

        train_data2, test_data2, train_labels, test_labels = train_test_split(data2, y, test_size=0.3, random_state=ind)
        model2 = get_modele(modele, train_data2, train_labels)
        if modele == 'NN':
            pred_after = model2.predict_classes(test_data2[i].reshape(1,-1))[0,0]
        else:
            pred_after = model2.predict([test_data2[i,:]])


        if ((pred_after-pred_before) != 0):
          nb_diff = nb_diff + 1
          
    return(nb_diff/n_test)