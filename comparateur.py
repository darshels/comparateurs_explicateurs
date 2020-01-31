#import comparateurs_explicateurs.evaluateurs.evaluateur_breakdown as bd
import comparateurs_explicateurs.evaluateurs.evaluateur_ibreakdown as ibd
import comparateurs_explicateurs.evaluateurs.evaluateur_lime as elime
import comparateurs_explicateurs.evaluateurs.evaluateur_shap as eshap
import numpy as np


def comparateur(data, y, feature_names, modele):
    np.set_printoptions(suppress=True)
    n_f = data.shape[1]
    #mauvaises_pred_bd = 0
    mauvaises_pred_ibd = 0
    mauvaises_pred_lime = 0
    mauvaises_pred_shap = 0
    #Faire tourner les algos pour un nombre de variables trustworthy alors de 1 à n_f-1 (n_f nombre
    #total de variables à expliquer) et sommer les taux retournés par les algos
    for i in range (1,n_f):
        print(i, "/", n_f)
        #Arbre, SVM, Random Forest, KNN, Logistic Regression, NN
        #mauvaises_pred_bd = mauvaises_pred_bd + bd.eval_bd(data, y, feature_names, i, modele)
        mauvaises_pred_ibd = mauvaises_pred_ibd + ibd.eval_ibd(data, y, feature_names, i, modele)
        mauvaises_pred_lime = mauvaises_pred_lime + elime.eval_lime(data, y, feature_names, i, modele)
        mauvaises_pred_shap = mauvaises_pred_shap + eshap.eval_shap(data, y, feature_names, i, modele)
        
        
    #diviser la somme pour obtenir un taux moyen pour chaque algo pour ces données et ce modèle
    #mauvaises_pred_bd = mauvaises_pred_bd/(n_f-1)
    mauvaises_pred_ibd = mauvaises_pred_ibd/(n_f-1)
    mauvaises_pred_lime = mauvaises_pred_lime/(n_f-1)
    mauvaises_pred_shap = mauvaises_pred_shap/(n_f-1)
    
    result = []
    #print("bd :", mauvaises_pred_bd)
    print("ibd :", mauvaises_pred_ibd)
    print("lime :", mauvaises_pred_lime)
    print("shap :", mauvaises_pred_shap)
    
    #result.append(mauvaises_pred_bd)
    result.append(mauvaises_pred_ibd)
    result.append(mauvaises_pred_lime)
    result.append(mauvaises_pred_shap)
    
    
    return(result)
    



