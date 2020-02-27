import comparateurs_explicateurs.evaluateurs.evaluateur_ibd as eibd
import comparateurs_explicateurs.evaluateurs.evaluateur_lime as elime
import comparateurs_explicateurs.evaluateurs.evaluateur_shap as eshap
import comparateurs_explicateurs.evaluateurs.evaluateur_ibd2 as eibd2
import comparateurs_explicateurs.evaluateurs.evaluateur_ibd3 as eibd3 
import shap
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tensorflow import keras

def get_modele(modele, train_data, train_labels, test_data):
    # Modele et explicateur
    if (modele == "Arbre"):
        model = DecisionTreeClassifier().fit(train_data,y=train_labels)
        tmp_time = time.time()
        shap_values = shap.TreeExplainer(model)
        time_res = time.time() - tmp_time
    elif (modele == "Regression Logistique"):
        model = LogisticRegression().fit(train_data, train_labels)
        tmp_time = time.time()
        shap_values = shap.KernelExplainer(model.predict_proba,train_data)
        time_res = time.time() - tmp_time
    elif (modele == "KNN"):
        model = KNeighborsClassifier(n_neighbors=5).fit(train_data, train_labels)
        tmp_time = time.time()
        shap_values = shap.KernelExplainer(model.predict_proba,train_data)
        time_res = time.time() - tmp_time
    elif (modele == "Random Forest"):
        model = RandomForestClassifier().fit(train_data, train_labels)
        tmp_time = time.time()
        shap_values = shap.TreeExplainer(model)
        time_res = time.time() - tmp_time
    elif (modele == "SVM"):
        model = SVC(probability=True).fit(train_data, train_labels)
        tmp_time = time.time()
        shap_values = shap.KernelExplainer(model.predict_proba,train_data)
        time_res = time.time() - tmp_time
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
        tmp_time = time.time()
        def proba_predic(data):
            return model.predict(data)
        shap_values = shap.KernelExplainer(proba_predic,train_data)
        time_res = time.time() - tmp_time
    else: 
        return("Ce nom de modèle n'est pas pris en charge")
    
    return model, shap_values, time_res

def comparateur(nom, data, y, feature_names, modele, nb_cross_validation):
    first_time = time.time()
    np.set_printoptions(suppress=True)
    n_f = data.shape[1]
    mauvaises_pred_ibd = 0
    mauvaises_pred_lime = 0
    mauvaises_pred_shap = 0
    mauvaises_pred_ibd2 = 0
    mauvaises_pred_ibd3 = 0
    
    time_ibd = 0
    time_lime = 0
    time_shap = 0
    time_ibd2 = 0
    time_ibd3 = 0
    
    #Faire tourner les algos pour un nombre de variables trustworthy alors de 1 à n_f-1 (n_f nombre
    #total de variables à expliquer) et sommer les taux retournés par les algos
    
    with open("Benchmark_"+nom+"_"+modele+".txt", "a+") as f: 
        
        print("\n\n ------------------------ \n\n", file=f)
        
        for ind in range(nb_cross_validation):            
            print("************", ind + 1, "/", nb_cross_validation, "************")
            train_data, test_data, train_labels, test_labels = train_test_split(data, y, test_size=0.3, random_state=ind)
        
            model, shap_values, time_res = get_modele(modele, train_data, train_labels, test_data)
            time_shap += time_res
            for i in range (1,n_f):
                print("****** CV : {}/{} - n : {}/{} ******".format(ind + 1, nb_cross_validation, i, n_f - 1))
    
                tmp_time = time.time()
                mauvaises_pred_ibd += eibd.eval_ibd(data, y, feature_names, i, modele, model, ind)
                time_ibd += time.time() - tmp_time
                print("ibd : {:.3f} s, score = {}".format(time_ibd, mauvaises_pred_ibd))
                
                tmp_time = time.time()
                mauvaises_pred_lime += elime.eval_lime(data, y, feature_names, i, modele, model, ind)
                time_lime += time.time() - tmp_time
                print("lime : {:.3f} s, score = {}".format(time_lime, mauvaises_pred_lime))
                
                tmp_time = time.time()
                mauvaises_pred_shap += eshap.eval_shap(data, y, feature_names, i, modele, model, shap_values,ind)
                time_shap += time.time() - tmp_time
                print("shap : {:.3f} s, score = {}".format(time_shap, mauvaises_pred_shap))
                
                tmp_time = time.time()
                mauvaises_pred_ibd2 += eibd2.eval_ibd2(data, y, feature_names, i, modele, model, ind)
                time_ibd2 += time.time() - tmp_time
                print("ibd2 : {:.3f} s, score = {}".format(time_ibd2, mauvaises_pred_ibd2))
                
                tmp_time = time.time()
                mauvaises_pred_ibd3 += eibd3.eval_ibd3(data, y, feature_names, i, modele, model, ind)
                time_ibd3 += time.time() - tmp_time
                print("ibd3 : {:.3f} s, score = {}".format(time_ibd3, mauvaises_pred_ibd3))
            
        #diviser la somme pour obtenir un taux moyen pour chaque algo pour ces données et ce modèle
        mauvaises_pred_ibd /= nb_cross_validation*(n_f-1)
        mauvaises_pred_lime /= nb_cross_validation*(n_f-1)
        mauvaises_pred_shap /= nb_cross_validation*(n_f-1)
        mauvaises_pred_ibd2 /= nb_cross_validation*(n_f-1)
        mauvaises_pred_ibd3 /= nb_cross_validation*(n_f-1)
        
        result = []
        
        print("\nTaux d'erreur pour chaque librairie", file=f)
        print("ibd : {:.8f}".format(mauvaises_pred_ibd), file=f)
        print("lime : {:.8f}".format(mauvaises_pred_lime), file=f)
        print("shap : {:.8f}".format(mauvaises_pred_shap), file=f)
        print("ibd2 : {:.8f}".format(mauvaises_pred_ibd2), file=f)
        print("ibd3 : {:.8f}".format(mauvaises_pred_ibd3), file=f)
        print("Taux d'erreur pour chaque librairie")
        print("ibd : {:.8f}".format(mauvaises_pred_ibd))
        print("lime : {:.8f}".format(mauvaises_pred_lime))
        print("shap : {:.8f}".format(mauvaises_pred_shap))
        print("ibd2 : {:.8f}".format(mauvaises_pred_ibd2))
        print("ibd3 : {:.8f}".format(mauvaises_pred_ibd3))
        
        print("\nTemps pour chaque librairie", file=f)
        print("ibd : {:.3f} s".format(time_ibd), file=f)
        print("lime : {:.3f} s".format(time_lime), file=f)
        print("shap : {:.3f} s".format(time_shap), file=f)
        print("ibd2 : {:.3f} s".format(time_ibd2), file=f)
        print("ibd3 : {:.3f} s".format(time_ibd3), file=f)
        print("\nTemps total : {:.3f} s".format(time.time() - first_time), file=f)
        print("Temps pour chaque librairie")
        print("ibd : {:.3f} s".format(time_ibd))
        print("lime : {:.3f} s".format(time_lime))
        print("shap : {:.3f} s".format(time_shap))
        print("ibd2 : {:.3f} s".format(time_ibd2))
        print("ibd3 : {:.3f} s".format(time_ibd3))
        print("Temps total : {:.3f} s".format(time.time() - first_time))
        
        result.append(mauvaises_pred_ibd)
        result.append(mauvaises_pred_lime)
        result.append(mauvaises_pred_shap)
        result.append(mauvaises_pred_ibd2)   
        result.append(mauvaises_pred_ibd3)
    
    return(result)
    



