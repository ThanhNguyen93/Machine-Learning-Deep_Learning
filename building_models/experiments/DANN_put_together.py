import json
from keras.utils import to_categorical

import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score, multilabel_confusion_matrix
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def CONFOUNDING_RENAME(data):
    confound = ['sex', 'Race_Cat_combine', 'Age_combine']
    new_name = ['sex_cat', 'race', 'age']
    for i, j in zip(confound, new_name):
#         print(data[i].value_counts())
        data[j] = data[i].astype('category').cat.codes
        data[j] = data[j].astype('category')
#         print(data[j].value_counts(), '\n')
    return data

       
def FILTER_DATA(data):
    fil = data.filter(['Strata', 'sex_cat','race', 'age','osteo_predict','BMI_Avg_imputed',
     'Alcohol_Prior', 'Tobacco_Prior','Drug_antipsych_prior', 'Drug_Estrogens_prior', 'Drug_Glucocorticoids_prior','Drug_Nsaids_prior', 'Drug_Opiates_prior', 'Drug_Thiazide_prior','Drug_Loop_Diuretic_Prior',
      'Drug_Pp_inhibitors_prior',  'Drug_Progesterone_prior','Drug_Seizure_prior',  'Drug_Ssris_prior','Drug_Tc_antidepress_prior',  'HeartDisease_Prior',  'Liver_Prior', 'PulmDisease_Prior',  'CNS_Disease_Prior',
 'Malignancy_Prior','Hyponatremia_Prior','Chronic_Hyponatremia', 'Recent_Hyponatremia',                                        
    'Calcium_Closest_Osteo_decile','Calcium_Avg_Prior_decile','Calcium_Avg_Ever_decile',
    'Sodium_Closest_Osteo_decile',   'Sodium_Avg_Prior_decile','Sodium_Worst_Prior_decile',
      'Sodium_Avg_Ever_decile','Sodium_Worst_Ever_decile']) 
    return fil

def CONVERT_STRING_FLOAT(data):
    var_decile=['Calcium_Closest_Osteo_decile', 'Calcium_Avg_Prior_decile','Calcium_Avg_Ever_decile',
            'Sodium_Closest_Osteo_decile','Sodium_Avg_Prior_decile','Sodium_Worst_Prior_decile',
   'Sodium_Avg_Ever_decile','Sodium_Worst_Ever_decile']
    for i in var_decile:
        data[i] = data[i].apply(lambda x: str(x))
       # print(i, data[i].dtypes, '\n')
    return data

def OHE(data, confounding):
    confounding_z = data[confounding]
    data_1 = data.copy()
    
    numeric_subset = data_1.select_dtypes('int64')
    object_subset = data_1.select_dtypes('object')
    categorical_subset = data_1.select_dtypes('category')
    float_subset = data_1.select_dtypes('float')
    
    # One hot encode
    object_ohe = pd.get_dummies(object_subset)
    categorical = pd.get_dummies(categorical_subset)
    
    # Join the two dataframes using concat. Make sure to use axis = 1 to perform a column bind
    data_2 = pd.concat([confounding_z, numeric_subset, float_subset, object_ohe, categorical], axis = 1)

    print('OHE shape: ', data_2.shape, '\n')
    print('Cols:', data_2.columns)
    return pd.DataFrame(data=data_2)

def TRAIN_MODEL(ohe_data, abow, confounding):
    X = ohe_data.drop('osteo_predict', axis = 1)
    Y = ohe_data.filter(['Strata', 'osteo_predict'])

    fold = KFold(3, shuffle = True, random_state = 12345)
    strata = ohe_data['Strata'].unique() 

    all_preds = np.full(ohe_data.shape[0], 100)
    probability = np.ones(ohe_data.shape[0])

    for train_index, test_index in fold.split(strata):

        train_index_strata = strata[train_index]
        test_index_strata = strata[test_index] 

        X_train = X.loc[X['Strata'].isin(train_index_strata)]
        X_train = X_train.drop(['Strata'], axis = 1)

        X_test = X.loc[X['Strata'].isin(test_index_strata)]
        X_test = X_test.drop(['Strata'], axis = 1)

        y_train = Y.loc[Y['Strata'].isin(train_index_strata)]['osteo_predict']
        y_test = Y.loc[Y['Strata'].isin(test_index_strata)]['osteo_predict']

        z_train = X_train[confounding]
        z_test = X_test[confounding]
        
        if confounding == 'sex_cat':
            #remove confounding from X
            X_train = X_train.drop(['sex_cat', 'sex_cat_0', 'sex_cat_1'], axis = 1)
            X_test = X_test.drop(['sex_cat', 'sex_cat_0', 'sex_cat_1'], axis = 1)

        elif confounding =='race':          
            #remove confounding from X
            X_train = X_train.drop(['race', 'race_0', 'race_1', 'race_2'], axis = 1)
            X_test = X_test.drop(['race', 'race_0', 'race_1', 'race_2'], axis = 1)

        elif confounding == 'age':
            #remove confounding from X
            X_train = X_train.drop(['age', 'age_0','age_1', 'age_2', 'age_3', 'age_4'], axis = 1)
            X_test = X_test.drop(['age', 'age_0','age_1', 'age_2', 'age_3', 'age_4'], axis = 1)

        #train model
        K.clear_session()
        abow.fit(X_train, y_train, z=z_train, validation_data = (X_test, [y_test, z_test]),  
                 epochs=epochs, batch_size= 128, verbose = 0)

        ypred_da = abow.label_clf.predict(X_test).round().flatten().astype(int)
        all_preds[y_test.index] = ypred_da

        probs = abow.predict_probability(X_test)
        probability[y_test.index] = probs[:, 0]

    print('\n==================== wait for me..... ==================\n')

    PRINT_EPOCHS(abow, lr)
 #   return ohe_data
   
    ohe_data['all_preds'] = all_preds
    ohe_data['probability'] = probability

    print('classification_report of whole model: \n', classification_report(Y['osteo_predict'], ohe_data['all_preds']), '\n')
    print('confusion_matrix of whole model: \n', confusion_matrix(Y['osteo_predict'], ohe_data['all_preds']), '\n')
    AUC = roc_auc_score(ohe_data['osteo_predict'], ohe_data['probability'])
    print('AUC of whole model: ', AUC)
         
   
###################################

def PRINT_EPOCHS(abow):
    print(abow.h.history.keys())

    y_loss = abow.h.history['y_loss']
    z_loss = abow.h.history['z_loss']
    y_acc = abow.h.history['y_accuracy']
    z_acc = abow.h.history['z_accuracy']

    val_y_loss = abow.h.history['val_y_loss']
    val_z_loss = abow.h.history['val_z_loss']
    val_y_acc = abow.h.history['val_y_accuracy']
    val_z_acc = abow.h.history['val_z_accuracy']
 
    ############
    y_loss_plt = plt.plot(y_loss, '--')
    z_loss_plt = plt.plot(z_loss, '--')
    
    plt.plot(val_y_loss, color = y_loss_plt[0].get_color())
    plt.plot(val_z_loss, color = z_loss_plt[0].get_color())
    
    plt.title('y_z_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['y_train_loss', 'z_train_loss', 'y_test_loss', 'z_test_loss'], loc='best')
    plt.show()
    
    ######
    y_acc_plt = plt.plot(y_acc, '--')
    z_acc_plt = plt.plot(z_acc, '--')
    
    plt.plot(val_y_acc, color = y_acc_plt[0].get_color())
    plt.plot(val_z_acc, color = z_acc_plt[0].get_color())
    plt.title('y_z_accuracy')
#     plt.title('y_z_accuracy, ' + 'lr = ' + str(lr))
    
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['y_train_acc', 'z_train_acc', 'y_test_acc', 'z_test_acc'], loc='best')
    plt.show()
    
#################################
def SUBGROUP_ERR(OHE, original_data, demographic):
    
    race = ['race_0', 'race_1', 'race_2']
    age = ['age_0','age_1', 'age_2', 'age_3', 'age_4']
    sex = ['sex_cat_0', 'sex_cat_1']

    subgroup_value = original_data[demographic].value_counts()
    print("subgroups: \n", subgroup_value)
    
    if demographic == 'race':
        for i in race: 
            subgroup_1 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_1['osteo_predict'], subgroup_1['Y_all_preds']))
            print(confusion_matrix(subgroup_1['osteo_predict'],subgroup_1['Y_all_preds']))
            AUC = roc_auc_score(subgroup_1['osteo_predict'], subgroup_1['Y_probability'])
            print('AUC of: ',i,  AUC)                    
            print('----------------\n')
            
    elif demographic == 'age':
        for i in age: 
            subgroup_2 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_2['osteo_predict'], subgroup_2['Y_all_preds']))
            print(confusion_matrix(subgroup_2['osteo_predict'],subgroup_2['Y_all_preds']))
            AUC = roc_auc_score(subgroup_2['osteo_predict'], subgroup_2['Y_probability'])
            print('AUC of: ',i,  AUC)                
            print('----------------\n')
            
    elif demographic == 'sex_cat':
        print('\n')
        for i in sex:
            subgroup_3 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_3['osteo_predict'], subgroup_3['Y_all_preds']))
            print(confusion_matrix(subgroup_3['osteo_predict'],subgroup_3['Y_all_preds']))
            AUC = roc_auc_score(subgroup_3['osteo_predict'], subgroup_3['Y_probability'])
            print('AUC of: ',i,  AUC)          
            print('----------------\n')

def DANN_TRAIN_TOGETHER(original_data, abow, demographic, epochs, lr):

    rename = CONFOUNDING_RENAME(original_data)
    extract = FILTER_DATA(rename)
    osteo= CONVERT_STRING_FLOAT(extract)

    ohe_data = OHE(osteo, demographic)
    print('\n+++++++++++++++++++++++++++++++++++++++++++\n')
    df_model = TRAIN_MODEL(ohe_data, abow, demographic, epochs, lr)
    print('\n+++++++++++++++++++++++++++++++++++++++++++\n')
    demog = ['sex_cat', 'age', 'race']
    for i in demog:
        SUBGROUP_ERR(df_model, original_data, i)


    
