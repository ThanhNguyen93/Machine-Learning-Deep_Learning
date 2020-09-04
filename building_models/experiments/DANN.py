#DANN_TRAIN_TOGETHER
import os
import os.path
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("/Users/thanhng/adaptive_confound")

from adaptive_confound import utils
from importlib import reload
import adaptive_confound.utils as acu
import adaptive_confound.control as acc
import adaptive_confound.topic_model as actm
import adaptive_confound.confound_detection as accd
import json
from keras.utils import to_categorical

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import KFold, cross_val_predict, cross_validate


#from sklearn.cross_validation import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

def CONFOUNDING_RENAME(data):
    confound = ['sex', 'Race_Cat_combine', 'Age_combine']
    new_name = ['sex_cat', 'race', 'age']
    for i, j in zip(confound, new_name):
        print(data[i].value_counts())
        data[j] = data[i].astype('category').cat.codes
        data[j] = data[j].astype('category')
        print(data[j].value_counts(), '\n')
    return data
 
def FILTER_DATA(data):
    fil = data.filter(['Strata', 'sex_cat','race', 'age','osteo_predict',
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

    # One hot encode
    object = pd.get_dummies(object_subset)
    categorical = pd.get_dummies(categorical_subset)
    
    # Join the two dataframes using concat. Make sure to use axis = 1 to perform a column bind
    data_2 = pd.concat([confounding_z, numeric_subset, object, categorical], axis = 1)

    print('OHE shape: ', data_2.shape, '\n')
    print('Cols:', data_2.columns)
    return pd.DataFrame(data=data_2)

def TRAIN_MODEL(ohe_data, abow, confounding, z_type):
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
        
        if z_type == 'discrete': 
            z = X_train[confounding]
            
        if z_type == 'category' and confounding == 'sex_cat':
            z = X_train.filter(['sex_cat_0', 'sex_cat_1'])
           
        if z_type == 'category' and confounding == 'race':
            z = X_train.filter(['race_0', 'race_1', 'race_2'])
            
        if z_type == 'category' and confounding == 'age':
            z = X_train.filter(['age_0','age_1', 'age_2', 'age_3', 'age_4'])

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
        abow.fit(X_train, y_train, z=z, epochs=80, batch_size= 128, verbose = 0)

        ypred_da = abow.label_clf.predict(X_test).round().flatten().astype(int)
        all_preds[y_test.index] = ypred_da
        
        PRINT_EPOCHS(abow)
        
        print('\n==================== wait for me..... ==================\n')
        
    ohe_data['all_preds'] = all_preds
    print('classification_report of whole model: \n', classification_report(Y['osteo_predict'], ohe_data['all_preds']), '\n')
    print('confusion_matrix of whole model: \n', confusion_matrix(Y['osteo_predict'], ohe_data['all_preds']), '\n')

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y['osteo_predict'], ohe_data['all_preds'])
    auc_keras = auc(fpr_keras, tpr_keras)
    print('AUC of whole model: ', auc_keras)
    
    return ohe_data

def PRINT_EPOCHS(abow): 
    print(abow.h.history.keys())
    y_loss = abow.h.history['y_loss']
    z_loss = abow.h.history['z_loss']
    y_acc = abow.h.history['y_acc']
    z_acc = abow.h.history['z_acc']

    # summarize history for accuracy
    plt.plot(y_acc)
    plt.plot(z_acc)
    #        plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['y_acc', 'z_acc'], loc='upper right')
    plt.show()
    # # summarize history for loss

    plt.plot(y_loss)
    plt.plot(z_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['y_loss', 'z_loss'], loc='upper left')
    plt.show()

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
            print(classification_report(subgroup_1['osteo_predict'], subgroup_1['all_preds']))
            print(confusion_matrix(subgroup_1['osteo_predict'],subgroup_1['all_preds']))
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(subgroup_1['osteo_predict'], subgroup_1['all_preds'])
            auc_keras = auc(fpr_keras, tpr_keras)
            print('AUC of: ',i,  auc_keras)          
            print('----------------\n')
            
    elif demographic == 'age':
        for i in age: 
            subgroup_2 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_2['osteo_predict'], subgroup_2['all_preds']))
            print(confusion_matrix(subgroup_2['osteo_predict'],subgroup_2['all_preds']))
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(subgroup_2['osteo_predict'], subgroup_2['all_preds'])
            auc_keras = auc(fpr_keras, tpr_keras)
            print('AUC of: ',i,  auc_keras)          
            print('----------------\n')
            
    elif demographic == 'sex_cat':
        print('\n')
        for i in sex:
            subgroup_3 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_3['osteo_predict'], subgroup_3['all_preds']))
            print(confusion_matrix(subgroup_3['osteo_predict'],subgroup_3['all_preds']))
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(subgroup_3['osteo_predict'], subgroup_3['all_preds'])
            auc_keras = auc(fpr_keras, tpr_keras)
            print('AUC of: ',i,  auc_keras)          
            print('----------------\n')

def DANN_TRAIN_TOGETHER(original_data, abow, demographic, z_type):
    '''
    z_type = 'discrete'/'category'
    '''
    rename = CONFOUNDING_RENAME(original_data)
    extract = FILTER_DATA(rename)
    osteo= CONVERT_STRING_FLOAT(extract)

    ohe_data = OHE(osteo, demographic)
    print('\n+++++++++++++++++++++++++++++++++++++++++++\n')
    df_model = TRAIN_MODEL(ohe_data, abow, demographic, z_type)
    print('\n+++++++++++++++++++++++++++++++++++++++++++\n')
    demog = ['sex_cat', 'age', 'race']
    for i in demog:
        SUBGROUP_ERR(df_model, original_data, i)


    
