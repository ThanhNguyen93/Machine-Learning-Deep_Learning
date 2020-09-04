
import json
from keras.utils import to_categorical

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras import optimizers
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import KFold, cross_val_predict, cross_validate

from keras.callbacks import History 
from keras.constraints import maxnorm
import tensorflow as tf
from tensorflow.keras import layers



#from sklearn.cross_validation import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score


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

def OHE(data):
    data_1 = data.copy()
    
    numeric_subset = data_1.select_dtypes('int64')
    object_subset = data_1.select_dtypes('object')
    categorical_subset = data_1.select_dtypes('category')

    # One hot encode
    object = pd.get_dummies(object_subset)
    categorical = pd.get_dummies(categorical_subset)
    
    # Join the two dataframes using concat. Make sure to use axis = 1 to perform a column bind
    data_2 = pd.concat([numeric_subset, object, categorical], axis = 1)

    print('OHE shape: ', data_2.shape, '\n')
#     print('Cols:', data_2.columns)
    return pd.DataFrame(data=data_2)

    
def RESET_WEIGHTS(model):
    session = K.get_session()
    for layer in model.layers: 
        if isinstance(layer, keras.engine.network.Network):
            reset_weights(layer)
            continue
        for v in layer.__dict__.values():
            if hasattr(v, 'initializer'):
                v.initializer.run(session=session)

def TRAIN_MODEL(ohe_data, ann):
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
       
        #train model
        print('X_train.shape: ', X_train.shape, 'X_test.shape: ', X_test.shape)
        RESET_WEIGHTS(ann)
        history = ann.fit(X_train, y_train, epochs=80, batch_size=164, verbose = 0, validation_data = [X_test, y_test])

        ypred_da = ann.predict(X_test).round().flatten().astype(int)
        all_preds[y_test.index] = ypred_da
        
        probs = ann.predict_proba(X_test)
        probability[y_test.index] = probs[:, 0]
  
        print('\nfinish another fold' )
        print('==================== wait for me..... ==================\n')
         
    ohe_data['all_preds'] = all_preds
    ohe_data['probability'] = probability
      
    PRINT_EPOCHS(history)

    print('classification_report of whole model:\n', classification_report(ohe_data['osteo_predict'], ohe_data['all_preds']), '\n')
    print('confusion_matrix of whole model: \n', confusion_matrix(ohe_data['osteo_predict'], ohe_data['all_preds']), '\n')
    AUC = roc_auc_score(ohe_data['osteo_predict'], ohe_data['probability'])
    print('AUC of whole model: ', AUC)
    
    return history, ohe_data   
 
def PRINT_EPOCHS(history): 
    print(history.history.keys())
    
    # summarize history for accuracy
    acc = plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'], 'r--', color = acc[0].get_color())

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['y_train_acc', 'y_test_acc'], loc='upper left')
    plt.show()

    # summarize history for loss

    loss = plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'], 'b--', color = acc[0].get_color())

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['y_train_loss', 'y_test_loss'], loc='upper left')
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
            AUC = roc_auc_score(subgroup_1['osteo_predict'], subgroup_1['probability'])
            print('AUC of: ',i,  AUC)          
            print('----------------\n')
            
    elif demographic == 'age':
        for i in age: 
            subgroup_2 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_2['osteo_predict'], subgroup_2['all_preds']))
            print(confusion_matrix(subgroup_2['osteo_predict'],subgroup_2['all_preds']))
            AUC = roc_auc_score(subgroup_2['osteo_predict'], subgroup_2['probability'])
            print('AUC of: ',i,  AUC)          
            print('----------------\n')
            
    elif demographic == 'sex_cat':
        print('\n')
        for i in sex:
            subgroup_3 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_3['osteo_predict'], subgroup_3['all_preds']))
            print(confusion_matrix(subgroup_3['osteo_predict'],subgroup_3['all_preds']))
            AUC = roc_auc_score(subgroup_3['osteo_predict'], subgroup_3['probability'])
            print('AUC of: ',i,  AUC)          
            print('----------------\n')

def ANN_TRAIN_TOGETHER(original_data, ann):
    rename = CONFOUNDING_RENAME(original_data)
    extract = FILTER_DATA(rename)
    osteo= CONVERT_STRING_FLOAT(extract)
    ohe_data = OHE(osteo)
    print('\n+++++++++++++++++++++++++++++++++++++++++++\n')
    history, df_model = TRAIN_MODEL(ohe_data, ann)
    print('\n+++++++++++++++++++++++++++++++++++++++++++\n')
    demog = ['sex_cat', 'age', 'race']
    for i in demog:
        SUBGROUP_ERR(df_model, original_data, i)


 #########TRAIN SEPARATELY
def EXTRACT_DATA(data, col, value):
    '''
    value = 0,1,2,3,4 (depend on number of subgroups)
    col = 'sex_cat', 'race', 'age'
    '''
    data_extract = data.loc[data[col] == value]
    print('************************************')
    print('Now working on this demographic:', col)
#    print(data[col].value_counts(), '\n')
    print ('extracting: ', value, 'of column: ', col)
#     print('after extracting: ', data_extract.shape) 

    data_extract_1 = data_extract.reset_index()
    data_extract_2 = data_extract_1.drop(['index', col], axis = 1)  
    print('new data shape', data_extract_2.shape)
    return pd.DataFrame(data=data_extract_2)


def ANN_TRAIN_SEPARATELY(original_data, demographic, ann):
    '''
    demographic = 'sex_cat', 'race', 'age'
    '''
    
    rename = CONFOUNDING_RENAME(original_data)
    extract = FILTER_DATA(rename)
    osteo= CONVERT_STRING_FLOAT(extract)
    title = osteo[demographic].value_counts()
#     title = 0, 1, 2, 3, 4 (depend on number of subgroups)
    
    for i in title.index:
        data_extract = EXTRACT_DATA(osteo, demographic, i)
        print(' ')
        ohe_data = OHE(data_extract)
        print('\n+++++++++++++++++++++++++++++++++++++++++++\n')
        hist, df_model = TRAIN_MODEL(ohe_data, ann)
        print('\n+++++++++++++++++++++++++++++++++++++++++++\n')
    
