import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_predict, cross_validate
import sklearn.linear_model as linear_model

from sklearn.cross_validation import cross_val_score
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt



def OHE(data):
    data_1 = data.copy()
    
    numeric_subset = data_1.select_dtypes('number')
    categorical_subset = data_1.select_dtypes('object')

    # One hot encode
    categorical_subset = pd.get_dummies(categorical_subset) 

    # Join the two dataframes using concat. Make sure to use axis = 1 to perform a column bind
    data_2 = pd.concat([numeric_subset, categorical_subset], axis = 1)

    print('OHE shape: ', data_2.shape, '\n')
    print('Cols:', data_2.columns)
    return pd.DataFrame(data=data_2)


def TRAIN_MODEL(data, classifier):
    fold = KFold(10, shuffle = True, random_state = 12345)
    strata = data['Strata'].unique() 

    all_preds = np.full(data.shape[0], 100)
    probability = np.ones(data.shape[0])

    for i, (train_index, test_index) in enumerate(fold.split(strata)):

        train_index_strata = strata[train_index]
        test_index_strata = strata[test_index] 

        X_train = data.loc[data['Strata'].isin(train_index_strata)].drop('osteo_predict', axis = 1)
        X_test = data.loc[data['Strata'].isin(test_index_strata)].drop('osteo_predict', axis = 1)

        y_train = data.loc[data['Strata'].isin(train_index_strata)]['osteo_predict']
        y_test = data.loc[data['Strata'].isin(test_index_strata)]['osteo_predict']

        if classifier == 'RF':
            lr = RandomForestClassifier(n_estimators=100, random_state = 12345)
        
        if classifier == 'LR':
            lr = linear_model.LogisticRegression()     
            
        fit = lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        all_preds[y_test.index] = y_pred

        probs = lr.predict_proba(X_test)
        probability[y_test.index] = probs[:, 1]

    #     print(classification_report(y_test, y_pred))_3['Strata'].unique() 

    data['all_pred'] = all_preds
    data['probability'] = probability
    print('\nfinal result of the whole model: ')
    print(classification_report(data['osteo_predict'], data['all_pred']), '\n')
    print(confusion_matrix(data['osteo_predict'], data['all_pred']), '\n') 
    print('ROC_AUC of', classifier, ' ',roc_auc_score(data['osteo_predict'], data['probability']))
    dataframe = pd.DataFrame(data=data)
    return dataframe, lr



def SUBGROUP_ERR(OHE, original_data, demographic):
    
    race = ['Race_Cat_combine_black', 'Race_Cat_combine_others', 'Race_Cat_combine_white']
    age = ['Age_combine_from_30_to_50', 'Age_combine_from_50_to_70', 
           'Age_combine_from_70_to_80', 'Age_combine_less_than_30','Age_combine_more_than_80']
    sex = ['sex_F', 'sex_M']   

    subgroup_value = original_data[demographic].value_counts()
    print("subgroups:", subgroup_value)
    
    if demographic == 'Race_Cat_combine': 
        for i in race: 
            subgroup_1 = OHE.loc[OHE[i] == 1]
            print(i, classification_report(subgroup_1['osteo_predict'], subgroup_1['all_pred']))
            print(confusion_matrix(subgroup_1['osteo_predict'],subgroup_1['all_pred']))
            print('ROC_AUC of', i, roc_auc_score(subgroup_1['osteo_predict'], subgroup_1['probability']))
            print('----')
            
    elif demographic == 'Age_combine':
        for i in age: 
            subgroup_2 = OHE.loc[OHE[i] == 1]
            print(i, classification_report(subgroup_2['osteo_predict'], subgroup_2['all_pred']))
            print(confusion_matrix(subgroup_2['osteo_predict'],subgroup_2['all_pred']))
            print('ROC_AUC of',i, roc_auc_score(subgroup_2['osteo_predict'], subgroup_2['probability']))
            print('------')
            
    elif demographic == 'sex': 
        for i in sex:
            subgroup_3 = OHE.loc[OHE[i] == 1]
            print(i, classification_report(subgroup_3['osteo_predict'], subgroup_3['all_pred']))
            print(confusion_matrix(subgroup_3['osteo_predict'],subgroup_3['all_pred']))
            print('ROC_AUC of',i, roc_auc_score(subgroup_3['osteo_predict'], subgroup_3['probability']))
            print('-----')


def TRAIN_TOGETHER(original_data, classifier, demographic):
    ohe = OHE(original_data)
    print('')
    df_model, lr = TRAIN_MODEL(ohe, classifier)
    print('')
    SUBGROUP_ERR(df_model, original_data, demographic)



### TRAIN SEPARATE

def EXTRACT_DATA(data, col, value):
    data_extract = data.loc[data[col] == value]
    print(data[col].value_counts(), '\n')
    print ('extracting: ', value, 'of column: ', col)
#     print('after extracting: ', data_extract.shape) 

    data_extract_1 = data_extract.reset_index()
    data_extract_2 = data_extract_1.drop(['index', col], axis = 1)  
    print('new data shape', data_extract_2.shape)
    return pd.DataFrame(data=data_extract_2)


def TRAIN_SEPARATELY(original_data, demographic, classifier):
    title = original_data[demographic].value_counts()
    
    for i in title.index:
        data_extract = EXTRACT_DATA(original_data, demographic, i)
        print(' ')
        ohe = OHE(data_extract)
        print(' ')
        TRAIN_MODEL(ohe, classifier)
        print('\n ------------------')