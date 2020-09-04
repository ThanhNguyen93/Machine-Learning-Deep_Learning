import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_predict, cross_validate
import sklearn.linear_model as linear_model

#from sklearn.cross_validation import cross_val_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, accuracy_score
#from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV


def FILTER_DATA(data):
    fil = data.filter(['Strata','sex','race_combine', 'Age_combine','osteo_predict', 'BMI_Avg_imputed',
                'Alcohol_Prior', 'Tobacco_Prior','Drug_antipsych_prior', 'Drug_Estrogens_prior',
                'Drug_Glucocorticoids_prior','Drug_Nsaids_prior', 'Drug_Opiates_prior',
                'Drug_Thiazide_prior','Drug_Loop_Diuretic_Prior','Drug_Pp_inhibitors_prior',
                'Drug_Progesterone_prior','Drug_Seizure_prior','Drug_Ssris_prior',
                'Drug_Tc_antidepress_prior','HeartDisease_Prior','Liver_Prior','PulmDisease_Prior',
                'CNS_Disease_Prior','Malignancy_Prior',
                'Hyponatremia_Prior','Chronic_Hyponatremia','Recent_Hyponatremia',
                'Median_Recent_Hypo_Cat_edit', 'Lowest_Recent_Hypo_Cat_edit',
                'Calcium_Closest_Osteo_decile','Calcium_Avg_Prior_decile','Calcium_Avg_Ever_decile',
                'Sodium_Closest_Osteo_decile','Sodium_Avg_Prior_decile','Sodium_Avg_Ever_decile',
                'Sodium_Worst_Prior_decile','Sodium_Worst_Ever_decile',
                'Calcium_Closest_Osteo_cat', 'Calcium_Avg_Prior_cat','Calcium_Avg_Ever_cat',
                'Sodium_Closest_Osteo_cat','Sodium_Avg_Prior_cat','Sodium_Avg_Ever_cat',
                'Sodium_Worst_Prior_cat','Sodium_Worst_Ever_cat'])
    print('shape after filter: ', fil.shape)
    return fil

'''convert categorical data (except binary discrete var) to OHE, these categorical var were read as str'''
'''need to convert str to float because python read these var as continuous(int64), not categorical(object), thus can't OHE'''
'''there're 10 vars'''
def CONVERT_STRING_FLOAT(data):
    var_decile= ['Median_Recent_Hypo_Cat_edit', 'Lowest_Recent_Hypo_Cat_edit',
                'Calcium_Closest_Osteo_decile','Calcium_Avg_Prior_decile','Calcium_Avg_Ever_decile',
                'Sodium_Closest_Osteo_decile', 'Sodium_Avg_Prior_decile','Sodium_Avg_Ever_decile',
                'Sodium_Worst_Prior_decile','Sodium_Worst_Ever_decile']
    for i in var_decile:
        data[i] = data[i].apply(lambda x: str(x))
       # print(i, data[i].dtypes, '\n')
    print('shape before OHE: ', data.shape)
    return data


def OHE(data):
    data_1 = data.copy()
    numeric_subset = data_1.select_dtypes('int64')
    float_subset = data_1.select_dtypes('float')
    categorical_subset = data_1.select_dtypes('object')

    # One hot encode
    categorical_subset = pd.get_dummies(categorical_subset)

    # Join the two dataframes using concat. Make sure to use axis = 1 to perform a column bind
    data_2 = pd.concat([numeric_subset, float_subset, categorical_subset], axis = 1)

    print('OHE shape: ', data_2.shape, '\n')
    print('Cols:', data_2.columns)
    return pd.DataFrame(data=data_2)


def TRAIN_MODEL_ML(data, classifier, tune):
    X = data.drop('osteo_predict', axis = 1)
    Y = data.filter(['Strata', 'osteo_predict'])

    fold = KFold(10, shuffle = True, random_state = 12345)
    strata = data['Strata'].unique()

    all_preds = np.full(data.shape[0], 100)
    probability = np.ones(data.shape[0])

    training_acc = []
    testing_acc = []

    for train_index, test_index in fold.split(strata):

        train_index_strata = strata[train_index]
        test_index_strata = strata[test_index]

        X_train = X.loc[X['Strata'].isin(train_index_strata)]
        X_train = X_train.drop(['Strata'], axis = 1)

        X_test = X.loc[X['Strata'].isin(test_index_strata)]
        X_test = X_test.drop(['Strata'], axis = 1)

        y_train = Y.loc[Y['Strata'].isin(train_index_strata)]['osteo_predict']
        y_test = Y.loc[Y['Strata'].isin(test_index_strata)]['osteo_predict']

        if classifier == 'LR':
            lr = linear_model.LogisticRegression(random_state = 12345, solver = 'lbfgs')
            #class_weight = {0:1, 1:2}, penalty = 'l2'

        if classifier == 'RF':
            # Create a based model
            lr = RandomForestClassifier(random_state = 12345)
            if tune == 'true':
                param_grid = {
                    'bootstrap': [True],
                    'max_depth': [80, 90, 100, 110],
                    'max_features': [2, 3],
                    'min_samples_leaf': [3, 4, 5],
                    'min_samples_split': [8, 10, 12],
                    'n_estimators': [100, 200, 300, 1000]
                            }
                # Instantiate the grid search model
                grid_search = GridSearchCV(estimator = lr, param_grid = param_grid, 
                                          cv = 3, n_jobs = -1, verbose = 1)
                # Fit the grid search to the data
                grid_search.fit(X_train, y_train)
                print(grid_search.best_params_)
      
                best_grid = grid_search.best_estimator_
                grid_accuracy = evaluate(best_grid, X_test, y_test)
                print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
               
        if classifier == 'SVM_linear':
           svm_linear = LinearSVC(max_iter = 1000, random_state = 12345, C = 0.001, loss='hinge')
           lr = CalibratedClassifierCV(svm_linear)

        if classifier == 'SVM_rbf':
            lr = svm.SVC(kernel='rbf', probability = True, max_iter = 1000, C= 100, gamma = 30)
           
        if classifier == 'AdaBoost':
            logistic_regression = linear_model.LogisticRegression(random_state = 12345, solver = 'lbfgs')
            lr = AdaBoostClassifier(n_estimators=100,
                                    base_estimator = logistic_regression, 
                                    learning_rate=1)

        if classifier == 'XGB':
            lr =GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                     max_depth=3, random_state=42)
            
        fit = lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        all_preds[y_test.index] = y_pred

        probs = lr.predict_proba(X_test)
        probability[y_test.index] = probs[:, 1]

        training_acc.append(fit.score(X_train, y_train))
        testing_acc.append(fit.score(X_test, y_test))
        
    data['all_pred'] = all_preds
    data['probability'] = probability
    print('\nfinal result of %s model: ' % classifier)
    print(classification_report(data['osteo_predict'], data['all_pred']), '\n')
    print(confusion_matrix(data['osteo_predict'], data['all_pred']), '\n')

    print('ROC_AUC of', classifier, '', roc_auc_score(data['osteo_predict'], data['probability']))
    return data, training_acc, testing_acc, lr

def PLOT_ACCURACY(training_acc, testing_acc):
    plt.plot(training_acc)
    plt.plot(testing_acc)
    plt.xlabel('10-fold CV')
    plt.ylabel('accuracy')
    plt.legend(['training_acc', 'testing_acc'])
    plt.title('Model accuracy')
    plt.show()

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
            print(i)
            print(classification_report(subgroup_1['osteo_predict'], subgroup_1['all_pred']))
            print(confusion_matrix(subgroup_1['osteo_predict'],subgroup_1['all_pred']))
            print('ROC_AUC of', i, roc_auc_score(subgroup_1['osteo_predict'], subgroup_1['probability']))
            print('----')

    elif demographic == 'Age_combine':
        for i in age:
            subgroup_2 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_2['osteo_predict'], subgroup_2['all_pred']))
            print(confusion_matrix(subgroup_2['osteo_predict'],subgroup_2['all_pred']))
            print('ROC_AUC of',i, roc_auc_score(subgroup_2['osteo_predict'], subgroup_2['probability']))
            print('------')

    elif demographic == 'sex':
        for i in sex:
            subgroup_3 = OHE.loc[OHE[i] == 1]
            print(i)
            print(classification_report(subgroup_3['osteo_predict'], subgroup_3['all_pred']))
            print(confusion_matrix(subgroup_3['osteo_predict'],subgroup_3['all_pred']))
            print('ROC_AUC of',i, roc_auc_score(subgroup_3['osteo_predict'], subgroup_3['probability']))
            print('-----')


def TRAIN_TOGETHER(original_data, classifier, demographic):
    extract = FILTER_DATA(original_data)
    osteo= CONVERT_STRING_FLOAT(extract)
    ohe = OHE(osteo)
    print('')
    df_model, training_acc, testing_acc= TRAIN_MODEL_ML(ohe, classifier)
    print('')
    PLOT_ACCURACY(training_acc, testing_acc)
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
        fil = FILTER_DATA(data_extract)
        convert= CONVERT_STRING_FLOAT(fil)
        print(' ')
        ohe = OHE(convert)
        print(' ')
        df_model, training_acc, testing_acc= TRAIN_MODEL_ML(ohe, classifier)
        print('\n ------------------')
        PLOT_ACCURACY(training_acc, testing_acc)
        print('\n ------- ENDING 1 SUBGROUP --------------')
