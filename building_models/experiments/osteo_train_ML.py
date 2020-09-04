#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:11:19 2020

@author: thanhng
"""

from ML_train_models import *
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

path = '/Users/thanhng/is-Thanh/data/osteo_clean.csv'
data = pd.read_csv(path)


'''filter data '''
cols_interested= ['Strata','sex','race_combine', 'Age_combine','osteo_predict', 'BMI_Avg_imputed',
                'Alcohol_Prior', 'Tobacco_Prior','Drug_antipsych_prior', 'Drug_Estrogens_prior',
                'Drug_Glucocorticoids_prior','Drug_Nsaids_prior', 'Drug_Opiates_prior',
                'Drug_Thiazide_prior','Drug_Loop_Diuretic_Prior','Drug_Pp_inhibitors_prior',
                'Drug_Progesterone_prior','Drug_Seizure_prior','Drug_Ssris_prior',
                'Drug_Tc_antidepress_prior','HeartDisease_Prior','Liver_Prior','PulmDisease_Prior',
                'CNS_Disease_Prior','Malignancy_Prior','Hyponatremia_Prior','Chronic_Hyponatremia','Recent_Hyponatremia',
                'Median_Recent_Hypo_Cat_edit', 'Lowest_Recent_Hypo_Cat_edit',
                'Calcium_Closest_Osteo_decile','Calcium_Avg_Prior_decile','Calcium_Avg_Ever_decile',
                'Sodium_Closest_Osteo_decile','Sodium_Avg_Prior_decile','Sodium_Avg_Ever_decile',
                'Sodium_Worst_Prior_decile','Sodium_Worst_Ever_decile',
                'Calcium_Closest_Osteo_cat', 'Calcium_Avg_Prior_cat','Calcium_Avg_Ever_cat',
                'Sodium_Closest_Osteo_cat','Sodium_Avg_Prior_cat','Sodium_Avg_Ever_cat',
                'Sodium_Worst_Prior_cat','Sodium_Worst_Ever_cat']

data_filter = data.filter(cols_interested)



'''convert int64 to object dtypes to prepare for OHE '''
#check dtypes first
data_filter.dtypes

data_prepare = CONVERT_STRING_FLOAT(data_filter)

data_prepare.dtypes

#OHE (from 46 to 139 cols)
data_OHE = OHE(data_prepare)

#LR

data_LR, train_LR, test_LR, lr = TRAIN_MODEL_ML(data_OHE, 'LR')

PLOT_ACCURACY(train_LR, test_LR)

data_LR['osteo_predict'] = data_LR['osteo_predict'].astype(int)

data_LR.columns

data_LR['all_pred']

data_LR['probability']

data_LR['osteo_predict']

log_loss(data_LR['osteo_predict'], data_LR['probability'])

#plot log_loss
yhat = [x*0.01 for x in range(0, 101)]
# evaluate predictions for a 0 true value
losses_0 = [log_loss([0], [x], labels=[0,1]) for x in yhat]
# evaluate predictions for a 1 true value
losses_1 = [log_loss([1], [x], labels=[0,1]) for x in yhat]

# plot input to loss
pyplot.plot(yhat, losses_0, label='true=0')
pyplot.plot(yhat, losses_1, label='true=1')
pyplot.legend()
pyplot.show()


#plot ROC

fpr,tpr,threshold=roc_curve(data_LR['osteo_predict'], data_LR['probability'])

plt.plot([0,1],[0,1],linestyle="-",label="Generalized Model Curve")
plt.plot(fpr,tpr,marker=".",label="ROC CURVE")
plt.legend()
plt.show()

###feature importance
#The positive scores indicate a feature that predicts class 1, 
#whereas the negative scores indicate a feature that predicts class 0.
feature_importance = lr.coef_[0]

def FEATURE_IMPORTANCE(data_OHE, feature_importance):
    feature_rank = np.array([feature_importance]).T   
    cols_name = np.array([data_OHE.columns[1:-1]]).T    
    feature_importance_df=pd.DataFrame(np.hstack((cols_name, feature_rank)), columns=['feature', 'importance']).sort_values(by='importance', ascending=False)
    return feature_importance_df

lr_feature_importance = FEATURE_IMPORTANCE(feature_importance)

lr_feature_importance.head(20)

# summarize feature importance
for i,v in enumerate(feature_importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(feature_importance))], feature_importance)
pyplot.show()


#RF

data_RF, train_RF, test_RF, lr = TRAIN_MODEL_ML(data_OHE, 'RF')

PLOT_ACCURACY(train_RF, test_RF)

feature_importance = lr.feature_importances_

#SVM

data_SVM_linear, train_SVM_lin, test_SVM_lin, svm = TRAIN_MODEL_ML(data_OHE, 'SVM_linear')

train_SVM_lin

test_SVM_lin

PLOT_ACCURACY(train_SVM_lin, test_SVM_lin)


data_SVM_rbf, train_SVM_rbf, test_SVM_rbf, svm_rbf = TRAIN_MODEL_ML(data_OHE, 'SVM_rbf')


####gradientboosting

data_gradboosting, train_gradboosting, test_gradboosting, gradboosting = TRAIN_MODEL_ML(data_OHE, 'XGB')

PLOT_ACCURACY(train_gradboosting, test_gradboosting)

def PLOT_ROC(data_model):
    fpr,tpr,threshold=roc_curve(data_model['osteo_predict'], data_model['probability'])
    plt.plot([0,1],[0,1],linestyle="-",label="Generalized Model Curve")
    plt.plot(fpr,tpr,marker=".",label="ROC CURVE")
    plt.legend()
    plt.show()
    
PLOT_ROC(data_gradboosting)
    

###plot ROC of multiple classifiers


TRAIN_MODEL_ML(data_OHE, 'RF', 'true')


classifiers = ['LR', 'RF', 'XGB', 'AdaBoost']
def RUN_MULTIPLE_MODELS(classifiers):
    result = []
    for classifier in classifiers:    
        model_data, train_acc, test_acc, clf = TRAIN_MODEL_ML(data_OHE, classifier)
        fpr,tpr,threshold=roc_curve(model_data['osteo_predict'], model_data['probability'])
        auc = roc_auc_score(model_data['osteo_predict'], model_data['probability'])
        result.append({'classifier': classifier, 
                          'FPR': fpr, 
                          'TPR': tpr, 
                          'AUC': auc, 
                          'training_acc': train_acc, 
                          'testing_acc': test_acc})
    
    result_df = pd.DataFrame(result, columns=['classifier','FPR','TPR','AUC',
                                      'training_acc', 'testing_acc'])

    result_df.set_index('classifier', inplace=True)
    return result_df


result_df = RUN_MULTIPLE_MODELS(classifiers)

def PLOT_MULTIPLE_ROC(result_df):
    #plot multiple AUC in 1 plot   
    fig = plt.figure(figsize=(8,6))
    for i in result_df.index:
        plt.plot(result_df.loc[i]['FPR'], 
                 result_df.loc[i]['TPR'], 
                 label="{}, AUC={:.3f}".format(i, result_df.loc[i]['AUC']))
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)
    plt.title('ROC Curve Comparison', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')
    plt.show()

#box-plot of training and testing acc

# boxplot algorithm comparison
def BOXPLOT_RESULTS(result_df, col):
    fig = plt.figure()
    fig.suptitle('%s comparison' % col)
    ax = fig.add_subplot(111)
    plt.boxplot(result_df[col])
    ax.set_xticklabels(classifiers)
    plt.show()
    
BOXPLOT_RESULTS(result_df, 'testing_acc')
BOXPLOT_RESULTS(result_df, 'training_acc')

for i in classifiers:
    print('classifier: ', i)
    PLOT_ACCURACY(result_df.loc[i]['training_acc'], result_df.loc[i]['testing_acc'])
    
