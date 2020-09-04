#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:56:13 2020

@author: thanhng
"""


import pandas as pd
import numpy as np

from data_cleaning import *

path = '/Users/thanhng/is-Thanh/data/osteo_impute_final.csv'

path = '/Users/thanhng/is-Thanh/data/osteo_clean_no_Decile.csv'
data = pd.read_csv(path)

'''take avg of BMI_avg'''
###take avg of BMI_avg groupby _imputation_

index_in_dataset = list(range(1,(data.shape[0]//5)+1)) *5
data['index_in_dataset'] = index_in_dataset


#take only imputation, index_in_dataset, BMI_Avg
filter_data= data.filter(['_Imputation_', 'index_in_dataset', 'BMI_Avg', 'Age_Enc_Date1'])
filter_data_1 = filter_data.groupby('index_in_dataset').agg(np.mean)


#rename BMI_Avg_Edit
filter_data_1 = filter_data_1.rename(columns={'BMI_Avg':'BMI_Avg_imputed'}).reset_index(drop=True)
filter_data_1['BMI_Avg_imputed']


#extract first imputation
data_1 = data[data['_Imputation_'] ==1]
#add BMI_Avg_imputed col to data_1
data_1['BMI_Avg_imputed'] = filter_data_1['BMI_Avg_imputed']

'''recode Y '''
data_1['osteo_predict'] = data_1['Osteoporosis_code'].copy()

###recode Y
data_1['osteo_predict'] = data_1['osteo_predict'].replace(733.00, 1)
data_1['osteo_predict'] = data_1['osteo_predict'].replace(733.01, 1)
data_1['osteo_predict'] = data_1['osteo_predict'].replace(733.09, 1)
data_1['osteo_predict'] = data_1['osteo_predict'].replace(733.02, 1)
data_1['osteo_predict'] = data_1['osteo_predict'].replace(733.03, 1)

# fill NA = 0, NA = no disease
data_1['osteo_predict'].fillna(0, inplace=True)

#check result: if diagnosis w disease = 1, otherwise = 0
print(data_1['Osteoporosis_code'].value_counts())
print('\n', data_1['osteo_predict'].value_counts())


#check gender

check_across_subgroup(data_1, 'osteo_predict', 'sex', 'no')



'''recode AGE using 'Age_Enc_Date1'''

for i in range(data_1.shape[0]):
    age = data_1['Age_Enc_Date1'][i]
    if age <30:
        data_1.loc[i, 'Age_combine'] = 'less_than_30'
    if age >=30 and age <50:
        data_1.loc[i, 'Age_combine'] = 'from_30_to_50'
    if age >=50 and age <70:
        data_1.loc[i, 'Age_combine'] = 'from_50_to_70'
    if age >=70 and age <80:
        data_1.loc[i, 'Age_combine'] = 'from_70_to_80'
    if age >=80:
        data_1.loc[i, 'Age_combine'] = 'more_than_80'

check_across_subgroup(data_1, 'osteo_predict', 'Age_combine', 'no')

'''recode RACE using 'Race_Cat' '''
race_dict = {'0': 'others',
            '1': 'black',
            '2': 'white',
            '3': 'others',
            '4':'others',
            '5': 'others',
            '6': 'others',
             '7': 'others'
            }
#combine race
data_1['race_combine'] = data_1['Race_Cat'].copy()
RECODE_DEVICE(data_1, 'race_combine', race_dict)

check_across_subgroup(data_1, 'osteo_predict', 'race_combine', 'no')


'''patientid'''

# remove duplicate patientID in osteo dataset
###duplicated patientid but different Enc_date1--> keep the first Enc_date1 of duplicated patient
data_1['is_dup_patientID'] = data_1['patientid'].duplicated()
data_nodup = data_1.loc[data_1['is_dup_patientID'] == False]

print('# of duplicate: ', data_1.loc[data_1['is_dup_patientID'] == True].shape[0])
print('old shape: ', data_1.shape)
print('new shape: ', data_nodup.shape )

'''imbalance strata due to dropping patientid'''

missing_strata = data_nodup.groupby('Strata').filter(lambda x: len(x) ==1)['Strata']
missing_strata

data_nodup_1 = data_nodup.drop(missing_strata.index, axis = 0)
print('before drop missing in strata, data.shape = ', data_nodup.shape)
print('after dropping: ', data_nodup_1.shape)


''''Lowest_Recent_Hypo_Cat_edit''''
#fillna with 0
data_nodup_1['Lowest_Recent_Hypo_Cat_edit'] = data_nodup_1['Lowest_Recent_Hypo_Cat'].fillna(0)
data_nodup_1['Lowest_Recent_Hypo_Cat_edit'].value_counts()


''''Median_Recent_Hypo_Cat_edit''''
#fillna with 0
data_nodup_1['Median_Recent_Hypo_Cat_edit'] = data_nodup_1['Median_Recent_Hypo_Cat'].fillna(0)
data_nodup_1['Median_Recent_Hypo_Cat_edit'].value_counts()



''''Calcium_Avg_Ever''''
# fillna with mean of no_missing
cal_mean = np.mean(data_nodup_1[data_npdup_1['Calcium_Avg_Ever'].notnull()]['Calcium_Avg_Ever'])
data_nodup_1['Calcium_Avg_Ever'] = data_nodup_1['Calcium_Avg_Ever'].fillna(cal_mean)


''''Sodium_Avg_Ever''''
# fillna with mean of no_missing
sodium_mean = np.mean(data_nodup_1[data_nodup_1['Sodium_Avg_Ever'].notnull()]['Sodium_Avg_Ever'])
data_nodup_1['Sodium_Avg_Ever'] = data_nodup_1['Sodium_Avg_Ever'].fillna(sodium_mean)

#####################

data_nodup_1=pd.read_csv(path)




'''create decile and discretize continuous variables '''

#create discretize var + add to df
dic = {'Calcium_Closest_Osteo': 'Calcium_Closest_Osteo_decile',
        'Calcium_Avg_Prior': 'Calcium_Avg_Prior_decile',
        'Calcium_Avg_Ever': 'Calcium_Avg_Ever_decile',
#     'Sodium_Closest_Osteo': 'Sodium_Closest_Osteo_decile',
        'Sodium_Avg_Prior': 'Sodium_Avg_Prior_decile',
        'Sodium_Avg_Ever': 'Sodium_Avg_Ever_decile',
        'Sodium_Worst_Prior': 'Sodium_Worst_Prior_decile',
        'Sodium_Worst_Ever': 'Sodium_Worst_Ever_decile'}

for var,new_var in dic.items():
    decile(data_nodup_1, var, new_var)


#sodium_closest_osteo has error, need to fix by doing pct_rank_qcut
data_nodup_1['Sodium_Closest_Osteo_decile'] = pct_rank_qcut(data_nodup_1['Sodium_Closest_Osteo'], 10)
    
    

'''create presence/absence variables '''

#create absence/present var + add to df
presence_absence_dict = {'Calcium_Closest_Osteo': 'Calcium_Closest_Osteo_cat',
                          'Calcium_Avg_Prior': 'Calcium_Avg_Prior_cat',
                          'Calcium_Avg_Ever': 'Calcium_Avg_Ever_cat',
                         'Sodium_Closest_Osteo': 'Sodium_Closest_Osteo_cat',
                         'Sodium_Avg_Prior': 'Sodium_Avg_Prior_cat',
                         'Sodium_Avg_Ever': 'Sodium_Avg_Ever_cat',
                         'Sodium_Worst_Prior': 'Sodium_Worst_Prior_cat',
                         'Sodium_Worst_Ever': 'Sodium_Worst_Ever_cat'}

for var, new_var in presence_absence_dict.items():
    print('var: ', var)
    present_absence(data_nodup_1, var, new_var)
    print(' ')


# extract needed columns

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

data_filter= data_nodup_1.filter(cols_interested)

[col for col in cols_interested if col not in data_filter.columns]

output='/Users/thanhng/is-Thanh/data/osteo_clean.csv'
data_filter.to_csv(output, index=False)
