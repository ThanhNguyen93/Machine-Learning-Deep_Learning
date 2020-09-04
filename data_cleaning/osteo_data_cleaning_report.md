###report data cleaning

###impute BMI



#### demographics variables: age, gender, race
  for age: use 'Age_Enc_Date1', combine ages into 5 subcategories
  for race: use 'race_cat' instead of 'race', because 'race' doesn't have balance Y
  for gender: use 'gender', keep the same

#### patientid + strata:
    there are 8 duplicated patientid, same people but different encounter_day, thus I only keep the 1st encounter_day

      --> dropping imbalance pair in strata due to dropping duplicated in patientid

#### discrete variable:   
      - decide calcium and Sodium_Closest_Osteo
      - 'Lowest_Recent_Hypo_Cat': fillna with 0
      - 'Median_Recent_Hypo_Cat': fillna with 0


### continuous variable:
    'Calcium_Avg_Ever' & 'Sodium_Avg_Ever': fillna with mean of non_na

    - create decile and discretize variables:
        dic = {'Calcium_Closest_Osteo': 'Calcium_Closest_Osteo_decile',
                'Calcium_Avg_Prior': 'Calcium_Avg_Prior_decile',
                'Calcium_Avg_Ever': 'Calcium_Avg_Ever_decile',

          #     'Sodium_Closest_Osteo': 'Sodium_Closest_Osteo_decile',
                'Sodium_Avg_Prior': 'Sodium_Avg_Prior_decile',
                'Sodium_Avg_Ever': 'Sodium_Avg_Ever_decile',
                'Sodium_Worst_Prior': 'Sodium_Worst_Prior_decile',
                'Sodium_Worst_Ever': 'Sodium_Worst_Ever_decile'}

    - presence/absence variables: present = 1, absence = 0
        presence_absence_dict =
              {'Calcium_Closest_Osteo':'Calcium_Closest_Osteo_cat',
                'Calcium_Avg_Prior':'Calcium_Avg_Prior_cat',
                'Calcium_Avg_Ever':'Calcium_Avg_Ever_cat',

    #           'Sodium_Closest_Osteo':'Sodium_Closest_Osteo_decile',
                'Sodium_Avg_Prior':'Sodium_Avg_Prior_cat',
                'Sodium_Avg_Ever':'Sodium_Avg_Ever_cat',
                'Sodium_Worst_Prior':'Sodium_Worst_Prior_cat',
                'Sodium_Worst_Ever':'Sodium_Worst_Ever_cat'}



#### Y variable:
        recode 'Osteoporosis_code', create a variable 'osteo_predict' with 1 and 0
