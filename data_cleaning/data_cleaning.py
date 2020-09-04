import pandas as pd
import numpy as np
import re


def check_missing(data, var):
    print(var)
    print('# of missing: ', data[var].isna().sum(), 'out of', data.shape[0])
    print('% of missing: ',  data[var].isna().sum()/data.shape[0], '\n')

def decile(data, var, new_var):
    '''create 10 bins and discretize data
    '''
    print(pd.qcut(data[var],10).value_counts())
    data[new_var] = pd.qcut(data[var],10, labels=False)
    return pd.qcut(data[var],10, labels=False)

def pct_rank_qcut(series, n):
    '''special case for errors in qcut
    here missing = 0
    '''
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).values.argmax()
    print(series.rank(pct=1).apply(f).value_counts())
    return series.rank(pct=1).apply(f)
#data_osteo['Sodium_Closest_Osteo_decile'] = pct_rank_qcut(data_osteo['Sodium_Closest_Osteo'], 10)

def present_absence(data, var, new_var):
    ''' create var: 0 = missing, 1 = non-missing
    '''
    print('# of missing: ', data[var].isna().sum())

    not_missing = pd.notna(data[var])
    b = []
    for i in not_missing:
        if i == True:
            b.append(1)
        if i == False:
            b.append(0)

    import collections
    counter=collections.Counter(b)
    print(counter)
    data[new_var] = b

def extract_outliers(data, variable, threshold, side):
    if side == "greater":
        print('# of outliers: ', data.loc[data[variable] >= threshold][variable].count())
        return data.loc[data[variable] >= threshold][variable]
    if side == "less":
        print('# of outliers: ', data.loc[data[variable] <= threshold][variable].count())
        return data.loc[data[variable] <= threshold]
    if side == 'equal':
        print('# of outliers: ', data.loc[data[variable] == threshold][variable].count())
        return data.loc[data[variable] == threshold]


def check_across_subgroup(data, col, row, percentage):
    if percentage == 'yes':
        return pd.crosstab(index=data[row], columns=data[col], margins = True)/data.shape[0]
    if percentage == 'no':
        return pd.crosstab(index=data[row], columns=data[col], margins = True)

def extract_missing_check_subgroup(data, var1, var2, var3):
    print('# of rows: \n', data[var1].value_counts())
    calculate_missing = extract_outliers(data, var1, 0, 'equal')
    print(check_across_subgroup(calculate_missing, var2, var3))

def RECODE_DEVICE(data, col, abbr_dict):
    '''use to fix typo, abbreviation, or recode a term, using (key, value) in dictionary '''
    for i, text in enumerate(data[col]):
        for key, val in abbr_dict.items():
#             key = key.lower()
#             val = val.lower()
            regex_key = r'\b' + key + r'\b'
            result = re.search(regex_key, str(text), flags = re.IGNORECASE)
            if result:
                data.loc[i, col] = val
    return data[col]


### OHE

def convert_to_str(data, var):
    print('before convert: ', data[var].dtypes)
    data[var] = data[var].astype(str)
    print('after convert: ', data[var].dtypes)
