
__Demographic on gender__: 

| osteo_predict | 0.0	 | 1.0 | Total
| --- | --- | --- | --- |
| Female | 24,551 |	24,547 |49,098|
| Male | 3,400 |	3,400| 6,800|
| Total |	27,951 |	27,947	| 55,898|

__WITH STRATA__: Gender_Train_together: 
 
| osteo_predict | pos	 | neg |
| --- | --- | --- |
| reality_True | 3716.3 |  1594.2|
| reality_False | 1136.3| 4174.2 |

Female_train_together: 

| osteo_predict | pos	 | neg |
| --- | --- | --- |
| reality_True | 3261.8| 1411. |
| reality_False | 996. | 3676.8 |

Male_train_together: 

| osteo_predict | pos	 | neg |
| --- | --- | --- |
| reality_True | 459.8 | 187.5|
| reality_False |144.3 |503. |

Gender_train together:

         precision    recall  f1-score   support
    0.0       0.77      0.70      0.73     27941
    1.0       0.72      0.79      0.75     27941
    avg/total 0.74      0.74      0.74     55882

Female_train_together: 

        precision    recall  f1-score   support
    0.0       0.77      0.70      0.73     24541
    1.0       0.72      0.79      0.75     24541
    avg/total 0.74      0.74      0.74     49082
    
 Male_train_together: 
 
    precision    recall  f1-score   support
    0.0       0.76      0.71      0.73      3400
    1.0       0.73      0.78      0.75      3400
    avg/total 0.74      0.74      0.74      6800
    
 Female_separate: 
 
            precision    recall  f1-score   support
    0.0       0.77      0.70      0.73     24551
    1.0       0.72      0.79      0.75     24547
    avg / total       0.75      0.74      0.74     49098
    
Female_Separate: 

| osteo_predict | pos	 | neg |
| --- | --- | --- |
| reality_True | 3254.7|  1419.|
| reality_False | 981.3| 3692.|

Male_separate: 

        precision    recall  f1-score   support
        0.0       0.75      0.72      0.74      3400
        1.0       0.73      0.76      0.75      3400
        avg / total       0.74      0.74      0.74      6800

Male_separate: 

| osteo_predict | pos	 | neg |
| --- | --- | --- |
| reality_True |  465.4|  180.|
| reality_False | 152.6| 492.8|

**********************
 __Race demographics__: 
 
| osteo_predict | 0.0	 | 1.0 | Total
| --- | --- | --- | --- |
| White | 18,512 (33.13%)	|18,218  (32.60%) |36,730 (65.72%)|
| Black| 8,596 (15.38%)|	8,627 (15.43%)|17,223 (30.82%)|
| Others |833 (1.491%)|	1,096 (1.96%)|	1,929 (3.45%)|
|Total| 27,941 (50%)|27,941 (50%)|	55,882 (100%)|

Race_train_together: 

             precision    recall  f1-score   support
        0.0       0.77      0.70      0.73     27941
        1.0       0.72      0.79      0.75     27941
        avg / total       0.74      0.74      0.74     55882

__white_train_together:__    

                  precision recall    f1-score support
        0.0       0.76      0.69      0.72     18512
        1.0       0.71      0.77      0.74     18218
        avg/total 0.73      0.73      0.73     36730
        
__black_train_together:__   

        precision    recall  f1-score   support
        0.0       0.79      0.72      0.75      8596
        1.0       0.75      0.80      0.77      8627
        avg / total       0.77      0.76      0.76     17223


__others_train_together:__ 

        precision    recall  f1-score   support
        0.0       0.75      0.59      0.66       833
        1.0       0.73      0.85      0.79      1096
        avg / total       0.74      0.74      0.73      1929


ON AVG: confusion matrix for __white__: 

| osteo_predict | pos	 | neg |
| --- | --- | --- |
| reality_True | 2443.9 |  1080.5|
| reality_False | 788.8| 2679.8 |

ON AVG: confusion matrix for __black__: 
 
| osteo_predict | pos	 | neg |
| --- | --- | --- |
| reality_True | 1185. |  452.5|
| reality_False| 320.| 1323.1 |

ON AVG: confusion matrix for __others__: 

| osteo_predict | pos	 | neg |
| --- | --- | --- |
| reality_True | 93.1 |  65.1|
| reality_False| 30.4|178.|
