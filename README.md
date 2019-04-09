# Machine-Learning
thesis project. Building predictive modeling using statistical methods, machine learning and deep learning techniques

There are 2 datasets, both are case-control study:
1. osteoporosis, case-control for osteoporosis
2. bone-fracture, case-control for bone-fracture

Notice this is case-control study, i.e. CV must based on STRATA instead of doing normally

Techniques to be applied:
- Logistic Regression
- SVM (non-linear)
- Random Forest (non-linear) 
- ANN 

Update results:
- because this is case-control study, there should be no bias in subgroups/demographic, i.e. AUC should be the same across subgroups. 
But current results do not support that!!! Train using LR and RF to track non-linear relationship, both LR and RF not the same either train subgroup together or train them separately 
