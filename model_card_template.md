# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest Classification were trained.

* Model version: 1.0.0
* Model date: 31 Dec 2023

## Intended Use
The model can be used for predicting income classes on census data. There are two income classes >50K and <=50K (binary classification task).
## Training Data
The UCI Census Income Data Set was used for training. Further information on the dataset can be found at https://archive.ics.uci.edu/ml/datasets/census+income
For training 80% of the 32561 rows were used (26561 instances) in the training set.

## Evaluation Data
For evaluation 20% of the 32561 rows were used (6513 instances) in the test set.

## Metrics
The model was evaluated using precision, recall and f1 score.
The model has a precision of 0.74, recall of 0.65 and f1 score of 0.69.

## Ethical Considerations
This dataset does not accurately represent salary distributions and should not be utilized to form presumptions about the income of specific group of people.

## Caveats and Recommendations
To further enhance performance, one can explore hyperparameter tuning. In addition, we can use the K-fold cross validation `KFold` instaed of `train_test_spilt` from sklearn library. Another valuable recommendation is to engage in feature engineering. 