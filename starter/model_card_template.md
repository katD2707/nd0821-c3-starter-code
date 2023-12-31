# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

## Intended Use

## Training Data

## Evaluation Data

## Metrics
The model was evaluated using precision, recall and f1 score.
The model has a precision of 0.74, recall of 0.65 and f1 score of 0.69.

## Ethical Considerations
This dataset does not accurately represent salary distributions and should not be utilized to form presumptions about the income of specific group of people.

## Caveats and Recommendations
To further enhance performance, one can explore hyperparameter tuning. In addition, we can use the K-fold cross validation `KFold` instaed of `train_test_spilt` from sklearn library. Another valuable recommendation is to engage in feature engineering. 