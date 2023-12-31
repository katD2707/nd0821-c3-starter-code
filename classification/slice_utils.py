"""
This module outputs the performance of the model on slices of the data for categorical features

Author: KatD2707
Date: 2023-31-12
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from .model_utils import compute_model_metrics
from .data_utils import process_data
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

DATA_PATH = 'data/census_clean.csv'

# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def test_performance():
    """
    Test the performance of the model on slices of the data for categorical features.
    """
    # Load data
    data = pd.read_csv(DATA_PATH)
    # Split data
    _, test = train_test_split(data, test_size=0.2, random_state=42)
    # Load model
    model = joblib.load("saved_models/model.pkl")
    encoder = joblib.load("saved_models/encoder.pkl")
    lb = joblib.load("saved_models/lb.pkl")

    slice_metrics = []

    logging.info("Saving performance metrics for slices to slice_output.txt")
    for feature in cat_features:
        for cls in test[feature].unique():
            df_temp = test[test[feature] == cls]
            X_test, y_test, _, _ = process_data(
                df_temp,
                cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False)
            y_pred = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            row = f"{feature} - {cls} :: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
            slice_metrics.append(row)

            with open('slice_output.txt', 'w') as file:
                for row in slice_metrics:
                    file.write(row + '\n')


if __name__ == "__main__":
    test_performance()