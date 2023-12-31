# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the classification code.
import logging
import joblib
import pandas as pd
from data_utils import process_data
from model_utils import train_model, compute_model_metrics, inference

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s %(message)s"
                    )

# Add code to load in the data.
logging.info("Importing data...")
data = pd.read_csv("../data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.
logging.info("Pre-processing data...")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
     training=False, encoder=encoder, lb=lb)

# Train and save a model.
logging.info("Training model...")
model = train_model(X_train, y_train)

# Evaluate the model's performance.
logging.info("Evaluating model accuracy")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Save the models.
logging.info("Saving models...")
joblib.dump(model, "../saved_models/model.pkl")
joblib.dump(encoder, "../saved_models/encoder.pkl")
joblib.dump(lb, "../saved_models/lb.pkl")
