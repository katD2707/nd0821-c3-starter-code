"""
This module contains unit tests for the model rf in model_utils.py

Author: KatD2707
Date: 2023-31-12
"""
from pathlib import Path
import logging
import pandas as pd
import pytest
import numpy as np
from classification.model_utils import train_model, compute_model_metrics


def test_model_type():
    """
    Test the type of the model.
    """
    X_train1 = np.random.rand(100, 10)
    y_train1 = np.random.randint(0, 2, 100)
    model1 = train_model(X_train1, y_train1)
    assert type(model1).__name__ == 'RandomForestClassifier'


def test_load_data():
    """
    Check the data received.
    """
    data = pd.read_csv("data/census_clean.csv")
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0]>0
    assert data.shape[1]>0


def test_output_type():
    """
    Test the type of the output.
    """
    y1 = np.random.randint(0, 2, 10)
    preds1 = np.random.randint(0, 2, 10)
    precision1, recall1, fbeta1 = compute_model_metrics(y1, preds1)
    assert isinstance(precision1, float)
    assert isinstance(recall1, float)
    assert isinstance(fbeta1, float)

