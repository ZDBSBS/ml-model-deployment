import pandas as pd
import numpy as np
from starter.starter.ml.data import process_data


def test_process_data_training_mode():
    data = pd.DataFrame({
        "age": [25, 45, 35],
        "workclass": ["Private", "Self-emp", "Private"],
        "salary": ["<=50K", ">50K", "<=50K"],
    })

    X, y, encoder, lb = process_data(
        data,
        categorical_features=["workclass"],
        label="salary",
        training=True,
    )

    assert X.shape[0] == 3
    assert y.shape[0] == 3
    assert encoder is not None
    assert lb is not None


def test_process_data_inference_mode():
    data = pd.DataFrame({
        "age": [30],
        "workclass": ["Private"],
        "salary": ["<=50K"],
    })

    X_train, y_train, encoder, lb = process_data(
        data,
        categorical_features=["workclass"],
        label="salary",
        training=True,
    )

    X_inf, y_inf, _, _ = process_data(
        data,
        categorical_features=["workclass"],
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert X_inf.shape[0] == 1