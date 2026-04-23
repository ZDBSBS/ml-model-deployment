# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import os
import joblib

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("starter/data/census.csv")
data.columns = data.columns.str.strip()

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)

os.makedirs("starter/model", exist_ok=True)
joblib.dump(model, "starter/model/model.joblib")
joblib.dump(encoder, "starter/model/encoder.joblib")
joblib.dump(lb, "starter/model/lb.joblib")

# Optional: print test performance
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {fbeta}")

# Compute performance on slices of the data
slice_feature = "education"

with open("slice_output.txt", "w") as f:
    for value in test[slice_feature].unique():
        slice_data = test[test[slice_feature] == value]

        X_slice, y_slice, _, _ = process_data(
            slice_data,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds_slice = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)

        f.write(f"Slice: {slice_feature} = {value}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {fbeta}\n")
        f.write("-" * 40 + "\n")