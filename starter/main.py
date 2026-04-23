# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

import sys
from pathlib import Path

# Make sure project root is on PYTHONPATH (for sanitycheck compatibility)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference


# Initialize FastAPI app
app = FastAPI(title="ML Model Inference API")

# Load trained artifacts
MODEL_PATH = "starter/model"
model = joblib.load(f"{MODEL_PATH}/model.joblib")
encoder = joblib.load(f"{MODEL_PATH}/encoder.joblib")
lb = joblib.load(f"{MODEL_PATH}/lb.joblib")

# Categorical features (must match training)
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Pydantic model with example values
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML model inference API!"}


@app.post("/model")
def predict(data: CensusData):
    # Convert input to DataFrame
    input_dict = data.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    # Process data (no training)
    X, _, _, _ = process_data(
        input_df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run inference
    preds = inference(model, X)

    # Decode prediction
    prediction = lb.inverse_transform(preds)[0].strip()

    return {"prediction": prediction}
