import numpy as np
from starter.starter.ml.model import train_model, inference, compute_model_metrics


def test_train_model_returns_model():
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, size=20)

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")


def test_inference_output_shape():
    X = np.random.rand(10, 4)
    y = np.random.randint(0, 2, size=10)

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(y)


def test_compute_model_metrics_returns_floats():
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)