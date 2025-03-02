import pytest
import pandas as pd
from src.model_pipeline import (
    load_data_from_csv,
    prepare_data,
    train_model,
    evaluate_model,
)


@pytest.fixture
def sample_data():
    data = load_data_from_csv("data/data.csv")
    if data is None:
        pytest.skip("Skipping tests: data.csv not found")
    return data


def test_load_data(sample_data):
    assert sample_data is not None, "Data loading failed"
    assert isinstance(sample_data, pd.DataFrame), "Data should be a DataFrame"
    assert "Churn" in sample_data.columns, "Churn column missing"


def test_prepare_data(sample_data):
    X_train, X_test, y_train, y_test, scaler, encoders = prepare_data(
        sample_data
    )
    assert X_train.shape[0] > 0, "X_train should not be empty"
    assert X_test.shape[0] > 0, "X_test should not be empty"
    assert len(y_train) == X_train.shape[0], "y_train length mismatch"
    assert len(y_test) == X_test.shape[0], "y_test length mismatch"
    assert scaler is not None, "Scaler should be initialized"
    assert isinstance(encoders, dict), "Encoders should be a dictionary"
    assert len(encoders) == 3, "Expected 3 encoders for categorical columns"


def test_train_model(sample_data):
    X_train, _, y_train, _, _, _ = prepare_data(sample_data)
    model = train_model(X_train, y_train)
    assert model is not None, "Model training failed"
    assert hasattr(model, "predict"), "Model should have predict method"
    assert model.score(X_train, y_train) > 0, "Model should have positive \
        training score"


def test_evaluate_model(sample_data):
    X_train, X_test, y_train, y_test, _, _ = prepare_data(sample_data)
    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert isinstance(report, str), "Report should be a string"
    assert "precision" in report, "Report should contain precision metric"
