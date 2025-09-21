import pytest
# TODO: add necessary import
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Define categorical features, mirroring train_model.py
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

@pytest.fixture(scope="session")
def data():
    """
    Fixture to load the census data once for all tests.
    """
    df = pd.read_csv("data/census.csv", skipinitialspace=True)
    return df

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_shape(data):
    """
    Tests if the processed data has the correct shape and that
    the number of features has increased after one-hot encoding.
    """
    # GIVEN: The raw census dataframe

    # WHEN: The data is processed
    X, y, _, _, _ = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    # THEN: The output shapes should be correct
    assert X.shape[0] == y.shape[0], "Number of rows in X and y should match."
    assert X.shape[1] > data.shape[1], "Number of columns in X should be greater than original data due to encoding."


# TODO: implement the second test. Change the function name and input as needed
def test_train_model(data):
    """
    Tests if the train_model function returns a trained model object.
    """
    # GIVEN: The raw data
    train, _ = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, _, _, _ = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    # WHEN: The model is trained
    model = train_model(X_train, y_train)

    # THEN: The model object should not be None
    assert model is not None, "The trained model should not be None."


# TODO: implement the third test. Change the function name and input as needed
def test_inference_and_metrics(data):
    """
    Tests the full pipeline: training, inference, and evaluation,
    ensuring the F1 score is above a minimum threshold.
    """
    # GIVEN: A trained model and test data
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary", training=False, encoder=encoder, lb=lb, scaler=scaler
    )
    model = train_model(X_train, y_train)

    # WHEN: Inference is performed and metrics are computed
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)

    # THEN: The F1 score should be above a reasonable minimum
    assert f1 > 0.5, f"F1 score of {f1:.2f} is below the 0.5 threshold."
