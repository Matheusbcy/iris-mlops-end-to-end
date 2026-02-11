import logging
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("src.model_evaluation.evaluate_model")


def load_model():
    model_path = "models/model.joblib"
    model = joblib.load(model_path)
    return model


def load_encoder() -> LabelEncoder:
    encoder_path = "artifacts/target_label_encoder.joblib"
    encoder = joblib.load(encoder_path)
    return encoder


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    data_path = "data/processed/test_processed.csv"
    logger.info(f"Loading test data from {data_path}")
    data = pd.read_csv(data_path)
    X = data.drop("species", axis=1)
    y = data["species"]
    return X, y


def evaluate_model(
    model, encoder: LabelEncoder, X: pd.DataFrame, y_true: pd.Series
) -> None:
    y_true_encoded = encoder.transform(y_true)

    y_pred_encoded = model.predict(X)

    report = classification_report(
        y_true_encoded,
        y_pred_encoded,
        output_dict=True,
    )
    cm = confusion_matrix(y_true_encoded, y_pred_encoded).tolist()

    evaluation = {"classification_report": report, "confusion_matrix": cm}

    logger.info(
        "Classification Report:\n"
        + classification_report(
            y_true_encoded, y_pred_encoded
        )
    )

    evaluation_path = "metrics/evaluation.json"
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=2)


def main() -> None:
    model = load_model()
    encoder = load_encoder()
    X, y = load_test_data()
    evaluate_model(model, encoder, X, y)
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()