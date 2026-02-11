import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

logger = logging.getLogger("src.model_training.train_model")
logging.basicConfig(level=logging.INFO)


def load_data() -> pd.DataFrame:
    train_path = "data/processed/train_processed.csv"
    logger.info(f"Loading feature data from {train_path}")
    train_data = pd.read_csv(train_path)
    return train_data


def load_params() -> dict[str, float | int | str]:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def prepare_data(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    X_train = train_data.drop("species", axis=1)
    y_train = train_data["species"]

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    return X_train, y_train_encoded, encoder


def create_model(params: dict[str, int | float | str]) -> SVC:
    svm = SVC(
        kernel=params["kernel"],
        C=params["C"],
        tol=params["tol"],
    )
    return svm

def save_training_artifacts(model: SVC, encoder: LabelEncoder) -> None:
    artifacts_dir = "artifacts"
    models_dir = "models"

    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.joblib")
    encoder_path = os.path.join(artifacts_dir, "target_label_encoder.joblib")

    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    logger.info(f"Saving encoder to {encoder_path}")
    joblib.dump(encoder, encoder_path)


def train_model(train_data: pd.DataFrame, params: dict[str, int | float | str]) -> None:
    X_train, y_train, encoder = prepare_data(train_data)
    model = create_model(params=params)

    logger.info("Training model...")
    model.fit(X_train, y_train)

    save_training_artifacts(model, encoder)

    metrics = {"accuracy": float(model.score(X_train, y_train))}
    metrics_path = "metrics/training.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training metrics saved to {metrics_path}")


def main() -> None:
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()
