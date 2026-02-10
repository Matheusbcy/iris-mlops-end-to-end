import logging
import os
import yaml

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


logger = logging.getLogger("src.data_preprocessing.preprocess_data")


def load_data() -> pd.DataFrame:
    input_path = "data/raw/raw.csv"
    logger.info(f"Loading raw data from {input_path}")
    data = pd.read_csv(input_path)
    return data


def load_params() -> dict[str, float | int]:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess_data"]


def preprocess_data(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Preprocessing data...")

    X = dataset.drop("species", axis=1)
    y = dataset["species"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def split_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = load_params()
    logger.info("Splitting data into train and test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_seed"],
    )

    train_data = X_train.copy()
    train_data["species"] = y_train

    test_data = X_test.copy()
    test_data["species"] = y_test

    return train_data, test_data


def save_artifacts(
    train_data: pd.DataFrame, test_data: pd.DataFrame, le: LabelEncoder
) -> None:
    data_dir = "data/preprocessed"
    logger.info(f"Saving processed data to {data_dir}")

    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    test_path = os.path.join(data_dir, "test_preprocessed.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    label_encoder_path = os.path.join("artifacts", "[features]_label_encoder.joblib")
    logger.info(f"Saving label encoder to {label_encoder_path}")
    joblib.dump(le, label_encoder_path)


def main() -> None:
    raw_data = load_data()
    X, y, le = preprocess_data(raw_data)
    train_data, test_data = split_data(X, y)
    save_artifacts(train_data, test_data, le)
    logger.info("Data preprocessing completed")


if __name__ == "__main__":
    main()
