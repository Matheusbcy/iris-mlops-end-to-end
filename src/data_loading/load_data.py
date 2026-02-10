import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("src.data_loading.load_data")


def fetch_data() -> pd.DataFrame:
    logger.info("Fetching data...")
    
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    
    dataset = pd.read_csv("data/iris.data", names = columns)
    
    dataset.drop_duplicates(inplace=True)
    
    return dataset


def save_data(data: pd.DataFrame) -> None:
    output_path = "data/raw/raw.csv"
    logger.info(f"Saving raw data to {output_path}")
    data.to_csv(output_path, index=False)


def main() -> None:
    raw_data = fetch_data()
    save_data(raw_data)
    logger.info("Data loading completed")


if __name__ == "__main__":
    main()
