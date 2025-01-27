import numpy as np
import pandas as pd
from LifeExpectancyPrediction.logger import logging
import os



def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the dataset from the given file path.

    Args:
    data_path: str: File path to the dataset

    Returns:
    pd.DataFrame: Loaded dataset
    """
    if not os.path.exists(data_path):
        logging.error(f"File not found at: {data_path}")
        return None

    # Load the dataset
    data = pd.read_csv(data_path)
    logging.info("Data loaded successfully.")

    return data


if __name__ == "__main__":
    data_path = "D:/Life-Expectancy-Prediction/notebooks/data/raw/Life Expectancy Data.csv"
    data = load_data(data_path)
    print(data.head())
