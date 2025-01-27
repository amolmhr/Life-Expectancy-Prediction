import numpy as np
import pandas as pd
from LifeExpectancyPrediction.logger import logging
from LifeExpectancyPrediction.exception import CustomException
from LifeExpectancyPrediction.components.data_ingestion import load_data

def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data transformation on the given dataset.

    Args:
    data: pd.DataFrame: Input dataset

    Returns:
    Transformed dataset 
    """
    df = data.copy()

    # Filling missing values as per the research jupyternotebook EDA exploration
    # Fill the population missiiing values with the mean of the population countrywise
    df['Population'] = df.groupby('Country')['Population'].transform(lambda x: x.fillna(x.mean()))
    df['Population'] = df.groupby('Status')['Population'].transform(lambda x: x.fillna(x.mean()))
    logging.info("Population missing values filled.The missing values now are {}".format(df['Population'].isnull().sum()))  


    # Fill the GDP missing values with the mean of the GDP countrywise
    df['GDP'] = df.groupby('Country')['GDP'].transform(lambda x: x.fillna(x.mean()))
    df['GDP'] = df.groupby('Status')['GDP'].transform(lambda x: x.fillna(x.mean()))
    logging.info("GDP missing values filled.The missing values now are {}".format(df['GDP'].isnull().sum()))

    # Fill the Hepatitis B missing values with the mean of the Hepatitis B countrywise
    df['Hepatitis B'] = df.groupby('Country')['Hepatitis B'].transform(lambda x: x.fillna(x.mean()))
    df['Hepatitis B'] = df.groupby('Status')['Hepatitis B'].transform(lambda x: x.fillna(x.mean()))
    logging.info("Hepatitis B missing values filled.The missing values now are {}".format(df['Hepatitis B'].isnull().sum()))

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df.groupby("Country")[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df.groupby("Status")[col].transform(lambda x: x.fillna(x.mean()))
        logging.info(f"{col} missing values filled. The missing values now are {df[col].isnull().sum()}")

    # Dropping the country & year column
    df.drop("Country", axis=1, inplace=True)
    df.drop("Year", axis=1, inplace=True)
    logging.info("Country and Year columns dropped.")

    # Assigning 0 to developing and 1 to developed
    df["Status"] = df["Status"].map({"Developing":0, "Developed":1})
    logging.info("Status column transformed.")

    return df

def main():
    # Reading the data from the data folder
    data_path = ("D:/Life-Expectancy-Prediction/notebooks/data/raw/Life Expectancy Data.csv")
    data = load_data(data_path)
    transformed_data= transform_data(data)

    # Storing the transformed data in the data processed folder
    transformed_data.to_csv("D:/Life-Expectancy-Prediction/notebooks/data/processed/processed_data.csv", index=False)
    logging.info("Transformed data stored in the data processed folder.")

if __name__ == "__main__":
    main()






