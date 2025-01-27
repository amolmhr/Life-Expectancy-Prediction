import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LifeExpectancyPrediction.logger import logging

# Importing the libraries XGB Regressor, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

def split_data(X, y):
    """
    Split the data into training and test sets.

    Parameters
    ----------
    X : pandas DataFrame
        The input features.
    y : pandas Series
        The target variable.

    Returns
    -------
    Tuple
        A tuple containing the training and test sets: (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and test sets.")

    return X_train, X_test, y_train, y_test

def Standardize_data(X_train, X_test):
    """
    Standardize the input features.

    Parameters
    ----------
    X_train : pandas DataFrame
        The training input features.
    X_test : pandas DataFrame
        The test input features.

    Returns
    -------
    Tuple
        A tuple containing the standardized training and test sets: (X_train_std, X_test_std).
    """
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    logging.info("Input features standardized.")
    return X_train_std, X_test_std

def train_model(X_train, y_train, model_name):
    """
    Train a machine learning model.

    Parameters
    ----------
    X_train : pandas DataFrame
        The training input features.
    y_train : pandas Series
        The training target variable.
    model_name : str
        The name of the model to train.

    Returns
    -------
    sklearn model
        The trained machine learning model.
    """
    if model_name == "XGBRegressor":
        model = XGBRegressor()
        logging.info("XGBRegressor model created.")
    elif model_name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor()
        logging.info("DecisionTreeRegressor model created.")
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor()
        logging.info("RandomForestRegressor model created.")
    elif model_name == "GradientBoostingRegressor":
        model = GradientBoostingRegressor()
        logging.info("GradientBoostingRegressor model created.")
    else:
        raise ValueError(f"Invalid model name: {model_name}")
        logging.error(f"Invalid model name: {model_name}")

    model.fit(X_train, y_train)
    logging.info("Model trained.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a machine learning model.

    Parameters
    ----------
    model : sklearn model
        The trained machine learning model.

    Returns
    -------
    dict
        A dictionary containing the model evaluation metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R^2 Score": r2
    }
    logging.info("Model evaluated.")
    return metrics


def train_and_evaluate_model(data):
    """ 
    Train and evaluate multiple machine learning models.
    """
    X = data.drop("Life expectancy ", axis=1)
    y = data["Life expectancy "]
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_std, X_test_std = Standardize_data(X_train, X_test)
    
    models = ["XGBRegressor", "DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor"]
    mse = []
    mae = []
    r2 = []
    trained_models={}
    for model_name in models:
        model = train_model(X_train_std, y_train, model_name)
        trained_models[model_name] = model
        metrics = evaluate_model(model, X_test_std, y_test)
        mse.append(metrics["Mean Squared Error"])
        mae.append(metrics["Mean Absolute Error"])
        r2.append(metrics["R^2 Score"])

    # Creating a dataframe for model comparison
    model_comparison = pd.DataFrame({
        "Model": models,
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R2 Score": r2
    })
    
    # Storing the model comparison in the data processed folder
    base_dir = "D:/Life-Expectancy-Prediction/artifacts"
    model_results_path = os.path.join(base_dir, "model_results")
    results_file_path = os.path.join(model_results_path, "model_comparison.csv")

    try:
        os.makedirs(model_results_path, exist_ok=True)  # Create required directories
        model_comparison.to_csv(results_file_path, index=False)
        logging.info(f"Model comparison saved to {results_file_path}")
    except PermissionError as e:
        logging.error(f"PermissionError: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

    return trained_models,model_comparison

def save_best_model(trained_models,model_comparison):
    """
    Save the best model based on the evaluation metrics.
    """
    best_model = model_comparison.loc[model_comparison["Mean Squared Error"].idxmin()]
    model_name = best_model["Model"]
    model = trained_models[model_name]
    model_path = os.path.join("D:/Life-Expectancy-Prediction/artifacts", f"{model_name}.pkl")

    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Best model saved to {model_path}")
    except PermissionError as e:
        logging.error(f"PermissionError: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

    

if __name__ == "__main__":
    data = pd.read_csv("D:/Life-Expectancy-Prediction/notebooks/data/processed/processed_data.csv")
    train_and_evaluate_model(data)