# Making training pipeline 
import os
from LifeExpectancyPrediction.components.model_trainer import split_data, Standardize_data, train_and_evaluate_model,save_best_model
from LifeExpectancyPrediction.logger import logging
from LifeExpectancyPrediction.exception import CustomException
from LifeExpectancyPrediction.components.data_ingestion import load_data
from LifeExpectancyPrediction.components.data_transformation import transform_data


if __name__ == "__main__":
    # Reading the data from the data folder
    data_path = ("D:/Life-Expectancy-Prediction/notebooks/data/raw/Life Expectancy Data.csv")
    data = load_data(data_path)
    transformed_data= transform_data(data)
    
    # training & Evaluating the model
    trained_models,model_comparison= train_and_evaluate_model(transformed_data)
    print(model_comparison)
    logging.info("Training pipeline completed.")
    save_best_model(trained_models,model_comparison)

    
    
