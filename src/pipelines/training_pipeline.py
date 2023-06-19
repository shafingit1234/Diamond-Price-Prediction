import os
import sys
from src.logger import logging
from src.exception import customException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
# We will execute the data ingestion module, through this pipeline.
if __name__ == '__main__':
    # create an object of dataingestion module.
    obj = DataIngestion()
    # execute initiate_data_ingestion() method, that will return us train and test split data and will create useful artifacts directory.
    train_data_path , test_data_path = obj.initiate_data_ingestion()
    # check whether we have received the train_set or not.
    # for this execute python src/pipelines/training_pipeline.py, if no error then you will see the desired train_data_path and test_data_path
    print("train " , train_data_path)
    print("test " , test_data_path)

    # initialize data transformation
    data_transformation = DataTransformation()
    train_arr , test_arr, _preprobj = data_transformation.initiate_data_transformation(train_data_path ,test_data_path)
    # Find the best model
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)

