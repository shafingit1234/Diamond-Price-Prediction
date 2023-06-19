from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines to create pipelines
from sklearn.pipeline import Pipeline
# Column transformer will be used to combine two different pipelines.
from sklearn.compose import ColumnTransformer
# Using these imported libraries we are going to automate whole EDA process.
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.exception import customException
from src.logger import logging
# import save_object function from the utils.py file
from src.utils import save_object

# Data Transformation Config
# Here we are going to use dataclass, and we will try to form path for pickle files.
# Just like Data Ingestion, Data Transformation will have to go from two steps, Data Transformation Config + Data Transformation
# Data transformation config involves defining class variables, without the use of self keyword.
# Previously, in the data ingestion phase, we read the data then formed training data and testing data.
# Now we are going to use that output (train and test data) to create pickle files so that we can directly pass the train and test data to this file and retrieve preprocessed or transformed data
# 
@dataclass
class DataTransformationconfig:
    # in artifacts I want to create a preprocessor.pkl file
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')



# Data Transformation class
class DataTransformation:
    def __init__(self):
        # data_transformation_config of this class gets access to configured variables.
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')
            # Below lines of codes are similar to what we did in model training.ipynb file
            # Define which columns should be ordinal-encoded and which should be scaled.
            
            categorical_cols = ['cut' , 'color' , 'clarity']
            numerical_cols = ['carat' , 'depth' , 'table' , 'x' , 'y' , 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            # Time to use pipeline for automated eda
            logging.info("Pipeline Initiated")

            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    # Stragtegy used here is median imputation to handle missing values.
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                # then we are going to perform standardization over the numerical feature.
                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    # Here strategy used is mode imputation
                ('imputer',SimpleImputer(strategy='most_frequent')),
                # Ordinal Encoding will convert the categorical feature to numerical feature based on their ranks.
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                # After that standardization will be performed.
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
                # After handling categorical and numerical value, we are going to combine both pipelines.
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            # Return the preprocessor object which is going to handle categorical and numerical data.
            logging.info('Pipeline Completed')
            return preprocessor


        except Exception as e:
            logging.info('Error in Data Transformation')
            raise customException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info(f'Train DataFrame Head: \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: \n{test_df.head().to_string()}')
            logging.info('Obtaining preprocessing object')
            # After reading the files, create the preprocessor file
            preprocessing_obj = self.get_data_transformation_object()
            # find target column name and drop both target coloumn and useless coloumn.
            target_column_name = 'price'
            drop_columns = [target_column_name , 'id']

            # features splitting into independent and dependent features.

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis = 1)
            target_feature_test_df = test_df[target_column_name]
            
            
            # # apply the transformation
            

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets")

            # train_arr = np.c_[input_feature_train_arr , np.array(target_feature_train_df)]
            # test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            

            # print("train_arr goes below")
            # print(train_arr)
            # print("test_arr goes below")
            # print(test_arr)

            # In utils.py we are going to write code for pickle file.
            # Utils.py will help in creating the pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info('Preprocessor pickle is created and saved!!')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Exception occured in the initiate_datatransformation')
            raise customException(e, sys)
