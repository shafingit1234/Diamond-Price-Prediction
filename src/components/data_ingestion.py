import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import customException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## Initialize the data ingestion configuration

@dataclass
class DataIngestionconfig:
    # here without using self keyword, I am defining and initializing the class variable all thanks to dataclass.
    # once file is read, store it in artifacts so that you don't need to read it all over again
    raw_data_path = os.path.join('artifacts' , 'raw.csv')
    # After reading I need to find out train and test data which will be the output of data ingestion module, and will be sent to the data transformation module.
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts' , 'test.csv')
    # We are creating reference for train, test and read data in artifacts folder, so that we can use them in one click without re executing function deriving them again and again.

# Create a data ingestion class.
# Here we will need to use self keyword for binding since we are not using @dataclass
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            # pass
            # read the csv file and store it as a dataframe
            df = pd.read_csv(os.path.join('notebooks/data' , 'gemstone.csv'))
            logging.info('Dataset read as pandas DataFrame')
            # create a directory in artifacts named as raw_data_path
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path) , exist_ok = True)
            # store csv material in that raw_data_path
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info("Train Test Split")
            # find the train and test split dataset.
            train_set, test_set = train_test_split(df, test_size = 0.30 , random_state = 42)
            # create a directory named train_data_path in artifacts storing train-set
            train_set.to_csv(self.ingestion_config.train_data_path , index=False, header= True)
            # Create directory named test_data_path in artifacts storing test-set
            test_set.to_csv(self.ingestion_config.test_data_path , index=False, header= True)

            logging.info('Ingestion of data is completed!!')

            return (
                # return the output of data ingestion module, which is train and test split, will be used in data transformation module.
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Error Occured in Data Ingestion Config')
