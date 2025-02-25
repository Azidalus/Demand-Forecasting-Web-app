import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Paths to raw, train and test data files
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion component')
        try:
            df = pd.read_csv('data\file.csv')
            logging.info('Read the dataset as dataframe')

            # Create train data folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Train test split initiated')
        except:
            pass

def get_data(csv_file, forecast_horizon):


    return X_train, y_train, X_test, y_test