import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrainer

# Config with paths to raw, train and test data files
@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_in_csv):
        logging.info('Entered the data ingestion component')
        try:
            #df = pd.read_csv('data\file.csv')
            df = pd.read_csv(data_in_csv)
            logging.info('Read the dataset as dataframe')

            # Create train data folder and save 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            #df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            # Save CSV to that dir
            logging.info('Train test split initiated')

            # Split data into train/test and save as CSV files to the dir
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=1, shuffle=False)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__name__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))