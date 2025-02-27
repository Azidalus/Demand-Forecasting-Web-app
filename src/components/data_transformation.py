import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median'))
                ]    
            )
            logging.info('Numerical columns encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'sales'
            numerical_columns = ['']
        except:
            pass