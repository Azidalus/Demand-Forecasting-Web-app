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
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['']
            ['week_of_year'] = data['date'].dt.week
            ['month'] = data['date'].dt.month
            ['day_of_year'] = data['date'].dt.dayofyear
            ['quarter'] = data['date'].dt.quarter
            ['day_of_month'] = data['date'].dt.quarter

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='mean'))
                    ('scaler', StandardScaler())
                    # outlier removal
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

            #logging.info('Obtaining preprocessing object')
            #preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = 'sales'
            #feature_columns = ['date']
            # Features
            input_features_train_df = train_df('date')
            target_feature_train_df = train_df(target_column_name)
            input_features_test_df = test_df('date')
            target_feature_test_df = test_df(target_column_name)

            logging.info('Applying preprocessing object on train and test df')
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            # Combine input and target features into train/test sets
            logging.info('Combining input and target features into train/test sets')
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]
            

            # save_object(
            #     file_path = self.data_transformation_config.preprocessor_obj_file_path,
            #     obj = preprocessing_obj
            # )
            # logging.info('Saved preprocessing object')

            return(
                train_arr,
                test_arr,
                #self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)