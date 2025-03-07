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

    def create_features(self, df):
        try:
            ['week_of_year'] = df['date'].dt.week
            ['month'] = df['date'].dt.month
            ['day_of_year'] = df['date'].dt.dayofyear
            ['quarter'] = df['date'].dt.quarter
            ['day_of_month'] = df['date'].dt.quarter
            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')

            #logging.info('Obtaining preprocessing object')
            #preprocessing_obj = self.get_data_transformer_object()
            
            # Create time features
            train_df = self.create_features(train_df)
            test_df = self.create_features(test_df)
            logging.info('Features successfully created')

            target_column_name = 'sales'
            numerical_columns = ['week_of_year','month','day_of_year','quarter','day_of_month']

            # Split df by train/test and feature type
            input_features_train_df = train_df('date')
            target_feature_train_df = train_df(target_column_name)
            input_features_test_df = test_df('date')
            target_feature_test_df = test_df(target_column_name)

            logging.info('Applying preprocessing object on train and test df')
            #logging.info('Applying preprocessing object on train and test df')
            input_features_train_arr
            #input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            #input_features_test_arr = preprocessing_obj.transform(input_features_test_df)
            # num_pipeline = Pipeline(
            #     steps = [
            #         ('imputer', SimpleImputer(strategy='mean'))
            #         ('scaler', StandardScaler())
            #         # outlier removal
            #     ]    
            # )
            # logging.info('Numerical columns encoding completed')

            # preprocessor = ColumnTransformer(
            #     [
            #         ('num_pipeline', num_pipeline, numerical_columns)
            #     ]
            # )

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