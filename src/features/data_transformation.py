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
    preprocessor_obj_file_path = os.path.join('data', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def create_features(self, df):
        try:
            df['time'] = np.arange(len(df))
            df['day_of_month'] = df['Date'].dt.day
            df['month'] = df['Date'].dt.month
            df['day_of_year'] = df['Date'].dt.dayofyear
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['quarter'] = df['Date'].dt.quarter
            return df

        except Exception as e:
            raise CustomException(e, sys)
        
    def preprocess(df, scale):
        # convert date to dt
        num_pipeline = Pipeline(
            steps = [
                ('imputer', SimpleImputer(strategy='mean'))
                ('scaler', StandardScaler())
                # outlier removal
            ]    
        )
        # Scale data if needed
        return 0 #preprocessing_obj
        
    def initiate_data_transformation(self, df, scale=False, create_time_ftrs=False):
        try:
            '''
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

            logging.info('Preprocessing train/test dfs')
            #logging.info('Applying preprocessing object on train and test df')
            input_features_train_arr = SimpleImputer(strategy='mean')
            #input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            #input_features_test_arr = preprocessing_obj.transform(input_features_test_df)
            # 
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
            '''
            logging.info('Entered data transformation component')
            # Do later
            logging.info('Preprocessing data')
            #df = preprocess()
            # Convert to date format
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True)
            # Replace zero values with last non-zero value
            df['Units'] = df['Units'].replace(0, pd.NA).ffill()

            # Create time features if needed
            if create_time_ftrs:
                logging.info('Creating features')
                df = self.create_features(df)

            #how to scale???
            '''
            logging.info('Splitting data')
            X = df.drop('Sales', axis='columns').set_index('Date')
            y = df['Units']
            '''
            logging.info('Data successfully transformed')

            return(df)
        
        except Exception as e:
            raise CustomException(e, sys)
        
'''
data_df = pd.read_csv
data_transformation = DataTransformation()
all_data, y = data_transformation.initiate_data_transformation(data_df)
'''