import sys
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, model_name, model, all_data, forecast_horizon):
        try:
            '''
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_preprocessed = preprocessor.transform(data)
            predictions = model.predict(data_preprocessed)
            '''
            logging.info('Entered predictor component')
            #model = auto_arima(data, seasonal=True, m=7)
            if model_name == 'naive':
                predictions = np.full_like(range(forecast_horizon), fill_value=all_data['Units'].iloc[-1])
            else:
                predictions = model.predict(forecast_horizon)

            '''
            if model_name == 'XGBoost':
                X = all_data.reset_index().drop(['Units','Date'], axis=1)
                y = all_data['Units']
                model.fit(X, y)
                predictions = model.predict(forecast_horizon)
                
            elif model_name == 'ARIMA':
                predictions = model.predict(forecast_horizon)
            '''

            logging.info('Prediction made successfully')
            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)  


# class CustomData:
#     def __init__(self,
#                  date: datetime,
#                  sales: int): 

#         self.date = date
#         self.sales = sales

#     def get_data_as_dataframe(self):
#         try:
#             custom_data_input_dict = {
#                 'date': [self.date],
#                 'sales': [self.sales],
#             }

#             return pd.DataFrame(custom_data_input_dict)
        
#         except Exception as e:
#             raise CustomException(e, sys)