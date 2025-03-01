import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

class CustomData:
    def __init__(self,
        feature1: date): 

        self.feature1 = feature1

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'feature1': [self.feature1],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)