import sys
import pandas as pd
from src.exception import CustomException
from src.utils import save_object

class TrainPipeline:
    def __init__(self):
        pass

    def train(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            
            data_preprocessed = preprocessor.transform(features)
            predictions = model.predict(data_preprocessed)
            save_object(file_path=preprocessor_path, best_model)
            
            return predictions
        
        except Exception as e:
            raise CustomException(e, sys)  