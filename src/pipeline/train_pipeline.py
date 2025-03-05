import sys
import pandas as pd
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import save_object

class TrainPipeline:
    def __init__(self):
        pass

    def train(self, data, forecast_horizon):
        try:
            #preprocessor_path = 'artifacts\preprocessor.pkl'
            #preprocessor = load_object(file_path=preprocessor_path)
            #data_preprocessed = preprocessor.transform(data)

            # Train models, choose the best model and save it as .pkl
            model_trainer = ModelTrainer()
            test_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            return test_score
        
        except Exception as e:
            raise CustomException(e, sys) 