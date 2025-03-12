import os
import sys
#from data_ingestion.py import
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X, y, preprocessor_path):
        try:
            '''
            logging.info('Split train and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {'Linear regression': LinearRegression(),
                      'Ridge': Ridge(),
                      'Lasso': Lasso(),
                      'XGBoost': XGBRegressor()}

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                                models=models)
            
            # Get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Best model's score is < 0.6")
            logging.info('Found best model on both train and test datasets')

            # Save best model
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            y_pred = best_model.predict(X_test)
            test_score = score(y_test, y_pred)
            '''
            param_grid = param['']

            time_series_cv = TimeSeriesSplit(n_splits=6)
            gs = GridSearchCV(model, cv=time_series_cv, param_grid=param_grid)
            gs.fit(X, y)
            results.append(cv['test_score'].mean())

            return test_score, best_model

        except Exception as e:
            raise CustomException(e, sys)
