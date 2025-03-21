import os
import sys
#from data_ingestion.py import
#from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from pmdarima import auto_arima
from sklearn.model_selection import GridSearchCV, cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from dataclasses import dataclass

sys.path.insert(0, 'C:\\Users\\Vector\\Documents\\GitHub\\Demand-Forecasting-Web-app')
from src.exception import CustomException
from src.logger import logging
#from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class TrainPipeline:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(X_train, y_train, X_test, y_test, models, param):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = param[list(models.keys())[i]]

                if models[i] != 'ARIMA':
                    # Perform cross val grid search on data, find best params, and set them to model
                    time_series_split = TimeSeriesSplit(test_size=90)
                    gs = GridSearchCV(model, param_grid, cv=time_series_split, scoring='neg_root_mean_squared_error')
                    gs.fit(X, y)
                    model.set_params(**gs.best_params_)

                    print("Best CV score: ", np.abs(gs.best_score_))

                    # Train model
                    model.fit(X_train, y_trai)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                else:
                    # train arima on full data
                    # get evaluation score from cross val

                train_model_score = score(y_train, y_train_pred)
                test_model_score = score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)


    def train(self, all_data, forecast_horizon, preprocessor_path=None):
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
            
            logging.info('Entered model trainer component')
            train = all_data[:-forecast_horizon]
            test = all_data[-forecast_horizon:]

            logging.info('Starting to train model')
            sarima = auto_arima(train, seasonal=True, m=7)
            xgboost
            #sarima_params = sarima.
            predictions = sarima.predict(n_periods=len(test))
            test_score = root_mean_squared_error(test, predictions)

            # Save test graph
            logging.info('Model training completed')
            return test_score #, sarima_params

        except Exception as e:
            raise CustomException(e, sys)
