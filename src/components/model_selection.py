import os
import sys
import numpy as np
import operator

#from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from pmdarima import auto_arima, ARIMA
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

class SelectionPipeline:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, all_data, models, param_grid, forecast_horizon):
        try:
            report = {}

            # Iterate over all models
            for model_name, model in models.items():
                #para = param_grid[model_name]
                ts_split = TimeSeriesSplit(n_splits=8, test_size=forecast_horizon)

                if model_name == 'naive':
                    logging.info('Training naive')
                    X = all_data['Units']
                    naive_errors = []
                    # Do cross-val on full data and get evaluation score
                    for train_idx, test_idx in ts_split.split(X):
                        train, test = X.iloc[train_idx], X.iloc[test_idx]
                        naive_preds = np.full_like(test, fill_value=train.iloc[-1])
                        naive_MSE = root_mean_squared_error(test, naive_preds)
                        naive_errors.append(naive_MSE)
                    model_score = np.mean(naive_errors)
                    model = None

                elif model_name == 'ARIMA':
                    # Train auto_arima on full data to get best params
                    logging.info('Training ARIMA')
                    X = all_data[['Date','Units']].set_index('Date')
                    best_arima = auto_arima(X, seasonal=True, suppress_warnings=True)
                    arima_params = best_arima.order
                    arima_errors = []
                    # Do cross-val on full data with these params and get evaluation score
                    for train_idx, test_idx in ts_split.split(X):
                        train, test = X.iloc[train_idx], X.iloc[test_idx]
                        arima_model = ARIMA(order=arima_params).fit(train)
                        arima_preds = arima_model.predict(n_periods=len(test))
                        arima_MSE = root_mean_squared_error(test, arima_preds)
                        arima_errors.append(arima_MSE)
                    model_score = np.mean(arima_errors)
                    model = best_arima

                else:
                    # Perform cross-val grid search on data, find best params, and set them to model
                    logging.info('Training XGBoost')
                    X = all_data.reset_index().drop(['Units','Date'], axis=1)
                    y = all_data['Units']
                    gs = GridSearchCV(model, param_grid, cv=ts_split, scoring='neg_root_mean_squared_error')
                    gs.fit(X, y)
                    model.set_params(**gs.best_params_)
                    model.fit(X, y)
                    model_score = np.abs(gs.best_score_)
                    
                # Add model and model score to final report
                report[model_name] = (model, model_score)

            # Get best model and best score
            logging.info('Determining best model')
            best_model_name = max(report.items(), key=lambda v: v[1][1])[0]
            best_model = max(report.items(), key=lambda v: v[1][1])[1][0]
            best_model_score = max(report.items(), key=lambda v: v[1][1])[1][1]

            return report, best_model_name, best_model, best_model_score 

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
            #train = all_data[:-forecast_horizon]
            #test = all_data[-forecast_horizon:]

            logging.info('Starting to train models')
            #sarima = auto_arima(train, seasonal=True, m=7)
            #predictions = sarima.predict(n_periods=len(test))
            #test_score = root_mean_squared_error(test, predictions)

            models = {
                      'naive': None,
                      'ARIMA': None,
                      'XGBoost': XGBRegressor(random_state=1)
                     }
            
            param_grid = {
                          'learning_rate': [0.01, 0.1],
                          'n_estimators': [100, 300, 500],
                          'sub_sample': [0, 0.4, 0.8]
                         }

            # Get report, best model score and best model from the report
            model_report, best_model_name, best_model, best_model_score = self.evaluate_models(all_data, models, param_grid, forecast_horizon)
            
            if best_model_score < 0.6:
                raise CustomException("Best model's score is < 0.6")

            # Save test graph
            logging.info(f'Model selection completed (Best model {best_model_name}, {best_model_score})')
            return best_model_name, best_model, best_model_score 

        except Exception as e:
            raise CustomException(e, sys)
