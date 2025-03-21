import os
import sys
import numpy as np
import pandas as pd
#import dill
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        #with open(file_path, 'wb') as file_obj:
        #    dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

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
                # Perform randomized grid search on train data and find best params
                gs = RandomizedSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

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
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)