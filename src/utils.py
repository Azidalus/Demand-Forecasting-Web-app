import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        pass
    except:
        pass

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Perform randomized grid search on train data 
            gs = RandomizedSearchCV(model, para, cv=cv, n_jobs=n_jobs)
            gs.fit(X_train, y_train)

            # Train model
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = score(y_train, y_train_pred)
            test_model_score = score(y_test, y_test_pred)
    except:
        pass