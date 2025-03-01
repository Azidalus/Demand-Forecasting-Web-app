import os
import sys
import numpy as np
import pandas as pd

def save_object(file_path, obj):
    try:
        pass
    except:
        pass

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            # Train model
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = score(y_train, y_train_pred)
            test_model_score = score(y_test, y_test_pred)
    except:
        pass