import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataIngestionConfig:
    pass



def get_data(csv_file, forecast_horizon):


    return X_train, y_train, X_test, y_test