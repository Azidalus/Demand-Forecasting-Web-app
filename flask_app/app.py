from flask import Flask, request, render_template, session
from fileinput import filename
import numpy as np
import pandas as pd
import os
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

app = Flask(__name__)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        
        #data = CustomData(
            #date = request.form.get('date'),
            #sales = request.form.get('sales')
        #)

        # Get data from forms
        forecast_horizon = request.form.get('forecast_horizon')
        data_in_csv = request.files.get('file')
        # data_in_csv = request.files['file']
        # Extract uploaded file name
        data_filename = secure_filename(data_in_csv.filename)
        data_in_csv.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],
                                                          data_filename)
 
        #return render_template('file uploaded.html')

        # Convert data to pandas df
        #data_df = data.get_data_as_dataframe()
        #print(data_df)
        # Receive data and split it into train and test CSV files
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion(data_in_csv=data_in_csv)
        
        # Preprocess data
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

        # Train models, choose the best model and save it as .pkl 
        train_pipeline = TrainPipeline()
        test_score = train_pipeline.train(train_arr, test_arr, forecast_horizon)

        # Make prediction with the best model
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(test_arr, forecast_horizon)
        
        return render_template('home.html', accuracy=test_score*100, results=results[0])