from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

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
        # Get data from forms
        data = CustomData(
            date = request.form.get('date'),
            sales = request.form.get('sales')
        )

        # Convert it to pandas df
        data_df = data.get_data_as_dataframe()
        print(data_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data_df)
        
        return render_template('home.html', results=results[0])