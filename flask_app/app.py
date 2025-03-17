import streamlit as st
#import Flask, request, render_template, session
#from fileinput import filename
import numpy as np
import pandas as pd
import os
#from src.pipeline.train_pipeline import TrainPipeline
from src.components.model_train import TrainPipeline
from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation

def set_state():
    st.session_state['predict_btn'] = 1

# Flag to indicate that 'Predict' button was pressed and prediction initiated
st.session_state['predict_btn'] = 0

# Display starter elements
st.title('Demand forecasting')
st.markdown('Upload your sales data as a CSV file. \nThe file *must* contain 2 columns named `date` and `sales`')
uploaded_CSV_obj = st.file_uploader('Choose file', type='csv')

# When CSV is uploaded, do:
if uploaded_CSV_obj: 
    # Display next part of app
    st.selectbox('Forecast for: ', '1 week')
    st.button('Predict', on_click=set_state())
else:
    pass

if st.session_state['predict_btn'] == 1:
    progress_bar = st.progress(0)
    #for percent_complete in range(100):
    #   time.sleep(0.01)
    #   progress_bar.progress(percent_complete + 1, text=progress_text)

    # Receive data, save it, and convert to df
    data_ingestion = DataIngestion()
    data_df = data_ingestion.initiate_data_ingestion(CSV_obj=uploaded_CSV_obj)
    #train_path, test_path = data_ingestion.initiate_data_ingestion(data_in_csv=uploaded_CSV)
        
    # Preprocess data
    data_transformation = DataTransformation()
    all_data, y = data_transformation.initiate_data_transformation(data_df)

    # Train models, choose the best model and save it as .pkl 
    # Train simple ARIMA model
    train_pipeline = TrainPipeline()
    test_score, test_graph, best_model_params = train_pipeline.train(all_data, forecast_horizon=30)

    # Make prediction with the best model
    predict_pipeline = PredictPipeline()
    predictions = predict_pipeline.predict(all_data, best_model_params, forecast_horizon=30)

    # Output test predictions graph with error
    #test_graph

    # Output predictions graph
    chart_data = pd.DataFrame(columns=["Date", "All_data", "Preds"])
    chart_data['All_data'] = all_data['Units']
    chart_data['Preds'] = None
    chart_data['Preds'][len(all_data): ] = predictions
    chart_data['Date'] = all_data['Units'][0] date extend

    st.line_chart(
        chart_data,
        x="Date",
        y=["All_data", "Preds"],
        color=["#FF0000", "#0000FF"],  
    )


#app = Flask(__name__)

# Route for home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predictdata', methods=['GET','POST'])
# def predict_datapoint():
    # if request.method == 'GET':
    #     return render_template('home.html')
    # else:
        
    #     #data = CustomData(
    #         #date = request.form.get('date'),
    #         #sales = request.form.get('sales')
    #     #)

    #     # Get data from forms
    #     forecast_horizon = request.form.get('forecast_horizon')
    #     data_in_csv = request.files.get('file')
    #     # data_in_csv = request.files['file']
    #     # Extract uploaded file name
    #     data_filename = secure_filename(data_in_csv.filename)
    #     data_in_csv.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
    #     session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],
    #                                                       data_filename)
 
    #     #return render_template('file uploaded.html')

    #     # Convert data to pandas df
    #     #data_df = data.get_data_as_dataframe()
    #     #print(data_df)
    #     # Receive data and split it into train and test CSV files
    #     data_ingestion = DataIngestion()
    #     train_path, test_path = data_ingestion.initiate_data_ingestion(data_in_csv=data_in_csv)
        
    #     # Preprocess data
    #     data_transformation = DataTransformation()
    #     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    #     # Train models, choose the best model and save it as .pkl 
    #     train_pipeline = TrainPipeline()
    #     test_score = train_pipeline.train(train_arr, test_arr, forecast_horizon)

    #     # Make prediction with the best model
    #     predict_pipeline = PredictPipeline()
    #     results = predict_pipeline.predict(test_arr, forecast_horizon)
        
    #     return render_template('home.html', accuracy=test_score*100, results=results[0])