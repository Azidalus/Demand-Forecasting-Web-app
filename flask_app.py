from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData

app = Flask(__name__)

# Route for home page
@app.route('/')
def index():