

from flask import render_template, url_for, redirect, jsonify, request, Flask
from pycaret.classification import *
import numpy as np
import pandas as pd

app = Flask(__name__)

model = load_model('deployment_1')
cols=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', \
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'] 

@app.route('/') #main page
def home():    
    return render_template('home.html',pred='quality')

@app.route('/predict', methods=['POST'])
def predict():
    int_features= [x for x in request.form.values()]
    final= np.array(int_features)
    data_unseen= pd.DataFrame([final], columns=cols)
    prediction=predict_model(model, data=data_unseen)
    prediction=int(prediction.Label[0])
    return render_template('home.html',pred='{}'.format(prediction))

app.run()
