# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Load the XGBoost CLassifier model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        SkinThickness_NAN = 0
        Insulin_NAN = 0

        f = open("xgboost.pkl", "rb")
        classifier = pickle.load(f)
        inp = pd.DataFrame({"Pregnancies": [preg], "Glucose": [glucose],
                            "BloodPressure": [bp], 'SkinThickness': [st],
                            'Insulin': [insulin],
                            'BMI': [bmi], 'DiabetesPedigreeFunction': [dpf],
                            'Age': [age], 'SkinThickness_NAN': [SkinThickness_NAN],
                            'Insulin_NAN': [Insulin_NAN]})

        my_prediction = classifier.predict(inp)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)