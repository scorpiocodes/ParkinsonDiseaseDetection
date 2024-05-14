from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

import pdd

app = Flask(__name__)

# Load ML model
model = joblib.load("ParkinsonsPredictor.sav")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    formData = request.form.to_dict()
    features = [float(request.form[field]) for field in request.form]
    print(features)

    scaler = pdd.scaler
    scaled_features = scaler.transform(np.array(features).reshape(1,-1))
    print(scaled_features)

    # Make prediction using your model
    prediction = model.predict(scaled_features)[0]

    if prediction == 1:
        return render_template('result.html', prediction="Positive", formData = formData)
    if prediction == 0:
        return render_template('result.html', prediction="Negative", formData = formData)

if __name__ == '__main__':
    app.run(debug=True)