from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('models/churn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = [
        float(request.form['tenure']),
        float(request.form['monthlycharges']),
        float(request.form['totalcharges'])
    ]

    # Add remaining dummy features
    while len(features) < 28:
        features.append(0)

    prediction = model.predict([features])

    result = "Customer Will Churn" if prediction[0] == 1 else "Customer Will Stay"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)