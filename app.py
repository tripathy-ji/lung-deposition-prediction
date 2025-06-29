from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder='frontend', static_url_path='')

model = None

@app.before_first_request
def load_model():
    global model
    if os.path.exists('deposition_model.joblib'):
        model = joblib.load('deposition_model.joblib')
        print("Model loaded successfully.")
    else:
        print("Model file missing!")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    br = float(data['breathingRate'])
    pdia = float(data['particleDiameter'])
    lr = data['lungRegion']

    features = pd.DataFrame([{
        "Breathing_Rate": br,
        "Particle_Diameter": pdia,
        "Lung_Region": lr
    }])

    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
