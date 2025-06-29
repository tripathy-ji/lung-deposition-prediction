from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load the trained model
model_path = 'models/deposition_model.joblib'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at '{model_path}'.")
model = joblib.load(model_path)
print("Model loaded successfully.")

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print(f"Received data: {data}")

        # Extract input values
        br = float(data['breathingRate'])
        pdia = float(data['particleDiameter'])
        lr = data['lungRegion']

        # Create a DataFrame with correct column names
        input_df = pd.DataFrame([{
            'Breathing_Rate': br,
            'Particle_Diameter': pdia,
            'Lung_Region': lr
        }])

        print(f"Input DataFrame:\n{input_df}")

        # Make prediction
        prediction = model.predict(input_df)

        return jsonify({'prediction': float(prediction[0])})

    except KeyError as e:
        return jsonify({'error': f"Missing key: {e.args[0]}"}), 400
    except ValueError:
        return jsonify({'error': 'Invalid input: numbers required'}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
