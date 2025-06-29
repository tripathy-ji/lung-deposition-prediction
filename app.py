from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Load your trained model
model_path = 'models/deposition_model.joblib'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at '{model_path}'. Please place it inside the 'models' folder.")
model = joblib.load(model_path)
print("Model loaded successfully.")

# Serve your frontend (index.html)
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Expect the same keys your frontend sends:
        br = float(data['breathingRate'])
        pdia = float(data['particleDiameter'])
        lr = data['lungRegion']

        # Build features array for your model.
        # If your model expects three columns, include lr (after encoding).
        # Example with just numeric features:
        X = np.array([[br, pdia]])

        prediction = model.predict(X)
        return jsonify({'prediction': float(prediction[0])})
    except KeyError as e:
        return jsonify({'error': f"Missing JSON key: {e.args[0]}"}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
