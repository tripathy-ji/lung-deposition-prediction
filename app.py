from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')

# Enable CORS (Allows frontend to call backend without browser blocking)
CORS(app)

# Load your trained model
model_path = 'model.pkl'
model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
else:
    print("Model file not found. Please check the path.")

# Serve your frontend (index.html)
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        particle_size = float(data.get('particle_size', 0))

        if model is None:
            return jsonify({'error': 'Model not loaded.'}), 500

        prediction = model.predict(np.array([[particle_size]]))
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# Run the app locally (Not required on Render, but useful for testing)
if __name__ == '__main__':
    app.run(debug=True)
