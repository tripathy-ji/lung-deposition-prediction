from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder='frontend', static_url_path='')

model = None

# Path to your model file inside models/
model_path = os.path.join('models', 'deposition_model.joblib')

# Load model at startup
with app.app_context():
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at {model_path}")

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
    try:
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
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment if available
    app.run(debug=True, host='0.0.0.0', port=port)
