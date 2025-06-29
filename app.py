from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd

app = Flask(__name__, static_folder='frontend', static_url_path='')
model = joblib.load('deposition_model.joblib')

# Debug print
print("MODEL EXPECTS:", getattr(model, 'feature_names_in_', None))

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    br = float(data['breathingRate'])
    pdia = float(data['particleDiameter'])
    lr = data['lungRegion']
    
    # Correct column names to match trained model
    features = pd.DataFrame([{
        "Breathing_Rate": br,
        "Particle_Diameter": pdia,
        "Lung_Region": lr
    }])
    
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
