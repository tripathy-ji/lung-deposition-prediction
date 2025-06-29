import joblib
import pandas as pd
import os

# Load model
model_path = os.path.join("models", "deposition_model.joblib")

if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    exit()

model = joblib.load(model_path)
print("Model loaded successfully.")

# Provide input with EXACT expected column names
input_data = pd.DataFrame([{
    "Breathing_Rate": 15.0,
    "Particle_Diameter": 2.5,
    "Lung_Region": "Upper Airways"  # Change this based on valid categories from your training
}])

# Make prediction
prediction = model.predict(input_data)
print(f"Prediction: {prediction[0]}")
