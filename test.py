import joblib
import pandas as pd

# Load the trained model
model_path = 'models/deposition_model.joblib'
model = joblib.load(model_path)
print("Model loaded successfully.")

# Take user input from the terminal
br = float(input("Enter Breathing Rate (e.g., 15): "))
pdia = float(input("Enter Particle Diameter (e.g., 2.5): "))
lr = input("Enter Lung Region (e.g., Upper Airways, Whole Lung, etc.): ")

# Prepare the input in the correct format
input_data = pd.DataFrame([{
    'Breathing_Rate': br,
    'Particle_Diameter': pdia,
    'Lung_Region': lr
}])

# Make the prediction
prediction = model.predict(input_data)
print(f"Prediction: {prediction[0]}")
