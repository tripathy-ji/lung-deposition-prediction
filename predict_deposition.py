import joblib
import pandas as pd
import re

# Load the saved model
try:
    model = joblib.load('deposition_model.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'deposition_model.joblib' not found. Please run the training script first.")
    exit()

# Prediction function
def predict_deposition(breathing_rate, particle_diameter, lung_region):
    try:
        # Convert inputs to appropriate types
        breathing_rate = float(breathing_rate)
        particle_diameter = float(re.sub(r' micron$', '', str(particle_diameter)))
        
        # Validate inputs
        if breathing_rate <= 0 or particle_diameter <= 0:
            raise ValueError("Breathing Rate and Particle Diameter must be positive numbers.")
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Breathing_Rate': [breathing_rate],
            'Particle_Diameter': [particle_diameter],
            'Lung_Region': [lung_region]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        return prediction
    
    except ValueError as ve:
        print(f"Error: {ve}")
        return None
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Get user input
print("\nEnter the following details to predict Deposition Efficiency:")
breathing_rate = input("Breathing Rate (L/min): ")
particle_diameter = input("Particle Diameter (µm, e.g., 3 or 3 micron): ")
lung_region = input("Lung Region (e.g., Upper Airways, Trachea, Whole Lung): ")

# Make prediction
result = predict_deposition(breathing_rate, particle_diameter, lung_region)
if result is not None:
    print(f"\nPredicted Deposition Efficiency: {result:.2f}%")