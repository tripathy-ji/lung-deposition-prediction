
# Lung Deposition Efficiency Prediction using Machine Learning

## 📦 Project Overview

This project focuses on building a machine learning-based predictive model to estimate lung particle deposition efficiency based on key physiological and experimental parameters. The goal is to support research on inhaled drug delivery and aerosol behavior in human lungs.

A live version of this project is deployed at:

🔗 **[Test the Web Application Here](https://lung-deposition-prediction.onrender.com/)**

## 📝 How the Dataset was Created

- Multiple peer-reviewed research papers on lung deposition were carefully studied.
- Relevant figures and tables showing relationships between particle size, breathing rate, lung region, and deposition efficiency were identified.
- Data points were extracted using tools like WebPlotDigitizer to ensure accurate digitization.
- A master dataset (`final.csv`) was compiled, containing clean, structured data with standardized lung region terminology.

### Dataset Columns:

| Column Name                | Description                                          |
|----------------------------|------------------------------------------------------|
| Paper Title                | Source research paper                                |
| Figure Number              | Specific figure from the paper                       |
| Sub-Figure Number          | Sub-part of the figure (if applicable)               |
| Particle Diameter (µm)     | Size of inhaled particles (micrometers)              |
| Breathing Rate (L/min)     | Airflow rate during inhalation (liters/min)          |
| Lung Region                | Standardized lung region label                       |
| Deposition Efficiency (%)  | Measured particle deposition efficiency              |

⚠️ Some columns like Dosage and Stokes Number exist but are incomplete and **NOT used for modeling**.

## 🤖 Model Development

The following steps were performed:

- Data cleaning and preprocessing
- Label encoding for lung region categories
- Polynomial feature engineering
- Model training using various approaches:
  - Neural Networks
  - XGBoost Regressor
  - Stacking Ensemble (final selected model)
- Hyperparameter tuning using Optuna

### Final Model Performance:

- **Test R² Score**: ~81%
- **Reliable prediction capability** on unseen data

The final trained model is saved as:

```
models/deposition_model.joblib
```

## 🛠 How to Use Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training script (optional if retraining needed):

```bash
python train_model.py
```

3. Run the test script to check model prediction in terminal:

```bash
python test_model.py
```

4. Run the Flask web application:

```bash
gunicorn app:app
```

Or for local development/testing:

```bash
python app.py
```

5. Open the browser and go to:

```
http://127.0.0.1:5000
```

6. Enter input parameters in the web form:
   - **Breathing Rate (L/min)** — accepts integers like 15, 30, etc.
   - **Particle Diameter (µm)** — accepts decimals like 2.5
   - **Lung Region** — must match available options (e.g., Trachea, Upper Airways, Whole Lung)

7. The web app returns the predicted deposition efficiency (%) based on the trained model.

## 🌐 Hosted Version (Recommended)

The project is live on Render and can be accessed here:

🔗 **[https://lung-deposition-prediction.onrender.com/](https://lung-deposition-prediction.onrender.com/)**

You can directly test the model predictions on the hosted website.

## 🚀 Potential Future Work

- Improve model accuracy towards 90% with further hyperparameter tuning.
- Extend dataset with more real experimental data.
- Enhance frontend design for better user experience.
- Explore other ML approaches like CatBoost, LightGBM, etc.

## ⚡ Notes

- The model is trained on limited real data extracted from literature; prediction accuracy depends on dataset quality.
- Always interpret model results as estimates to support, not replace, experimental validation.

## 👨‍🔬 Research Significance

This tool aims to assist researchers, students, and developers working on:

- Pulmonary drug delivery systems
- Aerosol science
- Inhalation toxicology studies
- AI/ML applications in biomedical engineering
