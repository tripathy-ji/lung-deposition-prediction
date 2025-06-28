import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import xgboost as xgb
import catboost as cb
import optuna
import matplotlib.pyplot as plt

# Load data
file_path = r"E:\internship project by sir\FULL AND FINAL DATASET\final.csv"
data = pd.read_csv(file_path)

# Clean and preprocess numeric columns
def clean_numeric_column(column):
    data[column] = data[column].astype(str).str.replace(r'[^\d.-]', '', regex=True)
    return pd.to_numeric(data[column], errors='coerce')

numeric_cols = ['Breathing Rate (L/min)', 'Particle Diameter (µm)', 'Dosage', 'Stokes Number']
for col in numeric_cols:
    data[col] = clean_numeric_column(col)

# Cap Deposition Efficiency at 100%
data['Deposition Efficiency (%)'] = data['Deposition Efficiency (%)'].clip(upper=100)

# Feature engineering: Add log transforms and interactions
data['log_Particle_Diameter'] = np.log1p(data['Particle Diameter (µm)'])
data['log_Dosage'] = np.log1p(data['Dosage'])
data['Breath_Diam_Interaction'] = data['Breathing Rate (L/min)'] * data['Particle Diameter (µm)']

# Preprocess data
imputer = SimpleImputer(strategy='median')
X_numeric = imputer.fit_transform(data[['Breathing Rate (L/min)', 'Particle Diameter (µm)', 'Dosage', 'Stokes Number', 'log_Particle_Diameter', 'log_Dosage', 'Breath_Diam_Interaction']])

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_numeric_poly = poly.fit_transform(X_numeric)

# Encode categorical variable
X_lung_region = data['Lung Region'].values.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_lung_region_encoded = encoder.fit_transform(X_lung_region)

# Combine features
X = np.hstack((X_numeric_poly, X_lung_region_encoded))
y = data['Deposition Efficiency (%)']

# Remove rows with NaN
mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
X, y = X[mask], y[mask]

# Scale features
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

# Split data (70% train, 15% validation, 15% test)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / (1 - 0.15), random_state=42)

# Objective function for Optuna
def objective(trial):
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1.0, 10.0, log=True),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1.0, 10.0, log=True),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0)
    }
    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 100, 500),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('cat_depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1.0, 10.0, log=True)
    }
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
        'max_depth': trial.suggest_int('rf_max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10)
    }

    xgb_model = xgb.XGBRegressor(**xgb_params)
    cat_model = cb.CatBoostRegressor(**cat_params, verbose=0)
    rf_model = RandomForestRegressor(**rf_params, random_state=42)

    ensemble = StackingRegressor(
        estimators=[('xgb', xgb_model), ('cat', cat_model), ('rf', rf_model)],
        final_estimator=LinearRegression()
    )
    ensemble.fit(X_train, y_train)
    y_val_pred = ensemble.predict(X_val)
    return r2_score(y_val, y_val_pred)

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params

# Train models with best parameters
xgb_params = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
cat_params = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}
rf_params = {k.replace('rf_', ''): v for k, v in best_params.items() if k.startswith('rf_')}

xgb_model = xgb.XGBRegressor(**xgb_params)
cat_model = cb.CatBoostRegressor(**cat_params, verbose=0)
rf_model = RandomForestRegressor(**rf_params, random_state=42)

stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('cat', cat_model), ('rf', rf_model)],
    final_estimator=LinearRegression()
)
stacking_model.fit(X_train, y_train)

# Evaluate model
y_pred = stacking_model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
print(f"Stacking Test R²: {test_r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='r2')
print(f"Stacking Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

# Prediction function
def predict_deposition(model, scaler_X, encoder, poly, breathing_rate, particle_diameter, lung_region, dosage, stokes_number):
    input_data = np.array([[breathing_rate, particle_diameter, dosage, stokes_number, 
                          np.log1p(particle_diameter), np.log1p(dosage), 
                          breathing_rate * particle_diameter]])
    input_data = SimpleImputer(strategy='median').fit_transform(input_data)
    input_numeric_poly = poly.transform(input_data)
    input_lung_region = encoder.transform([[lung_region]])
    input_data = np.hstack((input_numeric_poly, input_lung_region))
    input_data = scaler_X.transform(input_data)
    return model.predict(input_data)[0]

# Example prediction
print("\nExample prediction:")
dep_eff = predict_deposition(stacking_model, scaler_X, encoder, poly, 30, 2.1, 'Upper Airways', 10, 0.1)
print(f"Predicted Deposition Efficiency: {dep_eff:.2f}%")