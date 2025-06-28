import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
import optuna
import joblib
import re

# Load the dataset
file_path = r"E:\internship project by sir\FULL AND FINAL DATASET\final.csv"
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()
except pd.errors.ParserError:
    print("Error: Failed to parse CSV file")
    exit()

# Select required columns
required_columns = ['Breathing Rate (L/min)', 'Particle Diameter (µm)', 'Our Lung Region', 'Deposition Efficiency (%)']
if not all(col in data.columns for col in required_columns):
    print("Error: Required columns missing in the dataset")
    exit()
data = data[required_columns]

# Rename columns for simplicity
data.columns = ['Breathing_Rate', 'Particle_Diameter', 'Lung_Region', 'Deposition_Efficiency']

# Clean Particle_Diameter: remove 'micron' and convert to float
try:
    data['Particle_Diameter'] = data['Particle_Diameter'].apply(lambda x: re.sub(r' micron$', '', str(x)))
    data['Particle_Diameter'] = pd.to_numeric(data['Particle_Diameter'], errors='coerce')
except Exception as e:
    print(f"Error cleaning Particle_Diameter: {e}")
    exit()

# Clean Breathing_Rate: ensure it's numeric
try:
    data['Breathing_Rate'] = pd.to_numeric(data['Breathing_Rate'], errors='coerce')
except Exception as e:
    print(f"Error cleaning Breathing_Rate: {e}")
    exit()

# Drop rows where target is missing
data = data.dropna(subset=['Deposition_Efficiency'])

# Define features
numeric_features = ['Breathing_Rate', 'Particle_Diameter']
categorical_features = ['Lung_Region']

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Prepare features and target
X = data.drop('Deposition_Efficiency', axis=1)
y = data['Deposition_Efficiency']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimize CatBoost hyperparameters
def objective(trial):
    try:
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
            'task_type': 'GPU',
            'devices': '0',
            'verbose': 0
        }
        model = CatBoostRegressor(**params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        kf = KFold(n_splits=5)
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            scores.append(r2_score(y_val, y_pred))
        return np.mean(scores)
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        return np.nan

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params

# Define and train the stacking model
catboost_model = CatBoostRegressor(**best_params, task_type='GPU', devices='0')
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

stacking_model = StackingRegressor(
    estimators=[('catboost', catboost_model), ('xgb', xgb_model)],
    final_estimator=CatBoostRegressor(iterations=100, verbose=0, task_type='GPU', devices='0')
)

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', stacking_model)
])

try:
    full_pipeline.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# Evaluate the model
try:
    y_pred = full_pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    print(f"Test R² Score: {test_r2:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit()

# Save the model
try:
    joblib.dump(full_pipeline, 'deposition_model.joblib')
    print("Model saved as 'deposition_model.joblib'")
except Exception as e:
    print(f"Error saving model: {e}")