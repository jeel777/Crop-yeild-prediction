import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import time
import joblib

print("Starting model training process (v3 with Imputation and y-NaN drop)...")
start_time = time.time()

try:
    df = pd.read_csv("../../dataset/data_long_format_v7_full.csv")
except FileNotFoundError:
    print("ERROR: 'data_long_format_v7_full.csv' not found.")
    print("Please run the 'convert_to_long_format_v7.py' script first.")
    exit()

print(f"Data loaded successfully. Shape: {df.shape}")

target = 'YIELD'    


missing_y_count = df[target].isnull().sum()
if missing_y_count > 0:
    print(f"Found {missing_y_count} rows with missing YIELD. Dropping them.")
    df = df.dropna(subset=[target])
    print(f"New data shape after dropping: {df.shape}")

y = df[target]
X = df.drop(columns=[
    'YIELD', 
    'AREA', 
    'PRODUCTION', 
    'Dist_Code', 
    'State_Code'
])
print(f"Target (y): {target}")

categorical_features = [
    'State_Name', 
    'Dist_Name', 
    'Crop'
]
numerical_features = [
    col for col in X.columns if col not in categorical_features
]

print(f"\nIdentified {len(categorical_features)} categorical features.")
print(f"Identified {len(numerical_features)} numerical features.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)
print(f"Data split: {len(X_train)} training rows, {len(X_test)} testing rows.")

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    oob_score=True
)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

print("\n--- Training Random Forest Model ---")
print(f"Training on {len(X_train)} rows (with imputation)...")
clf.fit(X_train, y_train)
print("Training complete.")
training_time = time.time() - start_time
print(f"Total time taken: {training_time:.2f} seconds")

print("\n--- Model Evaluation ---")
try:
    print(f"Model OOB Score: {clf.named_steps['model'].oob_score_:.4f}")
except AttributeError:
    print("OOB Score not available.")

print("Making predictions on the test set...")
y_pred = clf.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R-squared ($R^2$): {r2:.4f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f} (Kg per ha)")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (Kg per ha)")
print("\nProcess Finished.")

print("\n--- Top 20 Most Important Features ---")
try:
    rf_model = clf.named_steps['model']
    preprocessor = clf.named_steps['preprocessor']
    
    cat_features_out = preprocessor.named_transformers_['cat'] \
                                   .named_steps['encoder'] \
                                   .get_feature_names_out(categorical_features)
    num_features_out = numerical_features
    
    all_feature_names = np.concatenate([num_features_out, cat_features_out])
    
    importances = pd.Series(
        rf_model.feature_importances_, 
        index=all_feature_names
    ).sort_values(ascending=False)
    
    print(importances.head(20))
except Exception as e:
    print(f"Could not calculate feature importances: {e}")

print("\n--- Saving Model ---")
model_filename = 'crop_yield_model_v3.joblib'
joblib.dump(clf, model_filename)
print(f"Model saved successfully as: {model_filename}")