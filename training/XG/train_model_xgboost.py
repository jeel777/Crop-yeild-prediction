import pandas as pd
from xgboost import XGBRegressor
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

# --- 1. LOAD THE TRAINING DATA ---
try:
    df = pd.read_csv("../../dataset/train_data.csv")
except FileNotFoundError:
    print("ERROR: 'train_data.csv' not found.")
    print("Please run 'create_train_test_split.py' script first in dataset/ folder.")
    print("\nTo create the split:")
    print("  cd ../../dataset/")
    print("  python create_train_test_split.py")
    exit()

print(f"Training data loaded successfully. Shape: {df.shape}")

# --- 2. DEFINE TARGET and HANDLE ITS MISSING VALUES ---
target = 'YIELD'

# The train_data.csv should already have NaN values removed, but double-check
missing_y_count = df[target].isnull().sum()
if missing_y_count > 0:
    print(f"Found {missing_y_count} rows with missing YIELD. Dropping them.")
    # Drop all rows where the 'YIELD' value is NaN
    df = df.dropna(subset=[target])
    print(f"New data shape after dropping: {df.shape}")

# --- 3. DEFINE FEATURES (X) and TARGET (y) ---
y = df[target]
X = df.drop(columns=[
    'YIELD', 
    'AREA', 
    'PRODUCTION', 
    'Dist_Code', 
    'State_Code'
])
print(f"Target (y): {target}")

# --- 4. IDENTIFY CATEGORICAL & NUMERICAL FEATURES ---
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

# --- 5. SPLIT DATA INTO TRAINING AND TESTING SETS ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)
print(f"Data split for validation: {len(X_train)} training rows, {len(X_test)} validation rows.")
print(f"Note: This is a validation split. Final testing should use test_data.csv")

# --- 6. CREATE A PREPROCESSING AND MODELING PIPELINE ---
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
model = XGBRegressor(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1
)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# --- 7. TRAIN THE MODEL ---
print("\n--- Training XGBoost Model ---")
print(f"Training on {len(X_train)} rows (with imputation)...")
clf.fit(X_train, y_train)
print("Training complete.")
training_time = time.time() - start_time
print(f"Total time taken: {training_time:.2f} seconds")

# --- 8. EVALUATE THE MODEL ---
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

# --- 9. FEATURE IMPORTANCES ---
print("\n--- Top 20 Most Important Features ---")
try:
    xgb_model = clf.named_steps['model']
    preprocessor = clf.named_steps['preprocessor']
    
    cat_features_out = preprocessor.named_transformers_['cat'] \
                                   .named_steps['encoder'] \
                                   .get_feature_names_out(categorical_features)
    num_features_out = numerical_features
    
    all_feature_names = np.concatenate([num_features_out, cat_features_out])
    
    importances = pd.Series(
        xgb_model.feature_importances_, 
        index=all_feature_names
    ).sort_values(ascending=False)
    
    print(importances.head(20))
except Exception as e:
    print(f"Could not calculate feature importances: {e}")

# --- 10. SAVE THE FINAL MODEL ---
print("\n--- Saving Model ---")
model_filename = 'models/crop_yield_model_xgb_latest.joblib'
joblib.dump(clf, model_filename)
print(f"Model saved successfully as: {model_filename}")
print(f"\n⚠️  IMPORTANT: This model was trained on train_data.csv")
print(f"   For final evaluation, test on test_data.csv (held-out data)")

