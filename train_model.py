import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import time

print("Starting model training process...")
start_time = time.time()

# --- 1. LOAD THE LONG-FORMAT DATA ---
try:
    df = pd.read_csv("data_long_format_v4.csv")
except FileNotFoundError:
    print("ERROR: 'data_long_format_v4.csv' not found.")
    print("Please make sure the file is in the same folder as this script.")
    exit()

print(f"Data loaded successfully. Shape: {df.shape}")

# --- 2. DEFINE FEATURES (X) and TARGET (y) ---

# Our target is what we want to predict
target = 'YIELD'
y = df[target]

# Our features are the columns we use to make the prediction.
# We DROP the target ('YIELD') and its 'cheater' columns ('AREA', 'PRODUCTION').
# We also drop simple ID codes that aren't useful (we use names).
X = df.drop(columns=[
    'YIELD', 
    'AREA', 
    'PRODUCTION', 
    'Dist_Code', 
    'State_Code'
])

print(f"Target (y): {target}")
print(f"Features (X): {X.columns.to_list()}")

# --- 3. IDENTIFY CATEGORICAL & NUMERICAL FEATURES ---
# The model needs to know which columns are text (categorical)
# and which are numbers (numerical).

categorical_features = [
    'State_Name', 
    'Dist_Name', 
    'Crop'
]

# All other columns in X are numerical
numerical_features = [
    col for col in X.columns if col not in categorical_features
]

print(f"\nIdentified {len(categorical_features)} categorical features.")
print(f"Identified {len(numerical_features)} numerical features.")

# --- 4. SPLIT DATA INTO TRAINING AND TESTING SETS ---
# We train the model on 80% of the data and test it on 20%
# to see how it performs on "unseen" data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42  # Ensures reproducible results
)

print(f"Data split: {len(X_train)} training rows, {len(X_test)} testing rows.")

# --- 5. CREATE A PREPROCESSING AND MODELING PIPELINE ---

# Step 5a: Create a 'preprocessor' to handle features
# We use 'OneHotEncoder' for categorical features to turn
# text like 'RICE' into numbers. 'handle_unknown='ignore''
# prevents errors if test data has a rare category.
# 'remainder='passthrough'' means all numerical columns are
# left unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Step 5b: Define the model
# n_estimators=100 means it will build 100 "trees" in the forest.
# random_state=42 ensures you get the same result every time.
# n_jobs=-1 uses all available CPU cores to speed up training.
model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    oob_score=True  # A special score for Random Forest
)

# Step 5c: Create the final pipeline
# This chains the steps: first 'preprocess', then 'model'.
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# --- 6. TRAIN THE MODEL ---
print("\n--- Training Random Forest Model ---")
print("This may take a few minutes...")
clf.fit(X_train, y_train)
print("Training complete.")
training_time = time.time() - start_time
print(f"Total time taken: {training_time:.2f} seconds")

# --- 7. EVALUATE THE MODEL ---
print("\n--- Model Evaluation ---")

# A) Out-of-Bag (OOB) Score
# A quick score from the model itself on data it didn't see.
# Closer to 1.0 is better.
print(f"Model OOB Score: {clf.named_steps['model'].oob_score_:.4f}")

# B) Predictions on the Test Set
print("Making predictions on the test set...")
y_pred = clf.predict(X_test)

# C) Key Metrics
# R-squared: "Coefficient of Determination". 
# How much of the yield's variance is explained by the model?
# 1.0 is perfect, 0.0 is no better than just guessing the average.
r2 = r2_score(y_test, y_pred)
print(f"R-squared ($R^2$): {r2:.4f}")

# MAE: Mean Absolute Error
# The average error of a prediction, in Kg per ha.
# Lower is better.
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f} (Kg per ha)")

# RMSE: Root Mean Squared Error
# Similar to MAE, but penalizes large errors more.
# Lower is better.
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (Kg per ha)")

print("\n--- Process Finished ---")

# --- BONUS: FEATURE IMPORTANCES ---
# (A bit more advanced, but very useful)
print("\n--- Top 20 Most Important Features ---")
try:
    # Get the model from the pipeline
    rf_model = clf.named_steps['model']
    
    # Get the preprocessor from the pipeline
    preprocessor = clf.named_steps['preprocessor']
    
    # Get the feature names AFTER OneHotEncoding
    # e.g., 'Crop' becomes 'Crop_RICE', 'Crop_JOWAR', etc.
    cat_features_out = preprocessor.named_transformers_['cat'] \
                                   .get_feature_names_out(categorical_features)
    
    # Combine with the original numerical feature names
    all_feature_names = np.concatenate([cat_features_out, numerical_features])
    
    # Create a nice Pandas Series to see them
    importances = pd.Series(
        rf_model.feature_importances_, 
        index=all_feature_names
    ).sort_values(ascending=False)
    
    # Print the top 20
    print(importances.head(20))

except Exception as e:
    print(f"Could not calculate feature importances: {e}")


# --- ADD THIS AT THE END OF train_model.py ---

import joblib

print("\n--- Saving Model ---")
model_filename = 'crop_yield_model.joblib'
joblib.dump(clf, model_filename)
print(f"Model saved successfully as: {model_filename}")