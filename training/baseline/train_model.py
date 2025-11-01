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

try:
    df = pd.read_csv("data_long_format_v4.csv")
except FileNotFoundError:
    print("ERROR: 'data_long_format_v4.csv' not found.")
    print("Please make sure the file is in the same folder as this script.")
    exit()

print(f"Data loaded successfully. Shape: {df.shape}")


target = 'YIELD'
y = df[target]


X = df.drop(columns=[
    'YIELD', 
    'AREA', 
    'PRODUCTION', 
    'Dist_Code', 
    'State_Code'
])

print(f"Target (y): {target}")
print(f"Features (X): {X.columns.to_list()}")



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
    random_state=42  # Ensures reproducible results
)

print(f"Data split: {len(X_train)} training rows, {len(X_test)} testing rows.")


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    oob_score=True  # A special score for Random Forest
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

print("\n--- Training Random Forest Model ---")
print("This may take a few minutes...")
clf.fit(X_train, y_train)
print("Training complete.")
training_time = time.time() - start_time
print(f"Total time taken: {training_time:.2f} seconds")

print("\n--- Model Evaluation ---")


print(f"Model OOB Score: {clf.named_steps['model'].oob_score_:.4f}")

print("Making predictions on the test set...")
y_pred = clf.predict(X_test)


r2 = r2_score(y_test, y_pred)
print(f"R-squared ($R^2$): {r2:.4f}")


mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f} (Kg per ha)")


rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (Kg per ha)")

print("\n--- Process Finished ---")


print("\n--- Top 20 Most Important Features ---")
try:
    rf_model = clf.named_steps['model']
    
    preprocessor = clf.named_steps['preprocessor']
    
    cat_features_out = preprocessor.named_transformers_['cat'] \
                                   .get_feature_names_out(categorical_features)
    
    all_feature_names = np.concatenate([cat_features_out, numerical_features])
    
    importances = pd.Series(
        rf_model.feature_importances_, 
        index=all_feature_names
    ).sort_values(ascending=False)
    
    print(importances.head(20))

except Exception as e:
    print(f"Could not calculate feature importances: {e}")



import joblib

print("\n--- Saving Model ---")
model_filename = 'crop_yield_model.joblib'
joblib.dump(clf, model_filename)
print(f"Model saved successfully as: {model_filename}")