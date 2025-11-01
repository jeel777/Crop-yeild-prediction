import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV 
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import time
import joblib

print("Starting model training process (v4 with HYPERPARAMETER TUNING)...")
start_time = time.time()

try:
    df = pd.read_csv("dataset/data_long_format_v7_full.csv")
except FileNotFoundError:
    print("ERROR: 'data_long_format_v7_full.csv' not found.")
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

categorical_features = ['State_Name', 'Dist_Name', 'Crop']
numerical_features = [col for col in X.columns if col not in categorical_features]

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
model = RandomForestRegressor(random_state=42, n_jobs=-1)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

print("\nDefining parameter search grid...")
param_grid = {
    'model__n_estimators': [100, 200], # A smaller grid to be faster for you
    'model__max_depth': [10, 20, 30, None],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 1.0] # 'sqrt' or 'auto' (old) vs 1.0 (all)
}

random_search = RandomizedSearchCV(
    estimator=clf, # Our pipeline
    param_distributions=param_grid, # Our settings grid
    n_iter=10,        # Number of combinations to try
    cv=3,             # Number of cross-validation folds
    verbose=2,        # This will print progress!
    random_state=42,
    n_jobs=-1         # Use all available CPU cores
)

print("\n--- Starting Hyperparameter Tuning (RandomizedSearchCV) ---")
print("This will take significantly longer than before. Please be patient.")
print("It is testing 10 combinations with 3-fold CV (total 30 fits)...")

random_search.fit(X_train, y_train)

print("--- Tuning Complete ---")

print("\n--- Best Parameters Found ---")
print(random_search.best_params_)

best_model = random_search.best_estimator_

print("\n--- Model Evaluation (using BEST model) ---")
print("Making predictions on the test set...")
y_pred = best_model.predict(X_test) # Use .predict() on the best model

r2 = r2_score(y_test, y_pred)
print(f"R-squared ($R^2$): {r2:.4f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f} (Kg per ha)")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (Kg per ha)")
print("\nProcess Finished.")

print("\n--- Top 20 Most Important Features (from BEST model) ---")
try:
    rf_model = best_model.named_steps['model']
    preprocessor = best_model.named_steps['preprocessor']
    
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

print("\n--- Saving Tuned Model ---")
model_filename = 'crop_yield_model_v4_tuned.joblib'

joblib.dump(best_model, model_filename)
print(f"Tuned model saved successfully as: {model_filename}")