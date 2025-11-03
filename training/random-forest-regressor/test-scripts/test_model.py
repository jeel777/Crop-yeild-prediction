import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Define File Paths ---
# Make sure this path points to your NEWLY trained model in Google Drive
MODEL_PATH = "/content/drive/MyDrive/ML_Models/crop_yield_model_v5_tuned.joblib"
DATA_PATH = "testdatav2.csv"  # Assumes it's in the root of your Colab session

# --- 2. Configuration (Based on your new training script) ---
TARGET_COLUMN = 'YIELD'
COLUMNS_TO_DROP_FOR_X = [
    'YIELD', 
    'AREA', 
    'PRODUCTION', 
    'Dist_Code', 
    'State_Code'
]
# --- NO CLEANING FIXES ARE NEEDED ---


def load_data(path):
    """Loads CSV data from the specified path."""
    print(f"\nLoading test data from {path}...")
    if not os.path.exists(path):
        print(f"--- ERROR: File not found at {path} ---")
        print("Please check your file path and try again.")
        return None
    try:
        data = pd.read_csv(path)
        print("Test data loaded successfully.")
        return data
    except Exception as e:
        print(f"--- ERROR: Could not read CSV file ---")
        print(f"Details: {e}")
        return None

def load_model(path):
    """Loads the .joblib model pipeline from the specified path."""
    print(f"\nLoading model pipeline from {path}...")
    if not os.path.exists(path):
        print(f"--- ERROR: Model file not found at {path} ---")
        print("Please check your file path and try again.")
        return None
    try:
        model_pipeline = joblib.load(path)
        print("Model pipeline loaded successfully.")
        return model_pipeline
    except Exception as e:
        print(f"--- ERROR: Could not load model file ---")
        print(f"Details: {e}")
        return None

def evaluate_model():
    """Main function to load data, load model, predict, and evaluate."""
    
    # Load model and data
    model = load_model(MODEL_PATH)
    test_data = load_data(DATA_PATH)
    
    if model is None or test_data is None:
        print("\nExiting due to errors.")
        return

    # Validate that the target column exists
    if TARGET_COLUMN not in test_data.columns:
        print(f"--- ERROR: Target column '{TARGET_COLUMN}' not found in {DATA_PATH} ---")
        print("Please check your test data.")
        return
        
    try:
        y_test = test_data[TARGET_COLUMN]
        
        # Create X_test by dropping the same columns as in training
        X_test = test_data.drop(columns=COLUMNS_TO_DROP_FOR_X, errors='ignore') 
        
        print(f"\nUsing '{TARGET_COLUMN}' as the target (y).")
        print(f"Creating features (X) by dropping: {', '.join(COLUMNS_TO_DROP_FOR_X)}")
        print(f"Shape of X_test for prediction: {X_test.shape}")

        # --- ALL FIXES REMOVED ---
        # The new pipeline will handle all NaNs automatically.
        
        print("Making predictions...")
        y_pred = model.predict(X_test)
        print("Predictions complete.")

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Print the evaluation statistics
        print("\n--- 📈 Model Evaluation Stats ---")
        print(f"  R-squared (R²):           {r2:.4f}")
        print(f"  Mean Absolute Error (MAE):  {mae:.4f}")
        print(f"  Mean Squared Error (MSE):   {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print("------------------------------------\n")
        
        print("--- 💡 How to Interpret These Stats ---")
        print(f"* R-squared (R²):      Of all the variation in '{TARGET_COLUMN}', {r2*100:.2f}% is explained by the model.")
        print(f"* MAE & RMSE:          These represent the average prediction error, in the same units as '{TARGET_COLUMN}'.")
        print("                        (Lower is better).")
        
    except Exception as e:
        print("\n--- 🛑 PREDICTION ERROR ---")
        print("An unexpected error occurred. This should not happen if the model trained correctly.")
        print("\nOriginal Error Message:")
        print(f"   {e}")

if __name__ == "__main__":
    evaluate_model()