import joblib
import pandas as pd

print("--- Loading Model for Prediction ---")
model_filename = 'crop_yield_model.joblib'
try:
    # Load the entire pipeline (preprocessor + model)
    model = joblib.load(model_filename)
except FileNotFoundError:
    print(f"ERROR: Model file '{model_filename}' not found.")
    print("Please run the 'train_model.py' script first to create it.")
    exit()

print("Model loaded successfully.")

# --- CREATE YOUR NEW DATA FOR PREDICTION ---
# *** ACTION REQUIRED ***
# You must provide ALL the features that the model was trained on.
# This means all 96 columns (State_Name, Dist_Name, Crop, all 92 weather/input cols)
#
# Here is an example with a few key columns. You must fill in ALL of them.
# The column names MUST match your training data exactly.

my_test_data = {
    # Categorical Features
    'State_Name': 'GUJARAT',
    'Dist_Name': 'AHMEDABAD',
    'Crop': 'RICE',
    
    # Numerical Features (Examples)
    'Year': 2026,
    'GROSS_CROPPED_AREA_(1000_ha)': 350.5,
    'NITROGEN_CONSUMPTION_(tons)': 8500.0,
    'PHOSPHATE_CONSUMPTION_(tons)': 4200.0,
    'POTASH_CONSUMPTION_(tons)': 1500.0,
    'TOTAL_FERTILISER_CONSUMPTION_(tons)': 14200.0,
    'TOTAL_AGRICULTURAL_LABOUR_POPULATION_(1000_Number)': 450.0,
    'GROSS_IRRIGATED_AREA_(1000_ha)': 210.0,
    
    # Weather Features (Examples - YOU MUST FILL IN ALL 84)
    'JANUARY_MAXIMUM_TEMPERATURE_(Centigrate)': 28.5,
    'FEBRUARY_MAXIMUM_TEMPERATURE_(Centigrate)': 30.1,
    'MARCH_MAXIMUM_TEMPERATURE_(Centigrate)': 35.2,
    # ... (and so on for all other temp, precip, windspeed, etc. columns)
    'Rainy_JUN-SEP_PERCIPITATION_(Millimeters)': 850.0,
    'Autumn_OCT-DEC_WINDSPEED_(Meter_per_second)': 1.2
    # ... (etc. until all 96 features are provided)
}

# --- MAKE THE PREDICTION ---
# The model expects a DataFrame, so we convert the dictionary.
# The [0] at the end means "give me the first (and only) prediction".
try:
    # Convert dictionary to a DataFrame
    # 'index=[0]' is important to tell pandas this is a single row
    X_predict = pd.DataFrame(my_test_data, index=[0])
    
    # Use the model to predict
    prediction = model.predict(X_predict)[0]
    
    print("\n--- Prediction Complete ---")
    print(f"Predicted Yield: {prediction:.2f} (Kg per ha)")

except ValueError as e:
    print(f"\n--- ERROR ---")
    print("The model's features and your test data do not match.")
    print("This usually means 'my_test_data' is missing some columns.")
    print(f"Details: {e}")