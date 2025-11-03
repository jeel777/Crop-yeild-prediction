import joblib
import pandas as pd

print("--- Loading Model and Data for Prediction Test ---")

# --- 1. LOAD THE SAVED MODEL ---
model_filename = 'training/crop_yield_model.joblib'
try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    print(f"ERROR: Model file '{model_filename}' not found.")
    print("Please run the 'train_model_v3.py' script first to create it.")
    exit()
print("Model loaded successfully.")

# --- 2. LOAD THE FULL DATASET TO TEST ON ---
data_filename = 'dataset/data_long_format_v7_full.csv'
try:
    df = pd.read_csv(data_filename)
except FileNotFoundError:
    print(f"ERROR: Data file '{data_filename}' not found.")
    exit()
print(f"Data file '{data_filename}' loaded successfully.")

# --- 3. PREPARE ONE RANDOM ROW FOR TESTING ---
# We must first drop any rows where YIELD is NaN, just like in training
df = df.dropna(subset=['YIELD'])

# Select one random row from the dataset
test_row = df.sample(n=1, random_state=42) # random_state=42 makes it repeatable

print("\n--- Test Data (1 Random Row) ---")
print(f"Location: {test_row['Dist_Name'].values[0]}, {test_row['State_Name'].values[0]}")
print(f"Crop: {test_row['Crop'].values[0]}")
print(f"Year: {test_row['Year'].values[0]}")

# Separate the 'correct answer' (y) from the features (X)
# The model expects all columns *except* the ones we dropped during training
y_actual = test_row['YIELD'].values[0]

X_predict = test_row.drop(columns=[
    'YIELD', 
    'AREA', 
    'PRODUCTION', 
    'Dist_Code', 
    'State_Code'
])

# --- 4. MAKE AND SHOW THE PREDICTION ---
# The model (which is a pipeline) will automatically
# handle imputation and one-hot encoding for this one row.
prediction = model.predict(X_predict)[0]

print("\n--- PREDICTION vs. ACTUAL ---")
print(f"   Actual Yield: {y_actual:.2f} (Kg per ha)")
print(f"Predicted Yield: {prediction:.2f} (Kg per ha)")
print("-------------------------------")

# Calculate the difference
difference = prediction - y_actual
print(f"Difference: {difference:+.2f} (Kg per ha)")