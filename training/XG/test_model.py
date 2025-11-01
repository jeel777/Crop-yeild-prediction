import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

print("=" * 60)
print("Model Testing on testdata.csv")
print("=" * 60)

# --- 1. LOAD AND CONVERT TEST DATA (Wide to Long Format) ---
print("\n--- Step 1: Loading and Converting Test Data ---")
try:
    df_wide = pd.read_csv("../../dataset/testdata.csv")
except FileNotFoundError:
    print("ERROR: 'testdata.csv' not found in dataset folder.")
    exit()

print(f"Loaded test data (wide format): {df_wide.shape}")

# Clean column names
df_wide.columns = df_wide.columns.str.strip().str.replace(' ', '_')

# Define the columns that identify a unique row (the predictors)
id_vars = [
    'Dist_Code', 'Year', 'State_Code', 'State_Name', 'Dist_Name',
    'GROSS_CROPPED_AREA_(1000_ha)', 'NITROGEN_CONSUMPTION_(tons)',
    'PHOSPHATE_CONSUMPTION_(tons)', 'POTASH_CONSUMPTION_(tons)',
    'TOTAL_FERTILISER_CONSUMPTION_(tons)',
    'GROSS_IRRIGATED_AREA_(1000_ha)',
    'JANUARY_MAXIMUM_TEMPERATURE_(Centigrate)',
    'FEBRUARY_MAXIMUM_TEMPERATURE_(Centigrate)',
    'MARCH_MAXIMUM_TEMPERATURE_(Centigrate)',
    'APRIL_MAXIMUM_TEMPERATURE_(Centigrate)',
    'MAY_MAXIMUM_TEMPERATURE_(Centigrate)',
    'JUNE_MAXIMUM_TEMPERATURE_(Centigrate)',
    'JULY_MAXIMUM_TEMPERATURE_(Centigrate)',
    'AUGUST_MAXIMUM_TEMPERATURE_(Centigrate)',
    'SEPTEMBER_MAXIMUM_TEMPERATURE_(Centigrate)',
    'OCTOBER_MAXIMUM_TEMPERATURE_(Centigrate)',
    'NOVEMBER_MAXIMUM_TEMPERATURE_(Centigrate)',
    'DECEMBER_MAXIMUM_TEMPERATURE_(Centigrate)',
    'Winter_JAN-FEB_MAXIMUM_TEMPERATURE_(Centigrate)',
    'Summer_MAR-MAY_MAXIMUM_TEMPERATURE_(Centigrate)',
    'Rainy_JUN-SEP_MAXIMUM_TEMPERATURE_(Centigrate)',
    'Autumn_OCT-DEC_MAXIMUM_TEMPERATURE_(Centigrate)',
    'JANUARY_MINIMUM_TEMPERATURE_(Centigrate)',
    'FEBRUARY_MINIMUM_TEMPERATURE_(Centigrate)',
    'MARCH_MINIMUM_TEMPERATURE_(Centigrate)',
    'APRIL_MINIMUM_TEMPERATURE_(Centigrate)',
    'MAY_MINIMUM_TEMPERATURE_(Centigrate)',
    'JUNE_MINIMUM_TEMPERATURE_(Centigrate)',
    'JULY_MINIMUM_TEMPERATURE_(Centigrate)',
    'AUGUST_MINIMUM_TEMPERATURE_(Centigrate)',
    'SEPTEMBER_MINIMUM_TEMPERATURE_(Centigrate)',
    'OCTOBER_MINIMUM_TEMPERATURE_(Centigrate)',
    'NOVEMBER_MINIMUM_TEMPERATURE_(Centigrate)',
    'DECEMBER_MINIMUM_TEMPERATURE_(Centigrate)',
    'Winter_JAN-FEB_MINIMUM_TEMPERATURE_(Centigrate)',
    'Summer_MAR-MAY_MINIMUM_TEMPERATURE_(Centigrate)',
    'Rainy_JUN-SEP_MINIMUM_TEMPERATURE_(Centigrate)',
    'Autumn_OCT-DEC_MINIMUM_TEMPERATURE_(Centigrate)',
    'JANUARY_PERCIPITATION_(Millimeters)',
    'FEBRUARY_PERCIPITATION_(Millimeters)',
    'MARCH_PERCIPITATION_(Millimeters)',
    'APRIL_PERCIPITATION_(Millimeters)',
    'MAY_PERCIPITATION_(Millimeters)',
    'JUNE_PERCIPITATION_(Millimeters)',
    'JULY_PERCIPITATION_(Millimeters)',
    'AUGUST_PERCIPITATION_(Millimeters)',
    'SEPTEMBER_PERCIPITATION_(Millimeters)',
    'OCTOBER_PERCIPITATION_(Millimeters)',
    'NOVEMBER_PERCIPITATION_(Millimeters)',
    'DECEMBER_PERCIPITATION_(Millimeters)',
    'Winter_JAN-FEB_PERCIPITATION_(Millimeters)',
    'Summer_MAR-MAY_PERCIPITATION_(Millimeters)',
    'Rainy_JUN-SEP_PERCIPITATION_(Millimeters)',
    'Autumn_OCT-DEC_PERCIPITATION_(Millimeters)',
    'JANUARY_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'FEBRUARY_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'MARCH_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'APRIL_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'MAY_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'JUNE_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'JULY_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'AUGUST_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'SEPTEMBER_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'OCTOBER_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'NOVEMBER_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'DECEMBER_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'Winter_JAN-FEB_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'Summer_MAR-MAY_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'Rainy_JUN-SEP_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'Autumn_OCT-DEC_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)',
    'JAN_WINDSPEED_(Meter_per_second)',
    'FEB_WINDSPEED_(Meter_per_second)',
    'MARCH_WINDSPEED_(Meter_per_second)',
    'APRIL_WINDSPEED_(Meter_per_second)',
    'MAY_WINDSPEED_(Meter_per_second)',
    'JUNE_WINDSPEED_(Meter_per_second)',
    'JULY_WINDSPEED_(Meter_per_second)',
    'AUG_WINDSPEED_(Meter_per_second)',
    'SEPT_WINDSPEED_(Meter_per_second)',
    'OCT_WINDSPEED_(Meter_per_second)',
    'NOV_WINDSPEED_(Meter_per_second)',
    'DEC_WINDSPEED_(Meter_per_second)',
    'Winter_JAN-FEB_WINDSPEED_(Meter_per_second)',
    'Summer_MAR-MAY_WINDSPEED_(Meter_per_second)',
    'Rainy_JUN-SEP_WINDSPEED_(Meter_per_second)',
    'Autumn_OCT-DEC_WINDSPEED_(Meter_per_second)'
]

# Define crop columns
area_cols = ['RICE_AREA_(1000_ha)', 'PEARL_MILLET_AREA_(1000_ha)', 'CHICKPEA_AREA_(1000_ha)', 'GROUNDNUT_AREA_(1000_ha)', 'SUGARCANE_AREA_(1000_ha)']
prod_cols = ['RICE_PRODUCTION_(1000_tons)', 'PEARL_MILLET_PRODUCTION_(1000_tons)', 'CHICKPEA_PRODUCTION_(1000_tons)', 'GROUNDNUT_PRODUCTION_(1000_tons)', 'SUGARCANE_PRODUCTION_(1000_tons)']
yield_cols = ['RICE_YIELD_(Kg_per_ha)', 'PEARL_MILLET_YIELD_(Kg_per_ha)', 'CHICKPEA_YIELD_(Kg_per_ha)', 'GROUNDNUT_YIELD_(Kg_per_ha)', 'SUGARCANE_YIELD_(Kg_per_ha)']
crop_names = ['RICE', 'PEARL_MILLET', 'CHICKPEA', 'GROUNDNUT', 'SUGARCANE']

# Convert to long format
def melt_metric(df, id_vars, value_vars, metric_name, crop_names):
    df_melted = df.melt(
        id_vars=id_vars, 
        value_vars=value_vars, 
        var_name='Crop_Name', 
        value_name=metric_name
    )
    crop_map = dict(zip(value_vars, crop_names))
    df_melted['Crop'] = df_melted['Crop_Name'].map(crop_map)
    df_melted = df_melted.drop('Crop_Name', axis=1)
    return df_melted

print("Converting wide format to long format...")
df_area = melt_metric(df_wide, id_vars, area_cols, 'AREA', crop_names)
df_prod = melt_metric(df_wide, id_vars, prod_cols, 'PRODUCTION', crop_names)
df_yield_data = melt_metric(df_wide, id_vars, yield_cols, 'YIELD', crop_names)

merge_keys = id_vars + ['Crop']
df_test = pd.merge(df_area, df_prod, on=merge_keys, how='outer')
df_test = pd.merge(df_test, df_yield_data, on=merge_keys, how='outer')

print(f"Converted to long format: {df_test.shape}")

# --- 2. PREPARE TEST DATA (Same as training) ---
print("\n--- Step 2: Preparing Test Data ---")
target = 'YIELD'

# Drop rows with missing YIELD (actual target values we want to test against)
missing_y_count = df_test[target].isnull().sum()
if missing_y_count > 0:
    print(f"Dropping {missing_y_count} rows with missing YIELD values.")
    df_test = df_test.dropna(subset=[target])
    print(f"Remaining rows: {df_test.shape[0]}")

# Filter out rows where YIELD is 0 (likely no crop planted)
initial_count = len(df_test)
df_test = df_test[df_test[target] > 0]
filtered_count = initial_count - len(df_test)
if filtered_count > 0:
    print(f"Filtered out {filtered_count} rows with YIELD = 0")

print(f"Final test dataset: {len(df_test)} rows")

# Extract features and target (same as training)
y_test = df_test[target]
X_test = df_test.drop(columns=[
    'YIELD', 
    'AREA', 
    'PRODUCTION', 
    'Dist_Code', 
    'State_Code'
])

print(f"Features (X): {X_test.shape[1]} columns")
print(f"Target (y): {len(y_test)} values")

# --- 3. LOAD TRAINED MODEL ---
print("\n--- Step 3: Loading Trained Model ---")
model_filename = 'crop_yield_model_v3.joblib'
try:
    model = joblib.load(model_filename)
    print(f"Model loaded successfully from: {model_filename}")
except FileNotFoundError:
    print(f"ERROR: Model file '{model_filename}' not found.")
    print("Please train the model first using train_model_xgboost.py")
    exit()

# --- 4. MAKE PREDICTIONS ---
print("\n--- Step 4: Making Predictions ---")
print(f"Predicting on {len(X_test)} test samples...")
y_pred = model.predict(X_test)
print("Predictions complete!")

# --- 5. CALCULATE METRICS ---
print("\n" + "=" * 60)
print("TEST RESULTS & STATISTICS")
print("=" * 60)

# Basic metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate accuracy percentage (using percentage error)
percentage_errors = np.abs((y_test - y_pred) / y_test) * 100
mean_percentage_error = np.mean(percentage_errors)
accuracy = 100 - mean_percentage_error

# Alternative: Within-threshold accuracy (predictions within 10%, 20%, 30% of actual)
within_10_pct = np.sum(percentage_errors <= 10) / len(percentage_errors) * 100
within_20_pct = np.sum(percentage_errors <= 20) / len(percentage_errors) * 100
within_30_pct = np.sum(percentage_errors <= 30) / len(percentage_errors) * 100

print(f"\n📊 MODEL PERFORMANCE METRICS:")
print(f"   R-squared (R²):                    {r2:.4f} ({r2*100:.2f}%)")
print(f"   Mean Absolute Error (MAE):         {mae:.2f} Kg per ha")
print(f"   Root Mean Squared Error (RMSE):    {rmse:.2f} Kg per ha")
print(f"\n📈 ACCURACY STATISTICS:")
print(f"   Mean Percentage Error:            {mean_percentage_error:.2f}%")
print(f"   Overall Accuracy:                  {accuracy:.2f}%")
print(f"\n🎯 PREDICTION ACCURACY (Within Tolerance):")
print(f"   Predictions within 10% of actual: {within_10_pct:.2f}%")
print(f"   Predictions within 20% of actual: {within_20_pct:.2f}%")
print(f"   Predictions within 30% of actual: {within_30_pct:.2f}%")

# Additional statistics
print(f"\n📋 ADDITIONAL STATISTICS:")
print(f"   Total test samples:                {len(y_test)}")
print(f"   Actual YIELD range:                {y_test.min():.2f} - {y_test.max():.2f} Kg per ha")
print(f"   Predicted YIELD range:             {y_pred.min():.2f} - {y_pred.max():.2f} Kg per ha")
print(f"   Actual YIELD mean:                 {y_test.mean():.2f} Kg per ha")
print(f"   Predicted YIELD mean:              {y_pred.mean():.2f} Kg per ha")

# Error distribution
print(f"\n📉 ERROR DISTRIBUTION:")
print(f"   Median percentage error:           {np.median(percentage_errors):.2f}%")
print(f"   Min percentage error:              {percentage_errors.min():.2f}%")
print(f"   Max percentage error:               {percentage_errors.max():.2f}%")
print(f"   Std deviation of errors:           {percentage_errors.std():.2f}%")

print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)

