import pandas as pd

print("--- Missing Data Checker ---")

# --- 1. LOAD YOUR ORIGINAL WIDE-FORMAT DATA ---
input_file = "ICRISAT_Dataset.csv"  # The 12,803-row file

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"ERROR: File '{input_file}' not found.")
    exit()

print(f"Successfully loaded file '{input_file}'.")
print(f"Total rows in file: {len(df)}")

# --- 2. CLEAN COLUMN NAMES ---
df.columns = df.columns.str.strip().str.replace(' ', '_')

# --- 3. DEFINE PREDICTOR COLUMNS ---
predictor_columns = [
    'Dist_Code', 'Year', 'State_Code', 'State_Name', 'Dist_Name',
    'GROSS_CROPPED_AREA_(1000_ha)', 'NITROGEN_CONSUMPTION_(tons)',
    'PHOSPHATE_CONSUMPTION_(tons)', 'POTASH_CONSUMPTION_(tons)',
    'TOTAL_FERTILISER_CONSUMPTION_(tons)',
    'TOTAL_AGRICULTURAL_LABOUR_POPULATION_(1000_Number)',
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
    'RainS_JUN-SEP_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)', # My V5 script had a typo here
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

# Fix a typo in my previous id_vars list so this script doesn't crash
# This column probably exists in your file
if 'Rainy_JUN-SEP_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)' in df.columns:
    predictor_columns.remove('RainS_JUN-SEP_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)')
    predictor_columns.append('Rainy_JUN-SEP_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)')
else:
    # If not, we just drop it to avoid an error
    if 'RainS_JUN-SEP_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)' in predictor_columns:
         predictor_columns.remove('RainS_JUN-SEP_ACTUAL_EVAPOTRANSPIRATION_(Millimeters)')


# --- 4. CHECK FOR MISSING DATA ---
# This will count how many rows are "clean" (have no NaNs)
num_complete_rows = len(df.dropna(subset=predictor_columns))

num_rows_with_nan = len(df) - num_complete_rows

# --- 5. PRINT REPORT ---
print("\n--- MISSING DATA REPORT ---")
print(f"Total rows loaded: {len(df)}")
print(f"Rows with *NO* missing data in predictors: {num_complete_rows}")
print(f"Rows with *AT LEAST ONE* missing value (NaN): {num_rows_with_nan}")
print("---------------------------------")

print("\nThe pivot_table in the conversion script is DROPPING")
print(f"all {num_rows_with_nan} rows that have missing data.")
print(f"\nThis leaves it with {num_complete_rows} 'clean' rows.")
print(f"Multiplied by 5 crops, this gives approx:")
print(f"{num_complete_rows} x 5 = {num_complete_rows * 5} rows (your 5996 is in this range).")

print("\n--- Columns with the Most Missing Data ---")
# This will show you WHICH columns are the problem
missing_counts = df[predictor_columns].isnull().sum()
print(missing_counts[missing_counts > 0].sort_values(ascending=False))