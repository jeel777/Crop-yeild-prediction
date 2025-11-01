import pandas as pd

print("--- Duplicate Row Checker ---")

# --- 1. LOAD YOUR ORIGINAL WIDE-FORMAT DATA ---
# *** ACTION REQUIRED ***
# Change this to the name of your 12,803-row CSV file
input_file = "ICRISAT_Dataset.csv" 

try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"ERROR: File '{input_file}' not found.")
    exit()

print(f"Successfully loaded file '{input_file}'.")
print(f"Total rows in file: {len(df)}")

# --- 2. CLEAN COLUMN NAMES ---
# This is the same logic as the conversion script
df.columns = df.columns.str.strip().str.replace(' ', '_')

# --- 3. DEFINE PREDICTOR COLUMNS ---
# These are the columns that define a "unique entry"
# (all the id_vars from the last script)
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

# --- 4. CHECK FOR DUPLICATES ---
# We check for duplicates based *only* on the predictor columns.
# We keep 'first' to get the count of unique rows.
num_unique_rows = len(df.drop_duplicates(subset=predictor_columns, keep='first'))

num_duplicate_rows = len(df) - num_unique_rows

# --- 5. PRINT REPORT ---
print("\n--- DUPLICATE REPORT ---")
print(f"Total rows loaded: {len(df)}")
print(f"Unique predictor sets (District, Year, Weather, etc.): {num_unique_rows}")
print(f"Duplicate predictor sets (rows to be averaged): {num_duplicate_rows}")
print("--------------------------")

if num_duplicate_rows > 0:
    print("\nThis means your file contains multiple rows for the same")
    print("District/Year/Weather combination. This is why the")
    print("conversion script is 'collapsing' them by averaging.")
    
    # Let's find and show an example
    print("\nFinding an example of a duplicate entry...")
    # This gets a list of True/False for duplicates
    dupes = df.duplicated(subset=predictor_columns, keep=False)
    
    if dupes.any():
        # Get all rows that are duplicates
        df_dupes = df[dupes]
        
        # Sort them so we can see them grouped together
        df_dupes_sorted = df_dupes.sort_values(by=['Dist_Name', 'Year'])
        
        print("--- Example of Duplicate Predictor Rows ---")
        # Print the first 5 rows from the sorted duplicates
        print(df_dupes_sorted.head())
        print("---------------------------------------------")
        print("\nNotice how the 'Dist_Name', 'Year', and weather data might")
        print("be identical, but the CROP columns (AREA, YIELD) are different.")
        print("The conversion script averages these crop values.")

else:
    print("\nNo duplicates found based on predictor columns.")
    print("If this is the case, my explanation was wrong.")