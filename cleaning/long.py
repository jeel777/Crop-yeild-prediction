import pandas as pd
import numpy as np

print("Starting data conversion (Version 7, Memory-Safe)...")

input_file = "ICRISAT_Dataset.csv" 
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"ERROR: File '{input_file}' not found.")
    exit()

print(f"Loaded file shape: {df.shape}")

print("Cleaning column names...")
df.columns = df.columns.str.strip().str.replace(' ', '_')

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

area_cols = ['RICE_AREA_(1000_ha)', 'PEARL_MILLET_AREA_(1000_ha)', 'CHICKPEA_AREA_(1000_ha)', 'GROUNDNUT_AREA_(1000_ha)', 'SUGARCANE_AREA_(1000_ha)']
prod_cols = ['RICE_PRODUCTION_(1000_tons)', 'PEARL_MILLET_PRODUCTION_(1000_tons)', 'CHICKPEA_PRODUCTION_(1000_tons)', 'GROUNDNUT_PRODUCTION_(1000_tons)', 'SUGARCANE_PRODUCTION_(1000_tons)']
yield_cols = ['RICE_YIELD_(Kg_per_ha)', 'PEARL_MILLET_YIELD_(Kg_per_ha)', 'CHICKPEA_YIELD_(Kg_per_ha)', 'GROUNDNUT_YIELD_(Kg_per_ha)', 'SUGARCANE_YIELD_(Kg_per_ha)']

crop_names = ['RICE', 'PEARL_MILLET', 'CHICKPEA', 'GROUNDNUT', 'SUGARCANE']

def melt_metric(df, id_vars, value_vars, metric_name, crop_names):
    """Helper function to melt one metric."""
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

print("Melting AREA data...")
df_area = melt_metric(df, id_vars, area_cols, 'AREA', crop_names)

print("Melting PRODUCTION data...")
df_prod = melt_metric(df, id_vars, prod_cols, 'PRODUCTION', crop_names)

print("Melting YIELD data...")
df_yield = melt_metric(df, id_vars, yield_cols, 'YIELD', crop_names)

merge_keys = id_vars + ['Crop']

print("Merging AREA and PRODUCTION data...")
df_final = pd.merge(df_area, df_prod, on=merge_keys, how='outer')

print("Merging YIELD data...")
df_final = pd.merge(df_final, df_yield, on=merge_keys, how='outer')

output_filename = 'data_long_format_v7_full.csv'
print(f"Saving final long-format file as: {output_filename}")
df_final.to_csv(output_filename, index=False)

print("\n--- SUCCESS! ---")
print(f"File converted and saved as: {output_filename}")
print(f"Final data shape: {df_final.shape}")
print("This file contains all your data, including NaNs.")