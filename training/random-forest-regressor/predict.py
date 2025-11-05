"""
Crop Yield Prediction - Interactive CLI Tool

This script allows users to input crop and location details via command line
and get yield predictions from the trained Random Forest model.
"""

import joblib
import pandas as pd
import numpy as np
import sys
import os

def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print("🌾 CROP YIELD PREDICTION SYSTEM 🌾")
    print("=" * 70)

def load_model():
    """Load the trained model"""
    model_path = 'models/crop_yield_model_rf_latest.joblib'
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model file not found at: {model_path}")
        print("\nPlease train the model first:")
        print("  cd baseline/")
        print("  python train_modelv2.py")
        sys.exit(1)
    
    try:
        model = joblib.load(model_path)
        print("\n✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        sys.exit(1)

def get_crop_type():
    """Get crop type from user"""
    print("\n📋 CROP SELECTION")
    print("-" * 70)
    crops = ['RICE', 'PEARL_MILLET', 'CHICKPEA', 'GROUNDNUT', 'SUGARCANE']
    
    for i, crop in enumerate(crops, 1):
        print(f"  {i}. {crop}")
    
    while True:
        try:
            choice = input("\nSelect crop number (1-5): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(crops):
                return crops[choice_idx]
            else:
                print("❌ Invalid choice. Please enter a number between 1 and 5.")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input. Please enter a number.")

def get_location():
    """Get location from user"""
    print("\n📍 LOCATION DETAILS")
    print("-" * 70)
    
    state_name = input("Enter State Name (e.g., Chhattisgarh): ").strip()
    dist_name = input("Enter District Name (e.g., Durg): ").strip()
    
    return state_name, dist_name

def get_year():
    """Get year from user"""
    print("\n📅 YEAR")
    print("-" * 70)
    
    while True:
        try:
            year = input("Enter year (e.g., 2024): ").strip()
            year_int = int(year)
            if 1990 <= year_int <= 2030:
                return year_int
            else:
                print("❌ Please enter a realistic year between 1990 and 2030.")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input. Please enter a valid year.")

def get_float_input(prompt, default=None, allow_empty=True):
    """Get float input from user with optional default"""
    while True:
        try:
            value = input(prompt).strip()
            if value == "" and allow_empty:
                return default
            return float(value)
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def get_agricultural_inputs():
    """Get agricultural input data"""
    print("\n🚜 AGRICULTURAL INPUTS")
    print("-" * 70)
    print("(Press Enter to use average values if you don't have specific data)")
    
    gross_cropped_area = get_float_input(
        "Gross Cropped Area (1000 ha) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    nitrogen = get_float_input(
        "Nitrogen Consumption (tons) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    phosphate = get_float_input(
        "Phosphate Consumption (tons) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    potash = get_float_input(
        "Potash Consumption (tons) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    total_fertilizer = get_float_input(
        "Total Fertilizer Consumption (tons) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    irrigated_area = get_float_input(
        "Gross Irrigated Area (1000 ha) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    return {
        'GROSS_CROPPED_AREA_(1000_ha)': gross_cropped_area,
        'NITROGEN_CONSUMPTION_(tons)': nitrogen,
        'PHOSPHATE_CONSUMPTION_(tons)': phosphate,
        'POTASH_CONSUMPTION_(tons)': potash,
        'TOTAL_FERTILISER_CONSUMPTION_(tons)': total_fertilizer,
        'GROSS_IRRIGATED_AREA_(1000_ha)': irrigated_area
    }

def get_weather_data():
    """Get weather data (simplified - monthly averages)"""
    print("\n🌦️  WEATHER DATA")
    print("-" * 70)
    print("(Press Enter to use seasonal averages if you don't have monthly data)")
    
    use_simple = input("Use simple seasonal inputs? (y/n) [y]: ").strip().lower()
    
    if use_simple in ['', 'y', 'yes']:
        return get_seasonal_weather()
    else:
        return get_monthly_weather()

def get_seasonal_weather():
    """Get simplified seasonal weather data"""
    print("\nEnter seasonal averages:")
    
    summer_max_temp = get_float_input(
        "Summer Max Temperature (°C) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    rainy_max_temp = get_float_input(
        "Rainy Season Max Temperature (°C) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    summer_min_temp = get_float_input(
        "Summer Min Temperature (°C) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    rainy_precip = get_float_input(
        "Rainy Season Precipitation (mm) [Press Enter for avg]: ",
        default=None,
        allow_empty=True
    )
    
    # Create a dictionary with all weather columns (use None for missing)
    weather_data = {col: None for col in get_all_weather_columns()}
    
    # Fill in the seasonal aggregates if provided
    if summer_max_temp is not None:
        weather_data['Summer_MAR-MAY_MAXIMUM_TEMPERATURE_(Centigrate)'] = summer_max_temp
    if rainy_max_temp is not None:
        weather_data['Rainy_JUN-SEP_MAXIMUM_TEMPERATURE_(Centigrate)'] = rainy_max_temp
    if summer_min_temp is not None:
        weather_data['Summer_MAR-MAY_MINIMUM_TEMPERATURE_(Centigrate)'] = summer_min_temp
    if rainy_precip is not None:
        weather_data['Rainy_JUN-SEP_PERCIPITATION_(Millimeters)'] = rainy_precip
    
    return weather_data

def get_monthly_weather():
    """Get detailed monthly weather data"""
    print("\nThis would require 80+ weather inputs.")
    print("For simplicity, using seasonal mode...")
    return get_seasonal_weather()

def get_all_weather_columns():
    """Return list of all weather column names"""
    return [
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

def make_prediction(model):
    """Main prediction workflow"""
    # Collect user inputs
    crop = get_crop_type()
    state_name, dist_name = get_location()
    year = get_year()
    ag_inputs = get_agricultural_inputs()
    weather_data = get_weather_data()
    
    # Build the input dataframe
    input_data = {
        'Year': year,
        'State_Name': state_name,
        'Dist_Name': dist_name,
        'Crop': crop,
    }
    
    # Add agricultural inputs
    input_data.update(ag_inputs)
    
    # Add weather data
    input_data.update(weather_data)
    
    # Create DataFrame (model expects certain columns)
    df_input = pd.DataFrame([input_data])
    
    # Make prediction
    print("\n" + "=" * 70)
    print("🔮 MAKING PREDICTION...")
    print("=" * 70)
    
    try:
        prediction = model.predict(df_input)[0]
        
        print("\n" + "=" * 70)
        print("✅ PREDICTION RESULTS")
        print("=" * 70)
        print(f"\n📊 Input Summary:")
        print(f"   Crop:     {crop}")
        print(f"   Location: {dist_name}, {state_name}")
        print(f"   Year:     {year}")
        
        print(f"\n🌾 PREDICTED YIELD: {prediction:.2f} Kg/ha")
        
        # Provide context
        if prediction < 500:
            print("   📉 Low yield - consider improving inputs or crop selection")
        elif prediction < 1500:
            print("   📊 Moderate yield - typical for many crops")
        else:
            print("   📈 High yield - excellent conditions!")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR during prediction: {e}")
        print("\nPossible issues:")
        print("  - Location not in training data (model will use imputation)")
        print("  - Missing critical features")
        print("  - Model compatibility issue")

def main():
    """Main entry point"""
    print_banner()
    
    model = load_model()
    
    while True:
        make_prediction(model)
        
        print("\n" + "=" * 70)
        again = input("\nMake another prediction? (y/n) [y]: ").strip().lower()
        if again in ['n', 'no']:
            print("\n👋 Thank you for using the Crop Yield Prediction System!")
            print("=" * 70 + "\n")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting... Thank you for using the system!")
        sys.exit(0)
