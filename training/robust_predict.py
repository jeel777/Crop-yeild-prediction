"""
Robust Crop Yield Prediction with Missing Column Handling

This script can make predictions even when some input columns are completely missing.
It will fill missing columns with appropriate default values.
"""

import joblib
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path='random-forest-regressor/models/crop_yield_model_rf_latest.joblib'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model file not found at: {model_path}")
        sys.exit(1)
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        sys.exit(1)

def get_expected_columns():
    """Return the expected column names for the model"""
    # These are the columns the model was trained on (excluding YIELD, AREA, etc.)
    return [
        'Year',
        'State_Name',
        'Dist_Name',
        'Crop',
        'GROSS_CROPPED_AREA_(1000_ha)',
        'NITROGEN_CONSUMPTION_(tons)',
        'PHOSPHATE_CONSUMPTION_(tons)',
        'POTASH_CONSUMPTION_(tons)',
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

def ensure_all_columns(df, expected_columns):
    """
    Ensure DataFrame has all expected columns.
    Missing columns are added with NaN values (will be imputed by model).
    
    Args:
        df: Input DataFrame
        expected_columns: List of column names the model expects
    
    Returns:
        DataFrame with all expected columns in correct order
    """
    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    
    if missing_cols:
        print(f"\n⚠️  WARNING: {len(missing_cols)} columns missing from input data:")
        for col in sorted(list(missing_cols))[:10]:  # Show first 10
            print(f"   - {col}")
        if len(missing_cols) > 10:
            print(f"   ... and {len(missing_cols) - 10} more")
        print(f"\n✅ Adding missing columns with NaN (will be imputed by model)")
        
        # Add missing columns with NaN
        for col in missing_cols:
            df[col] = np.nan
    
    if extra_cols:
        print(f"\n⚠️  INFO: {len(extra_cols)} extra columns in input (will be ignored):")
        for col in sorted(list(extra_cols))[:5]:
            print(f"   - {col}")
    
    # Return only expected columns in correct order
    return df[expected_columns]

def predict_robust(model, input_data):
    """
    Make predictions with robust column handling
    
    Args:
        model: Trained sklearn pipeline/model
        input_data: DataFrame with input features (may have missing columns)
    
    Returns:
        numpy array of predictions
    """
    expected_cols = get_expected_columns()
    
    print(f"\n📊 Input data check:")
    print(f"   Provided columns: {len(input_data.columns)}")
    print(f"   Expected columns: {len(expected_cols)}")
    
    # Ensure all columns present
    input_data_fixed = ensure_all_columns(input_data, expected_cols)
    
    print(f"\n✅ Data prepared for prediction: {input_data_fixed.shape}")
    
    # Make prediction
    predictions = model.predict(input_data_fixed)
    
    return predictions

def load_and_predict_from_csv(csv_path, model):
    """Load test data from CSV and make predictions"""
    print(f"\n📂 Loading test data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"   ❌ File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return None
    
    # Remove columns that shouldn't be in features
    columns_to_remove = ['YIELD', 'AREA', 'PRODUCTION', 'Dist_Code', 'State_Code']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Make predictions
    predictions = predict_robust(model, df)
    
    return predictions

def main():
    """Main function"""
    print("=" * 80)
    print("🔬 ROBUST CROP YIELD PREDICTION")
    print("   (Handles Missing Columns)")
    print("=" * 80)
    
    # Load model
    print("\n1. Loading model...")
    model = load_model()
    
    # Check if test data file provided
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = '../dataset/test_data.csv'
    
    # Make predictions
    print("\n2. Making predictions...")
    predictions = load_and_predict_from_csv(csv_path, model)
    
    if predictions is not None:
        print("\n" + "=" * 80)
        print("✅ PREDICTIONS COMPLETE")
        print("=" * 80)
        print(f"\n📊 Prediction Statistics:")
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Mean predicted yield: {predictions.mean():.2f} kg/ha")
        print(f"   Std deviation: {predictions.std():.2f} kg/ha")
        print(f"   Min predicted yield: {predictions.min():.2f} kg/ha")
        print(f"   Max predicted yield: {predictions.max():.2f} kg/ha")
        
        print(f"\n📝 First 10 predictions:")
        for i, pred in enumerate(predictions[:10], 1):
            print(f"   {i:2d}. {pred:8.2f} kg/ha")
        
        # Optionally save predictions
        save = input("\n💾 Save predictions to file? (y/n) [n]: ").strip().lower()
        if save in ['y', 'yes']:
            output_file = 'predictions.csv'
            pd.DataFrame({'Predicted_Yield_kg_ha': predictions}).to_csv(output_file, index=False)
            print(f"   ✅ Saved to: {output_file}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

