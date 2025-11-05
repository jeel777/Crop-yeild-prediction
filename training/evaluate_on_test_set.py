"""
Evaluate trained models on the held-out test set.

This script evaluates the trained Random Forest and XGBoost models
on test_data.csv to get unbiased performance metrics.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
import sys

def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 80)
    print("📊 MODEL EVALUATION ON HELD-OUT TEST SET 📊")
    print("=" * 80)

def load_test_data():
    """Load the held-out test dataset"""
    print("\n1. Loading test data...")
    test_path = '../dataset/test_data.csv'
    
    if not os.path.exists(test_path):
        print(f"❌ ERROR: Test data not found at {test_path}")
        print("\nPlease create the train-test split first:")
        print("  cd ../dataset/")
        print("  python create_train_test_split.py")
        sys.exit(1)
    
    df = pd.read_csv(test_path)
    print(f"   ✅ Test data loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Check for missing yields
    missing_yield = df['YIELD'].isnull().sum()
    if missing_yield > 0:
        print(f"   ⚠️  Dropping {missing_yield} rows with missing YIELD")
        df = df.dropna(subset=['YIELD'])
    
    return df

def prepare_test_data(df):
    """Prepare features and target from test data"""
    print("\n2. Preparing test data...")
    
    target = 'YIELD'
    y_test = df[target]
    
    X_test = df.drop(columns=[
        'YIELD', 
        'AREA', 
        'PRODUCTION', 
        'Dist_Code', 
        'State_Code'
    ])
    
    print(f"   Features (X): {X_test.shape[1]} columns")
    print(f"   Target (y): {len(y_test)} samples")
    print(f"   YIELD range: {y_test.min():.2f} to {y_test.max():.2f} kg/ha")
    
    return X_test, y_test

def load_model(model_path):
    """Load a trained model"""
    if not os.path.exists(model_path):
        print(f"   ⚠️  Model not found: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"   ✅ Loaded: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"   ❌ Error loading {model_path}: {e}")
        return None

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and print comprehensive metrics"""
    print("\n" + "-" * 80)
    print(f"📈 {model_name} Evaluation Results")
    print("-" * 80)
    
    # Make predictions
    print("   Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Percentage-based metrics
    percentage_errors = np.abs((y_test - y_pred) / y_test) * 100
    mean_percentage_error = np.mean(percentage_errors)
    median_percentage_error = np.median(percentage_errors)
    
    # Within-threshold accuracy
    within_10_pct = np.sum(percentage_errors <= 10) / len(percentage_errors) * 100
    within_20_pct = np.sum(percentage_errors <= 20) / len(percentage_errors) * 100
    within_30_pct = np.sum(percentage_errors <= 30) / len(percentage_errors) * 100
    
    # Print results
    print(f"\n   📊 PRIMARY METRICS:")
    print(f"      R² Score:                    {r2:.4f} ({r2*100:.2f}%)")
    print(f"      Mean Absolute Error (MAE):   {mae:.2f} kg/ha")
    print(f"      Root Mean Squared Error:     {rmse:.2f} kg/ha")
    
    print(f"\n   📈 ACCURACY STATISTICS:")
    print(f"      Mean Percentage Error:       {mean_percentage_error:.2f}%")
    print(f"      Median Percentage Error:     {median_percentage_error:.2f}%")
    
    print(f"\n   🎯 PREDICTION ACCURACY (Within Tolerance):")
    print(f"      Within 10% of actual:        {within_10_pct:.2f}%")
    print(f"      Within 20% of actual:        {within_20_pct:.2f}%")
    print(f"      Within 30% of actual:        {within_30_pct:.2f}%")
    
    print(f"\n   📋 PREDICTION STATISTICS:")
    print(f"      Actual YIELD range:          {y_test.min():.2f} - {y_test.max():.2f} kg/ha")
    print(f"      Predicted YIELD range:       {y_pred.min():.2f} - {y_pred.max():.2f} kg/ha")
    print(f"      Actual YIELD mean:           {y_test.mean():.2f} kg/ha")
    print(f"      Predicted YIELD mean:        {y_pred.mean():.2f} kg/ha")
    
    return {
        'name': model_name,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mean_pct_error': mean_percentage_error,
        'within_20_pct': within_20_pct
    }

def compare_models(results):
    """Print comparison table"""
    if len(results) < 2:
        return
    
    print("\n" + "=" * 80)
    print("🏆 MODEL COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Model':<30} {'R²':<10} {'MAE':<12} {'RMSE':<12} {'Within 20%':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<30} "
              f"{result['r2']:.4f}    "
              f"{result['mae']:>8.2f}    "
              f"{result['rmse']:>8.2f}    "
              f"{result['within_20_pct']:>8.2f}%")
    
    # Determine best model
    print("\n" + "-" * 80)
    best_r2 = max(results, key=lambda x: x['r2'])
    best_mae = min(results, key=lambda x: x['mae'])
    
    print(f"🥇 Best R² Score:    {best_r2['name']} ({best_r2['r2']:.4f})")
    print(f"🥇 Best MAE:         {best_mae['name']} ({best_mae['mae']:.2f} kg/ha)")

def main():
    """Main evaluation workflow"""
    print_banner()
    
    # Load test data
    df_test = load_test_data()
    X_test, y_test = prepare_test_data(df_test)
    
    # Load models
    print("\n3. Loading trained models...")
    
    rf_model = load_model('random-forest-regressor/models/crop_yield_model_rf_latest.joblib')
    xgb_model = load_model('XG/models/crop_yield_model_xgb_latest.joblib')
    
    if rf_model is None and xgb_model is None:
        print("\n❌ No models found! Please train models first.")
        print("\nFor Random Forest:")
        print("  cd random-forest-regressor/baseline/")
        print("  python train_modelv2.py")
        print("\nFor XGBoost:")
        print("  cd XG/")
        print("  python train_model_xgboost.py")
        sys.exit(1)
    
    # Evaluate models
    print("\n4. Evaluating models on test set...")
    results = []
    
    if rf_model is not None:
        result = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        results.append(result)
    
    if xgb_model is not None:
        result = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        results.append(result)
    
    # Compare models
    compare_models(results)
    
    # Final notes
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 80)
    print("\n📝 Notes:")
    print("   - These scores are on HELD-OUT test data (never seen during training)")
    print("   - Lower R² than training is EXPECTED and indicates proper evaluation")
    print("   - R² of 0.4-0.6 is typical for crop yield prediction")
    print("   - MAE shows average error in kg/ha")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Evaluation interrupted. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

