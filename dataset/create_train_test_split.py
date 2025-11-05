"""
Create a proper train-test split to avoid data leakage.

This script splits the full dataset into:
- Training set (80%): Used ONLY for training the model
- Test set (20%): Used ONLY for final evaluation (never seen during training)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 70)
print("Creating Proper Train-Test Split")
print("=" * 70)

# Load the full dataset
print("\n1. Loading full dataset...")
df = pd.read_csv("dataset/data_long_format_v7_full.csv")
print(f"   Total rows: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")

# Check for missing yields
missing_yield = df['YIELD'].isnull().sum()
print(f"\n2. Data quality check:")
print(f"   Rows with missing YIELD: {missing_yield:,}")

# Remove rows with missing YIELD values (can't train/test on these)
if missing_yield > 0:
    print(f"   Dropping rows with missing YIELD...")
    df = df.dropna(subset=['YIELD'])
    print(f"   Remaining rows: {len(df):,}")

# Split the data: 80% training, 20% testing
# random_state=42 ensures reproducibility
print("\n3. Splitting data...")
print("   Strategy: Random 80/20 split (stratified would be better but complex)")

train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    shuffle=True  # Ensure random distribution
)

print(f"   Training set: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Test set:     {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")

# Save the splits
print("\n4. Saving splits to disk...")
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("   ✅ train_data.csv saved")
print("   ✅ test_data.csv saved")

# Show some statistics
print("\n5. Dataset statistics:")
print(f"\n   Training set:")
print(f"   - Years: {train_df['Year'].min()} to {train_df['Year'].max()}")
print(f"   - Unique crops: {train_df['Crop'].nunique()}")
print(f"   - Unique districts: {train_df['Dist_Name'].nunique()}")
print(f"   - YIELD range: {train_df['YIELD'].min():.2f} to {train_df['YIELD'].max():.2f}")

print(f"\n   Test set:")
print(f"   - Years: {test_df['Year'].min()} to {test_df['Year'].max()}")
print(f"   - Unique crops: {test_df['Crop'].nunique()}")
print(f"   - Unique districts: {test_df['Dist_Name'].nunique()}")
print(f"   - YIELD range: {test_df['YIELD'].min():.2f} to {test_df['YIELD'].max():.2f}")

print("\n" + "=" * 70)
print("✅ SPLIT COMPLETE!")
print("=" * 70)
print("\n⚠️  IMPORTANT:")
print("   - Use train_data.csv for ALL model training")
print("   - Use test_data.csv ONLY for final evaluation")
print("   - NEVER mix or swap these datasets")
print("=" * 70)

