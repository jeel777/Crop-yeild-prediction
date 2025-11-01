import pandas as pd
import numpy as np

# Load the full dataset
print("Loading dataset...")
df = pd.read_csv("ICRISAT_Dataset.csv")
print(f"Total rows in dataset: {len(df)}")

# Sample 11,000 random rows
n_samples = 11000
if len(df) < n_samples:
    print(f"Warning: Dataset has only {len(df)} rows, which is less than {n_samples}.")
    print(f"Using all {len(df)} rows instead.")
    test_df = df.copy()
else:
    test_df = df.sample(n=n_samples, random_state=42)
    print(f"Sampled {n_samples} random rows.")

# Save to testdata.csv
output_filename = "testdata.csv"
test_df.to_csv(output_filename, index=False)
print(f"\nTest data saved successfully as: {output_filename}")
print(f"Shape: {test_df.shape}")

