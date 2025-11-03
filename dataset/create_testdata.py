import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv("dataset/data_long_format_v7_full.csv")
print(f"Total rows in dataset: {len(df)}")

n_samples = 11000
if len(df) < n_samples:
    print(f"Warning: Dataset has only {len(df)} rows, which is less than {n_samples}.")
    print(f"Using all {len(df)} rows instead.")
    test_df = df.copy()
else:
    test_df = df.sample(n=n_samples, random_state=42)
    print(f"Sampled {n_samples} random rows.")

# Save to testdata.csv
output_filename = "testdatav2.csv"
test_df.to_csv(output_filename, index=False)
print(f"\nTest data saved successfully as: {output_filename}")
print(f"Shape: {test_df.shape}")

