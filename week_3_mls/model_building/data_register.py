import pandas as pd

RAW_PATH = "week_3_mls/data/machine-failure-prediction.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "UDI", "Type", "Air temperature", "Process temperature",
    "Rotational speed", "Torque", "Tool wear", "Failure",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Failure distribution:")
print(df["Failure"].value_counts())
