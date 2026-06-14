import pandas as pd

# Dataset load karo
file_path = r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\dataset\House_Rent_Dataset.csv"

df = pd.read_csv(file_path)

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

# Shape
print(f"\nRows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Columns
print("\nColumns:")
for col in df.columns:
    print(f"- {col}")

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nNumerical Summary:")
print(df.describe())

print("\n" + "="*60)
print("CHECK COMPLETED")
print("="*60)