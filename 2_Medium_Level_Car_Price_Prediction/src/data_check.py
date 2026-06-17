import pandas as pd

file_path = r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\dataset\car_price_dataset.csv"

df = pd.read_csv(file_path)

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())