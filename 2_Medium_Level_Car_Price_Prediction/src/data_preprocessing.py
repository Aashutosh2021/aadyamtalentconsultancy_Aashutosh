import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\dataset\car_price_dataset.csv"
)

print("Original Shape:", df.shape)

# Remove duplicates
df = df.drop_duplicates()

# Drop unnecessary column
df = df.drop(columns=["name"])

# Fill missing values
df["mileage(km/ltr/kg)"] = df["mileage(km/ltr/kg)"].fillna(
    df["mileage(km/ltr/kg)"].median()
)

df["engine"] = df["engine"].fillna(
    df["engine"].median()
)

df["max_power"] = pd.to_numeric(
    df["max_power"],
    errors="coerce"
)

df["max_power"] = df["max_power"].fillna(
    df["max_power"].median()
)

df["seats"] = df["seats"].fillna(
    df["seats"].median()
)

# Encode categorical columns
encoder = LabelEncoder()

categorical_columns = [
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nProcessed Shape:", df.shape)

# Save cleaned dataset
output_path = r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\outputs\cleaned_car_data.csv"

df.to_csv(output_path, index=False)

print("Cleaned dataset saved successfully!")