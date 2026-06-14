import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\dataset\House_Rent_Dataset.csv"
)

print("Original Shape:", df.shape)

# Select useful columns
df = df[
    [
        "BHK",
        "Size",
        "Bathroom",
        "City",
        "Furnishing Status",
        "Tenant Preferred",
        "Area Type",
        "Rent",
    ]
]

# Convert text columns to numbers
encoder = LabelEncoder()

categorical_columns = [
    "City",
    "Furnishing Status",
    "Tenant Preferred",
    "Area Type",
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nProcessed Dataset:")
print(df.head())

# Save cleaned dataset
output_path = r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\outputs\cleaned_data.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Location:", output_path)