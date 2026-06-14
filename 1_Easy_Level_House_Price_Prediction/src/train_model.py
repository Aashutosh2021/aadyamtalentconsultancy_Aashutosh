import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load cleaned dataset
df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\outputs\cleaned_data.csv"
)

# Features (Input)
X = df.drop("Rent", axis=1)

# Target (Output)
y = df["Rent"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\nModel Performance")
print("-" * 40)
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 2))

# Sample predictions
print("\nSample Predictions")
print("-" * 40)

for i in range(10):
    print(
        f"Actual Rent: {y_test.iloc[i]} | Predicted Rent: {round(predictions[i],2)}"
    )

model_path = r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\models\house_rent_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel Saved Successfully!")
print(model_path)    

# Load original dataset for readable prediction report
original_df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\dataset\House_Rent_Dataset.csv"
)

# Same split as model training
_, original_test = train_test_split(
    original_df,
    test_size=0.2,
    random_state=42
)

# Reset indexes
original_test = original_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Create prediction report
results_df = pd.DataFrame({
    "BHK": original_test["BHK"],
    "Size": original_test["Size"],
    "Bathroom": original_test["Bathroom"],
    "City": original_test["City"],
    "Furnishing_Status": original_test["Furnishing Status"],
    "Tenant_Preferred": original_test["Tenant Preferred"],
    "Area_Type": original_test["Area Type"],
    "Actual_Rent": y_test,
    "Predicted_Rent": predictions.round(2)
})

# Prediction error
results_df["Difference"] = (
    results_df["Actual_Rent"] -
    results_df["Predicted_Rent"]
).round(2)

# Save file
results_path = r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\outputs\prediction_results.csv"

results_df.to_csv(results_path, index=False)

print("\nPrediction Results Saved Successfully!")
print(results_path)

print("\nTop 10 Predictions:")
print(results_df.head(10))