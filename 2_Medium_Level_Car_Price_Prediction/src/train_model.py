import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Load cleaned dataset
df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\outputs\cleaned_car_data.csv"
)

# Features
X = df.drop("selling_price", axis=1)

# Target
y = df["selling_price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

print("\nModel Performance")
print("-" * 40)

print("R2 Score :", round(r2, 4))
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))

# Save Model
model_path = r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\models\car_price_model.pkl"

joblib.dump(model, model_path)

print("\nModel Saved Successfully!")
print(model_path)

# Prediction Results
results_df = X_test.copy()

results_df["Actual_Price"] = y_test.values
results_df["Predicted_Price"] = predictions.round(2)

results_df["Difference"] = (
    results_df["Actual_Price"] -
    results_df["Predicted_Price"]
).round(2)

results_df.to_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\outputs\prediction_results.csv",
    index=False
)

print("\nPrediction Results Saved!")
print(results_df.head(10))