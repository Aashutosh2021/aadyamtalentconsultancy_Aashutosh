import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load(
    r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\models\car_price_model.pkl"
)

df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\outputs\cleaned_car_data.csv"
)

X = df.drop("selling_price", axis=1)

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))
plt.bar(feature_df["Feature"], feature_df["Importance"])

plt.xticks(rotation=45)

plt.title("Feature Importance")
plt.tight_layout()

plt.savefig(
    r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\visualizations\feature_importance.png"
)

plt.show()