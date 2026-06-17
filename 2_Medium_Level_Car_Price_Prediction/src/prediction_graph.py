import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\outputs\prediction_results.csv"
)

plt.figure(figsize=(8,6))

plt.scatter(
    df.index,
    df["Actual_Price"],
    color="green",
    label="Actual Price",
    alpha=0.7
)

plt.scatter(
    df.index,
    df["Predicted_Price"],
    color="yellow",
    label="Predicted Price",                    
    alpha=0.7
)

plt.xlabel("Data Index")
plt.ylabel("Price")

plt.title("Actual vs Predicted Car Prices")

plt.savefig(
    r"E:\aadyamtalentconsultancy_Aashutosh\2_Medium_Level_Car_Price_Prediction\visualizations\actual_vs_predicted.png"
)

plt.show()
