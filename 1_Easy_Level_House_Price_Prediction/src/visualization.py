import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\outputs\cleaned_data.csv"
)

# Rent Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Rent"], bins=30)
plt.title("Rent Distribution")
plt.xlabel("Rent")
plt.ylabel("Count")

plt.savefig(
    r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\visualizations\rent_distribution.png"
)

plt.show()