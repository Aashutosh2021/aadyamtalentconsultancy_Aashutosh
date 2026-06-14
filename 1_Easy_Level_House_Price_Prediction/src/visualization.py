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

plt.figure(figsize=(8,6))
plt.scatter(df["Size"], df["Rent"])

plt.title("Size vs Rent")
plt.xlabel("Size (sq ft)")
plt.ylabel("Rent")

plt.savefig(
    r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\visualizations\size_vs_rent.png"
)

plt.figure(figsize=(8,6))
plt.scatter(df["Bathroom"], df["Rent"])

plt.title("Bathroom vs Rent")
plt.xlabel("Bathroom")
plt.ylabel("Rent")

plt.savefig(
    r"E:\aadyamtalentconsultancy_Aashutosh\1_Easy_Level_House_Price_Prediction\visualizations\bathroom_vs_rent.png"
)




plt.show()  