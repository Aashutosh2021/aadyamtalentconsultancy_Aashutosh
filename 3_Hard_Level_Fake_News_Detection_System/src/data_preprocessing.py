import pandas as pd

# Load datasets
fake_df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\dataset\Fake.csv"
)

true_df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\dataset\True.csv"
)

print("Original Shape")
print("Fake :", fake_df.shape)
print("True :", true_df.shape)

# Remove duplicates
fake_df = fake_df.drop_duplicates()
true_df = true_df.drop_duplicates()

# Label datasets
fake_df["label"] = 0
true_df["label"] = 1

# Merge
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep required columns
df = df[["title", "text", "subject", "label"]]

print("\nProcessed Shape :", df.shape)

print("\nClass Distribution")
print(df["label"].value_counts())

# Save
output_path = r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\outputs\cleaned_news_dataset.csv"

df.to_csv(output_path, index=False)

print("\nSaved Successfully!")
print(output_path)