import pandas as pd

# Load datasets
fake_df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\dataset\Fake.csv"
)

true_df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\dataset\True.csv"
)

print("="*60)
print("FAKE NEWS DATASET")
print("="*60)

print("\nFake Dataset Shape :", fake_df.shape)
print("True Dataset Shape :", true_df.shape)

print("\nFake Columns:")
print(fake_df.columns.tolist())

print("\nTrue Columns:")
print(true_df.columns.tolist())

print("\nMissing Values (Fake)")
print(fake_df.isnull().sum())

print("\nMissing Values (True)")
print(true_df.isnull().sum())

print("\nDuplicate Fake:", fake_df.duplicated().sum())
print("Duplicate True:", true_df.duplicated().sum())

print("\nFake Sample")
print(fake_df.head())

print("\nTrue Sample")
print(true_df.head())