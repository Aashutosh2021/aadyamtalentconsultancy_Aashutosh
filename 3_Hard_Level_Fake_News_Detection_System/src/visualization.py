import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\outputs\cleaned_news_dataset.csv"
)

df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = joblib.load(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\models\tfidf_vectorizer.pkl"
)

model = joblib.load(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\models\fake_news_model.pkl"
)

X_test = vectorizer.transform(X_test)

predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake","Real"],
    yticklabels=["Fake","Real"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\visualizations\confusion_matrix.png"
)

plt.figure(figsize=(6,4))

df["label"].value_counts().plot(kind="bar")

plt.xticks([0,1],["Fake","Real"])

plt.title("Class Distribution")

plt.savefig(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\visualizations\class_distribution.png"
)

plt.show()