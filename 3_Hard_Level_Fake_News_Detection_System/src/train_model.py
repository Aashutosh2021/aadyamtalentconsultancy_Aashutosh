import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\outputs\cleaned_news_dataset.csv"
)

# Combine title + text
df["content"] = df["title"] + " " + df["text"]

X = df["content"]
y = df["label"]

# ===========================
# Train Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ===========================
# TF-IDF
# ===========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ===========================
# Model
# ===========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ===========================
# Prediction
# ===========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nModel Performance")
print("-" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ===========================
# Save Model
# ===========================

joblib.dump(
    model,
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\models\fake_news_model.pkl"
)

joblib.dump(
    vectorizer,
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\models\tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully!")

# ===========================
# Prediction Results
# ===========================

results_df = pd.DataFrame({
    "News": X_test,
    "Actual": y_test.values,
    "Predicted": predictions
})

results_df["Actual"] = results_df["Actual"].map({
    0: "Fake",
    1: "Real"
})

results_df["Predicted"] = results_df["Predicted"].map({
    0: "Fake",
    1: "Real"
})

results_df.to_csv(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\outputs\prediction_results.csv",
    index=False
)

print("\nPrediction Results Saved!")

# ===========================
# Classification Report
# ===========================

with open(
    r"E:\aadyamtalentconsultancy_Aashutosh\3_Hard_Level_Fake_News_Detection_System\reports\classification_report.txt",
    "w"
) as f:

    f.write(classification_report(y_test, predictions))

print("Classification Report Saved!")

# ===========================
# Confusion Matrix
# ===========================

print("\nConfusion Matrix")

print(confusion_matrix(y_test, predictions))