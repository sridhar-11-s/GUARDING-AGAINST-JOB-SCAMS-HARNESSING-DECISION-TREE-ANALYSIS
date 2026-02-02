# Fake_job_predictor.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("fake_job_postings.csv")
data = data.fillna("")

# -----------------------------
# Combine text columns
# -----------------------------
data["text"] = (
    data["title"] + " " +
    data["company_profile"] + " " +
    data["description"] + " " +
    data["requirements"] + " " +
    data["benefits"]
)

X = data["text"]
y = data["fraudulent"]   # 0 = Real, 1 = Fake

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = MultinomialNB(class_prior=[0.5, 0.5])
model.fit(X_train_vec, y_train)

# (If you used Decision Tree instead, use this)
# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train_vec, y_train)

# -----------------------------
# Save model & vectorizer
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model training complete")
print("✅ model.pkl and vectorizer.pkl saved")
