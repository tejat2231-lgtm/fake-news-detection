import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Manual dataset create cheyyadam
data = {
    "text": [
        "India won the cricket world cup",
        "Aliens landed in Hyderabad yesterday",
        "Government launches new education policy",
        "Actor is 300 years old",
        "Scientists discovered new planet",
        "Man claims he can fly without wings"
    ],
    "label": [
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake"
    ]
}

df = pd.DataFrame(data)

print("\n--- Dataset Preview ---")
print(df)

# Features and labels
X = df["text"]
y = df["label"]

# Convert text to numerical format
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42
)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# User input test
user_news = input("\nEnter news text: ")
user_vector = vectorizer.transform([user_news])
prediction = model.predict(user_vector)

print("\nPrediction:", prediction[0])
