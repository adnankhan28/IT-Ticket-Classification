import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset (replace with your actual CSV filename if different)
df = pd.read_csv("IT_Service_Tickets.csv")

# Display first few records
print("First 5 records:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# Drop empty or null values
df.dropna(inplace=True)
df = df[df['Document'].str.strip() != '']  # Remove blank ticket descriptions

# Visualize the number of samples per ticket category
plt.figure(figsize=(10, 6))
sns.countplot(y=df['Topic_group'], order=df['Topic_group'].value_counts().index)
plt.title("Ticket Category Distribution")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# Split data into input (X) and label (y)
X = df['Document']
y = df['Topic_group']

# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Classification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save trained model and vectorizer
joblib.dump(model, "ticket_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and Vectorizer saved successfully.")
