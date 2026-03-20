"""
Step 0: Train and save the spam classifier.
Run this ONCE before any of the web apps.

    python train_model.py

Produces: spam_model.pkl (the pipeline) and a quick accuracy report.
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

# Binary classification: "rec.*" = ham, "talk.*" = spam-like
categories = ["rec.sport.baseball", "rec.autos", "talk.politics.guns", "talk.religion.misc"]
data = fetch_20newsgroups(subset="all", categories=categories, random_state=42)

# Relabel: rec.* → "Not Spam" (0), talk.* → "Spam" (1)
labels = [1 if data.target_names[t].startswith("talk") else 0 for t in data.target]
label_names = {0: "Not Spam", 1: "Spam"}

X_train, X_test, y_train, y_test = train_test_split(
    data.data, labels, test_size=0.2, random_state=42)

pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000, stop_words="english"),
    MultinomialNB()
)
pipeline.fit(X_train, y_train)

acc = pipeline.score(X_test, y_test)
print(f"Model accuracy: {acc:.1%}")
print(f"Saved to spam_model.pkl")

joblib.dump(pipeline, "spam_model.pkl")
