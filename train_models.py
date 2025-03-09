import pandas as pd
import numpy as np
import pickle
import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("Starting model training process...")
print("Downloading NLTK data...")

# Download required NLTK data
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

print("Preparing data from NLTK movie reviews...")

# Prepare data from NLTK movie reviews
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append({
            'text': ' '.join(movie_reviews.words(fileid)),
            'sentiment': 1 if category == 'pos' else 0
        })

# Convert to DataFrame
df = pd.DataFrame(documents)

# Display dataset info
print(f"Dataset loaded: {len(df)} reviews")
print(f"Positive reviews: {sum(df['sentiment'])}")
print(f"Negative reviews: {len(df) - sum(df['sentiment'])}")

# Split text and labels
texts = df['text'].values
labels = df['sentiment'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create feature extractors
print("Creating feature extractors...")
count_vectorizer = CountVectorizer(max_features=5000, min_df=5, stop_words='english')
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, stop_words='english')

# Transform text data to numerical features
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"Feature extraction complete. Vocabulary size: {len(count_vectorizer.vocabulary_)}")

# Train Naive Bayes model
print("Training Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_counts, y_train)

# Train Logistic Regression model
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, C=1.0)
lr_model.fit(X_train_tfidf, y_train)

# Evaluate models
nb_predictions = nb_model.predict(X_test_counts)
lr_predictions = lr_model.predict(X_test_tfidf)

print("\n--- Model Evaluation ---")
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))

# Save models and vectorizers
print("\nSaving models to disk...")
with open('models/naive_bayes.pkl', 'wb') as f:
    pickle.dump((nb_model, count_vectorizer), f)
    
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump((lr_model, tfidf_vectorizer), f)

print("Models trained and saved successfully!")
print("\nYou can now run the Flask application with 'python app.py'")