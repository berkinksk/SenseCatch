import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("Starting model training process...")
print("Downloading NLTK data...")

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('stopwords')
from nltk.corpus import movie_reviews, stopwords

# Function for text cleaning
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preparing data from NLTK movie reviews...")

# Prepare data from NLTK movie reviews with better cleaning
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        text = ' '.join(movie_reviews.words(fileid))
        cleaned_text = clean_text(text)
        documents.append({
            'text': cleaned_text,
            'sentiment': 1 if category == 'pos' else 0
        })

# Convert to DataFrame
df = pd.DataFrame(documents)

# Display dataset info
print(f"Dataset loaded: {len(df)} reviews")
print(f"Positive reviews: {sum(df['sentiment'])}")
print(f"Negative reviews: {len(df) - sum(df['sentiment'])}")

# Add some extremely positive and negative samples to improve confidence on obvious cases
obvious_samples = [
    {'text': 'this movie was amazing fantastic wonderful incredible brilliant loved it', 'sentiment': 1},
    {'text': 'best film ever seen perfect outstanding brilliant masterpiece', 'sentiment': 1},
    {'text': 'excellent superb magnificent outstanding remarkable phenomenal', 'sentiment': 1},
    {'text': 'terrible awful horrible worst garbage waste of time', 'sentiment': 0},
    {'text': 'dreadful pathetic disappointing boring stupid terrible', 'sentiment': 0},
    {'text': 'hate disliked awful terrible horrible worst ever', 'sentiment': 0}
]

# Add the obvious samples multiple times to increase their weight
for _ in range(10):
    df = pd.concat([df, pd.DataFrame(obvious_samples)], ignore_index=True)

# Split text and labels
texts = df['text'].values
labels = df['sentiment'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Create feature extractors
print("Creating feature extractors...")
stop_words = set(stopwords.words('english'))

# CountVectorizer with improved parameters
count_vectorizer = CountVectorizer(
    max_features=10000, 
    min_df=3, 
    max_df=0.9,
    ngram_range=(1, 2),  # Include unigrams and bigrams
    stop_words=stop_words,
    strip_accents='unicode'
)

# TfidfVectorizer with improved parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000, 
    min_df=3, 
    max_df=0.9,
    ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
    stop_words=stop_words,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
)

# Transform text data
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"CountVectorizer vocabulary size: {len(count_vectorizer.vocabulary_)}")
print(f"TfidfVectorizer vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# Train Naive Bayes model with improved hyperparameters
print("Training Naive Bayes model...")
nb_model = MultinomialNB(alpha=0.1)  # Slightly reduce smoothing
nb_model.fit(X_train_counts, y_train)

# Train Logistic Regression model with improved hyperparameters
print("Training Logistic Regression model...")
lr_model = LogisticRegression(
    C=5.0,  # Increase regularization strength
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    solver='liblinear'
)
lr_model.fit(X_train_tfidf, y_train)

# Evaluate models
nb_predictions = nb_model.predict(X_test_counts)
lr_predictions = lr_model.predict(X_test_tfidf)

print("\n--- Model Evaluation ---")
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))
print("\nNaive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

# Test the models on some obvious examples
test_examples = [
    "I loved this movie, it was awesome!",
    "This was the worst film I've ever seen, terrible acting.",
    "The movie was okay, nothing special."
]

print("\n--- Testing on Example Sentences ---")
print("Naive Bayes predictions:")
X_test_examples_counts = count_vectorizer.transform(test_examples)
for i, example in enumerate(test_examples):
    prediction = nb_model.predict(X_test_examples_counts[i:i+1])[0]
    proba = nb_model.predict_proba(X_test_examples_counts[i:i+1])[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f'"{example}" => {sentiment} ({confidence*100:.2f}% confidence)')

print("\nLogistic Regression predictions:")
X_test_examples_tfidf = tfidf_vectorizer.transform(test_examples)
for i, example in enumerate(test_examples):
    prediction = lr_model.predict(X_test_examples_tfidf[i:i+1])[0]
    proba = lr_model.predict_proba(X_test_examples_tfidf[i:i+1])[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f'"{example}" => {sentiment} ({confidence*100:.2f}% confidence)')

# Save models and vectorizers as tuples
print("\nSaving models to disk...")
with open('models/naive_bayes.pkl', 'wb') as f:
    pickle.dump((nb_model, count_vectorizer), f)
    
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump((lr_model, tfidf_vectorizer), f)

print("Models trained and saved successfully!")
print("\nYou can now run the Flask application with 'python app.py'")