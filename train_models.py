import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from sentiment_lexicon import SentimentLexiconFeatures

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

print("Starting model training process...")
print("Downloading NLTK data...")

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize

# ==== IMPROVED TEXT PREPROCESSING WITH NEGATION HANDLING ====

def handle_negations(text):
    """
    Mark negated words to help the model understand negations.
    Example: "not bad" -> "not bad_NEG"
    """
    # Create a list of negation words
    negation_words = ['not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 
                     'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t', 'isn\'t', 
                     'aren\'t', 'ain\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 
                     'hasn\'t', 'hadn\'t', 'won\'t', 'nor', 'neither']
    
    # Tokenize the text
    words = word_tokenize(text.lower())
    
    # Process negations
    in_negation = False
    result = []
    
    for word in words:
        if word in negation_words:
            in_negation = True
            result.append(word)
        elif word in ['.', '!', '?', ',', ';', ':', ')', ']']:
            # End negation scope at punctuation
            in_negation = False
            result.append(word)
        elif in_negation and word not in ['and', 'or', 'the', 'a', 'an', 'to', 'of', 'in']:
            # Mark negated content words
            result.append(word + '_NEG')
        else:
            result.append(word)
    
    return ' '.join(result)

def clean_text(text):
    """Enhanced text cleaning with negation handling"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep apostrophes for negations
    text = re.sub(r'[^\w\s\']', ' ', text)
    # Apply negation handling
    text = handle_negations(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==== END PREPROCESSING CODE ====

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

# Specialized datasets as before
print("Adding specialized negation examples...")
negation_examples = [
    {"text": "i don't think it was boring", "sentiment": 1},
    {"text": "i don't hate this movie", "sentiment": 1},
    {"text": "this movie wasn't bad at all", "sentiment": 1},
    {"text": "this wasn't as terrible as people say", "sentiment": 1},
    {"text": "not a bad film", "sentiment": 1},
    {"text": "not terrible", "sentiment": 1},
    {"text": "not the worst i've seen", "sentiment": 1},
    {"text": "didn't dislike it", "sentiment": 1},
    {"text": "isn't awful", "sentiment": 1},
    {"text": "can't complain about this movie", "sentiment": 1},
    {"text": "i don't think it was good", "sentiment": 0},
    {"text": "i don't like this movie", "sentiment": 0},
    {"text": "this movie wasn't great at all", "sentiment": 0},
    {"text": "this wasn't as good as people say", "sentiment": 0},
    {"text": "not a good film", "sentiment": 0},
    {"text": "not amazing", "sentiment": 0},
    {"text": "not the best i've seen", "sentiment": 0},
    {"text": "didn't enjoy it", "sentiment": 0},
    {"text": "isn't great", "sentiment": 0},
    {"text": "can't say i enjoyed this movie", "sentiment": 0},
]

print("Adding mental health/emotional content examples...")
emotional_examples = [
    {"text": "i want to hurt myself", "sentiment": 0},
    {"text": "i feel like killing myself", "sentiment": 0},
    {"text": "i am worthless", "sentiment": 0},
    {"text": "i hate myself", "sentiment": 0},
    {"text": "everything feels hopeless", "sentiment": 0},
    {"text": "i am so depressed", "sentiment": 0},
    {"text": "nobody cares about me", "sentiment": 0},
    {"text": "i feel so alone", "sentiment": 0},
    {"text": "i'm better off dead", "sentiment": 0},
    {"text": "i can't take it anymore", "sentiment": 0},
    {"text": "life is meaningless", "sentiment": 0},
    {"text": "no one would miss me", "sentiment": 0},
    {"text": "i'm a burden to everyone", "sentiment": 0},
    {"text": "i'm so anxious all the time", "sentiment": 0},
    {"text": "i'm a failure", "sentiment": 0},
]

print("Adding film terminology examples...")
film_examples = [
    {"text": "this film is so underrated", "sentiment": 1},
    {"text": "this is a cult classic", "sentiment": 1},
    {"text": "this movie is a hidden gem", "sentiment": 1},
    {"text": "king of comedy is brilliant", "sentiment": 1},
    {"text": "this comedy is hilarious", "sentiment": 1},
    {"text": "a thought-provoking film", "sentiment": 1},
    {"text": "this movie is overrated", "sentiment": 0},
    {"text": "this film is pretentious", "sentiment": 0},
    {"text": "the comedy falls flat", "sentiment": 0},
    {"text": "heavy-handed film", "sentiment": 0},
]

print("Adding high-confidence calibration examples...")
obvious_examples = [
    {"text": "this movie was amazing fantastic wonderful incredible brilliant loved it", "sentiment": 1},
    {"text": "best film ever seen perfect outstanding brilliant masterpiece", "sentiment": 1},
    {"text": "excellent superb magnificent outstanding remarkable phenomenal", "sentiment": 1},
    {"text": "i absolutely loved every second of this film", "sentiment": 1},
    {"text": "this movie brings me so much joy every time i watch it", "sentiment": 1},
    {"text": "one of the greatest films ever made without question", "sentiment": 1},
    {"text": "terrible awful horrible worst garbage waste of time", "sentiment": 0},
    {"text": "dreadful pathetic disappointing boring stupid terrible", "sentiment": 0},
    {"text": "hate disliked awful terrible horrible worst ever", "sentiment": 0},
    {"text": "i absolutely hated every second of this film", "sentiment": 0},
    {"text": "this movie was painful to watch and completely worthless", "sentiment": 0},
    {"text": "one of the worst films ever made without question", "sentiment": 0},
]

# Process all specialized examples with the new preprocessing
for examples in [negation_examples, emotional_examples, film_examples, obvious_examples]:
    for example in examples:
        example["text"] = clean_text(example["text"])

# Combine all specialized examples
specialized_examples = pd.DataFrame(negation_examples + emotional_examples + film_examples)

# Add specialized examples multiple times
for _ in range(10):
    df = pd.concat([df, specialized_examples], ignore_index=True)

# Add obvious examples even more times for confidence calibration
obvious_df = pd.DataFrame(obvious_examples)
for _ in range(20):
    df = pd.concat([df, obvious_df], ignore_index=True)

print(f"Final dataset size after adding specialized examples: {len(df)}")
print(f"Final positive examples: {sum(df['sentiment'])}")
print(f"Final negative examples: {len(df) - sum(df['sentiment'])}")

# Split text and labels
texts = df['text'].values
labels = df['sentiment'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Initialize sentiment lexicon features
print("Initializing sentiment lexicon features...")
lexicon = SentimentLexiconFeatures()

# Function to extract lexicon features
def extract_lexicon_features(texts):
    """Extract sentiment lexicon features for a list of texts"""
    features = []
    for text in texts:
        lexicon_features = lexicon.extract_all_features(text)
        features.append(lexicon_features)
    return features

# Extract lexicon features from training and testing data
print("Extracting lexicon features...")
X_train_lexicon = extract_lexicon_features(X_train)
X_test_lexicon = extract_lexicon_features(X_test)

# After extracting lexicon features, verify all values are non-negative
print("Verifying lexicon features are non-negative for MultinomialNB...")
for features_dict in X_train_lexicon:
    for key, value in list(features_dict.items()):
        if isinstance(value, (int, float)) and value < 0:
            # This shouldn't happen with our updated lexicon features,
            # but let's make absolutely sure
            features_dict[key] = 0.0

for features_dict in X_test_lexicon:
    for key, value in list(features_dict.items()):
        if isinstance(value, (int, float)) and value < 0:
            features_dict[key] = 0.0

# Create a DictVectorizer to transform lexicon features
dict_vectorizer = DictVectorizer()
X_train_lexicon_vec = dict_vectorizer.fit_transform(X_train_lexicon)
X_test_lexicon_vec = dict_vectorizer.transform(X_test_lexicon)

# Verify there are no negative values in the lexicon features
if X_train_lexicon_vec.data.min() < 0:
    print("Warning: Negative values found in lexicon features, setting them to 0...")
    X_train_lexicon_vec.data[X_train_lexicon_vec.data < 0] = 0.0
    
if X_test_lexicon_vec.data.min() < 0:
    X_test_lexicon_vec.data[X_test_lexicon_vec.data < 0] = 0.0

# Create feature extractors
print("Creating feature extractors...")
stop_words = 'english'

# CountVectorizer with improved parameters
count_vectorizer = CountVectorizer(
    max_features=15000,  # Increased from 10000 to capture more negated terms
    min_df=2,  # Reduced from 3 to include more rare negated features
    max_df=0.9,
    ngram_range=(1, 3),  # Increased from (1,2) to capture negation phrases
    stop_words=stop_words,
    strip_accents='unicode'
)

# TfidfVectorizer with improved parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=15000,  # Increased to match count_vectorizer
    min_df=2,  # Reduced to match count_vectorizer
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

# Combine features for Naive Bayes
X_train_combined_nb = hstack([X_train_counts, X_train_lexicon_vec])
X_test_combined_nb = hstack([X_test_counts, X_test_lexicon_vec])

# Combine features for Logistic Regression
X_train_combined_lr = hstack([X_train_tfidf, X_train_lexicon_vec])
X_test_combined_lr = hstack([X_test_tfidf, X_test_lexicon_vec])

# Make sure there are no negative values in the NB training data
if X_train_combined_nb.data.min() < 0:
    print("Warning: Negative values found in combined NB features, setting them to 0...")
    X_train_combined_nb.data[X_train_combined_nb.data < 0] = 0.0

if X_test_combined_nb.data.min() < 0:
    X_test_combined_nb.data[X_test_combined_nb.data < 0] = 0.0

# Train Naive Bayes model with improved hyperparameters
print("Training Naive Bayes model...")
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_combined_nb, y_train)

# Train Logistic Regression model with improved hyperparameters
print("Training Logistic Regression model...")
lr_model = LogisticRegression(
    C=5.0,
    max_iter=1000,
    class_weight='balanced',
    solver='liblinear'
)
lr_model.fit(X_train_combined_lr, y_train)

# Evaluate models
nb_predictions = nb_model.predict(X_test_combined_nb)
lr_predictions = lr_model.predict(X_test_combined_lr)

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

# Test the models on specific challenging examples
print("\n--- Testing on Challenging Examples ---")
challenge_examples = [
    "I don't think it was boring",
    "King of comedy is so underrated",
    "I want to hurt myself",
    "I loved this movie, it was awesome!",
    "This was the worst film I've ever seen, terrible acting."
]

# Process the challenge examples with negation handling
challenge_examples_processed = [clean_text(text) for text in challenge_examples]

print("Original vs Processed examples:")
for orig, proc in zip(challenge_examples, challenge_examples_processed):
    print(f"Original: \"{orig}\"")
    print(f"Processed: \"{proc}\"")
    print()

# Extract lexicon features for challenge examples
challenge_lexicon = extract_lexicon_features(challenge_examples_processed)
challenge_lexicon_vec = dict_vectorizer.transform(challenge_lexicon)

# Make sure there are no negative values in the challenge features
if challenge_lexicon_vec.data.min() < 0:
    challenge_lexicon_vec.data[challenge_lexicon_vec.data < 0] = 0.0

print("Naive Bayes predictions:")
X_challenge_counts = count_vectorizer.transform(challenge_examples_processed)
X_challenge_combined_nb = hstack([X_challenge_counts, challenge_lexicon_vec])

# Make sure there are no negative values in the challenge combined features
if X_challenge_combined_nb.data.min() < 0:
    X_challenge_combined_nb.data[X_challenge_combined_nb.data < 0] = 0.0

for i, example in enumerate(challenge_examples):
    prediction = nb_model.predict(X_challenge_combined_nb[i:i+1])[0]
    proba = nb_model.predict_proba(X_challenge_combined_nb[i:i+1])[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f'"{example}" => {sentiment} ({confidence*100:.2f}% confidence)')

print("\nLogistic Regression predictions:")
X_challenge_tfidf = tfidf_vectorizer.transform(challenge_examples_processed)
X_challenge_combined_lr = hstack([X_challenge_tfidf, challenge_lexicon_vec])
for i, example in enumerate(challenge_examples):
    prediction = lr_model.predict(X_challenge_combined_lr[i:i+1])[0]
    proba = lr_model.predict_proba(X_challenge_combined_lr[i:i+1])[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f'"{example}" => {sentiment} ({confidence*100:.2f}% confidence)')

# Save models and vectorizers as tuples
print("\nSaving models to disk...")
with open('models/naive_bayes.pkl', 'wb') as f:
    pickle.dump((nb_model, count_vectorizer, dict_vectorizer), f)
    
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump((lr_model, tfidf_vectorizer, dict_vectorizer), f)

print("Models trained and saved successfully!")
print("\nYou can now run the Flask application with 'python app.py'")