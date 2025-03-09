import nltk
import pickle
import random
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

# Create models directory if it doesn't exist
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

# Prepare data from movie reviews
print("Preparing data...")
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append({
            'text': movie_reviews.raw(fileid),
            'sentiment': 'positive' if category == 'pos' else 'negative'
        })

# Shuffle documents
random.shuffle(documents)

# Function to extract features from text
def extract_features(text):
    words = word_tokenize(text.lower())
    return {word: True for word in words}

# Prepare training and testing sets
print("Creating feature sets...")
featuresets = [(extract_features(doc['text']), doc['sentiment']) for doc in documents]
train_set, test_set = featuresets[:1600], featuresets[1600:]

# Train Naive Bayes classifier
print("Training Naive Bayes classifier...")
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy*100:.2f}%")

# Show most informative features
print("Most Informative Features:")
classifier.show_most_informative_features(20)

# Save the classifier
print("Saving model...")
with open('models/naive_bayes_nltk.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("Model trained and saved successfully!")