from flask import Flask, request, jsonify, render_template
import pickle
import re
import os
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Download NLTK data if needed
nltk.download('punkt')

# Load classifier
classifier = None
model_path = 'models/naive_bayes_nltk.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)

# Function to extract features from text
def extract_features(text):
    words = word_tokenize(text.lower())
    return {word: True for word in words}

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get text from request
    data = request.get_json()
    text = data['text']
    model_type = data['model']
    
    # Check if model is loaded
    if classifier is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_models.py first.'
        }), 500
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Extract features
    features = extract_features(cleaned_text)
    
    # Make prediction
    sentiment = classifier.classify(features)
    dist = classifier.prob_classify(features)
    confidence = dist.prob(sentiment) * 100
    
    # Get important words
    important_words = []
    informative_words = classifier.most_informative_features(10)
    for word, _ in informative_words:
        if word in features:
            importance = dist.prob('positive') if sentiment == 'positive' else dist.prob('negative')
            important_words.append({
                "word": word, 
                "importance": float(importance), 
                "sentiment": sentiment
            })
    
    # Return prediction
    return jsonify({
        'text': text,
        'sentiment': sentiment.capitalize(),
        'confidence': round(confidence, 2),
        'model': model_type,
        'important_words': important_words[:5]
    })

if __name__ == '__main__':
    app.run(debug=True)