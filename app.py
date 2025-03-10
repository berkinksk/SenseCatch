from flask import Flask, request, jsonify, render_template
import pickle
import re
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

app = Flask(__name__)

# Load models
models = {}
vectorizers = {}
model_paths = {
    'naive_bayes': 'models/naive_bayes.pkl',
    'logistic_regression': 'models/logistic_regression.pkl'
}

# Load all available models
for model_name, path in model_paths.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            model, vectorizer = pickle.load(f)
            models[model_name] = model
            vectorizers[model_name] = vectorizer
            print(f"Loaded model: {model_name}")
    else:
        print(f"Warning: Model {model_name} not found at {path}")

# Function to clean text
def clean_text(text):
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to get important words with their weights
def get_important_words(text, model_name, prediction, vectorizer, model):
    # Get feature names based on vectorizer type
    if isinstance(vectorizer, CountVectorizer) or isinstance(vectorizer, TfidfVectorizer):
        feature_names = vectorizer.get_feature_names_out()
    else:
        return []  # Return empty list if unknown vectorizer type
    
    # Transform the text
    X = vectorizer.transform([text])
    
    # Get feature importance based on model type
    if hasattr(model, 'coef_'):  # For logistic regression
        # For binary classification, get weights for the positive class
        coefficients = model.coef_[0]
        # Sort features by importance for the predicted class
        importance = coefficients if prediction == 1 else -coefficients
        
    elif hasattr(model, 'feature_log_prob_'):  # For Naive Bayes
        # Calculate log probability differences between positive and negative classes
        importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        if prediction == 0:  # For negative predictions, reverse importance
            importance = -importance
    else:
        return []  # Unsupported model type
    
    # Get non-zero features in the input text
    non_zero_features = X.nonzero()[1]
    
    # Get the importance scores for words in the text
    word_importance = [(feature_names[i], importance[i]) for i in non_zero_features]
    
    # Sort by absolute importance and take top 5
    word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    top_words = word_importance[:5]
    
    # Format the response
    result = []
    for word, score in top_words:
        sentiment = "positive" if score > 0 else "negative"
        # Scale the importance to a percentage
        importance_score = min(abs(score) * 10, 100)  # Scaling factor
        result.append({
            "word": word,
            "importance": float(importance_score),
            "sentiment": sentiment
        })
    
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get data from request
    data = request.get_json()
    text = data.get('text', '')
    model_type = data.get('model', 'naive_bayes')
    
    # Check if the selected model is available
    if model_type not in models:
        return jsonify({
            'error': f'Model {model_type} is not available. Please run train_models.py first.'
        }), 404
    
    # Get the model and vectorizer
    model = models[model_type]
    vectorizer = vectorizers[model_type]
    
    # Clean and process text
    cleaned_text = clean_text(text)
    
    # Vectorize the text
    X = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(X)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    # Get prediction probability
    try:
        proba = model.predict_proba(X)[0]
        confidence = proba[1] * 100 if prediction == 1 else proba[0] * 100
    except AttributeError:
        # If model doesn't support predict_proba
        confidence = 85.0  # Fallback confidence
    
    # Get important words
    important_words = get_important_words(cleaned_text, model_type, prediction, vectorizer, model)
    
    # Return prediction
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': round(confidence, 2),
        'model': model_type,
        'important_words': important_words
    })

if __name__ == '__main__':
    app.run(debug=True)