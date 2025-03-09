from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
import os

app = Flask(__name__)

# Check if model files exist before loading
model_files = {
    'naive_bayes': 'models/naive_bayes.pkl',
    'logistic_regression': 'models/logistic_regression.pkl'
}

# Initialize models as None
nb_model, count_vectorizer = None, None
lr_model, tfidf_vectorizer = None, None

# Load models if they exist
if os.path.exists(model_files['naive_bayes']):
    with open(model_files['naive_bayes'], 'rb') as f:
        nb_model, count_vectorizer = pickle.load(f)
    
if os.path.exists(model_files['logistic_regression']):
    with open(model_files['logistic_regression'], 'rb') as f:
        lr_model, tfidf_vectorizer = pickle.load(f)

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
    
    # Check if models are loaded
    if nb_model is None or lr_model is None:
        return jsonify({
            'error': 'Models not loaded. Please run train_models.py first.'
        }), 500
    
    # Clean text
    cleaned_text = clean_text(text)
    
    # Make prediction based on model type
    if model_type == 'naive_bayes':
        features = count_vectorizer.transform([cleaned_text])
        prediction = nb_model.predict(features)[0]
        confidence = float(np.max(nb_model.predict_proba(features)[0]))
        
        # Get top positive and negative words
        word_importance = {}
        for word, idx in count_vectorizer.vocabulary_.items():
            if word in cleaned_text.split():
                importance = nb_model.feature_log_prob_[1][idx] - nb_model.feature_log_prob_[0][idx]
                word_importance[word] = importance
        
        # Sort by importance
        important_words = sorted(word_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        important_words = [{"word": word, "importance": float(imp), "sentiment": "positive" if imp > 0 else "negative"} 
                          for word, imp in important_words]
        
    elif model_type == 'logistic_regression':
        features = tfidf_vectorizer.transform([cleaned_text])
        prediction = lr_model.predict(features)[0]
        confidence = float(np.max(lr_model.predict_proba(features)[0]))
        
        # Get top important words for logistic regression
        word_importance = {}
        for word, idx in tfidf_vectorizer.vocabulary_.items():
            if word in cleaned_text.split():
                importance = lr_model.coef_[0][idx]  # Coefficient indicates importance
                word_importance[word] = importance
        
        # Sort by importance
        important_words = sorted(word_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        important_words = [{"word": word, "importance": float(imp), "sentiment": "positive" if imp > 0 else "negative"} 
                          for word, imp in important_words]
    
    # Determine sentiment
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    # Return prediction
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 2),
        'model': model_type,
        'important_words': important_words
    })

if __name__ == '__main__':
    app.run(debug=True)