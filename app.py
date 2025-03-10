from flask import Flask, request, jsonify, render_template
import re
import os
from ensemble_model import SentimentEnsemble

app = Flask(__name__)

# Initialize the ensemble model
ensemble = SentimentEnsemble()

# Function to clean text (for consistency, though the ensemble handles this internally)
def clean_text(text):
    return ensemble.clean_text(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get data from request
    data = request.get_json()
    text = data.get('text', '')
    model_type = data.get('model', 'ensemble')  # Default to ensemble
    
    # Clean the text (the ensemble will do this internally, but for consistency)
    cleaned_text = clean_text(text)
    
    if model_type == 'ensemble':
        # Use the ensemble model
        prediction, confidence, important_words = ensemble.predict(text)
        sentiment = "Positive" if prediction == 1 else "Negative"
    else:
        # Check if the selected model is available in the ensemble
        if model_type not in ensemble.models:
            return jsonify({
                'error': f'Model {model_type} is not available.'
            }), 404
        
        # Use the specific model from the ensemble
        model = ensemble.models[model_type]
        vectorizer = ensemble.vectorizers[model_type]
        
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
        important_words = ensemble._extract_influential_words(cleaned_text, prediction, model_type)
    
    # Format confidence as percentage
    confidence_pct = confidence * 100 if confidence <= 1 else confidence
    
    # Return prediction
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': round(confidence_pct, 2),
        'model': model_type,
        'important_words': important_words
    })

if __name__ == '__main__':
    app.run(debug=True)