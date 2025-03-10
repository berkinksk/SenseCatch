from flask import Flask, request, jsonify, render_template
import re
import os
import traceback
import logging
from ensemble_model import SentimentEnsemble

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the ensemble model
try:
    logger.info("Initializing ensemble model...")
    ensemble = SentimentEnsemble()
    logger.info("Ensemble model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ensemble model: {str(e)}")
    logger.error(traceback.format_exc())
    ensemble = None

# Function to clean text
def clean_text(text):
    try:
        return ensemble.clean_text(text) if ensemble else text.lower()
    except Exception as e:
        logger.error(f"Error in clean_text: {str(e)}")
        return text.lower()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get data from request
        data = request.get_json()
        text = data.get('text', '')
        model_type = data.get('model', 'naive_bayes')  # Default to naive_bayes for now
        
        logger.info(f"Analyzing text: '{text}' with model: {model_type}")
        
        # Check if ensemble was initialized
        if ensemble is None:
            logger.error("Ensemble model was not initialized properly")
            return jsonify({
                'error': 'Sentiment model not available. Please try again later.'
            }), 500
        
        # Clean the text
        cleaned_text = clean_text(text)
        logger.info(f"Cleaned text: '{cleaned_text}'")
        
        # Simple fallback approach when ensemble fails
        # This ensures basic functionality even if the ensemble has issues
        if model_type == 'ensemble' and ensemble is not None:
            try:
                prediction, confidence, important_words = ensemble.predict(text)
                sentiment = "Positive" if prediction == 1 else "Negative"
            except Exception as e:
                logger.error(f"Error using ensemble model: {str(e)}")
                logger.error(traceback.format_exc())
                # Fall back to a specific model
                model_type = 'naive_bayes'
                logger.info(f"Falling back to {model_type} model")
        
        # If not using ensemble or ensemble failed
        if model_type != 'ensemble' or 'sentiment' not in locals():
            # Check if the selected model is available
            if model_type not in ensemble.models:
                logger.error(f"Model {model_type} not found in ensemble")
                return jsonify({
                    'error': f'Model {model_type} is not available.'
                }), 404
            
            # Use the specific model
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
                confidence = proba[1] if prediction == 1 else proba[0]
            except AttributeError as e:
                logger.error(f"Error getting prediction probability: {str(e)}")
                confidence = 0.85  # Fallback confidence
            
            # Get important words
            try:
                important_words = ensemble._extract_influential_words(cleaned_text, prediction, model_type)
            except Exception as e:
                logger.error(f"Error extracting influential words: {str(e)}")
                important_words = []
        
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
        
    except Exception as e:
        logger.error(f"Unhandled exception in analyze route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Server error: {str(e)}. Please try again.'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)