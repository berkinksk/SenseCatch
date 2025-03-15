import os
import logging
from flask import Flask, request, jsonify, render_template
import re
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app first, in case there are errors with other imports
app = Flask(__name__)

# Set NLTK data path explicitly
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Try to import nltk, but handle the error if it fails
try:
    import nltk
    nltk.data.path.insert(0, nltk_data_path)
    
    # Download essential NLTK data if not available
    try:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)
        nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path, quiet=True)
        nltk.download('maxent_ne_chunker', download_dir=nltk_data_path, quiet=True)
        nltk.download('words', download_dir=nltk_data_path, quiet=True)
        logger.info(f"NLTK resources downloaded successfully to {nltk_data_path}")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")
except ImportError:
    logger.warning("NLTK package not available. Fallback tokenization will be used.")
    nltk = None

# Simple fallback tokenization without relying on NLTK
def simple_tokenize(text):
    return text.lower().split()

# Import ensemble model with error handling
ensemble = None
try:
    from ensemble_model import SentimentEnsemble
    logger.info("Initializing ensemble model...")
    ensemble = SentimentEnsemble()
    logger.info("Ensemble model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ensemble model: {str(e)}")
    logger.error(traceback.format_exc())

# Function to clean text
def clean_text(text):
    try:
        # Delegate to ensemble's clean_text if available
        if ensemble and hasattr(ensemble, 'clean_text'):
            return ensemble.clean_text(text)
        
        # Fallback cleaning
        text = text.lower()
        # Remove special characters but keep apostrophes
        text = re.sub(r"[^a-z0-9'\\s]", ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
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
        model_type = data.get('model', 'naive_bayes')  # Default to naive_bayes
        
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
        
        # First handle obvious cases directly
        is_simple_case, simple_pred, simple_conf = ensemble._handle_simple_cases(text)
        if is_simple_case:
            sentiment = "Positive" if simple_pred == 1 else "Negative"
            confidence = simple_conf * 100
            
            # Create simple influential words for explanation
            important_words = []
            for word in simple_tokenize(text):
                if word in ["awesome", "amazing", "excellent", "great", "good", "love", 
                           "terrible", "awful", "horrible", "hate", "bad", "worst"]:
                    sentiment_type = "positive" if word in ["awesome", "amazing", "excellent", "great", "good", "love"] else "negative"
                    important_words.append({
                        "word": word,
                        "importance": 95.0,
                        "sentiment": sentiment_type
                    })
            
            important_words = important_words[:5]  # Take up to 5 words
            
            logger.info(f"Simple case detected: {sentiment} with {confidence}% confidence")
            return jsonify({
                'text': text,
                'sentiment': sentiment,
                'confidence': round(confidence, 2),
                'model': model_type,
                'important_words': important_words
            })
        
        # Try using specific model prediction with proper error handling
        try:
            # Use the specific requested model (not the ensemble) for more diverse results
            prediction, confidence, important_words = ensemble.predict_with_specific_model(text, model_type)
            
            if prediction is None:
                raise ValueError(f"Model {model_type} returned None prediction")
            
            # Check if confidence falls in the neutral range (0.4-0.6)
            is_neutral = 0.4 <= confidence <= 0.6
            
            # Determine sentiment label
            if is_neutral:
                sentiment_label = "Neutral"
            else:
                sentiment_label = "Positive" if prediction == 1 else "Negative"
                
            confidence_percent = confidence * 100  # Convert to percentage
            
            # Return the prediction
            return jsonify({
                'text': text,
                'sentiment': sentiment_label,
                'confidence': round(confidence_percent, 2),
                'model': model_type,
                'important_words': important_words
            })
        except Exception as e:
            logger.error(f"Error in specific model prediction: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to ensemble prediction
            try:
                prediction, confidence, important_words = ensemble.predict(text)
                
                # Check if confidence falls in the neutral range (0.4-0.6)
                is_neutral = 0.4 <= confidence <= 0.6
                
                # Determine sentiment label
                if is_neutral:
                    sentiment_label = "Neutral"
                else:
                    sentiment_label = "Positive" if prediction == 1 else "Negative"
                    
                confidence_percent = confidence * 100  # Convert to percentage
                
                logger.info(f"Using ensemble fallback for {text}: {sentiment_label} with {confidence_percent}% confidence")
                
                return jsonify({
                    'text': text,
                    'sentiment': sentiment_label,
                    'confidence': round(confidence_percent, 2),
                    'model': model_type + " (ensemble fallback)",
                    'important_words': important_words
                })
            except Exception as nested_e:
                logger.error(f"Error in ensemble fallback: {str(nested_e)}")
                # Fall back to simple case analysis
                return analyze_simple_case(text, model_type)
    except Exception as e:
        logger.error(f"Unhandled exception in analyze route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Server error: {str(e)}. Please try again.'
        }), 500

def analyze_simple_case(text, model_type):
    """Fallback analysis when model prediction fails"""
    # Simple rule-based analysis
    text_lower = text.lower()
    
    # Check for obvious positive terms
    positive_terms = ["good", "great", "excellent", "amazing", "awesome", "love", "nice", "enjoy", "like", "best"]
    negative_terms = ["bad", "terrible", "awful", "horrible", "worst", "hate", "dislike", "poor", "waste", "boring"]
    
    pos_count = sum(1 for term in positive_terms if term in text_lower)
    neg_count = sum(1 for term in negative_terms if term in text_lower)
    
    # Check for negation
    negations = ["not", "don't", "doesn't", "didn't", "no", "never"]
    has_negation = any(neg in text_lower for neg in negations)
    
    # Check for contrast markers
    contrast_markers = ["but", "however", "although", "though", "despite", "yet"]
    has_contrast = any(marker in text_lower for marker in contrast_markers)
    
    # Check for neutral indicators
    neutral_indicators = ["average", "mediocre", "ok", "okay", "fine", "neither", "nor"]
    has_neutral = any(indicator in text_lower for indicator in neutral_indicators)
    
    # Simple logic for sentiment
    if has_neutral or (pos_count == neg_count and pos_count > 0):
        # Likely neutral
        sentiment = "Neutral"
        confidence = 60
    elif has_negation:
        # Negation flips the sentiment
        if pos_count > neg_count:
            sentiment = "Negative"
            confidence = min(65 + (pos_count * 5), 90)
        elif neg_count > pos_count:
            sentiment = "Positive"
            confidence = min(65 + (neg_count * 5), 90)
        else:
            # No clear sentiment with negation
            sentiment = "Neutral"
            confidence = 60
    elif has_contrast:
        # With contrast markers, be more cautious
        if pos_count > neg_count + 2:
            sentiment = "Positive"
            confidence = 70
        elif neg_count > pos_count + 2:
            sentiment = "Negative"
            confidence = 70
        else:
            sentiment = "Neutral"
            confidence = 60
    else:
        if pos_count > neg_count:
            sentiment = "Positive"
            confidence = min(70 + (pos_count * 5), 95)
        elif neg_count > pos_count:
            sentiment = "Negative"
            confidence = min(70 + (neg_count * 5), 95)
        else:
            # Neutral or unclear sentiment
            sentiment = "Neutral"
            confidence = 60
    
    # Generate simple word importance
    important_words = []
    words = simple_tokenize(text)
    
    for word in words:
        if word in positive_terms:
            important_words.append({
                "word": word,
                "importance": 80.0,
                "sentiment": "positive" if not has_negation else "negative"
            })
        elif word in negative_terms:
            important_words.append({
                "word": word,
                "importance": 80.0,
                "sentiment": "negative" if not has_negation else "positive"
            })
    
    important_words = important_words[:5]  # Limit to 5 words
    
    logger.info(f"Using fallback analysis: {sentiment} with {confidence}% confidence")
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': round(confidence, 2),
        'model': model_type + " (fallback)",
        'important_words': important_words
    })

if __name__ == '__main__':
    app.run(debug=True)