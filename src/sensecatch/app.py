import os
import logging
from flask import Flask, request, jsonify, render_template
import re
import traceback
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app with explicit template & static folders (project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEMPLATES_PATH = os.path.join(PROJECT_ROOT, 'templates')
STATIC_PATH = os.path.join(PROJECT_ROOT, 'static')
app = Flask(__name__, template_folder=TEMPLATES_PATH, static_folder=STATIC_PATH, static_url_path='')

# Enable CORS for frontend hosted on a different domain (e.g., Vercel)
try:
    allowed_origins = os.environ.get('ALLOWED_ORIGINS')
    if allowed_origins:
        origins_list = [o.strip() for o in allowed_origins.split(',') if o.strip()]
        CORS(app, resources={
            r"/analyze": {
                "origins": origins_list,
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"],
                "expose_headers": ["Content-Type"],
                "supports_credentials": False,
                "max_age": 600
            }
        })
        logger.info(f"CORS enabled for origins: {origins_list}")
    else:
        # Default: allow all origins
        CORS(app, resources={
            r"/analyze": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type"],
                "max_age": 600
            }
        })
        logger.info("CORS enabled for all origins")
except Exception as e:
    logger.error(f"CORS configuration error: {e}")

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
    from src.sensecatch.ensemble_model import SentimentEnsemble
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

# Lightweight health check endpoint for Render
@app.route('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model_initialized': ensemble is not None
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get data from request
        data = request.get_json()
        text = data.get('text', '')
        # Valid model values: naive_bayes, logistic_regression, linear_svc, nbsvm, distilbert, stack, rule_based
        model_type = data.get('model', 'naive_bayes')  # Default to naive_bayes
        
        logger.info(f"Analyzing text: '{text}' with model: {model_type}")
        
        # Check if ensemble was initialized
        if ensemble is None:
            logger.error("Ensemble model was not initialized properly")
            return jsonify({
                'error': 'Sentiment model not available. Please try again later.'
            }), 500
        
        # Clean the text for logging purposes
        cleaned_text = clean_text(text)
        logger.info(f"Cleaned text: '{cleaned_text}'")
        
        # Use the model prediction with new API format
        try:
            # Try with the specific requested model
            result = ensemble.predict(text, specific_model=model_type)
            
            # Result is now a dictionary with all needed information
            sentiment_label = result.get("sentiment", "Neutral")
            confidence = result.get("confidence", 50.0)  # Already in percentage
            important_words = result.get("influential_words", [])
            model_used = result.get("model_used", model_type)
            
            logger.info(f"Analysis result: {sentiment_label} with {confidence:.2f}% confidence")
            
            # Return the prediction
            return jsonify({
                'text': text,
                'sentiment': sentiment_label,
                'confidence': round(confidence, 2),
                'model': model_used,
                'important_words': important_words
            })
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to a basic analysis
            return jsonify({
                'text': text,
                'sentiment': "Neutral",
                'confidence': 50.0,
                'model': "error_fallback",
                'important_words': []
            })
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e)
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
    negations = ["not", "don't", "doesn't", "didn't", "no", "never", "cannot", "nor", "neither"]
    has_negation = any(neg in text_lower for neg in negations)
    
    # Identify words in negation scope
    words = simple_tokenize(text)
    negation_scope = {}
    for i, word in enumerate(words):
        if word in negations or any(neg in word for neg in ["n't"]):
            # Mark the next 3 words (or until end of sentence) as in negation scope
            for j in range(i+1, min(i+4, len(words))):
                negation_scope[words[j]] = True
                # End negation scope at punctuation
                if words[j].endswith(('.', '!', '?', ',')):
                    break
    
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
    
    for word in words:
        # Check if word is in negation scope
        word_in_negation = word in negation_scope
        
        if word in positive_terms:
            important_words.append({
                "word": word,
                "importance": 80.0,
                "sentiment": "positive",  # Keep raw sentiment for color coding
                "negated": word_in_negation  # Add negation information
            })
        elif word in negative_terms:
            important_words.append({
                "word": word,
                "importance": 80.0,
                "sentiment": "negative",  # Keep raw sentiment for color coding
                "negated": word_in_negation  # Add negation information
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