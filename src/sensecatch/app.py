import os
import logging

import nltk
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.sensecatch.ensemble_model import SentimentEnsemble, RAW_MODELS

# force=True replaces the DEBUG setup the engine module applies on import.
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
# The engine logs review text at INFO level, so keep only its warnings and errors.
logging.getLogger('src.sensecatch.ensemble_model').setLevel(logging.WARNING)

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

# NLTK data the app needs, with the path NLTK uses to find each package.
NLTK_RESOURCES = [
    ('punkt', 'tokenizers/punkt'),
    ('stopwords', 'corpora/stopwords'),
    ('wordnet', 'corpora/wordnet.zip'),
    ('omw-1.4', 'corpora/omw-1.4.zip'),
    ('vader_lexicon', 'sentiment/vader_lexicon.zip'),
    ('sentiwordnet', 'corpora/sentiwordnet'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
    ('maxent_ne_chunker', 'chunkers/maxent_ne_chunker'),
    ('words', 'corpora/words'),
]

# The Docker image fills NLTK_DATA at build time. Locally, ./nltk_data is used.
nltk_data_dir = os.environ.get('NLTK_DATA') or os.path.join(os.getcwd(), 'nltk_data')
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)
for name, path in NLTK_RESOURCES:
    try:
        nltk.data.find(path)
    except LookupError:
        logger.info(f"Downloading NLTK resource: {name}")
        nltk.download(name, download_dir=nltk_data_dir, quiet=True)

VALID_MODELS = RAW_MODELS | {'rule_based'}
ANALYSIS_FAILED = 'Something went wrong while analyzing the text. Please try again.'

ensemble = None
try:
    logger.info("Initializing ensemble model...")
    ensemble = SentimentEnsemble()
    logger.info("Ensemble model initialized successfully")
except Exception:
    logger.exception("Error initializing ensemble model")


def preload_heavy_models():
    """Load DistilBERT and the stacked ensemble now, so the first visitor does not wait."""
    for name in ('distilbert', 'stack'):
        result = ensemble.predict('A good film.', specific_model=name)
        if str(result.get('model_used', '')).endswith('_error'):
            raise RuntimeError(f"{name} could not be loaded")


if ensemble is not None and os.environ.get('PRELOAD_MODELS') == '1':
    try:
        preload_heavy_models()
        logger.info("Preloaded DistilBERT and the stacked ensemble")
    except Exception:
        logger.exception("Could not preload the heavy models")


def error_response(message, status):
    return jsonify({'error': message}), status


@app.route('/')
def home():
    return render_template('index.html')


# Health check for the frontend and the uptime monitor.
@app.route('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model_initialized': ensemble is not None
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Send a JSON body with a 'text' field.", 400)

    text = data.get('text')
    if not isinstance(text, str) or not text.strip():
        return error_response('Please enter some text to analyze.', 400)

    model_type = data.get('model', 'naive_bayes')
    if not isinstance(model_type, str) or model_type not in VALID_MODELS:
        return error_response(f'Unknown model: {model_type}', 400)

    if ensemble is None:
        return error_response('The models are not available right now. Please try again in a moment.', 503)

    try:
        result = ensemble.predict(text, specific_model=model_type)
        model_used = result.get('model_used', model_type)
        # The engine marks a failed prediction with an "_error" model name.
        if str(model_used).endswith('_error'):
            raise RuntimeError(f"{model_type} could not make a prediction")
        response = {
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': round(float(result['confidence']), 2),
            'model': model_used,
            'important_words': result.get('influential_words', []),
        }
    except Exception:
        logger.exception(f"Prediction failed for model {model_type}")
        return error_response(ANALYSIS_FAILED, 500)

    logger.info(f"Analyzed with {model_used}: {response['sentiment']}")
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 7860)))
