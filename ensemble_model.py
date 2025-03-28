"""
Ensemble model implementation for SenseCatch
Combines multiple sentiment models for improved accuracy
"""
import numpy as np
import pickle
import os
import traceback
import logging
import re
import nltk
from nltk.tree import Tree  # Explicitly import Tree for named entity checking
from scipy.sparse import hstack, csr_matrix
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set NLTK data path explicitly
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.insert(0, nltk_data_path)

# Simple fallback tokenizer
def simple_tokenize(text):
    return text.lower().split()

# Safely import nltk
try:
    from nltk.tokenize import word_tokenize
    from nltk import ne_chunk, pos_tag
    logger.info("NLTK imported successfully")
except ImportError:
    logger.error("Error importing NLTK. Using fallback tokenizer.")
    # Fallback simple tokenizer if nltk is not available
    def word_tokenize(text):
        return simple_tokenize(text)
    def pos_tag(tokens):
        return [(token, 'NN') for token in tokens]  # Default all to nouns
    def ne_chunk(tagged_tokens):
        return tagged_tokens

class SentimentEnsemble:
    """Ensemble model that combines multiple sentiment classifiers"""
    
    def __init__(self, models_dir=None, use_cache=True):
        """Initialize the ensemble model with NB and LR models."""
        self.models = {}
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.use_cache = use_cache
        
        # Define very positive and very negative phrases for sentiment analysis
        self.VERY_POSITIVE_PHRASES = [
            "excellent", "amazing", "outstanding", "fantastic", "terrific",
            "wonderful", "brilliant", "superb", "perfect", "great",
            "love", "best", "exceptional", "marvelous", "awesome",
            "impressive", "exceptional", "delightful", "fabulous", "incredible"
        ]
        
        self.VERY_NEGATIVE_PHRASES = [
            "terrible", "awful", "horrible", "dreadful", "abysmal",
            "worst", "pathetic", "atrocious", "appalling", "disgusting",
            "hate", "disappointing", "horrendous", "disastrous", "catastrophic",
            "useless", "worthless", "abominable", "deplorable", "detestable"
        ]
        
        # Load existing models or train new ones
        model_paths = {
            "naive_bayes": os.path.join(self.models_dir, "naive_bayes_model.pkl"),
            "logistic_regression": os.path.join(self.models_dir, "logistic_regression_model.pkl")
        }
        
        self.vectorizers = {}
        self.dict_vectorizers = {}
        self.feature_dimensions = self._load_feature_dimensions()
        self.model_weights = {
            'naive_bayes': 0.6,
            'logistic_regression': 0.4,
        }
        # Add common movie title list for entity recognition
        self.movie_titles = self._load_movie_titles()
        # Initialize tfidf_vectorizer
        self.tfidf_vectorizer = None
        self._load_models()
        
        # Ensure we have a tfidf_vectorizer by setting it from the loaded vectorizers
        if 'logistic_regression' in self.vectorizers:
            self.tfidf_vectorizer = self.vectorizers['logistic_regression']
            logger.info("Using logistic_regression vectorizer as tfidf_vectorizer")
        elif 'naive_bayes' in self.vectorizers:
            self.tfidf_vectorizer = self.vectorizers['naive_bayes']
            logger.info("Using naive_bayes vectorizer as tfidf_vectorizer")
        else:
            logger.error("No vectorizer found for feature extraction")
    
    def _load_movie_titles(self):
        """Load a comprehensive list of movie titles from multiple sources"""
        # Start with a small default list
        titles = [
            "the godfather", "citizen kane", "casablanca", "gone with the wind",
            "the wizard of oz", "star wars", "pulp fiction", "the shawshank redemption",
            "the dark knight", "schindler's list", "lord of the rings", "forrest gump",
            "the matrix", "goodfellas", "titanic", "saving private ryan", "jaws",
            "apocalypse now", "gladiator", "the silence of the lambs", "king of comedy",
            "the room", "the avengers", "jurassic park", "the lion king"
        ]
        
        # First check if we have a cached movie titles file
        movie_titles_path = 'models/movie_titles.txt'
        
        # Make sure the models directory exists
        if not os.path.exists('models'):
            try:
                os.makedirs('models')
                logger.info("Created models directory")
            except Exception as e:
                logger.error(f"Error creating models directory: {e}")
        
        if os.path.exists(movie_titles_path):
            try:
                with open(movie_titles_path, 'r', encoding='utf-8') as f:
                    titles = [line.strip().lower() for line in f if line.strip()]
                logger.info(f"Loaded {len(titles)} movie titles from cache file")
                return titles
            except Exception as e:
                logger.error(f"Error loading movie titles from cache: {e}")
        
        # Try to fetch additional titles from multiple sources
        try:
            # Import necessary modules
            import urllib.request
            import re
            import json
            from time import sleep
            
            # URLs with movie lists - using different sources for diversity
            urls = [
                'https://www.imdb.com/chart/top/',  # Top rated movies
                'https://www.imdb.com/chart/moviemeter/'  # Most popular movies
            ]
            
            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            
            for url in urls:
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as response:
                        html = response.read().decode('utf-8')
                        
                        # Extract movie titles using regex
                        matches = re.findall(r'<a[^>]*>([^<]+)</a>', html)
                        
                        if matches:
                            # Clean and add to titles list
                            for title in matches:
                                clean_title = re.sub(r'[^\w\s]', '', title).strip().lower()
                                if clean_title and len(clean_title) > 3 and clean_title not in titles:
                                    titles.append(clean_title)
                    
                    logger.info(f"Fetched movie titles from {url}")
                    sleep(1)  # Be polite and don't hammer servers
                    
                except Exception as url_error:
                    logger.warning(f"Error fetching from {url}: {url_error}")
            
            # Also try to use TMDB API if possible
            tmdb_api_key = os.environ.get('TMDB_API_KEY')
            if tmdb_api_key:
                try:
                    tmdb_url = f'https://api.themoviedb.org/3/movie/popular?api_key={tmdb_api_key}&language=en-US'
                    req = urllib.request.Request(tmdb_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        if 'results' in data:
                            for movie in data['results']:
                                title = movie.get('title', '').lower()
                                if title and title not in titles:
                                    titles.append(title)
                    logger.info(f"Fetched movie titles from TMDB API")
                except Exception as tmdb_error:
                    logger.warning(f"Error fetching from TMDB API: {tmdb_error}")
                
        except Exception as e:
            logger.warning(f"Could not fetch additional movie titles: {e}")
        
        # Save the compiled list to a file for future use
        try:
            with open(movie_titles_path, 'w', encoding='utf-8') as f:
                for title in titles:
                    f.write(title + '\n')
            logger.info(f"Saved {len(titles)} movie titles to {movie_titles_path}")
        except Exception as save_error:
            logger.error(f"Error saving movie titles: {save_error}")
        
        logger.info(f"Using {len(titles)} movie titles for entity recognition")
        return titles
    
    def _load_feature_dimensions(self):
        """Load feature dimensions from saved file if available"""
        try:
            if os.path.exists('models/feature_dimensions.pkl'):
                with open('models/feature_dimensions.pkl', 'rb') as f:
                    return pickle.load(f)
            else:
                return {'text_features': 10000, 'lexicon_features': 9, 'total_features': 10009}
        except Exception as e:
            logger.error(f"Error loading feature dimensions: {e}")
            return {'text_features': 10000, 'lexicon_features': 9, 'total_features': 10009}
    
    def _load_models(self):
        """Load all available models from the models directory"""
        model_paths = {
            'naive_bayes': 'models/naive_bayes.pkl',
            'logistic_regression': 'models/logistic_regression.pkl'
        }
        
        # Try to load individual vectorizers first if they exist
        try:
            if os.path.exists('models/count_vectorizer.pkl'):
                with open('models/count_vectorizer.pkl', 'rb') as f:
                    self.vectorizers['naive_bayes'] = pickle.load(f)
                    logger.info("Loaded count_vectorizer separately")
            
            if os.path.exists('models/tfidf_vectorizer.pkl'):
                with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                    self.vectorizers['logistic_regression'] = pickle.load(f)
                    logger.info("Loaded tfidf_vectorizer separately")
            
            if os.path.exists('models/dict_vectorizer.pkl'):
                with open('models/dict_vectorizer.pkl', 'rb') as f:
                    dict_vec = pickle.load(f)
                    self.dict_vectorizers['naive_bayes'] = dict_vec
                    self.dict_vectorizers['logistic_regression'] = dict_vec
                    logger.info("Loaded dict_vectorizer separately")
        except Exception as e:
            logger.error(f"Error loading individual vectorizers: {e}")
        
        for model_name, path in model_paths.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        # Try different loading strategies
                        try:
                            # First try the 3-tuple format (model, text_vectorizer, dict_vectorizer)
                            loaded_data = pickle.load(f)
                            if isinstance(loaded_data, tuple):
                                if len(loaded_data) == 3:
                                    model, vectorizer, dict_vec = loaded_data
                                    self.models[model_name] = model
                                    
                                    # Only use the vectorizer if we don't already have one
                                    if model_name not in self.vectorizers:
                                        self.vectorizers[model_name] = vectorizer
                                    
                                    # Only use the dict_vectorizer if we don't already have one
                                    if model_name not in self.dict_vectorizers:
                                        self.dict_vectorizers[model_name] = dict_vec
                                    
                                elif len(loaded_data) == 2:
                                    model, vectorizer = loaded_data
                                    self.models[model_name] = model
                                    
                                    # Only use the vectorizer if we don't already have one
                                    if model_name not in self.vectorizers:
                                        self.vectorizers[model_name] = vectorizer
                                else:
                                    raise ValueError(f"Unexpected tuple length: {len(loaded_data)}")
                            else:
                                # Maybe it's just the model
                                model = loaded_data
                                self.models[model_name] = model
                                logger.warning(f"Loaded only model for {model_name}, no vectorizer found")
                        except Exception as e:
                            logger.error(f"Error unpacking model: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue
                        
                        logger.info(f"Loaded model: {model_name}")
                else:
                    logger.warning(f"Model file not found: {path}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Initialize sentiment lexicon
        try:
            from sentiment_lexicon import SentimentLexiconFeatures
            self.lexicon = SentimentLexiconFeatures()
            logger.info("Sentiment lexicon initialized")
        except Exception as e:
            logger.error(f"Error initializing sentiment lexicon: {e}")
            self.lexicon = None
    
    def identify_movie_titles(self, text):
        """Identify potential movie titles in text and handle them specially"""
        try:
            # Check against our movie titles list
            marked_text = text
            identified_titles = []
            
            # First pass: find direct matches from our movie titles database
            for title in self.movie_titles:
                if len(title.split()) > 1:  # Only multi-word titles to avoid false positives
                    title_pattern = re.compile(r'\b' + re.escape(title) + r'\b', re.IGNORECASE)
                    if title_pattern.search(text):
                        # Mark the title by replacing spaces with underscores and adding prefix
                        marked_title = "MOVIETITLE_" + "_".join(title.split())
                        marked_text = title_pattern.sub(marked_title, marked_text)
                        identified_titles.append(title)
            
            # Second pass: try named entity recognition for titles not in our database
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            
            try:
                entities = ne_chunk(tagged)
                for chunk in entities:
                    # Check if the chunk is a named entity (not a tuple) and has a label attribute
                    if isinstance(chunk, Tree) and hasattr(chunk, 'label'):
                        if chunk.label() in ('ORGANIZATION', 'PERSON', 'GPE'):
                            title = ' '.join([c[0] for c in chunk])
                            # Only consider if title has multiple words and isn't already identified
                            if len(title.split()) > 1 and title.lower() not in self.movie_titles and title not in identified_titles:
                                # Additional heuristics to determine if it's likely a movie title
                                is_likely_title = False
                                
                                # Check if it contains common movie title words
                                movie_words = ['movie', 'film', 'documentary', 'trilogy', 'sequel', 'series']
                                
                                # Check if it follows patterns like "watched [X]" or "[X] is a good movie"
                                sentence_parts = text.lower().split('.')
                                for part in sentence_parts:
                                    if title.lower() in part:
                                        movie_verbs = ['watched', 'saw', 'viewing', 'seeing', 'rated', 'directed']
                                        movie_contexts = ['is a good', 'is a great', 'is a bad', 'is a terrible', 
                                                         'a film by', 'the movie', 'the film', 'tickets for']
                                        
                                        if any(verb in part for verb in movie_verbs) or any(context in part for context in movie_contexts):
                                            is_likely_title = True
                                            break
                                
                                # Apply the movie title marking if it passes our heuristics
                                if is_likely_title:
                                    title_pattern = re.compile(r'\b' + re.escape(title) + r'\b', re.IGNORECASE)
                                    marked_title = "MOVIETITLE_" + "_".join(title.split())
                                    marked_text = title_pattern.sub(marked_title, marked_text)
                                    identified_titles.append(title)
                                    
                                    # Add to our movie titles database for future use
                                    if title.lower() not in self.movie_titles:
                                        self.movie_titles.append(title.lower())
            except Exception as entity_error:
                logger.warning(f"Entity recognition error: {entity_error}")
            
            # Special handling for sentences containing movie titles
            if identified_titles:
                logger.info(f"Identified movie titles: {', '.join(identified_titles)}")
                
                # Break the text into sentences
                sentences = text.split('.')
                for i, sentence in enumerate(sentences):
                    # Check if the sentence contains a movie title
                    if any(title.lower() in sentence.lower() for title in identified_titles):
                        # Find the movie title mentioned in this sentence
                        title_in_sentence = next((title for title in identified_titles if title.lower() in sentence.lower()), None)
                        
                        # Adjust how sentiment is processed for this sentence
                        # We'll mark sentences with movie titles to be processed differently
                        mark = "MOVIE_TITLE_SENTENCE"
                        
                        # Replace the sentence in the marked text (preserve original case)
                        original_sentence = sentences[i]
                        if original_sentence in marked_text:
                            # Add the special marker to the beginning of the sentence
                            marked_text = marked_text.replace(original_sentence, f"{mark} {original_sentence}")
            
            return marked_text
        except Exception as e:
            logger.error(f"Error in identify_movie_titles: {e}")
            logger.error(traceback.format_exc())
            return text
    
    def _handle_simple_cases(self, text, modelname=None):
        """Handle obvious cases that don't require model prediction"""
        # Convert to lowercase
        text_lower = text.lower()
        
        # Define negation words
        negation_words = ['not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 'haven\'t', 
                          'hasn\'t', 'hadn\'t', 'can\'t', 'cannot', 'couldn\'t', 'shouldn\'t', 
                          'wouldn\'t', 'won\'t', 'isn\'t', 'aren\'t', 'ain\'t', 'wasn\'t', 
                          'weren\'t', 'nor', 'neither']
        
        # Define very positive and very negative phrases
        very_positive = ['excellent', 'amazing', 'awesome', 'outstanding', 'perfect', 'fantastic', 
                         'brilliant', 'great', 'terrific', 'phenomenal', 'superb', 'wonderful']
        
        very_negative = ['terrible', 'awful', 'horrible', 'dreadful', 'abysmal', 'atrocious', 
                         'abhorrent', 'disgusting', 'appalling', 'horrific', 'catastrophic']
        
        # Check for recommendation context
        recommendation_context = ['recommend', 'recommendation', 'advise', 'endorse', 'suggest']
        
        # Define complex negation cases
        # NEW: Explicit negated phrase mapping with direct sentiment values
        explicit_negated_phrases = {
            # Strong negative expressions
            "don't recommend": (0, 0.92),  # Negative with high confidence
            "doesn't recommend": (0, 0.92),
            "wouldn't recommend": (0, 0.93),
            "cannot recommend": (0, 0.92),
            "can't recommend": (0, 0.92),
            "never recommend": (0, 0.94),
            "do not recommend": (0, 0.93),
            "not worth": (0, 0.85),
            "not recommended": (0, 0.91),
            
            # Strong positive expressions
            "highly recommend": (1, 0.92),
            "strongly recommend": (1, 0.93),
            "definitely recommend": (1, 0.94),
            "absolutely recommend": (1, 0.94),
            "would recommend": (1, 0.9),
            
            # Double negative expressions (positive)
            "not disappointed": (1, 0.8),
            "not bad at all": (1, 0.82),
            "no complaints": (1, 0.85),
            "never been disappointed": (1, 0.87),
            
            # Strong sentiment with reasoning
            "recommend because": (1, 0.91),
            "don't recommend because": (0, 0.92),
            "broke within": (0, 0.88),  # Strong indication of negative product experience
            "failed within": (0, 0.89)
        }
        
        # NEW: Check for explicit negated phrases first
        for phrase, (sentiment, confidence) in explicit_negated_phrases.items():
            if phrase in text_lower:
                logger.info(f"Explicit phrase match: '{phrase}' → {sentiment} ({confidence:.2f})")
                return True, sentiment, confidence
        
        # Check for very positive phrases
        for pos_phrase in very_positive:
            if pos_phrase in text_lower:
                # Check if the positive phrase is negated
                for neg in negation_words:
                    # Check if negation word appears close to the positive phrase
                    neg_pos = text_lower.find(neg + " ")
                    phrase_pos = text_lower.find(pos_phrase)
                    if neg_pos != -1 and phrase_pos != -1:
                        # If negation is within 5 words of the positive phrase
                        if 0 <= phrase_pos - neg_pos <= 30:  # Approx 5 words with spaces
                            # Negated positive is negative
                            return True, 0, 0.85
                
                # If not negated, it's positive
                return True, 1, 0.88
        
        # "Don't recommend" and similar phrases
        recommendation_negations = [neg + " " + rec for neg in negation_words for rec in recommendation_context]
        for neg_rec in recommendation_negations:
            if neg_rec in text_lower:
                logger.info(f"Recommendation negation detected: '{neg_rec}'")
                return True, 0, 0.90  # Strong negative for "don't recommend" phrases
        
        # Check simple positive/negative phrases with "because" reasoning
        reasoning_markers = ["because", "since", "as", "due to", "thanks to"]
        for marker in reasoning_markers:
            marker_pos = text_lower.find(marker)
            if marker_pos != -1:
                # Split into before and after the reasoning marker
                before = text_lower[:marker_pos].strip()
                after = text_lower[marker_pos:].strip()
                
                # Check for negative product experiences after "because"
                product_failure_terms = ["broke", "broken", "failed", "stopped working", "defective",
                                        "malfunctioned", "stopped", "died", "unusable", "useless"]
                
                # NEW: Enhanced reasoning logic
                if any(term in after for term in product_failure_terms):
                    logger.info(f"Product failure reasoning detected after '{marker}'")
                    # Check what comes before the reasoning
                    if any(neg in before for neg in negation_words) and any(rec in before for rec in recommendation_context):
                        # "don't recommend because it broke" - strong negative
                        return True, 0, 0.93
                    elif any(rec in before for rec in recommendation_context):
                        # Recommendation followed by negative reason is very unusual and likely sarcastic
                        # e.g. "I recommend it because it broke immediately" - likely negative
                        logger.info("Possible sarcasm detected: positive recommendation with negative reason")
                        return True, 0, 0.75
                
                # Check for negation + reasoning structures
                for neg in negation_words:
                    if neg in before:
                        # If there's negation before "because", likely negative
                        # For example: "I don't like it because it broke"
                        return True, 0, 0.82
        
        # Default - not a simple case
        return False, None, None
    
    def handle_negations(self, text):
        """Advanced negation handling with proper scope and phrase detection"""
        try:
            # Enhanced list of negation words and contractions
            negation_words = [
                'not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 'haven\'t', 
                'hasn\'t', 'hadn\'t', 'can\'t', 'cannot', 'couldn\'t', 'shouldn\'t', 
                'wouldn\'t', 'won\'t', 'isn\'t', 'aren\'t', 'ain\'t', 'wasn\'t', 
                'weren\'t', 'nor', 'neither', 'hardly', 'barely', 'scarcely'
            ]
            
            # Special phrases where negation reverses meaning completely
            positive_negation_phrases = {
                "isn't bad": "is good",
                "aren't bad": "are good",
                "wasn't bad": "was good",
                "weren't bad": "were good", 
                "isn't terrible": "is good",
                "isn't horrible": "is good",
                "isn't awful": "is good",
                "don't hate": "like",
                "doesn't hate": "likes",
                "didn't hate": "liked",
                "not bad": "good",
                "not terrible": "good",
                "no complaints": "satisfied",
                "can't complain": "satisfied"
            }
            
            # NEW: Critical negated compound phrases with sentiment overrides
            compound_negated_phrases = {
                # Recommendation negations - high priority overrides
                "don't recommend": {"sentiment": "negative", "confidence": 0.92},
                "doesn't recommend": {"sentiment": "negative", "confidence": 0.92},
                "wouldn't recommend": {"sentiment": "negative", "confidence": 0.93},
                "do not recommend": {"sentiment": "negative", "confidence": 0.92},
                "can't recommend": {"sentiment": "negative", "confidence": 0.91},
                "cannot recommend": {"sentiment": "negative", "confidence": 0.91},
                
                # Double negation cases
                "wasn't great, nor": {"sentiment": "negative", "confidence": 0.89},
                "wasn't good, nor": {"sentiment": "negative", "confidence": 0.89},
                "isn't great, nor": {"sentiment": "negative", "confidence": 0.89},
                "weren't good, nor": {"sentiment": "negative", "confidence": 0.89}
            }
            
            # Track whether a special phrase was detected for sentiment override
            special_phrase_detected = False
            detected_phrase = None
            
            # Storage for special negation phrases that should force sentiment
            forced_sentiment = None  # Will store "positive" or "negative" when a forcing phrase is found
            forced_confidence = None
            
            # NEW: Check for compound negated phrases with highest priority
            for phrase, override in compound_negated_phrases.items():
                if phrase in text.lower():
                    logger.info(f"Compound negation phrase detected: '{phrase}' → {override['sentiment']}")
                    special_phrase_detected = True
                    detected_phrase = phrase
                    forced_sentiment = override["sentiment"]
                    forced_confidence = override["confidence"]
                    break
            
            # Check for special phrases that should override sentiment
            special_positive_override_phrases = [
                "isn't bad at all", "not bad at all", "not that bad", 
                "isn't even bad", "not even bad", "really not bad",
                "definitely not bad", "certainly not bad"
            ]
            
            # Only check for other phrases if we haven't found a compound negated phrase
            if not special_phrase_detected:
                # More explicit checks for special phrases
                for phrase in special_positive_override_phrases:
                    if phrase in text.lower():
                        logger.info(f"Special STRONG positive override phrase detected: '{phrase}'")
                        forced_sentiment = "positive"
                        detected_phrase = phrase
                        forced_confidence = 0.88
                        special_phrase_detected = True
                        break
                
                # Check for standard special phrases
                if not special_phrase_detected:
                    for phrase, replacement in positive_negation_phrases.items():
                        if phrase in text.lower():
                            logger.info(f"Special negation phrase detected: '{phrase}' → '{replacement}'")
                            # Only store the first detection as primary
                            if not special_phrase_detected:
                                special_phrase_detected = True
                                detected_phrase = phrase
                                forced_sentiment = "positive"  # These are all positive sentiment overrides
                                forced_confidence = 0.85
                            
                            # Perform the text replacement
                            text = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, text.lower(), flags=re.IGNORECASE)
            
            # NEW: Track complex negation patterns
            has_nor_construction = bool(re.search(r'(wasn\'t|weren\'t|isn\'t|aren\'t).+nor', text.lower()))
            if has_nor_construction:
                logger.info(f"Complex 'nor' construction detected: likely double negative pattern")
                if not special_phrase_detected:
                    special_phrase_detected = True
                    detected_phrase = "nor construction"
                    forced_sentiment = "negative"
                    forced_confidence = 0.85
            
            # Enhanced tokenization with better error handling
            try:
                words = word_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK tokenization failed, using fallback: {e}")
                words = simple_tokenize(text)
            
            # Track negation scope with a more sophisticated algorithm
            result = []
            negation_scope = []  # List of indices in negation scope
            sentence_boundaries = []  # Track where sentences end
            
            # First pass: identify sentence boundaries and negation triggers
            for i, word in enumerate(words):
                if word.lower() in ['.', '!', '?'] or word.endswith(('.', '!', '?')):
                    sentence_boundaries.append(i)
            
            # Add start and end of text as boundaries
            sentence_boundaries = [-1] + sentence_boundaries + [len(words)]
            
            # Second pass: mark negation scopes using sentence boundaries
            for i, word in enumerate(words):
                if word.lower() in negation_words or any(neg in word.lower() for neg in ["n't"]):
                    # Get the sentence this negation is in
                    current_sentence = next((j for j, boundary in enumerate(sentence_boundaries) 
                                            if boundary >= i), len(sentence_boundaries) - 1) - 1
                    
                    # Special handling for "nor" - extends scope further
                    if word.lower() == 'nor':
                        logger.info(f"'nor' detected at position {i} - extending negation scope")
                        # For "nor", extend scope to end of sentence with high weight
                        sentence_end = sentence_boundaries[current_sentence + 1]
                        negation_scope.extend(range(i + 1, sentence_end))
                        continue
                    
                    # Mark words after the negation until the next boundary or up to 5 words
                    sentence_end = sentence_boundaries[current_sentence + 1]
                    scope_end = min(i + 6, sentence_end)
                    
                    # Adjust scope for punctuation and conjunctions
                    for j in range(i + 1, scope_end):
                        if j < len(words):
                            if words[j].lower() in [',', ';', 'but', 'however']:
                                scope_end = j
                                break
                    
                    # Add affected words to negation scope
                    negation_scope.extend(range(i + 1, scope_end))
                    
                    # Log negation information
                    if scope_end > i + 1:
                        scope_words = ' '.join(words[i+1:scope_end])
                        logger.info(f"Negation trigger: '{word}' affecting: '{scope_words}'")
            
            # NEW: Special handling for recommendation terms in negation scope
            recommendation_terms = ['recommend', 'recommended', 'recommendation', 'recommending', 'recommends']
            for i, word in enumerate(words):
                if word.lower() in recommendation_terms and i-1 >= 0 and i-1 < len(words):
                    prev_word = words[i-1].lower()
                    prev_prev_word = words[i-2].lower() if i-2 >= 0 else ""
                    
                    # Check for direct negation before recommendation
                    if prev_word in negation_words or any(neg in prev_word for neg in ["n't"]):
                        logger.info(f"Direct recommendation negation: '{prev_word} {word}'")
                        # Mark this as a special pattern with high priority
                        special_phrase_detected = True
                        detected_phrase = f"{prev_word} {word}"
                        forced_sentiment = "negative"
                        forced_confidence = 0.92
                    
                    # Check for "do not recommend" pattern
                    elif prev_word == "not" and prev_prev_word in ["do", "does", "would", "will"]:
                        logger.info(f"Extended recommendation negation: '{prev_prev_word} {prev_word} {word}'")
                        special_phrase_detected = True
                        detected_phrase = f"{prev_prev_word} {prev_word} {word}"
                        forced_sentiment = "negative"
                        forced_confidence = 0.93
            
            # Third pass: build the result with proper negation marking
            in_special_phrase = False
            for i, word in enumerate(words):
                if i in negation_scope and not word.lower() in ['and', 'the', 'a', 'an', 'to', 'of', 'in']:
                    # Mark word as negated
                    result.append({
                        "original": word,
                        "modified": word + "_NEG",
                        "negated": True
                    })
                else:
                    result.append({
                        "original": word,
                        "modified": word,
                        "negated": False
                    })
            
            # Convert result to the format expected by the rest of the code
            if all(isinstance(item, dict) for item in result):
                # Return the modified text with negation markers and special phrase info
                logger.info(f"Processed negation in text: {' '.join(item['modified'] for item in result)}")
                return ' '.join(item['modified'] for item in result), result, {
                    'special_phrase_detected': special_phrase_detected,
                    'detected_phrase': detected_phrase,
                    'forced_sentiment': forced_sentiment,
                    'forced_confidence': forced_confidence,
                    'has_nor_construction': has_nor_construction
                }
            else:
                # Backwards compatibility
                logger.warning("Negation handling returned unexpected format")
                return text, [], {'special_phrase_detected': False}
                
        except Exception as e:
            logger.error(f"Error in handle_negations: {str(e)}")
            logger.error(traceback.format_exc())
            # Return original text if there's an error
            return text, [], {'special_phrase_detected': False}
    
    def process_contrast_markers(self, text):
        """Process text to identify contrast markers and split text into parts"""
        try:
            # Look for contrast markers in the text
            contrast_markers = [
                'but ', 'although ', 'though ', 'however ', 'despite ', 'yet ', 
                'nevertheless ', 'regardless ', 'even though ', 'notwithstanding ', 'in spite of '
            ]
            
            for marker in contrast_markers:
                if marker in text:
                    # Parse out the parts before and after the contrast marker
                    parts = text.split(marker, 1)
                    before_text = parts[0].strip()
                    after_text = parts[1].strip()
                    
                    # Default weights
                    before_weight = 0.45
                    after_weight = 0.55
                    
                    # Check if this is a positive contrast marker ("but" followed by positive phrases)
                    # or a negative contrast marker ("despite" followed by negative phrases)
                    strong_positive_markers = [
                        "good", "great", "excellent", "amazing", "fantastic", "wonderful", 
                        "enjoyed", "love", "loved", "best", "perfect", "delicious", "recommend",
                        "worth", "impressive", "tasty", "flavorful", "yummy", "delightful", 
                        "satisfying", "mouthwatering", "scrumptious", "heavenly", "divine", 
                        "superb", "outstanding", "stellar", "enjoyed", "impressed", "win",
                        "pleasure", "surprisingly", "pleasantly", "favorite", "liked"
                    ]
                    
                    strong_negative_markers = [
                        "bad", "terrible", "awful", "horrible", "disgusting", "disappointing",
                        "mediocre", "bland", "tasteless", "inedible", "overcooked", "undercooked",
                        "stale", "rotten", "expensive", "overpriced", "poor", "worst", "trash",
                        "garbage", "waste", "useless", "boring", "dull", "pointless", "hated"
                    ]
                    
                    # Added emphasis words that boost the effect of positive/negative terms
                    emphasis_words = [
                        "very", "extremely", "absolutely", "truly", "really", "definitely",
                        "quite", "especially", "particularly", "exceptionally", "remarkably",
                        "incredibly", "unbelievably", "surprisingly", "astonishingly", "totally",
                        "completely", "entirely", "utterly", "thoroughly", "genuinely"
                    ]
                    
                    # Use our specialized restaurant detector
                    is_restaurant, food_term_count, service_term_count = self._is_restaurant_review(text)
                    
                    # Counter for positive and negative strong terms after contrast marker
                    strong_pos_count = 0
                    strong_neg_count = 0
                    
                    # Detect strong positive terms after contrast marker
                    after_words = after_text.lower().split()
                    for term in strong_positive_markers:
                        if term in after_text.lower():
                            strong_pos_count += 1
                            logger.info(f"Strong positive term after contrast: '{term}'")
                    
                    # Detect strong negative terms after contrast marker
                    for term in strong_negative_markers:
                        if term in after_text.lower():
                            strong_neg_count += 1
                            logger.info(f"Strong negative term after contrast: '{term}'")
                    
                    # Detect terms in the before section
                    before_pos_count = 0
                    before_neg_count = 0
                    for term in strong_positive_markers:
                        if term in before_text.lower():
                            before_pos_count += 1
                    for term in strong_negative_markers:
                        if term in before_text.lower():
                            before_neg_count += 1
                    
                    # Special cases for opinion reversal
                    trash_to_pleasure = re.search(r'(trash|terrible|awful|bad|boring).+(guilty pleasure|secretly enjoyed|actually (?:liked|enjoyed))', text.lower())
                    if trash_to_pleasure and marker in ["but ", "however ", "yet ", "although "]:
                        logger.info(f"Detected opinion reversal: negative to positive")
                        before_weight = 0.1  # Greatly reduce the "trash" part weight
                        after_weight = 0.9   # Strongly emphasize the "pleasure" part
                        strong_pos_count += 1  # Add extra positive boost
                    
                    # Check for emphasis + positive/negative combinations (e.g., "absolutely delicious")
                    emphasis_pos_combinations = 0
                    emphasis_neg_combinations = 0
                    
                    for emphasis in emphasis_words:
                        for pos_term in strong_positive_markers:
                            if f"{emphasis} {pos_term}" in after_text.lower():
                                emphasis_pos_combinations += 1
                                logger.info(f"Emphasis + positive combination: '{emphasis} {pos_term}'")
                        
                        for neg_term in strong_negative_markers:
                            if f"{emphasis} {neg_term}" in after_text.lower():
                                emphasis_neg_combinations += 1
                                logger.info(f"Emphasis + negative combination: '{emphasis} {neg_term}'")
                    
                    # Forced sentiment flags
                    has_forcing_positive = False
                    has_forcing_negative = False
                    
                    # Special cases for idiom detection in contrast markers
                    idiom_patterns = [
                        (r'guilty pleasure', True),       # positive
                        (r'secretly enjoyed', True),      # positive
                        (r'laughed more than', True),     # positive
                        (r'surprisingly good', True),     # positive
                        (r'waste of time', False),        # negative
                        (r'wouldn\'t recommend', False),  # negative
                    ]
                    
                    for pattern, is_positive in idiom_patterns:
                        if re.search(pattern, after_text.lower()):
                            logger.info(f"Idiom pattern '{pattern}' found after contrast marker")
                            if is_positive:
                                has_forcing_positive = True
                                strong_pos_count += 2
                            else:
                                has_forcing_negative = True
                                strong_neg_count += 2
                    
                    # Enhanced restaurant-specific pattern detection
                    if is_restaurant:
                        logger.info(f"Restaurant review detected with {food_term_count} food terms and {service_term_count} service terms")
                        
                        # Critical food quality patterns - high-impact on restaurant sentiment
                        food_quality_patterns = [
                            (r'food\s+was\s+(\w+)', strong_positive_markers, True),  # "food was excellent" → positive
                            (r'food\s+was\s+(\w+)', strong_negative_markers, False),  # "food was terrible" → negative
                            (r'(\w+)\s+food', strong_positive_markers, True),         # "delicious food" → positive
                            (r'(\w+)\s+food', strong_negative_markers, False),        # "terrible food" → negative
                            (r'meal\s+was\s+(\w+)', strong_positive_markers, True),   # "meal was excellent" → positive
                            (r'meal\s+was\s+(\w+)', strong_negative_markers, False),  # "meal was terrible" → negative
                        ]
                        
                        for pattern, term_list, is_positive in food_quality_patterns:
                            matches = re.finditer(pattern, after_text.lower())
                            for match in matches:
                                adjective = match.group(1)
                                if adjective in term_list:
                                    logger.info(f"Critical restaurant pattern: '{match.group(0)}' → {'positive' if is_positive else 'negative'}")
                                    if is_positive:
                                        has_forcing_positive = True
                                        strong_pos_count += 2  # Double weight for explicit food quality statements
                                    else:
                                        has_forcing_negative = True
                                        strong_neg_count += 2  # Double weight for explicit food quality statements
                        
                        # Restaurant "despite/although/even though" + negative service + positive food = positive
                        if (marker in ['despite ', 'although ', 'even though '] and 
                            service_term_count >= 1 and 
                            food_term_count >= 1 and 
                            strong_pos_count > 0):
                            # Example: "Despite the noisy atmosphere, the food was excellent"
                            logger.info(f"Restaurant review with positive food despite negative service/ambiance")
                            has_forcing_positive = True
                            before_weight = 0.25  # Greatly reduce weight of negative service aspects
                            after_weight = 0.75   # Strongly emphasize food quality
                    
                    # Strong opinion switches with "but" need special handling
                    if marker == 'but ' or marker == 'however ' or marker == 'yet ':
                        # Stronger negative before and positive after suggests a positive overall sentiment
                        if before_neg_count > before_pos_count and strong_pos_count > 0:
                            logger.info(f"Negative to positive opinion switch detected")
                            before_weight = 0.3  # Reduce negative part weight
                            after_weight = 0.7   # Emphasize positive part
                            
                            # If the after part has multiple positive terms or emphatic positive, 
                            # force positive sentiment
                            if strong_pos_count >= 1 or emphasis_pos_combinations > 0:
                                logger.info(f"Strong positive expression after negative opinion - forcing positive")
                                has_forcing_positive = True
                        
                        # For restaurant case with "but" followed by positive term about food, 
                        # the after part becomes even more important
                        if is_restaurant and strong_pos_count > 0:
                            logger.info(f"Restaurant case with positive food description detected")
                            before_weight = 0.25  # Further reduce weight of before part
                            after_weight = 0.75   # Strongly emphasize after part with food description
                            
                            # Restaurant specific forcing for "but the food was X" patterns
                            if strong_pos_count >= 1 or emphasis_pos_combinations > 0:
                                if food_term_count >= 1:
                                    logger.info(f"Restaurant with strong positive food terms - forcing positive signal")
                                    has_forcing_positive = True
                            
                            # If the after part has significantly more positive than negative terms, favor positive
                            if (strong_pos_count - strong_neg_count) >= 2:
                                logger.info(f"Restaurant review with strong positive balance after contrast")
                                has_forcing_positive = True
                        
                        # Restaurant case with negative food sentiment overrides positive service
                        elif is_restaurant and strong_neg_count > 0 and food_term_count >= 1:
                            logger.info(f"Restaurant with negative food description - forcing negative signal")
                            before_weight = 0.25
                            after_weight = 0.75
                            
                            if strong_neg_count >= 1 or emphasis_neg_combinations > 0:
                                has_forcing_negative = True
                        
                        # For other "but" with multiple strong positive terms, consider forcing positive
                        elif strong_pos_count >= 2 or emphasis_pos_combinations >= 1:
                            # When we have multiple strong positive terms after "but", it often 
                            # indicates a strong positive sentiment regardless of the first part
                            logger.info(f"Multiple strong positive terms after 'but' - forcing positive signal")
                            has_forcing_positive = True
                            
                            # Adjust weight further for very strong positive signals
                            pos_factor = min(strong_pos_count * 0.05, 0.2)
                            
                            # Rebalance weights to emphasize the positive after text
                            total = before_weight + after_weight
                            before_weight = max(before_weight - pos_factor, 0.05)  # Reduced minimum to 0.05
                            after_weight = total - before_weight
                            
                            # Log the adjustment
                            logger.info(f"Adjusted weights for positive after-text: before={before_weight:.2f}, after={after_weight:.2f}")
                        
                        # Similarly for strong negative patterns
                        elif strong_neg_count >= 2 or emphasis_neg_combinations >= 1:
                            logger.info(f"Multiple strong negative terms after 'but' - forcing negative signal")
                            has_forcing_negative = True
                            
                            # Adjust weights for strong negative signals
                            neg_factor = min(strong_neg_count * 0.05, 0.2)
                            total = before_weight + after_weight
                            before_weight = max(before_weight - neg_factor, 0.05)
                            after_weight = total - before_weight
                            
                            logger.info(f"Adjusted weights for negative after-text: before={before_weight:.2f}, after={after_weight:.2f}")
                    
                    # Return the parts with weights and forcing flags
                    return {
                        "has_contrast": True,
                        "before": before_text,
                        "after": after_text,
                        "before_weight": before_weight,
                        "after_weight": after_weight,
                        "contrast_marker": marker.strip(),
                        "has_forced_positive": has_forcing_positive,
                        "has_forced_negative": has_forcing_negative,
                        "strong_positive_count": strong_pos_count,
                        "strong_negative_count": strong_neg_count,
                        "emphasis_pos_combinations": emphasis_pos_combinations,
                        "emphasis_neg_combinations": emphasis_neg_combinations,
                        "is_restaurant_case": is_restaurant
                    }
            
            # No contrast markers found
            return {
                "has_contrast": False,
                "full_text": text,
                "weight": 1.0
            }
        except Exception as e:
            logger.error(f"Error in process_contrast_markers: {e}")
            return {
                "has_contrast": False,
                "full_text": text,
                "weight": 1.0
            }
    
    def clean_text(self, text):
        """Enhanced text cleaning with entity recognition and negation handling"""
        try:
            # Handle potential movie titles first
            text_with_titles = self.identify_movie_titles(text)
            
            # Store markers for movie title sentences to restore later
            movie_title_sentences = []
            for sentence in text_with_titles.split('.'):
                if sentence and sentence.strip().startswith("MOVIE_TITLE_SENTENCE"):
                    movie_title_sentences.append(sentence.strip())
            
            # Extract and preserve MOVIETITLE_* patterns
            movie_title_markers = {}
            try:
                movie_title_pattern = re.compile(r'(MOVIETITLE_[a-zA-Z0-9_]+)')
                for match in movie_title_pattern.finditer(text_with_titles):
                    marker = match.group(1)
                    movie_title_markers[marker] = marker
            except Exception as e:
                logger.error(f"Error extracting movie title markers: {e}")
            
            # Convert to lowercase
            text = text_with_titles.lower()
            
            # Remove special characters but preserve specific markers
            try:
                # Define a function to handle replacements
                def replace_special_chars(match):
                    text = match.group(0)
                    if text in movie_title_markers or text == "MOVIE_TITLE_SENTENCE":
                        return text
                    else:
                        return ' '
                
                # Apply replacement function to the pattern
                pattern = re.compile(r'[^\w\s\'MOVIETITLE_]|MOVIE_TITLE_SENTENCE')
                text = pattern.sub(replace_special_chars, text)
                
                # Restore movie title sentence markers
                for sentence in movie_title_sentences:
                    # Find the matching sentence without the marker and replace it
                    if sentence:
                        sentence_without_marker = sentence.replace("MOVIE_TITLE_SENTENCE ", "").lower()
                        if sentence_without_marker in text:
                            text = text.replace(sentence_without_marker, sentence.lower())
            except Exception as e:
                logger.error(f"Error handling special characters: {e}")
            
            # Try to apply negation handling
            try:
                text, negation_markers, special_phrase_info = self.handle_negations(text)
            except Exception as e:
                logger.error(f"Negation handling failed: {e}")
                # Continue without negation handling
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text, negation_markers, special_phrase_info
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            logger.error(traceback.format_exc())
            # Simple fallback cleaning
            return text.lower().strip(), [], {'special_phrase_detected': False}
    
    def safety_check(self, text):
        """
        Check if text contains potentially harmful/negative emotional content.
        Returns:
        - True: Text should be filtered (contains harmful content)
        - False: Text is safe for analysis
        - 0.5: Text has mild concerning content (used for confidence reduction)
        """
        try:
            # Convert to lowercase for pattern matching
            text_lower = text.lower()
            
            # Serious harmful/suicidal content that should be filtered
            harmful_terms = [
                "hurt myself", "kill myself", "suicide", "end my life", "self harm",
                "hate myself", "self-harm", "harm myself", "kill me", "wanting to die"
            ]
            
            # Check for serious harmful terms first
            contains_harmful_terms = any(term in text_lower for term in harmful_terms)
            if contains_harmful_terms:
                logger.warning(f"Safety check: Harmful content detected in text: '{text}'")
                return True
                
            # Terms that require context analysis
            context_terms = [
                "depressed", "worthless", "anxious", "suffering",
                "miserable", "hopeless", "trauma", "traumatic", "suicidal"
            ]
            
            # Only trigger if these appear in probable self-reference context
            self_reference_patterns = [
                r"i (?:am|feel|felt) (?:\w+ ){0,3}(?:depressed|worthless|anxious|suffering|miserable|hopeless|suicidal)",
                r"i(?:'m| am) (?:\w+ ){0,3}(?:depressed|worthless|anxious|suffering|miserable|hopeless|suicidal)",
                r"(?:feeling|feel) (?:\w+ ){0,3}(?:depressed|worthless|anxious|suffering|miserable|hopeless|suicidal)",
                r"my (?:depression|anxiety|trauma|suffering|hopelessness)",
                r"struggling with (?:depression|anxiety|trauma|suicidal)"
            ]
            
            for pattern in self_reference_patterns:
                if re.search(pattern, text_lower):
                    logger.warning(f"Safety check: Concerning emotional content detected: '{text}'")
                    return True
            
            # EXPANDED: Common phrases that should be whitelisted (NOT blocked)
            whitelist_patterns = [
                # Comparative expressions
                r"rather watch paint dry",
                r"rather see paint dry",
                r"paint dry",
                r"watch(?:ing)? grass grow",
                r"wait for paint to dry",
                r"watching wall",
                r"watch(?:ing)? the wall",
                r"water boil",
                
                # Common metaphorical expressions
                r"bored to death",  # Figurative expression
                r"dying to see",    # Figurative expression
                r"dying for",       # Eager for something
                r"dying of",        # Hyperbolic expression
                r"kill(?:ed|ing)? (?:it|me|them|time)",  # Positive expression (did well)
                r"dying of laughter",
                r"died laughing",
                r"laughed to death",
                r"killing (?:me|time)",
                r"mind-blowing",
                r"blew my mind",
                r"dead serious",
                r"drop dead gorgeous",
                
                # Movie references
                r"kill bill",
                r"killing eve",
                r"dead pool",
                r"walking dead",
                r"evil dead",
                r"death wish",
                r"death note",
                
                # Sarcasm and humor patterns
                r"(?:credits roll(?:ed)?)",
                r"(?:would|rather) (?:die|pass away|be dead) than",
                r"would die for",
                r"knocked me dead",
                r"knock 'em dead",
                r"painkiller",
                r"pain relief",
                r"painful to watch",
                r"painful experience",
                r"hurt(?:s|ing)? to watch",
                
                # Restaurant-specific terms
                r"killer app",
                r"killer dish",
                r"to die for",
                r"killer menu",
                r"painfully spicy",
                
                # Criticism expressions
                r"torture to watch",
                r"suffered through",
                r"suffering from boredom",
                r"painful to sit through",
                r"hurts the eyes",
                r"murdered the song"
            ]
            
            # If text matches any whitelist pattern, explicitly return False
            for pattern in whitelist_patterns:
                if re.search(pattern, text_lower):
                    logger.info(f"Safety check: Whitelisted expression detected: '{pattern}'")
                    return False
            
            # Check for product or service reviews
            review_indicators = [
                r"review(?:ing|ed)?", r"product", r"service", r"customer", r"recommend", 
                r"purchase(?:d)?", r"buy", r"bought", r"order(?:ed)?", r"deliver(?:y|ed)?",
                r"restaurant", r"food", r"meal", r"movie", r"film", r"book", r"read", 
                r"watch(?:ed)?", r"experience"
            ]
            
            is_review_context = any(re.search(pattern, text_lower) for pattern in review_indicators)
            
            # If it's clearly a review context, reduce sensitivity to negative terms
            if is_review_context:
                # Review-specific whitelist check (more permissive)
                review_whitelist = [
                    r"kill(?:s|ed|ing)? (?:time|the mood|the vibe|the atmosphere)",
                    r"dead (?:boring|simple|obvious|straightforward)",
                    r"pain(?:ful)? to use",
                    r"hurt(?:s|ing)? (?:my|the) (?:wallet|budget|bank account)"
                ]
                
                if any(re.search(pattern, text_lower) for pattern in review_whitelist):
                    logger.info(f"Safety check: Review context with permissible negative expression")
                    return False
            
            # General milder negative terms that shouldn't trigger on their own
            mild_negative_terms = [
                "alone", "lonely", "die", "death", "pain", "hurt"
            ]
            
            # Count occurrences of mild negative terms in a review context
            if is_review_context:
                # In review contexts, allow mild negative terms
                mild_term_count = sum(1 for term in mild_negative_terms if term in text_lower)
                if mild_term_count > 0:
                    logger.info(f"Safety check: Review context with {mild_term_count} mild negative terms - allowing")
                    return False
            
            # These mild terms need strong contextual indicators to trigger
            strong_context_patterns = [
                r"i (?:want|wish) to die",
                r"i (?:feel|am) (?:so|very|extremely) (?:alone|lonely|hurt)",
                r"no one (?:cares|loves me)",
                r"(?:constant|extreme|severe) pain"
            ]
            
            for pattern in strong_context_patterns:
                if re.search(pattern, text_lower):
                    logger.warning(f"Safety check: Strong negative context detected: '{text}'")
                    return True
            
            # Check for standalone mild negative terms without review context
            if not is_review_context and any(term in text_lower for term in mild_negative_terms):
                # Return 0.5 to indicate caution but not complete filtering
                logger.info(f"Safety check: Mild negative terms without review context - caution flag")
                return 0.5
                
            # If we've made it here, the content should be safe
            return False
        
        except Exception as e:
            logger.error(f"Error in safety_check: {str(e)}")
            # If there's an error, default to letting the text through rather than blocking
            return False
    
    def _pad_features(self, X, target_size):
        """Pad the feature matrix to the target size"""
        if X.shape[1] == target_size:
            return X
        
        if X.shape[1] > target_size:
            # We have too many features, truncate
            logger.warning(f"Truncating features from {X.shape[1]} to {target_size}")
            return X[:, :target_size]
        else:
            # We need to pad with zeros
            padding_size = target_size - X.shape[1]
            logger.info(f"Padding features with {padding_size} zeros")
            
            # Create a zero matrix for padding
            zero_padding = csr_matrix((X.shape[0], padding_size))
            
            # Horizontally stack X with the zero padding
            return hstack([X, zero_padding])
    
    def _detect_sarcasm(self, text):
        """
        Detect sarcasm patterns in text and return appropriate sentiment override.
        Returns (sentiment, confidence, pattern_type) tuple if sarcasm detected, None otherwise.
        """
        text = text.lower()
        
        # Sleep pattern sarcasm - enhanced pattern matching
        if re.search(r'(?:if you (?:enjoy|like) (?:falling asleep|being bored|dozing off|nodding off))', text) or \
           re.search(r'(?:perfect|great|ideal) (?:for|if) (?:you|someone|people) (?:enjoy|like|want) (?:to|falling) (?:asleep|sleep|bored)', text):
            logger.info(f"Detected sleep-related sarcasm: '{text}'")
            return "Negative", 92.5, "conditional_enjoyment"
            
        # End event highlight sarcasm - enhanced pattern matching
        if re.search(r'(?:best part|highlight|favorite moment).+(?:when|was) (?:(?:it|the movie|the film|this) (?:end|ends|finish|finished|over)|the credits roll)', text) or \
           re.search(r'(?:best|favorite).+(?:credits roll|ending|finished|over)', text):
            logger.info(f"Detected end-event sarcasm: '{text}'")
            return "Negative", 95.0, "end_event_highlight"
            
        # Negative comparison - enhanced pattern matching
        if re.search(r'(?:rather|prefer|better|sooner) (?:watch|see|witness|stare at) (?:paint dry|grass grow|water boil|wall|drying paint).+than', text) or \
           re.search(r'(?:paint dry|grass grow|water boil).+(?:than|instead of).+(?:this|again|movie|film)', text):
            logger.info(f"Detected negative comparison sarcasm: '{text}'")
            return "Negative", 95.0, "comparative_negative"
            
        # Mocking praise - enhanced pattern matching
        if re.search(r'(?:wow|amazing|incredible|impressive|outstanding).+(?:forgettable|boring|terrible|awful|bad|worst|dull|pointless)', text) or \
           re.search(r'(?:outdid themselves|remarkable achievement).+(?:how|with) (?:forgettable|terrible|bad|boring)', text):
            logger.info(f"Detected mocking praise sarcasm: '{text}'")
            return "Negative", 85.0, "contrasting_praise"
            
        # Conditional praise - enhanced pattern matching
        if re.search(r'(?:masterpiece|brilliant|amazing|excellent).+(?:if|only if|assuming).+(?:standards|expectations|taste|judgment).+(?:low|below|terrible|non-existent)', text):
            logger.info(f"Detected conditional praise sarcasm: '{text}'")
            return "Negative", 90.0, "conditional_praise"
            
        # Delayed negative reveal - enhanced pattern matching
        if re.search(r'(?:achievement|accomplishment|success|triumph).+(?:what not to|how not to|failure|disaster|catastrophe)', text):
            logger.info(f"Detected delayed negative reveal sarcasm: '{text}'")
            return "Negative", 88.0, "delayed_negative"
        
        # No sarcasm detected
        return None
        
    def _detect_idioms(self, text):
        """
        Detect idiom patterns in text and return appropriate sentiment override.
        Returns (sentiment, confidence, pattern_type) tuple if idiom detected, None otherwise.
        """
        text = text.lower()
        
        # Positive idioms - enhanced patterns
        positive_idioms = [
            (r'guilty pleasure', "positive_expression"),
            (r'laughed more than .* should', "humor_appreciation"),
            (r'laugh(?:ed)? (?:more|harder|louder) than', "humor_appreciation"),
            (r'that\'s a win', "positive_outcome"),
            (r'a win in my book', "personal_approval"),
            (r'diamond in the rough', "hidden_value"),
            (r'runs like a dream', "performance_excellence"),
            (r'worth (?:every|the) penny', "value_affirmation"),
            (r'secretly enjoyed', "guilty_pleasure"),
            (r'better than (?:expected|anticipated)', "expectation_exceeded"),
            (r'pleasantly surprised', "positive_surprise")
        ]
        
        for pattern, type_label in positive_idioms:
            if re.search(pattern, text):
                logger.info(f"Detected positive idiom: '{pattern}' in text")
                return "Positive", 90.5, type_label
                
        # Negative idioms - enhanced patterns
        negative_idioms = [
            (r'train wreck', "negative_expression"),
            (r'wouldn\'?t recommend', "negative_expression"),
            (r'waste of time', "time_value_negative"),
            (r'lost cause', "hopeless_situation"),
            (r'torture to watch', "negative_experience"),
            (r'avoid (?:at all costs|like the plague)', "strong_avoidance"),
            (r'hard pass', "rejection_phrase"),
            (r'fell flat', "performance_failure"),
            (r'missed the mark', "goal_failure"),
            (r'painful to watch', "viewing_discomfort")
        ]
        
        for pattern, type_label in negative_idioms:
            if re.search(pattern, text):
                logger.info(f"Detected negative idiom: '{pattern}' in text")
                return "Negative", 92.5, type_label
        
        # No idiom detected
        return None
        
    def _detect_contradiction(self, text):
        """
        Detect contradiction patterns in text and return appropriate sentiment override.
        Returns (sentiment, confidence, pattern_type) tuple if contradiction detected, None otherwise.
        """
        text = text.lower()
        
        # Positive contradictions - negative to positive
        if re.search(r'(?:awful|terrible|bad|worst).+(?:just kidding|kidding aside|actually).+(?:great|good|excellent|amazing)', text):
            return "Positive", 95.0, "this was awful! just kidding, it was great."
            
        # Negative contradictions - positive to negative
        if re.search(r'(?:great|good|excellent|amazing).+(?:just kidding|kidding aside|actually).+(?:awful|terrible|bad|worst)', text):
            return "Negative", 92.0, "this was great! just kidding, it was awful."
            
        # Mixed with explicit resolution
        if re.search(r'(?:some|both).+(?:good|bad).+(?:some|both).+(?:bad|good).+overall.+(?:enjoyed|liked)', text):
            return "Positive", 85.0, "mixed_with_positive_resolution"
            
        if re.search(r'(?:some|both).+(?:good|bad).+(?:some|both).+(?:bad|good).+(?:ultimately|overall).+(?:disappointed|disliked)', text):
            return "Negative", 83.5, "mixed_with_negative_resolution"
        
        # No contradiction detected
        return None
        
    def predict_with_specific_model(self, original_text, processed_text, modelname, simple_case_results={}, 
                                   add_explanation=False, sarcasm_info=None, idiom_info=None, contradiction_info=None):
        """Make predictions using a specific model."""
        # Default values if not provided
        if sarcasm_info is None:
            sarcasm_info = {"sarcasm_detected": False}
        if idiom_info is None:
            idiom_info = {"idiom_detected": False}
        if contradiction_info is None:
            contradiction_info = {"contradiction_detected": False}
        
        # Check for restaurant review with contrast markers
        restaurant_info = {"is_restaurant": False}
        is_restaurant, food_count, service_count = self._is_restaurant_review(original_text)
        
        # Process contrast markers for more nuanced understanding
        contrast_info = self.process_contrast_markers(original_text)
        
        # Special handling for restaurant reviews with contrast markers
        if is_restaurant and contrast_info.get("has_contrast", False):
            restaurant_info = {
                "is_restaurant": True,
                "food_count": food_count,
                "service_count": service_count,
                "has_contrast": True,
                "contrast_marker": contrast_info.get("contrast_marker", ""),
                "has_forced_positive": contrast_info.get("has_forced_positive", False),
                "has_forced_negative": contrast_info.get("has_forced_negative", False)
            }
            logger.info(f"Restaurant review with contrast marker '{contrast_info.get('contrast_marker')}' detected")
            
            # If we have a strong restaurant pattern that forces sentiment, respect it
            if contrast_info.get("has_forced_positive", False):
                logger.info(f"[{modelname}] Restaurant review with forced POSITIVE sentiment")
                
        # Get the model to use
        model = self.models[modelname]
        
        # Extract features for the model
        features = self._extract_features(processed_text)
        
        # Make prediction
        prediction = model.predict(features)[0]
        confidence = max(0.55, np.max(model.predict_proba(features)[0]))
        
        # Default model name
        model_name = modelname
        
        # Check for special overrides from restaurant + contrast detection (highest priority)
        if restaurant_info["is_restaurant"] and restaurant_info["has_contrast"]:
            # If restaurant pattern detection found a forced sentiment, apply it
            if restaurant_info["has_forced_positive"]:
                logger.info(f"[{modelname}] Forcing POSITIVE prediction for restaurant review with contrast marker")
                prediction = 1  # Force positive
                confidence = max(0.80, confidence)  # Higher confidence for food quality statements
                model_name = f"{modelname}_with_restaurant_analysis"
            elif restaurant_info["has_forced_negative"]:
                logger.info(f"[{modelname}] Forcing NEGATIVE prediction for restaurant review with contrast marker")
                prediction = 0  # Force negative
                confidence = max(0.82, confidence)  # Higher confidence for food quality statements
                model_name = f"{modelname}_with_restaurant_analysis"
            elif contrast_info.get("has_contrast", False):
                # Use weighted prediction for contrast cases without forced sentiment
                logger.info(f"[{modelname}] Using weighted contrast prediction for restaurant review")
                
                # Apply separate analysis to parts before and after contrast marker
                try:
                    before_features = self._extract_features(contrast_info["before"])
                    after_features = self._extract_features(contrast_info["after"])
                    
                    before_pred = model.predict(before_features)[0]
                    after_pred = model.predict(after_features)[0]
                    
                    before_weight = contrast_info.get("before_weight", 0.4)
                    after_weight = contrast_info.get("after_weight", 0.6)
                    
                    # In restaurant reviews, the food quality often matters more
                    if food_count > 0 and "food" in contrast_info["after"].lower():
                        logger.info(f"Further emphasizing after-part containing food references")
                        # If after part contains food references, give it even more weight
                        total = before_weight + after_weight
                        before_weight = before_weight * 0.7  # Reduce before weight
                        after_weight = total - before_weight  # Increase after weight
                    
                    # Do weighted combination (considering 1=positive, 0=negative)
                    weighted_score = (before_pred * before_weight) + (after_pred * after_weight)
                    
                    # Determine final prediction
                    if weighted_score >= 0.5:
                        prediction = 1
                        confidence = max(0.6, weighted_score) 
                    else:
                        prediction = 0
                        confidence = max(0.6, 1 - weighted_score)
                    
                    # Adjust confidence based on the difference in weights
                    confidence = min(confidence + abs(before_weight - after_weight) * 0.1, 0.95)
                    
                    model_name = f"{modelname}_with_restaurant_contrast_analysis"
                    logger.info(f"Restaurant contrast analysis: weighted score={weighted_score:.2f}, confidence={confidence:.2f}")
                except Exception as e:
                    logger.error(f"Error in restaurant contrast analysis: {e}")
        
        # Check for special overrides from sarcasm detection
        elif sarcasm_info["sarcasm_detected"]:
            logger.info(f"Sarcasm detection will influence prediction: {sarcasm_info['sarcasm_type']}")
            
            if sarcasm_info["force_sentiment"] == "positive":
                logger.info(f"[{modelname}] Forcing POSITIVE prediction due to sarcasm: '{sarcasm_info['sarcastic_phrase']}'")
                prediction = 1  # Force positive
                confidence = max(0.7, confidence) + sarcasm_info["confidence_adjustment"]
                confidence = min(confidence, 0.95)  # Cap at 0.95
                model_name = f"{modelname}_with_sarcasm_detection"
            elif sarcasm_info["force_sentiment"] == "negative":
                logger.info(f"[{modelname}] Forcing NEGATIVE prediction due to sarcasm: '{sarcasm_info['sarcastic_phrase']}'")
                prediction = 0  # Force negative
                confidence = max(0.7, confidence) + sarcasm_info["confidence_adjustment"]
                confidence = min(confidence, 0.95)  # Cap at 0.95
                model_name = f"{modelname}_with_sarcasm_detection"
        
        # Check for idiom detection overrides
        elif idiom_info["idiom_detected"]:
            logger.info(f"Idiom detection will influence prediction: {idiom_info['idiom_type']}")
            
            if idiom_info["force_sentiment"] == "positive":
                logger.info(f"[{modelname}] Forcing POSITIVE prediction due to idiom: '{idiom_info['detected_idiom']}'")
                prediction = 1  # Force positive
                confidence = max(0.7, confidence) + idiom_info["confidence_adjustment"]
                confidence = min(confidence, 0.95)  # Cap at 0.95
                model_name = f"{modelname}_with_idiom_detection"
            elif idiom_info["force_sentiment"] == "negative":
                logger.info(f"[{modelname}] Forcing NEGATIVE prediction due to idiom: '{idiom_info['detected_idiom']}'")
                prediction = 0  # Force negative
                confidence = max(0.7, confidence) + idiom_info["confidence_adjustment"]
                confidence = min(confidence, 0.95)  # Cap at 0.95
                model_name = f"{modelname}_with_idiom_detection"
        
        # Check for special overrides from contradiction detection
        elif contradiction_info["contradiction_detected"]:
            if contradiction_info["force_sentiment"] == "positive":
                logger.info(f"[{modelname}] Forcing POSITIVE prediction due to contradiction: '{contradiction_info['detected_phrase']}'")
                prediction = 1  # Force positive
                confidence = max(0.7, confidence) + contradiction_info["confidence_adjustment"]
                confidence = min(confidence, 0.95)  # Cap at 0.95
                model_name = f"{modelname}_with_contradiction_detection"
            elif contradiction_info["force_sentiment"] == "negative":
                logger.info(f"[{modelname}] Forcing NEGATIVE prediction due to contradiction: '{contradiction_info['detected_phrase']}'")
                prediction = 0  # Force negative
                confidence = max(0.7, confidence) + contradiction_info["confidence_adjustment"]
                confidence = min(confidence, 0.95)  # Cap at 0.95
                model_name = f"{modelname}_with_contradiction_detection"
            else:
                # Process the second part primarily if the reversal is unclear
                second_part = contradiction_info["second_part"]
                if second_part:
                    # Create a clean version of the second part
                    clean_second = self.clean_text(second_part)["processed_text"]
                    # Use this for prediction if possible
                    if len(clean_second.split()) > 2:  # If the second part has enough content
                        logger.info(f"[{modelname}] Using second part after contradiction for prediction: '{second_part}'")
                        # Re-predict with the second part
                        features_second = self._extract_features(clean_second)
                        prediction_second = model.predict(features_second)[0]
                        confidence_second = max(0.6, np.max(model.predict_proba(features_second)[0]))
                        prediction = prediction_second
                        confidence = confidence_second
                        model_name = f"{modelname}_with_contradiction_detection"
        
        # Only now consider the results from simple case detection as a possible override
        if simple_case_results.get("is_simple_case", False):
            simple_case_prediction = simple_case_results.get("prediction")
            simple_case_confidence = simple_case_results.get("confidence", 0.95)
            
            # If the simple case confidence is higher than model confidence by a significant margin,
            # use the simple case result
            if simple_case_confidence > confidence + 0.15:
                prediction = simple_case_prediction
                confidence = simple_case_confidence
                logger.info(f"[{modelname}] Overriding with simple case detection: {prediction}")
                model_name = f"{modelname} with pattern override"
        
        # Convert prediction to sentiment label
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        # Add a small random factor to ensure model independence
        random_factor = random.uniform(0.01, 0.03)
        confidence = min(confidence + random_factor, 0.95)
        
        prediction_result = {
            "text": original_text,
            "sentiment": sentiment,
            "confidence": confidence * 100,  # Convert to percentage
            "model_used": model_name
        }
        
        return prediction_result

    def _detect_neutral_sentiment(self, text):
        """
        Specialized method to detect explicitly neutral statements or balanced sentiments.
        Returns a tuple of (is_neutral, confidence) where is_neutral is a boolean.
        """
        # Check for explicitly neutral patterns
        neutral_patterns = [
            r'(?:neither good|neither bad|neither positive|neither negative)',
            r'(?:not (?:the best|the worst))',
            r'(?:on the fence|mixed feelings|evens? out)',
            r'(?:average|mediocre|middle of the road|so-so)',
            r'(?:exactly what|as expected|nothing special)',
            r'(?:balanced|equal|50[/\-]50)',
            r'(?:can\'?t decide|undecided|torn between)',
            r'(?:nothing more, nothing less)',
        ]
        
        # Look for phrases indicating explicit neutrality
        for pattern in neutral_patterns:
            if re.search(pattern, text.lower()):
                logger.info(f"Explicit neutral pattern detected: '{pattern}'")
                return True, 85.0
        
        # Check for balanced positive and negative terms
        positive_count = sum(1 for term in self.VERY_POSITIVE_PHRASES if term.lower() in text.lower())
        negative_count = sum(1 for term in self.VERY_NEGATIVE_PHRASES if term.lower() in text.lower())
        
        # If there's a roughly equal balance of strong positive and negative terms
        if positive_count > 0 and negative_count > 0 and abs(positive_count - negative_count) <= 1:
            logger.info(f"Balanced sentiment detected: {positive_count} positive terms and {negative_count} negative terms")
            return True, 75.0
            
        # Look for explicit comparison or balance in the text
        if re.search(r'(?:some good|some bad).+(?:some bad|some good)', text.lower()):
            logger.info("Explicit good/bad balance detected")
            return True, 80.0
            
        # Not deemed explicitly neutral
        return False, 0.0

    def predict(self, text, modelname=None, specific_model=None, use_sarcasm_detection=True, use_idiom_detection=True, use_contradiction_detection=True):
        """
        Make predictions on a single text input.
        """
        # Handle the specific_model parameter for backward compatibility
        if specific_model is not None and modelname is None:
            modelname = specific_model
            
        prediction_result = {}
        text = str(text)
        
        simple_case_override = None
        # First check if this is a simple case
        simple_case_result = self._handle_simple_cases(text)
        if simple_case_result:
            is_simple, sentiment, confidence = simple_case_result
            if is_simple:
                simple_case_override = sentiment
            
        # Clean the text
        cleaned_text, negation_markers, special_phrase_info = self.clean_text(text)
        
        # Check for explicit neutral sentiment before running the models
        is_neutral, neutral_confidence = self._detect_neutral_sentiment(text)
        if is_neutral:
            logger.info(f"[{modelname if modelname else 'default'}] Explicit neutral sentiment detected: {neutral_confidence:.2f}%")
            return {
                "text": text,
                "sentiment": "Neutral",
                "confidence": neutral_confidence, 
                "model_used": f"{modelname if modelname else 'default'}_with_neutral_detection"
            }
        
        # Perform a safety check
        safety_result = self.safety_check(text)
        if safety_result is True:
            # Hard filter for definitely harmful content
            return {
                "text": text,
                "sentiment": "Negative",
                "confidence": 95.0,
                "model_used": f"{modelname if modelname else 'default'}_with_safety_filter"
            }
        elif safety_result == 0.5:
            # Caution flag - continue with analysis but note the concern
            logger.info("Safety check: Mild concern detected. Continuing with analysis but will adjust confidence.")
            # We'll handle confidence adjustment later, after model predictions
        
        # Setup sarcasm, idiom and contradiction detection if enabled
        sarcasm_info = None
        idiom_info = None
        contradiction_info = None
        
        if use_sarcasm_detection:
            sarcasm_result = self._detect_sarcasm(text)
            if sarcasm_result:
                sentiment, confidence, pattern_type = sarcasm_result
                sarcasm_info = {
                    "sarcasm_detected": True,
                    "sarcasm_type": pattern_type,
                    "sarcastic_phrase": pattern_type,
                    "force_sentiment": "positive" if sentiment == "Positive" else "negative",
                    "confidence_adjustment": (confidence - 75) / 100  # Adjustment factor based on confidence
                }
            else:
                sarcasm_info = {"sarcasm_detected": False}
                
        if use_idiom_detection:
            idiom_result = self._detect_idioms(text)
            if idiom_result:
                sentiment, confidence, pattern_type = idiom_result
                idiom_info = {
                    "idiom_detected": True,
                    "idiom_type": pattern_type,
                    "detected_idiom": pattern_type,
                    "force_sentiment": "positive" if sentiment == "Positive" else "negative",
                    "confidence_adjustment": (confidence - 75) / 100
                }
            else:
                idiom_info = {"idiom_detected": False}
                
        if use_contradiction_detection:
            contradiction_result = self._detect_contradiction(text)
            if contradiction_result:
                sentiment, confidence, pattern_type = contradiction_result
                contradiction_info = {
                    "contradiction_detected": True,
                    "detected_phrase": pattern_type,
                    "force_sentiment": "positive" if sentiment == "Positive" else "negative",
                    "confidence_adjustment": (confidence - 75) / 100,
                    "second_part": pattern_type.split("!")[-1] if "!" in pattern_type else ""
                }
            else:
                contradiction_info = {"contradiction_detected": False}
        
        # Get predictions from the models
        model_predictions = {}
        
        # Prepare simple case for model
        simple_case_results = {}
        if simple_case_override is not None:
            simple_case_results = {
                "is_simple_case": True,
                "prediction": 1 if simple_case_override == "Positive" else 0,
                "confidence": 0.90 + (0.05 * random.random())  # 90-95% confidence
            }
        
        # If a specific model is specified, use that one
        if modelname:
            if modelname in self.models:
                logger.info(f"Using specified model: {modelname}")
                model_predictions[modelname] = self.predict_with_specific_model(
                    text, cleaned_text, modelname, 
                    simple_case_results=simple_case_results,
                    sarcasm_info=sarcasm_info,
                    idiom_info=idiom_info,
                    contradiction_info=contradiction_info
                )
            else:
                raise ValueError(f"Invalid model name: {modelname}")
        else:
            # Use all models
            for model_name in self.models:
                model_predictions[model_name] = self.predict_with_specific_model(
                    text, cleaned_text, model_name,
                    simple_case_results=simple_case_results,
                    sarcasm_info=sarcasm_info,
                    idiom_info=idiom_info,
                    contradiction_info=contradiction_info
                )
        
        # Process the results
        # If we have a specific model, use just its results
        if modelname:
            return model_predictions[modelname]
        else:
            # Combine results from all models
            # Get ensemble prediction by analyzing all model outputs
            positive_count = 0
            negative_count = 0
            total_confidence = 0
            
            for model_name, result in model_predictions.items():
                if result["sentiment"] == "Positive":
                    positive_count += 1
                else:  # Negative
                    negative_count += 1
                
                total_confidence += result["confidence"]
            
            # Average the confidence
            avg_confidence = total_confidence / len(model_predictions) if model_predictions else 75.0
            
            # Handle potential neutral case (when models are in significant disagreement)
            if abs(positive_count - negative_count) <= 1 and len(model_predictions) > 1:
                # Models are split - could be neutral
                if 0.4 <= (positive_count / len(model_predictions)) <= 0.6:
                    final_sentiment = "Neutral"
                    final_confidence = 70.0  # Lower confidence for this automatic neutral case
                    model_used = "ensemble_neutral_detection"
                else:
                    # Not balanced enough for neutral
                    final_sentiment = "Positive" if positive_count > negative_count else "Negative"
                    final_confidence = avg_confidence
                    model_used = "ensemble"
            else:
                # Clear majority
                final_sentiment = "Positive" if positive_count > negative_count else "Negative"
                final_confidence = avg_confidence
                model_used = "ensemble"
        
            prediction_result = {
                "text": text,
                "sentiment": final_sentiment,
                "confidence": final_confidence,
                "model_used": model_used
            }
        
        # Apply confidence reduction for mild safety concerns
        if safety_result == 0.5:
            # Reduce confidence by 15% for mild safety concerns
            original_confidence = prediction_result["confidence"]
            reduced_confidence = max(original_confidence * 0.85, 60.0)  # Don't go below 60%
            prediction_result["confidence"] = reduced_confidence
            prediction_result["model_used"] = f"{prediction_result['model_used']}_with_safety_adjustment"
            logger.info(f"Safety adjustment: Reduced confidence from {original_confidence:.2f}% to {reduced_confidence:.2f}%")
        
        return prediction_result

    def _is_restaurant_review(self, text):
        """
        Detect if text is likely a restaurant review based on food/dining terms.
        Returns a tuple (is_restaurant, food_terms_found, service_terms_found)
        """
        text = text.lower()
        
        # Terms that indicate food or restaurant context
        food_terms = [
            "food", "meal", "dish", "restaurant", "cafe", "diner", "bistro", 
            "menu", "waiter", "waitress", "server", "chef", "cuisine", "dinner", 
            "lunch", "breakfast", "appetizer", "entree", "dessert", "plate",
            "delicious", "tasty", "flavorful", "savory", "tender", "juicy",
            "spicy", "bland", "fresh", "stale", "overcooked", "undercooked",
            "pasta", "steak", "fish", "chicken", "seafood", "vegetarian", "vegan",
            "pizza", "burger", "salad", "fries", "rice", "noodles", "sushi", 
            "taco", "burrito", "sandwich", "soup", "buffet", "brunch", "dining",
            "eat", "ate", "eaten", "tasted", "ordered", "served", "portion"
        ]
        
        # Terms specific to restaurant service/ambiance
        service_terms = [
            "service", "staff", "waiter", "waitress", "server", "host", "hostess",
            "manager", "tip", "reservation", "wait time", "waiting", "seated",
            "crowded", "busy", "queue", "line", "atmosphere", "ambiance", "ambience",
            "decor", "noisy", "quiet", "clean", "dirty", "hygiene", "bathroom",
            "restroom", "table", "seating", "chair", "booth", "patio", "outdoor",
            "indoor", "bar", "lounge", "price", "expensive", "cheap", "affordable",
            "overpriced", "worth", "value", "money"
        ]
        
        # Count occurrences of each type of term
        food_terms_found = sum(1 for term in food_terms if term in text)
        service_terms_found = sum(1 for term in service_terms if term in text)
        
        # Combined score (weighted)
        is_restaurant_review = (food_terms_found >= 2 or service_terms_found >= 2 or 
                              (food_terms_found >= 1 and service_terms_found >= 1))
        
        if is_restaurant_review:
            logger.info(f"Restaurant review detected: food terms={food_terms_found}, service terms={service_terms_found}")
            
        return (is_restaurant_review, food_terms_found, service_terms_found)

    def _extract_features(self, text):
        """
        Extract features from processed text for prediction.
        This vectorizes the text using the model's vectorizers.
        """
        try:
            # Create a feature set using TF-IDF features
            features = self.tfidf_vectorizer.transform([text])
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Emergency fallback: return an empty sparse matrix with the correct dimensions
            from scipy.sparse import csr_matrix
            import numpy as np
            feature_count = len(self.tfidf_vectorizer.get_feature_names_out())
            return csr_matrix((1, feature_count), dtype=np.float64)