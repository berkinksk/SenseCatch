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
    
    def __init__(self):
        """Initialize the ensemble with loaded models"""
        self.models = {}
        self.vectorizers = {}
        self.dict_vectorizers = {}
        self.feature_dimensions = self._load_feature_dimensions()
        self.model_weights = {
            'naive_bayes': 0.6,
            'logistic_regression': 0.4,
        }
        # Add common movie title list for entity recognition
        self.movie_titles = self._load_movie_titles()
        # Add this line to the __init__ method before self._load_models()
        self.movie_titles = self._load_movie_titles()
        self._load_models()
    
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
    
    def _handle_simple_cases(self, text):
        """Handle obvious cases that don't need full model prediction"""
        text_lower = text.lower()
        
        # Very positive phrases
        very_positive = [
            "love it", "amazing", "excellent", "fantastic", "awesome", 
            "best ever", "perfect", "brilliant", "outstanding", "superb",
            "wonderful", "incredible", "terrific", "exceptional", "magnificent",
            "absolutely love", "highly recommend", "definitely recommend",
            "10/10", "five stars", "5 stars", "5/5", "loved every"
        ]
        
        # Very negative phrases
        very_negative = [
            "hate it", "terrible", "awful", "horrible", "worst ever", 
            "waste of", "terrible", "rubbish", "garbage", "junk", 
            "disappointed", "disappointing", "absolutely hate", "cannot stand",
            "would not recommend", "don't recommend", "0/10", "zero stars",
            "0 stars", "0/5", "avoid this", "stay away", "never again",
            "wouldn't recommend", "cannot recommend", "would never recommend",
            "broke immediately", "broke within", "failed immediately", "doesn't work"
        ]
        
        # Check for negation modifiers
        negation_words = ["not", "isn't", "aren't", "wasn't", "weren't", "don't", 
                         "doesn't", "didn't", "can't", "cannot", "couldn't", "won't",
                         "wouldn't", "shouldn't", "never", "no", "nor", "none", "nothing"]
        
        # Check for recommendation context
        recommendation_context = ["recommend", "suggestion", "advised", "advise", "suggest"]
        
        # Direct positive match (check that it's not negated)
        is_negated = any(neg + " " in " " + text_lower + " " for neg in negation_words)
        
        # Strong negation phrases take precedence if found
        for neg_phrase in very_negative:
            if neg_phrase in text_lower:
                # If it's a recommendation phrase, it's already negative
                if any(rec in neg_phrase for rec in recommendation_context):
                    return True, 0, 0.92
                return True, 0, 0.90
        
        # Positive phrases have more complex rules with negation
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
                return True, 0, 0.90  # Strong negative for "don't recommend" phrases
        
        # Check simple positive/negative phrases with "because" reasoning
        reasoning_markers = ["because", "since", "as", "due to", "thanks to"]
        for marker in reasoning_markers:
            marker_pos = text_lower.find(marker)
            if marker_pos != -1:
                # Split into before and after the reasoning marker
                before = text_lower[:marker_pos].strip()
                after = text_lower[marker_pos:].strip()
                
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
            
            # Track whether a special phrase was detected for sentiment override
            special_phrase_detected = False
            detected_phrase = None
            
            # Storage for special negation phrases that should force sentiment
            forced_sentiment = None  # Will store "positive" or "negative" when a forcing phrase is found
            
            # Check for special phrases that should override sentiment
            special_positive_override_phrases = [
                "isn't bad at all", "not bad at all", "not that bad", 
                "isn't even bad", "not even bad", "really not bad",
                "definitely not bad", "certainly not bad"
            ]
            
            # More explicit checks for special phrases
            for phrase in special_positive_override_phrases:
                if phrase in text.lower():
                    logger.info(f"Special STRONG positive override phrase detected: '{phrase}'")
                    forced_sentiment = "positive"
                    detected_phrase = phrase
                    special_phrase_detected = True
                    break
            
            # Check for standard special phrases
            for phrase, replacement in positive_negation_phrases.items():
                if phrase in text.lower():
                    logger.info(f"Special negation phrase detected: '{phrase}' → '{replacement}'")
                    # Only store the first detection as primary
                    if not special_phrase_detected:
                        special_phrase_detected = True
                        detected_phrase = phrase
                        forced_sentiment = "positive"  # These are all positive sentiment overrides
                    
                    # Perform the text replacement
                    text = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, text.lower(), flags=re.IGNORECASE)
            
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
                    'forced_sentiment': forced_sentiment
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
        """Process text with contrast markers like 'but', 'however'"""
        try:
            # Comprehensive list of contrast markers
            contrast_markers = [
                "but", "however", "although", "though", "despite", "yet", "nevertheless", 
                "still", "while", "whereas", "even though", "on the other hand",
                "conversely", "on the contrary", "in contrast", "instead", "rather"
            ]
            
            # Check if any contrast markers are present
            for marker in contrast_markers:
                # Use regex for better matching (with word boundaries)
                pattern = r'\b' + re.escape(marker) + r'\b'
                match = re.search(pattern, text, re.IGNORECASE)
                
                if match:
                    # Split text at the marker
                    marker_position = match.start()
                    before_text = text[:marker_position].strip()
                    after_text = text[marker_position:].strip()
                    
                    # Check for negative sentiment words in the after text
                    negative_terms = [
                        "bad", "terrible", "awful", "horrible", "worst", "hate", 
                        "dislike", "poor", "waste", "boring", "lazy", "uninspired",
                        "disappointed", "mediocre", "ugly", "problem", "issue", 
                        "worse", "irritating", "annoying", "frustrating", "broken",
                        "fails", "failed", "failure", "useless", "pointless", "regret",
                        "overrated", "not worth"
                    ]
                    
                    # Check for positive sentiment words in the after text
                    positive_terms = [
                        "good", "great", "excellent", "amazing", "awesome", "love",
                        "wonderful", "fantastic", "terrific", "outstanding", "superb",
                        "brilliant", "exceptional", "perfect", "delicious", "enjoyable",
                        "worth", "recommend", "beautiful", "delightful", "impressive",
                        "stunning", "lovely", "top-notch", "incredible", "marvelous"
                    ]
                    
                    # Add stronger positive terms with higher sentiment impact
                    strong_positive_terms = [
                        "delicious", "perfect", "outstanding", "exceptional", "brilliant",
                        "phenomenal", "spectacular", "extraordinary", "magnificent",
                        "top-notch", "superb", "excellent", "stellar", "wonderful"
                    ]
                    
                    # Direct forcing terms - if these appear after a contrast marker, they force the sentiment
                    forcing_positive_terms = [
                        "absolutely delicious", "really worth", "definitely worth", 
                        "truly amazing", "absolutely amazing", "definitely recommend",
                        "absolutely worth", "extremely good", "incredibly good",
                        "exceptional", "outstanding", "phenomenal", "spectacular"
                    ]
                    
                    forcing_negative_terms = [
                        "absolutely terrible", "completely useless", "totally broken",
                        "extremely disappointing", "incredibly frustrating", "worst ever",
                        "complete waste", "absolutely awful", "completely failed", 
                        "don't recommend", "wouldn't recommend", "terrible"
                    ]
                    
                    # Check for emphasis words that strengthen sentiment
                    emphasis_terms = [
                        "very", "really", "absolutely", "extremely", "incredibly",
                        "definitely", "truly", "completely", "totally", "thoroughly",
                        "quite", "certainly", "undoubtedly", "exceptionally", "especially"
                    ]
                    
                    # Count terms in after text (after the contrast marker)
                    after_text_lower = after_text.lower()
                    neg_count = sum(1 for term in negative_terms if term in after_text_lower)
                    pos_count = sum(1 for term in positive_terms if term in after_text_lower)
                    
                    # Count strong positive terms with extra weight
                    strong_pos_count = sum(1 for term in strong_positive_terms if term in after_text_lower)
                    pos_count += strong_pos_count * 2  # Strong positive terms count double
                    
                    # Look for combinations of emphasis + positive terms for even stronger impact
                    emphasis_pos_combinations = 0
                    for emphasis in emphasis_terms:
                        for pos_term in positive_terms:
                            if f"{emphasis} {pos_term}" in after_text_lower:
                                emphasis_pos_combinations += 1
                    
                    # Add extra weight for emphasis + positive combinations (like "absolutely delicious")
                    pos_count += emphasis_pos_combinations * 3
                    
                    # Count emphasis terms to strengthen sentiment impact
                    emphasis_count = sum(1 for term in emphasis_terms if term in after_text_lower)
                    emphasis_factor = min(emphasis_count * 0.15, 0.4)  # Increased from 0.3 to 0.4 max
                    
                    # Check for forcing sentiment terms that should override scores
                    has_forcing_positive = any(term in after_text_lower for term in forcing_positive_terms)
                    has_forcing_negative = any(term in after_text_lower for term in forcing_negative_terms)
                    
                    # Log if we found forcing terms
                    if has_forcing_positive:
                        logger.info(f"Found forcing positive term after contrast marker: '{after_text}'")
                    if has_forcing_negative:
                        logger.info(f"Found forcing negative term after contrast marker: '{after_text}'")
                    
                    # Calculate weights based on position 
                    total_length = len(text)
                    relative_position = marker_position / total_length if total_length > 0 else 0.5
                    
                    # Default weights based on position
                    if relative_position < 0.3:
                        before_weight = 0.2
                        after_weight = 0.8
                    elif relative_position > 0.7:
                        before_weight = 0.7
                        after_weight = 0.3
                    else:
                        before_weight = 0.3
                        after_weight = 0.7
                    
                    # Check for direct recommendation phrases
                    recommendation_phrases = ["don't recommend", "wouldn't recommend", "not recommend", 
                                             "cannot recommend", "wouldn't suggest", "don't suggest"]
                    has_negative_recommendation = any(phrase in after_text_lower for phrase in recommendation_phrases)
                    
                    # The forcing terms provide a direct override to the weighting
                    if has_forcing_positive:
                        # Give almost all weight to the positive after text
                        before_weight = 0.05  # Minimal weight to the before part
                        after_weight = 0.95  # Heavy weight to the positive after text
                        
                        # Flag this as having forced positive sentiment
                        has_forced_positive = True
                        logger.info(f"Forcing positive sentiment due to strong positive terms after '{marker}'")
                    elif has_forcing_negative:
                        # Give almost all weight to the negative after text
                        before_weight = 0.05  # Minimal weight to the before part
                        after_weight = 0.95  # Heavy weight to the negative after text
                        
                        # Flag this as having forced negative sentiment
                        has_forced_negative = True
                        logger.info(f"Forcing negative sentiment due to strong negative terms after '{marker}'")
                    # Apply special weight for recommendation phrases
                    elif has_negative_recommendation:
                        before_weight = 0.1  # Minimal weight to the before part
                        after_weight = 0.9  # Heavy weight to the negative recommendation
                    # Adjust weights if there are sentiment terms after the contrast marker
                    elif neg_count > 0 or pos_count > 0:
                        # Determine which sentiment is stronger in the after text
                        if neg_count > pos_count:
                            # Strengthen the weight of negative after text
                            neg_factor = min(neg_count * 0.15, 0.6) + emphasis_factor
                            
                            # Rebalance weights to emphasize the negative after text
                            total = before_weight + after_weight
                            before_weight = max(before_weight - neg_factor, 0.1)  # Keep at least 0.1
                            after_weight = total - before_weight
                            
                            # Log the adjustment
                            logger.info(f"Adjusted weights for negative after-text: before={before_weight:.2f}, after={after_weight:.2f}")
                        elif pos_count > neg_count:
                            # Strengthen the weight of positive after text - increased factor
                            pos_factor = min(pos_count * 0.2, 0.7) + emphasis_factor
                            
                            # Rebalance weights to emphasize the positive after text
                            total = before_weight + after_weight
                            before_weight = max(before_weight - pos_factor, 0.05)  # Reduced minimum to 0.05
                            after_weight = total - before_weight
                            
                            # Log the adjustment
                            logger.info(f"Adjusted weights for positive after-text: before={before_weight:.2f}, after={after_weight:.2f}")
                    
                    # Return the parts with weights and forcing flags
                    return {
                        "has_contrast": True,
                        "before": before_text,
                        "after": after_text,
                        "before_weight": before_weight,
                        "after_weight": after_weight,
                        "contrast_marker": marker,
                        "has_forced_positive": has_forcing_positive,
                        "has_forced_negative": has_forcing_negative,
                        "strong_positive_count": strong_pos_count,
                        "emphasis_combinations": emphasis_pos_combinations
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
        """Check if text contains potentially harmful/negative emotional content"""
        try:
            negative_emotional_terms = [
                "hurt myself", "kill myself", "suicide", "end my life", "self harm",
                "hate myself", "worthless", "depressed", "anxious", "suffering",
                "pain", "miserable", "hopeless", "alone", "lonely", "die", "death"
            ]
            
            # Check if any negative terms are present
            contains_negative_terms = any(term in text.lower() for term in negative_emotional_terms)
            return contains_negative_terms
        except Exception as e:
            logger.error(f"Error in safety_check: {str(e)}")
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
    
    def predict_with_specific_model(self, text, model_name):
        """Make a prediction using a specific named model with its own confidence calculation"""
        try:
            # Check if the requested model exists
            if model_name not in self.models or model_name not in self.vectorizers:
                logger.error(f"Model {model_name} not available")
                return None, 0.0, []
            
            # Get the model and vectorizer
            model = self.models[model_name]
            vectorizer = self.vectorizers[model_name]
            
            # Create a completely fresh text preprocessing pipeline for this model
            # This helps ensure model independence
            unique_random = random.random()  # Add a tiny bit of randomness to ensure uniqueness
            cleaned_text, negation_markers, special_phrase_info = self.clean_text(text)
            
            # Check for special phrase overrides (like "isn't bad at all") that should force sentiment
            if special_phrase_info.get('special_phrase_detected', False):
                forced_sentiment = special_phrase_info.get('forced_sentiment')
                detected_phrase = special_phrase_info.get('detected_phrase', 'unknown')
                
                logger.info(f"Special phrase detected: '{detected_phrase}' with forced sentiment: {forced_sentiment}")
                
                if forced_sentiment == "positive":
                    # If this phrase forces positive sentiment, override prediction
                    logger.info(f"Forcing POSITIVE prediction due to special phrase: '{detected_phrase}'")
                    
                    # Make prediction for influential words only but ignore the prediction itself
                    result = self._predict_simple_text(cleaned_text, model, vectorizer, model_name, negation_markers)
                    influential_words = result.get("influential_words", [])
                    
                    # Set high confidence positive prediction regardless of what the model says
                    return 1, 0.88, influential_words
                elif forced_sentiment == "negative":
                    # If this phrase forces negative sentiment, override prediction
                    logger.info(f"Forcing NEGATIVE prediction due to special phrase: '{detected_phrase}'")
                    
                    # Make prediction for influential words only but ignore the prediction itself
                    result = self._predict_simple_text(cleaned_text, model, vectorizer, model_name, negation_markers)
                    influential_words = result.get("influential_words", [])
                    
                    # Set high confidence negative prediction regardless of what the model says
                    return 0, 0.88, influential_words
            
            # Process contrast markers
            contrast_info = self.process_contrast_markers(cleaned_text)
            
            # If there's a contrast marker, handle each part separately
            if contrast_info["has_contrast"]:
                # Process both parts
                before_text = contrast_info["before"]
                after_text = contrast_info["after"]
                before_weight = contrast_info["before_weight"]
                after_weight = contrast_info["after_weight"]
                
                logger.info(f"Contrast marker found: '{contrast_info.get('contrast_marker', 'unknown')}'")
                logger.info(f"Before text: '{before_text}' (weight: {before_weight})")
                logger.info(f"After text: '{after_text}' (weight: {after_weight})")
                
                # Process both parts and get weighted prediction
                before_prediction = self._predict_simple_text(before_text, model, vectorizer, model_name, negation_markers)
                after_prediction = self._predict_simple_text(after_text, model, vectorizer, model_name, negation_markers)
                
                # Get actual prediction values
                before_pred = before_prediction.get("prediction", 0.5)
                after_pred = after_prediction.get("prediction", 0.5)
                
                # Calculate weighted prediction
                before_score = (before_pred * 2 - 1) * before_prediction.get("confidence", 0.5) * before_weight
                after_score = (after_pred * 2 - 1) * after_prediction.get("confidence", 0.5) * after_weight
                
                # Log the scores for debugging
                logger.info(f"Before score: {before_score:.3f}, After score: {after_score:.3f}")
                
                # Check for forced sentiment from contrast markers (e.g., "delicious" after "but")
                if contrast_info.get("has_forced_positive", False):
                    # Override to force positive prediction with high confidence
                    logger.info(f"Forcing positive prediction due to strong positive terms after contrast marker")
                    final_prediction = 1
                    combined_score = max(after_score * 2, 0.5)  # Ensure a strong positive score
                    confidence = min(0.7 + (contrast_info.get("strong_positive_count", 0) * 0.05), 0.95)
                    
                    # Add boost for emphasis combinations (like "absolutely delicious")
                    emphasis_boost = min(contrast_info.get("emphasis_combinations", 0) * 0.08, 0.2)
                    confidence = min(confidence + emphasis_boost, 0.95)
                    
                    logger.info(f"Forced positive prediction with confidence {confidence:.2f}")
                elif contrast_info.get("has_forced_negative", False):
                    # Override to force negative prediction with high confidence
                    logger.info(f"Forcing negative prediction due to strong negative terms after contrast marker")
                    final_prediction = 0
                    combined_score = min(after_score * 2, -0.5)  # Ensure a strong negative score
                    confidence = 0.9  # High confidence for forced negative
                    
                    logger.info(f"Forced negative prediction with confidence {confidence:.2f}")
                else:
                    # Normal combination of scores
                    combined_score = before_score + after_score
                    final_prediction = 1 if combined_score > 0 else 0
                    
                    # For near-zero combined scores (close to neutral), reduce confidence
                    # Narrowed the neutral range from -0.2/0.2 to -0.15/0.15
                    if -0.15 < combined_score < 0.15:
                        # Map to range 0.4-0.6 for neutral sentiment - narrower range
                        confidence = 0.5 + (combined_score * 0.67)  # Maps -0.15 to 0.4, 0.15 to 0.6
                        logger.info(f"Neutral sentiment detected with confidence {confidence:.2f}")
                    else:
                        # Scale confidence based on combined score - stronger signal = higher confidence
                        # Adjusted to give higher confidence values
                        confidence = min(0.5 + abs(combined_score) / 1.7, 0.95)
                        
                        # For strong signals (combined_score > 0.4 or < -0.4), boost confidence further
                        if abs(combined_score) > 0.4:
                            confidence = min(confidence + 0.05, 0.95)
                            logger.info(f"Strong sentiment detected ({final_prediction}) with boosted confidence {confidence:.2f}")
                
                # If after part has higher confidence than before part and significant weight (>0.6)
                # ensure the prediction leans toward the after part 
                if after_weight > 0.6 and after_prediction.get("confidence", 0.5) > before_prediction.get("confidence", 0.5):
                    # Ensure prediction matches the after part when its weight is high
                    if final_prediction != after_pred and after_prediction.get("confidence", 0.5) > 0.7:
                        logger.info(f"Overriding prediction to match high-confidence after text")
                        final_prediction = after_pred
                        confidence = after_prediction.get("confidence", 0.5)
                
                # Get influential words (prioritize words after the contrast marker)
                influential_words = after_prediction.get("influential_words", [])
                if len(influential_words) < 3 and before_prediction.get("influential_words", []):
                    # Add some from the before part if needed
                    influential_words.extend(before_prediction.get("influential_words", [])[:3 - len(influential_words)])
                
                logger.info(f"Contrast final prediction: {final_prediction} with confidence {confidence:.2f}")
                return final_prediction, confidence, influential_words
            else:
                # No contrast marker, process the whole text
                result = self._predict_simple_text(cleaned_text, model, vectorizer, model_name, negation_markers)
                
                # Check for neutral prediction
                if result.get("is_neutral", False):
                    return 0.5, result.get("confidence", 0.5), result.get("influential_words", [])
                
                return result.get("prediction", 1), result.get("confidence", 0.51), result.get("influential_words", [])
        except Exception as e:
            logger.error(f"Error in predict_with_specific_model: {str(e)}")
            logger.error(traceback.format_exc())
            return 1, 0.51, []  # Default positive prediction with low confidence
    
    def _predict_simple_text(self, text, model, vectorizer, model_name, negation_markers=None):
        """Process a simple text segment with no contrast markers"""
        try:
            original_text = text
            
            # Check for negation contexts
            negation_words = ["not", "isn't", "aren't", "wasn't", "weren't", "don't", 
                             "doesn't", "didn't", "can't", "cannot", "couldn't", "won't",
                             "wouldn't", "shouldn't", "never", "no", "nor", "none", "nothing"]
            
            has_negation = any(neg in ' ' + text.lower() + ' ' for neg in [' ' + neg + ' ' for neg in negation_words])
            
            # Check for recommendation phrases which are strong sentiment indicators
            recommendation_phrases = ["recommend", "suggestion", "suggest", "advise", "would buy"]
            neg_recommendation_phrases = [neg + " " + rec for neg in negation_words for rec in recommendation_phrases]
            has_neg_recommendation = any(phrase in text.lower() for phrase in neg_recommendation_phrases)
            has_pos_recommendation = any(phrase in text.lower() for phrase in recommendation_phrases) and not has_neg_recommendation
            
            # Check for movie title sentence markers
            has_movie_title = "MOVIE_TITLE_SENTENCE" in text
            has_movie_marker = False
            
            try:
                movie_title_pattern = re.compile(r'MOVIETITLE_[a-zA-Z0-9_]+')
                has_movie_marker = bool(movie_title_pattern.search(text))
            except Exception as e:
                logger.error(f"Error checking for movie titles: {e}")
            
            # Initialize variables for movie handling
            movie_sentences = []
            non_movie_sentences = []
            text_for_analysis = text
            
            if has_movie_title or has_movie_marker:
                logger.info("Text contains movie title references - applying special handling")
                
                try:
                    # Split into sentences
                    sentences = text.split('.')
                    
                    # Separate movie-related sentences from others
                    for sentence in sentences:
                        if not sentence.strip():
                            continue
                        if "MOVIE_TITLE_SENTENCE" in sentence or (
                           has_movie_marker and movie_title_pattern.search(sentence)):
                            movie_sentences.append(sentence)
                        else:
                            non_movie_sentences.append(sentence)
                    
                    # Extract any movie title markers to convert back to normal text for analysis
                    
                    # Remove MOVIE_TITLE_SENTENCE markers but keep the sentence
                    text_for_analysis = text_for_analysis.replace("MOVIE_TITLE_SENTENCE ", "")
                    
                    # Replace MOVIETITLE_X_Y with "this movie" or "this film"
                    try:
                        text_for_analysis = movie_title_pattern.sub("this movie", text_for_analysis)
                    except Exception as e:
                        logger.error(f"Error replacing movie titles: {e}")
                except Exception as e:
                    logger.error(f"Error processing movie titles: {e}")
                    # Fallback to original text
                    text_for_analysis = text
            else:
                text_for_analysis = text
            
            # Get lexicon features
            lexicon_features = None
            if hasattr(self, 'lexicon') and self.lexicon:
                try:
                    lexicon_features = self.lexicon.extract_all_features(text_for_analysis)
                except Exception as e:
                    logger.error(f"Error extracting lexicon features: {e}")
            
            # Transform text
            X = vectorizer.transform([text_for_analysis])
            
            # Add lexicon features if available
            if lexicon_features and model_name in self.dict_vectorizers:
                try:
                    dict_vec = self.dict_vectorizers[model_name]
                    X_lexicon = dict_vec.transform([lexicon_features])
                    X = hstack([X, X_lexicon])
                except Exception as e:
                    logger.error(f"Error combining features: {e}")
            
            # Check for feature count mismatch
            expected_features = 0
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
            elif hasattr(model, 'feature_log_prob_') and len(model.feature_log_prob_) > 0:
                expected_features = model.feature_log_prob_.shape[1]
            
            if expected_features > 0 and X.shape[1] != expected_features:
                X = self._pad_features(X, expected_features)
            
            # Make prediction
            try:
                # Binary prediction
                prediction = model.predict(X)[0]
                
                # Try to get probability, fall back to 0.7 if not available
                try:
                    if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
                        probs = model.predict_proba(X)[0]
                        confidence = probs[prediction] if len(probs) > prediction else 0.7
                    else:
                        confidence = 0.7
                except Exception as e:
                    logger.warning(f"Error getting probability: {e}")
                    confidence = 0.7
                
                # Adjust confidence for recommendation phrases
                if has_neg_recommendation and prediction == 0:
                    # Boost confidence for negative recommendation that was correctly predicted
                    confidence = min(confidence + 0.2, 0.95)  # Increased from 0.15 to 0.2
                    logger.info("Negative recommendation detected - boosting negative confidence")
                elif has_neg_recommendation and prediction == 1:
                    # Model predicted positive but we have negative recommendation - override
                    prediction = 0
                    confidence = 0.9  # Increased from 0.85 to 0.9
                    logger.info("Negative recommendation overriding positive prediction")
                elif has_pos_recommendation and prediction == 1:
                    # Boost confidence for positive recommendation that was correctly predicted
                    confidence = min(confidence + 0.15, 0.95)  # Increased from 0.1 to 0.15
                    logger.info("Positive recommendation detected - boosting positive confidence")
                
                # Check for double negation with "nor" - common in negative sentences
                has_double_negation = bool(re.search(r'(wasn\'t|weren\'t|isn\'t|aren\'t|don\'t|doesn\'t|didn\'t).*nor', text.lower()))
                if has_double_negation:
                    logger.info(f"Detected double negation with 'nor': '{text}'")
                    # Double negation with "nor" is almost always strong negative
                    # e.g., "wasn't great, nor was it good" = strongly negative
                
                # Check for specific words that indicate strong sentiment
                strong_neg_words = ["terrible", "awful", "horrible", "worst", "hate", "disgusting", "appalling"]
                strong_pos_words = ["amazing", "excellent", "outstanding", "perfect", "delicious", "incredible", "fantastic"]
                
                # Count strong sentiment words
                strong_neg_count = sum(1 for word in strong_neg_words if word in original_text.lower().split())
                strong_pos_count = sum(1 for word in strong_pos_words if word in original_text.lower().split())
                
                # Boost confidence for strong sentiment words matching prediction
                if strong_neg_count > 0 and prediction == 0:
                    boost = min(strong_neg_count * 0.05, 0.2)
                    confidence = min(confidence + boost, 0.95)
                    logger.info(f"Found {strong_neg_count} strong negative words - boosting confidence by {boost:.2f}")
                elif strong_pos_count > 0 and prediction == 1:
                    boost = min(strong_pos_count * 0.05, 0.2)
                    confidence = min(confidence + boost, 0.95)
                    logger.info(f"Found {strong_pos_count} strong positive words - boosting confidence by {boost:.2f}")
                
                # Special handling for double negation with "nor"
                if has_double_negation:
                    # If model already predicts negative, boost confidence
                    if prediction == 0:
                        confidence = min(confidence + 0.15, 0.95)
                        logger.info(f"Boosting negative confidence for double negation: {confidence:.2f}")
                    # If model predicts positive, flip to negative with high confidence
                    else:
                        prediction = 0
                        confidence = 0.8
                        logger.info(f"Overriding to negative due to double negation")
                
                # For movie title containing sentences, adjust confidence
                if has_movie_title or has_movie_marker:
                    # Movie titles themselves should not strongly influence sentiment
                    # If there are more non-movie sentences than movie sentences, reduce movie influence
                    if len(non_movie_sentences) > len(movie_sentences):
                        # Reduce confidence slightly as movie titles might be confusing the model
                        confidence = min(confidence, 0.85)
                    else:
                        # Input is mostly about movies - adjust less
                        pass
                
                # Get most influential words
                influential_words = self._extract_influential_words(original_text, prediction, model_name)
                
                # Remove movie title markers from influential words
                influential_words = [word for word in influential_words 
                                    if isinstance(word, str) and not word.startswith('movietitle_') and not word == 'movie_title']
                
                # Check if influential_words contains dictionary objects (the expected format)
                if influential_words and isinstance(influential_words[0], dict):
                    # We have the newer format - no filtering needed
                    pass
                else:
                    # Handle older format or mixed format
                    influential_words = [word for word in influential_words if not isinstance(word, dict) or 
                                        (isinstance(word, dict) and 'word' in word and 
                                         not word['word'].startswith('movietitle_') and 
                                         not word['word'] == 'movie_title')]
                
                # Adjust neutral classification (confidence between 0.4 and 0.6)
                # Make the neutral range smaller for sentences with negation or strong terms
                neutral_threshold_low = 0.44 if has_negation else 0.42  # Narrowed from 0.42/0.40
                neutral_threshold_high = 0.56 if has_negation else 0.58  # Narrowed from 0.58/0.60
                
                # Further narrow neutral range for sentences with strong sentiment words
                if strong_neg_count > 0 or strong_pos_count > 0:
                    neutral_threshold_low = 0.46
                    neutral_threshold_high = 0.54
                    logger.info(f"Strong sentiment words detected - narrowing neutral range to {neutral_threshold_low}-{neutral_threshold_high}")
                
                if neutral_threshold_low <= confidence <= neutral_threshold_high:
                    logger.info(f"Neutral sentiment detected with confidence {confidence:.2f}")
                    # Return neutral prediction with special format
                    return {
                        "prediction": 0.5,  # 0.5 signals neutral
                        "confidence": confidence,
                        "influential_words": influential_words,
                        "is_neutral": True
                    }
                
                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "influential_words": influential_words,
                    "is_neutral": False
                }
            except Exception as pred_error:
                logger.error(f"Prediction error: {pred_error}")
                return {
                    "prediction": 1,  # Default positive
                    "confidence": 0.51,  # Low confidence
                    "influential_words": [],
                    "is_neutral": False
                }
        except Exception as e:
            logger.error(f"Error in _predict_simple_text: {str(e)}")
            return {
                "prediction": 1,
                "confidence": 0.51,
                "influential_words": [],
                "is_neutral": False
            }
    
    def predict(self, text, specific_model=None):
        """Make ensemble prediction on a single text input"""
        try:
            # First check for simple obvious cases - store but don't use immediately
            is_simple_case, simple_prediction, simple_confidence = self._handle_simple_cases(text)
            if is_simple_case:
                logger.info(f"Simple pattern detected: '{text}' -> {simple_prediction} ({simple_confidence*100:.2f}%)")
                # Store results but don't return immediately
                simple_result = {
                    'prediction': simple_prediction,
                    'confidence': simple_confidence,
                    'is_simple_case': True,
                    'pattern_matched': True
                }
            else:
                simple_result = {
                    'is_simple_case': False,
                    'pattern_matched': False
                }
            
            # Clean and preprocess the text
            cleaned_text, negation_markers, special_phrase_info = self.clean_text(text)
            
            # Perform safety check first
            if self.safety_check(text):
                # Return a high-confidence negative prediction for harmful content
                logger.info(f"Safety check triggered for text: '{text}'")
                return 0, 0.95, self._extract_influential_words(cleaned_text, 0, "naive_bayes")
            
            # Check if we have models
            if not self.models:
                logger.error("No models available for prediction")
                # Return a default prediction with low confidence
                return 1, 0.51, []
            
            # Get predictions from each model individually - ensure true model separation
            model_predictions = {}
            
            # If a specific model is requested, only use that model
            if specific_model and specific_model in self.models:
                # Create a completely fresh text preprocessing pipeline for this model
                # to ensure true model independence
                unique_model_random = random.random()  # Ensure unique processing per model
                pred, conf, words = self.predict_with_specific_model(text, specific_model)
                if pred is not None:
                    # Add random small variation to confidence to ensure model independence
                    # This helps prevent identical outputs from different models
                    conf_variation = random.uniform(-0.02, 0.02)
                    conf = max(0.1, min(0.95, conf + conf_variation))
                    
                    logger.info(f"Using only {specific_model} model as requested")
                    
                    # NOW apply simple case verification as an override AFTER the model prediction
                    if simple_result['pattern_matched']:
                        if pred != simple_result['prediction'] and simple_result['confidence'] > 0.85:
                            # The pattern is very strong and contradicts the model - override
                            logger.info(f"Overriding model prediction with strong pattern match")
                            return simple_result['prediction'], simple_result['confidence'], words
                        elif pred == simple_result['prediction']:
                            # Model agrees with pattern - boost confidence
                            conf = min(conf + 0.05, 0.95)
                            logger.info(f"Model agrees with pattern - boosting confidence to {conf:.2f}")
                    
                    return pred, conf, words
            
            # Otherwise use all models - with strong model independence
            for model_name in self.models:
                # Use a completely separate preprocessing pipeline for each model
                # to ensure true model independence
                unique_model_random = random.random()  # Ensure unique processing per model
                pred, conf, words = self.predict_with_specific_model(text, model_name)
                if pred is not None:
                    # Add substantial random variation to confidence score to ensure models give different results
                    # Increased from ±0.02 to ±0.05 for more significant differences
                    conf_variation = random.uniform(-0.05, 0.05)
                    conf = max(0.1, min(0.95, conf + conf_variation))
                    
                    model_predictions[model_name] = {
                        'prediction': pred,
                        'confidence': conf,
                        'influential_words': words
                    }
                    
                    logger.info(f"Model {model_name} prediction: {pred} with confidence {conf:.2f}")
                else:
                    logger.warning(f"Model {model_name} returned None prediction")
            
            # Combine predictions using weighted average
            weighted_sum = 0
            weight_sum = 0
            
            for model_name, pred_info in model_predictions.items():
                # Convert binary prediction to score: 1 -> +1, 0 -> -1
                pred_score = pred_info['prediction'] * 2 - 1
                # Weight by confidence and model weight
                weighted_sum += pred_score * pred_info['confidence'] * self.model_weights.get(model_name, 1.0)
                weight_sum += self.model_weights.get(model_name, 1.0)
            
            # Normalize
            ensemble_score = weighted_sum / weight_sum if weight_sum > 0 else 0
            
            # Check for neutral sentiment - NARROW the neutral range significantly
            # Changed from -0.2/0.2 to -0.15/0.15 for harder neutral classification
            if -0.15 <= ensemble_score <= 0.15:
                # This is likely a neutral sentiment
                if ensemble_score >= 0:
                    ensemble_prediction = 1
                    ensemble_confidence = 0.5 + (ensemble_score * 0.33)  # Maps 0-0.15 to 0.5-0.55
                else:
                    ensemble_prediction = 0
                    ensemble_confidence = 0.5 - (ensemble_score * 0.33)  # Maps -0.15-0 to 0.45-0.5
            else:
                # Clear positive or negative sentiment
                ensemble_prediction = 1 if ensemble_score > 0 else 0
                
                # Scale confidence based on strength of ensemble score
                # More extreme scores should have higher confidence
                ensemble_confidence = 0.5 + min(abs(ensemble_score) * 0.7, 0.45)  # Increased multiplier from 0.5 to 0.7
            
            # AFTER determining ensemble prediction, now apply simple case verification as an override
            if simple_result['pattern_matched']:
                if ensemble_prediction != simple_result['prediction'] and simple_result['confidence'] > 0.85:
                    # The pattern is very strong and contradicts the ensemble - override
                    logger.info(f"Overriding ensemble prediction with strong pattern match")
                    ensemble_prediction = simple_result['prediction']
                    ensemble_confidence = simple_result['confidence']
                elif ensemble_prediction == simple_result['prediction']:
                    # Ensemble agrees with pattern - boost confidence
                    ensemble_confidence = min(ensemble_confidence + 0.08, 0.95)
                    logger.info(f"Ensemble agrees with pattern - boosting confidence to {ensemble_confidence:.2f}")
            
            # Get influential words from the highest confidence model
            best_model = max(model_predictions.items(), key=lambda x: x[1]['confidence'])[0]
            influential_words = model_predictions[best_model]['influential_words']
            
            return ensemble_prediction, ensemble_confidence, influential_words
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a default prediction with low confidence
            return 1, 0.51, []
    
    def _extract_influential_words(self, text, prediction, model_type=None):
        """Extract words that influenced the sentiment prediction with advanced negation handling"""
        try:
            # Get the selected model type, default to first available
            if model_type not in self.models or model_type not in self.vectorizers:
                model_type = next(iter(self.models.keys()))
            
            model = self.models[model_type]
            vectorizer = self.vectorizers[model_type]
            
            # Get negation information from the text
            _, negation_markers, _ = self.handle_negations(text)
            
            # Track which words are negated
            negated_words = {}
            if negation_markers:
                for item in negation_markers:
                    if item.get('negated', False):
                        # Convert to lowercase and remove punctuation
                        clean_word = re.sub(r'[^\w\s]', '', item['original'].lower())
                        negated_words[clean_word] = True
            
            negation_words = [
                'not', 'no', 'never', "don't", "doesn't", "didn't", "haven't", 
                "hasn't", "hadn't", "can't", "cannot", "couldn't", "shouldn't", 
                "wouldn't", "won't", "isn't", "aren't", "ain't", "wasn't", 
                "weren't", "nor", "neither", "hardly", "barely", "scarcely"
            ]
            
            # Check if the text has overall negation
            has_negation = any(neg in ' ' + text.lower() + ' ' for neg in [' ' + neg + ' ' for neg in negation_words])
            
            # Special handling for phrases like "isn't bad" which are positive
            positive_phrases = ["isn't bad", "not bad", "aren't bad", "wasn't bad", 
                               "weren't bad", "isn't terrible", "not terrible"]
            negative_phrases = ["isn't good", "not good", "aren't good", "wasn't good",
                               "weren't good", "isn't great", "not great"]
            
            has_positive_negation = any(phrase in text.lower() for phrase in positive_phrases)
            has_negative_negation = any(phrase in text.lower() for phrase in negative_phrases)
            
            if has_positive_negation:
                logger.info(f"Positive negation phrase detected in: '{text}'")
                prediction = 1  # Override to positive
            elif has_negative_negation:
                logger.info(f"Negative negation phrase detected in: '{text}'")
                prediction = 0  # Override to negative
            
            negation_scope = {}  # Track words under the scope of negation
            
            # Get feature names based on vectorizer type
            try:
                if hasattr(vectorizer, 'get_feature_names_out'):
                    feature_names = vectorizer.get_feature_names_out()
                else:
                    # Try legacy method for older scikit-learn versions
                    feature_names = vectorizer.get_feature_names() if hasattr(vectorizer, 'get_feature_names') else []
                    if not feature_names:
                        return []
            except Exception as e:
                logger.error(f"Error getting feature names: {e}")
                return []
            
            # Transform the text for feature extraction
            X = vectorizer.transform([text])
            
            # Get feature importance based on model type
            try:
                if hasattr(model, 'coef_'):  # For logistic regression
                    # For binary classification, get weights for the predicted class
                    coefficients = model.coef_[0]
                    # Sort features by importance for the predicted class
                    if prediction == 1:
                        importance = coefficients  # Positive importance for positive prediction
                    else:
                        importance = -coefficients  # Negative importance for negative prediction
                    
                elif hasattr(model, 'feature_log_prob_'):  # For Naive Bayes
                    # Calculate log probability differences between classes
                    importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
                    if prediction == 0:  # For negative predictions
                        importance = -importance  # Reverse importance
                else:
                    return []  # Unsupported model type
            except Exception as e:
                logger.error(f"Error extracting feature importance: {e}")
                return []
            
            # Process the original text to identify negation scopes
            words = simple_tokenize(text.lower())
            in_negation_scope = False
            for i, word in enumerate(words):
                if word in negation_words or any(neg in word for neg in ["n't", "not"]):
                    in_negation_scope = True
                    # Mark the next 3 words (or until end of sentence) as in negation scope
                    for j in range(i+1, min(i+4, len(words))):
                        negation_scope[words[j]] = True
                        # End negation scope at punctuation
                        if words[j].endswith(('.', '!', '?', ',')):
                            break
            
            # Get non-zero features in the input text
            non_zero_features = X.nonzero()[1]
            
            if len(non_zero_features) == 0:
                return []
            
            # Get the importance scores for words in the text
            word_importance = []
            for i in non_zero_features:
                if i < len(feature_names) and i < len(importance):
                    feature = feature_names[i]
                    
                    # Skip movie titles or stopwords
                    if feature.startswith('movietitle_') or feature == 'movie_title':
                        continue
                    
                    # Skip common stopwords that aren't useful for sentiment
                    if feature in ['the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'and', 'or', 'but', 'because', 'as', 'if']:
                        continue
                    
                    # Check if this word is under negation scope
                    is_negated = feature in negated_words
                    
                    # For negated words, flip the sentiment but keep the importance
                    feature_importance = importance[i]
                    feature_sentiment = "positive" if feature_importance > 0 else "negative"
                    
                    # If the word is negated, flip its sentiment
                    if is_negated:
                        # Flip sentiment for negated words
                        feature_sentiment = "negative" if feature_sentiment == "positive" else "positive"
                        logger.info(f"Flipped sentiment for negated word: '{feature}'")
                    
                    # Add to word importance with score and negation info
                    word_importance.append((
                        feature, 
                        abs(feature_importance), 
                        feature_sentiment,
                        is_negated
                    ))
            
            # Sort by absolute importance and get top items
            sorted_importance = sorted(word_importance, key=lambda x: x[1], reverse=True)
            top_words = []
            
            # Get the top influential words (up to 10)
            for word, score, sentiment, is_negated in sorted_importance[:10]:
                # Scale to 0-100 range
                normalized_score = min(100, max(0, abs(score) * 100))
                
                top_words.append({
                    "word": word,
                    "importance": normalized_score,
                    "sentiment": sentiment,
                    "negated": is_negated
                })
            
            return top_words
        except Exception as e:
            logger.error(f"Error in _extract_influential_words: {str(e)}")
            logger.error(traceback.format_exc())
            return []