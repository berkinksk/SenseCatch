"""
Sentiment lexicon utilities for SenseCatch
Provides pre-defined sentiment scores for words to enhance model accuracy
"""
import os
import json
import re
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('wordnet', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")

class SentimentLexiconFeatures:
    """Extract sentiment features from various lexicons"""
    
    def __init__(self):
        """Initialize lexicon resources"""
        # Initialize VADER
        try:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VADER: {e}")
            self.vader = None
        
        # Custom sentiment lexicon
        self.custom_lexicon = self._load_custom_lexicon()
    
    def _load_custom_lexicon(self):
        """Load custom lexicon with movie-specific sentiment words"""
        # Default values for movie-related terms
        lexicon = {
            # Positive movie terms
            "masterpiece": 1.0,
            "brilliant": 0.9,
            "superb": 0.9,
            "excellent": 0.8,
            "outstanding": 0.8,
            "incredible": 0.8,
            "amazing": 0.8,
            "fantastic": 0.8,
            "wonderful": 0.8,
            "compelling": 0.7,
            "gripping": 0.7,
            "powerful": 0.7,
            "hilarious": 0.7,
            "touching": 0.6,
            "enjoyed": 0.6,
            "love": 0.6,
            "recommend": 0.6,
            "underrated": 0.5,
            "classic": 0.5,
            "gem": 0.5,
            "awesome": 0.9,
            "great": 0.8,
            "good": 0.7,
            "fun": 0.6,
            "nice": 0.5,
            
            # Negative movie terms
            "terrible": -0.9,
            "awful": -0.9,
            "horrible": -0.9,
            "waste": -0.8,
            "boring": -0.8,
            "dull": -0.7,
            "disappointing": -0.7,
            "mediocre": -0.6,
            "overrated": -0.6,
            "bad": -0.6,
            "worst": -0.9,
            "poor": -0.6,
            "hate": -0.8,
            "dreadful": -0.8,
            "avoid": -0.7,
            "failure": -0.7,
            "stupid": -0.7,
            "pathetic": -0.8,
            "garbage": -0.9,
            "rubbish": -0.8
        }
        
        # Try to load from disk if exists
        try:
            if os.path.exists('models/custom_lexicon.json'):
                with open('models/custom_lexicon.json', 'r') as f:
                    loaded_lexicon = json.load(f)
                    lexicon.update(loaded_lexicon)
                    logger.info("Custom lexicon loaded from disk")
        except Exception as e:
            logger.error(f"Error loading custom lexicon: {e}")
        
        return lexicon
    
    def save_custom_lexicon(self):
        """Save custom lexicon to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            with open('models/custom_lexicon.json', 'w') as f:
                json.dump(self.custom_lexicon, f)
            logger.info("Custom lexicon saved to disk")
        except Exception as e:
            logger.error(f"Error saving custom lexicon: {e}")
    
    def get_vader_scores(self, text):
        """Get VADER sentiment scores for the text"""
        if not self.vader:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
        
        try:
            return self.vader.polarity_scores(text)
        except Exception as e:
            logger.error(f"Error getting VADER scores: {e}")
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
    
    def get_custom_lexicon_score(self, text):
        """Calculate sentiment score using custom lexicon"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        if not words:
            return 0.0
        
        # Calculate average sentiment
        total_score = 0.0
        found_words = 0
        
        for word in words:
            if word in self.custom_lexicon:
                total_score += self.custom_lexicon[word]
                found_words += 1
        
        # Return normalized score or 0 if no words found
        return total_score / max(found_words, 1) if found_words > 0 else 0.0
    
    # In sentiment_lexicon.py, update the get_sentiwordnet_score method:

# In sentiment_lexicon.py, update the get_sentiwordnet_score method:

def get_sentiwordnet_score(self, text):
    """Calculate sentiment score using SentiWordNet"""
    try:
        # Check if we have the required resources
        try:
            # Try to import the necessary resource
            from nltk.corpus import wordnet
            # If we get here, wordnet is available
        except (ImportError, LookupError) as e:
            logger.warning(f"WordNet resources not fully available, skipping SentiWordNet score: {e}")
            return 0.0
            
        words = re.findall(r'\b\w+\b', text.lower())
        pos_score = 0.0
        neg_score = 0.0
        count = 0
        
        for word in words:
            try:
                synsets = list(swn.senti_synsets(word))
                if synsets:
                    # Average over all synsets
                    word_pos = sum(s.pos_score() for s in synsets) / len(synsets)
                    word_neg = sum(s.neg_score() for s in synsets) / len(synsets)
                    pos_score += word_pos
                    neg_score += word_neg
                    count += 1
            except Exception as e:
                # Skip words that cause problems
                logger.debug(f"Error processing word '{word}' in SentiWordNet: {e}")
                continue
        
        if count == 0:
            return 0.0
        
        # Return normalized difference between positive and negative
        return (pos_score - neg_score) / count
    except Exception as e:
        logger.error(f"Error getting SentiWordNet score: {e}")
        return 0.0
    
    def extract_all_features(self, text):
        """Extract all sentiment lexicon features for a text"""
        features = {}
        
        # Get VADER scores
        vader_scores = self.get_vader_scores(text)
        features['vader_compound'] = vader_scores['compound']
        features['vader_pos'] = vader_scores['pos']
        features['vader_neg'] = vader_scores['neg']
        features['vader_neu'] = vader_scores['neu']
        
        # Get custom lexicon score
        features['custom_score'] = self.get_custom_lexicon_score(text)
        
        # Get SentiWordNet score 
        features['sentiwordnet'] = self.get_sentiwordnet_score(text)
        
        return features
    
    def update_lexicon(self, word, score):
        """Update the custom lexicon with a new word or score"""
        self.custom_lexicon[word.lower()] = float(score)
        self.save_custom_lexicon()