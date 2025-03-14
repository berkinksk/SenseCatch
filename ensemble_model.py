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
from scipy.sparse import hstack, csr_matrix

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
    logger.info("NLTK imported successfully")
except ImportError:
    logger.error("Error importing NLTK. Using fallback tokenizer.")
    # Fallback simple tokenizer if nltk is not available
    def word_tokenize(text):
        return simple_tokenize(text)

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
        self._load_models()
    
    def _load_feature_dimensions(self):
        """Load feature dimensions from saved file if available"""
        try:
            if os.path.exists('models/feature_dimensions.pkl'):
                with open('models/feature_dimensions.pkl', 'rb') as f:
                    return pickle.load(f)
            else:
                return {'text_features': 15000, 'lexicon_features': 9, 'total_features': 15009}
        except Exception as e:
            logger.error(f"Error loading feature dimensions: {e}")
            return {'text_features': 15000, 'lexicon_features': 9, 'total_features': 15009}
    
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
    
    def _handle_simple_cases(self, text):
        """Handle simple obvious cases directly"""
        text_lower = text.lower()
        
        # Direct pattern matching for very obvious cases
        obvious_positive = ["awesome", "amazing", "excellent", "great", "love", "wonderful", 
                           "brilliant", "fantastic", "superb", "perfect", "best"]
        obvious_negative = ["terrible", "awful", "horrible", "hate", "bad", "worst", 
                           "disappointing", "poor", "waste", "boring", "garbage"]
        
        # Check for obvious positive terms without negation
        if any(term in text_lower for term in obvious_positive) and not any(neg in text_lower for neg in ["not ", "n't ", "don't", "didn't", "doesn't"]):
            return True, 1, 0.98  # Positive with high confidence
            
        # Check for obvious negative terms without negation
        if any(term in text_lower for term in obvious_negative) and not any(neg in text_lower for neg in ["not ", "n't ", "don't", "didn't", "doesn't"]):
            return True, 0, 0.98  # Negative with high confidence
            
        return False, None, None
    
    def handle_negations(self, text):
        """Mark negated words to help the model understand negations"""
        try:
            # Create a list of negation words
            negation_words = ['not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 
                             'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t', 'isn\'t', 
                             'aren\'t', 'ain\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 
                             'hasn\'t', 'hadn\'t', 'won\'t', 'nor', 'neither']
            
            # Try NLTK tokenization first
            try:
                words = word_tokenize(text.lower())
            except Exception as e:
                logger.warning(f"NLTK tokenization failed, using fallback: {e}")
                words = simple_tokenize(text.lower())
            
            # Process negations
            in_negation = False
            result = []
            
            for word in words:
                if word in negation_words:
                    in_negation = True
                    result.append(word)
                elif word in ['.', '!', '?', ',', ';', ':', ')', ']']:
                    # End negation scope at punctuation
                    in_negation = False
                    result.append(word)
                elif in_negation and word not in ['and', 'or', 'the', 'a', 'an', 'to', 'of', 'in']:
                    # Mark negated content words
                    result.append(word + '_NEG')
                else:
                    result.append(word)
            
            return ' '.join(result)
        except Exception as e:
            logger.error(f"Error in handle_negations: {str(e)}")
            logger.error(traceback.format_exc())
            # Return original text if there's an error
            return text
    
    def process_contrast_markers(self, text):
        """Process text with contrast markers like 'but', 'however'"""
        try:
            # List of contrast markers
            contrast_markers = ["but", "however", "although", "though", "despite", "yet", "nevertheless", "still"]
            
            # Check if any contrast markers are present
            for marker in contrast_markers:
                if f" {marker} " in f" {text} ":
                    parts = text.split(f" {marker} ", 1)
                    if len(parts) == 2:
                        # Return the parts with weights
                        return {
                            "has_contrast": True,
                            "before": parts[0].strip(),
                            "after": parts[1].strip(),
                            "before_weight": 0.3,  # Give less weight to what comes before the contrast marker
                            "after_weight": 0.7    # Give more weight to what comes after the contrast marker
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
        """Enhanced text cleaning with negation handling"""
        try:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters but keep apostrophes for negations
            text = re.sub(r'[^\w\s\']', ' ', text)
            
            # Try to apply negation handling
            try:
                text = self.handle_negations(text)
            except Exception as e:
                logger.error(f"Negation handling failed: {e}")
                # Continue without negation handling
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            logger.error(traceback.format_exc())
            # Simple fallback cleaning
            return text.lower().strip()
    
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
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Process contrast markers
            contrast_info = self.process_contrast_markers(cleaned_text)
            
            # If there's a contrast marker, handle each part separately
            if contrast_info["has_contrast"]:
                # Process both parts
                before_text = contrast_info["before"]
                after_text = contrast_info["after"]
                before_weight = contrast_info["before_weight"]
                after_weight = contrast_info["after_weight"]
                
                # Process both parts and get weighted prediction
                before_prediction = self._predict_simple_text(before_text, model, vectorizer, model_name)
                after_prediction = self._predict_simple_text(after_text, model, vectorizer, model_name)
                
                # Calculate weighted prediction
                before_score = (before_prediction["prediction"] * 2 - 1) * before_prediction["confidence"] * before_weight
                after_score = (after_prediction["prediction"] * 2 - 1) * after_prediction["confidence"] * after_weight
                
                # Combine scores
                combined_score = before_score + after_score
                final_prediction = 1 if combined_score > 0 else 0
                
                # Scale confidence based on combined score
                confidence = min(0.5 + abs(combined_score) / 2, 0.99)
                
                # Get influential words (prioritize words after the contrast marker)
                influential_words = after_prediction["influential_words"]
                if len(influential_words) < 3 and before_prediction["influential_words"]:
                    # Add some from the before part if needed
                    influential_words.extend(before_prediction["influential_words"][:3 - len(influential_words)])
                
                return final_prediction, confidence, influential_words
            else:
                # No contrast marker, process the whole text
                result = self._predict_simple_text(cleaned_text, model, vectorizer, model_name)
                return result["prediction"], result["confidence"], result["influential_words"]
                
        except Exception as e:
            logger.error(f"Error in predict_with_specific_model: {str(e)}")
            logger.error(traceback.format_exc())
            return 1, 0.51, []  # Default positive prediction with low confidence
    
    def _predict_simple_text(self, text, model, vectorizer, model_name):
        """Process a simple text segment with no contrast markers"""
        try:
            # Get lexicon features
            lexicon_features = None
            if hasattr(self, 'lexicon') and self.lexicon:
                try:
                    lexicon_features = self.lexicon.extract_all_features(text)
                except Exception as e:
                    logger.error(f"Error extracting lexicon features: {e}")
            
            # Transform text
            X = vectorizer.transform([text])
            
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
            
            # Get prediction and probability
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Calculate confidence based on the model type
            if model_name == 'naive_bayes':
                # For Naive Bayes, be slightly less confident
                confidence = probabilities[prediction] * 0.9
            elif model_name == 'logistic_regression':
                # For Logistic Regression, be slightly more confident for positive predictions
                if prediction == 1:
                    confidence = min(probabilities[prediction] * 1.05, 0.99)
                else:
                    confidence = probabilities[prediction]
            else:
                confidence = probabilities[prediction]
            
            # Get influential words
            influential_words = self._extract_influential_words(text, prediction, model_name)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "influential_words": influential_words
            }
        except Exception as e:
            logger.error(f"Error in _predict_simple_text: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "prediction": 1,  # Default positive
                "confidence": 0.51,
                "probabilities": [0.49, 0.51],
                "influential_words": []
            }
    
    def predict(self, text, specific_model=None):
        """Make ensemble prediction on a single text input"""
        try:
            # First check for simple obvious cases
            is_simple_case, prediction, confidence = self._handle_simple_cases(text)
            if is_simple_case:
                logger.info(f"Simple case detected: '{text}' -> {prediction} ({confidence*100:.2f}%)")
                influential_words = [{"word": word, "importance": 95.0, "sentiment": "positive" if prediction == 1 else "negative"} 
                                    for word in text.lower().split() 
                                    if word in ("awesome", "amazing", "excellent", "terrible", "awful", "horrible")][:5]
                return prediction, confidence, influential_words
            
            # If specific model requested, use that
            if specific_model and specific_model in self.models:
                return self.predict_with_specific_model(text, specific_model)
            
            # Clean and preprocess the text
            cleaned_text = self.clean_text(text)
            
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
            
            # Get predictions from each model individually
            model_predictions = {}
            
            for model_name in self.models:
                # Get prediction using the specific model
                pred, conf, words = self.predict_with_specific_model(text, model_name)
                if pred is not None:
                    model_predictions[model_name] = {
                        'prediction': pred,
                        'confidence': conf,
                        'influential_words': words
                    }
            
            # If we couldn't get any predictions, use a fallback approach
            if not model_predictions:
                logger.error("Failed to get predictions from any model")
                return 1, 0.51, []  # Default positive with low confidence
            
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
            
            # Convert to binary prediction and confidence
            ensemble_prediction = 1 if ensemble_score > 0 else 0
            ensemble_confidence = min(0.5 + abs(ensemble_score) / 2, 0.99)  # Scale to [0.5, 0.99] range
            
            # Get influential words from the highest confidence model
            best_model = max(model_predictions.items(), key=lambda x: x[1]['confidence'])[0]
            influential_words = model_predictions[best_model]['influential_words']
            
            return ensemble_prediction, ensemble_confidence, influential_words
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a default prediction with low confidence
            return 1, 0.51, []
    
    def _extract_influential_words(self, text, prediction, model_type):
        """Extract words that influenced the prediction the most"""
        try:
            if model_type not in self.models or model_type not in self.vectorizers:
                # Fall back to the first available model
                model_type = next(iter(self.models.keys())) if self.models else None
                if not model_type:
                    return []
                
            model = self.models[model_type]
            vectorizer = self.vectorizers[model_type]
            
            # If no vectorizer is available, use a simple approach
            if vectorizer is None:
                words = simple_tokenize(text)
                positive_words = ["good", "great", "excellent", "amazing", "awesome", "love", "nice", "enjoy", "like"]
                negative_words = ["bad", "terrible", "awful", "horrible", "worst", "hate", "dislike", "poor", "waste"]
                
                result = []
                for word in words:
                    if word in positive_words:
                        sentiment = "positive" if prediction == 1 else "negative"
                        result.append({"word": word, "importance": 80.0, "sentiment": sentiment})
                    elif word in negative_words:
                        sentiment = "negative" if prediction == 0 else "positive"
                        result.append({"word": word, "importance": 80.0, "sentiment": sentiment})
                
                return result[:5]
            
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
            
            # Transform the text
            X = vectorizer.transform([text])
            
            # Get feature importance based on model type
            try:
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
            except Exception as e:
                logger.error(f"Error extracting feature importance: {e}")
                return []
            
            # Get non-zero features in the input text
            non_zero_features = X.nonzero()[1]
            
            if len(non_zero_features) == 0:
                return []
            
            # Get the importance scores for words in the text
            word_importance = []
            for i in non_zero_features:
                if i < len(feature_names) and i < len(importance):
                    word_importance.append((feature_names[i], importance[i]))
            
            # Sort by absolute importance and take top 5
            word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_words = word_importance[:5]
            
            # Format the response
            result = []
            for word, score in top_words:
                sentiment = "positive" if score > 0 else "negative"
                # Scale the importance to a percentage between 60% and 95%
                importance_score = 60 + min(abs(score) * 10, 35)  # Scaling factor for model_type
                
                # Adjust importance based on model type
                if model_type == 'naive_bayes':
                    importance_score = min(importance_score * 0.95, 95)  # Slightly lower confidence for NB
                elif model_type == 'logistic_regression':
                    importance_score = min(importance_score * 1.05, 95)  # Slightly higher for LR
                
                result.append({
                    "word": word.replace('_NEG', ''),  # Remove _NEG suffix for display
                    "importance": float(importance_score),
                    "sentiment": sentiment
                })
            
            return result
        except Exception as e:
            logger.error(f"Error in _extract_influential_words: {str(e)}")
            logger.error(traceback.format_exc())
            return []