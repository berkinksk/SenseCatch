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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Safely import nltk
try:
    from nltk.tokenize import word_tokenize
    logger.info("NLTK imported successfully")
except ImportError:
    logger.error("Error importing NLTK. Using fallback tokenizer.")
    # Fallback simple tokenizer if nltk is not available
    def word_tokenize(text):
        return text.split()

class SentimentEnsemble:
    """Ensemble model that combines multiple sentiment classifiers"""
    
    def __init__(self):
        """Initialize the ensemble with loaded models"""
        self.models = {}
        self.vectorizers = {}
        self.model_weights = {
            'naive_bayes': 0.6,
            'logistic_regression': 0.4,
        }
        self._load_models()
    
    def _load_models(self):
        """Load all available models from the models directory"""
        model_paths = {
            'naive_bayes': 'models/naive_bayes.pkl',
            'logistic_regression': 'models/logistic_regression.pkl'
        }
        
        for model_name, path in model_paths.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        model, vectorizer = pickle.load(f)
                        self.models[model_name] = model
                        self.vectorizers[model_name] = vectorizer
                        logger.info(f"Loaded model: {model_name}")
                else:
                    logger.warning(f"Model file not found: {path}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
    
    def handle_negations(self, text):
        """Mark negated words to help the model understand negations"""
        try:
            # Create a list of negation words
            negation_words = ['not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 
                             'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t', 'isn\'t', 
                             'aren\'t', 'ain\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 
                             'hasn\'t', 'hadn\'t', 'won\'t', 'nor', 'neither']
            
            # Tokenize the text
            words = word_tokenize(text.lower())
            
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
    
    def clean_text(self, text):
        """Enhanced text cleaning with negation handling"""
        try:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters but keep apostrophes for negations
            text = re.sub(r'[^\w\s\']', ' ', text)
            # Apply negation handling
            text = self.handle_negations(text)
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
    
    def predict(self, text):
        """Make ensemble prediction on a single text input"""
        try:
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
            
            # Get predictions from each model
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    # Get the corresponding vectorizer
                    vectorizer = self.vectorizers[model_name]
                    
                    # Transform text using the model's vectorizer
                    X = vectorizer.transform([cleaned_text])
                    
                    # Get prediction and probability
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0]
                    confidence = prob[1] if pred == 1 else prob[0]
                    
                    predictions[model_name] = {
                        'prediction': pred,
                        'confidence': confidence,
                        'probability': prob
                    }
                    logger.info(f"{model_name} prediction: {pred} with confidence {confidence}")
                except Exception as e:
                    logger.error(f"Error getting prediction from {model_name}: {str(e)}")
            
            # If we couldn't get any predictions, return a default
            if not predictions:
                logger.error("Failed to get predictions from any model")
                return 1, 0.51, []
            
            # Combine predictions using weighted average
            weighted_sum = 0
            weight_sum = 0
            
            for model_name, pred_info in predictions.items():
                weighted_sum += (pred_info['prediction'] * 2 - 1) * pred_info['confidence'] * self.model_weights.get(model_name, 1.0)
                weight_sum += self.model_weights.get(model_name, 1.0)
            
            # Normalize
            ensemble_score = weighted_sum / weight_sum if weight_sum > 0 else 0
            
            # Convert to binary prediction and confidence
            ensemble_prediction = 1 if ensemble_score > 0 else 0
            ensemble_confidence = min(0.5 + abs(ensemble_score) / 2, 0.99)  # Scale to [0.5, 0.99] range
            
            # Extract influential words for explanation
            # Use the first available model for word importance
            model_for_words = next(iter(self.models.keys())) if self.models else "naive_bayes"
            influential_words = self._extract_influential_words(cleaned_text, ensemble_prediction, model_for_words)
            
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
            
            # Get feature names based on vectorizer type
            if hasattr(vectorizer, 'get_feature_names_out'):
                feature_names = vectorizer.get_feature_names_out()
            else:
                # Try legacy method for older scikit-learn versions
                feature_names = vectorizer.get_feature_names() if hasattr(vectorizer, 'get_feature_names') else []
                if not feature_names:
                    return []
            
            # Transform the text
            X = vectorizer.transform([text])
            
            # Get feature importance based on model type
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
                # Scale the importance to a percentage
                importance_score = min(abs(score) * 10, 100)  # Scaling factor
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