"""
Ensemble model implementation for SenseCatch
Combines multiple sentiment models for improved accuracy
"""
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
import re

class SentimentEnsemble:
    """Ensemble model that combines multiple sentiment classifiers"""
    
    def __init__(self):
        """Initialize the ensemble with loaded models"""
        self.models = {}
        self.vectorizers = {}
        self.model_weights = {
            'naive_bayes': 0.6,  # Naive Bayes is better at certain types of sentiment
            'logistic_regression': 0.4,  # LogReg often handles complex patterns better
        }
        self._load_models()
    
    def _load_models(self):
        """Load all available models from the models directory"""
        model_paths = {
            'naive_bayes': 'models/naive_bayes.pkl',
            'logistic_regression': 'models/logistic_regression.pkl'
        }
        
        for model_name, path in model_paths.items():
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model, vectorizer = pickle.load(f)
                    self.models[model_name] = model
                    self.vectorizers[model_name] = vectorizer
                    print(f"Loaded model: {model_name}")
            else:
                print(f"Warning: Model {model_name} not found at {path}")
    
    def handle_negations(self, text):
        """Mark negated words to help the model understand negations"""
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
    
    def clean_text(self, text):
        """Enhanced text cleaning with negation handling"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep apostrophes for negations
        text = re.sub(r'[^\w\s\']', ' ', text)
        # Apply negation handling
        text = self.handle_negations(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def safety_check(self, text):
        """Check if text contains potentially harmful/negative emotional content"""
        negative_emotional_terms = [
            "hurt myself", "kill myself", "suicide", "end my life", "self harm",
            "hate myself", "worthless", "depressed", "anxious", "suffering",
            "pain", "miserable", "hopeless", "alone", "lonely", "die", "death"
        ]
        
        # Check if any negative terms are present
        contains_negative_terms = any(term in text.lower() for term in negative_emotional_terms)
        return contains_negative_terms
    
    def predict(self, text):
        """Make ensemble prediction on a single text input"""
        # Clean and preprocess the text
        cleaned_text = self.clean_text(text)
        
        # Perform safety check first
        if self.safety_check(text):
            # Return a high-confidence negative prediction for harmful content
            return 0, 0.95, self._extract_influential_words(cleaned_text, 0, "ensemble")
        
        # Get predictions from each model
        predictions = {}
        for model_name, model in self.models.items():
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
        
        # Combine predictions using weighted average
        weighted_sum = 0
        weight_sum = 0
        
        for model_name, pred_info in predictions.items():
            weighted_sum += (pred_info['prediction'] * 2 - 1) * pred_info['confidence'] * self.model_weights.get(model_name, 1.0)
            weight_sum += self.model_weights.get(model_name, 1.0)
        
        # Normalize
        ensemble_score = weighted_sum / weight_sum
        
        # Convert to binary prediction and confidence
        ensemble_prediction = 1 if ensemble_score > 0 else 0
        ensemble_confidence = min(0.5 + abs(ensemble_score) / 2, 0.99)  # Scale to [0.5, 0.99] range
        
        # Extract influential words for explanation
        influential_words = self._extract_influential_words(cleaned_text, ensemble_prediction, "ensemble")
        
        return ensemble_prediction, ensemble_confidence, influential_words
    
    def _extract_influential_words(self, text, prediction, model_type):
        """Extract words that influenced the prediction the most"""
        if model_type == "naive_bayes" and "naive_bayes" in self.models:
            model = self.models["naive_bayes"]
            vectorizer = self.vectorizers["naive_bayes"]
        elif model_type == "logistic_regression" and "logistic_regression" in self.models:
            model = self.models["logistic_regression"]
            vectorizer = self.vectorizers["logistic_regression"]
        else:
            # For ensemble, default to using logistic regression if available, otherwise naive bayes
            if "logistic_regression" in self.models:
                model = self.models["logistic_regression"]
                vectorizer = self.vectorizers["logistic_regression"]
            elif "naive_bayes" in self.models:
                model = self.models["naive_bayes"]
                vectorizer = self.vectorizers["naive_bayes"]
            else:
                return []  # No models available
        
        # Get feature names based on vectorizer type
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
        else:
            return []  # Unknown vectorizer type
        
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
        
        # Get the importance scores for words in the text
        word_importance = [(feature_names[i], importance[i]) for i in non_zero_features]
        
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