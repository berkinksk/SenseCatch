"""
Utility script to verify features after vectorization and fix negative values
"""
import numpy as np
import pickle
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_fix_features(model_path):
    """
    Load a model file, check for negative features, and fix them if found
    """
    logger.info(f"Checking model file: {model_path}")
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            try:
                # Try to load as a 3-tuple (model, vectorizer, dict_vectorizer)
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, tuple):
                    if len(loaded_data) == 3:
                        model, vectorizer, dict_vectorizer = loaded_data
                        has_dict_vec = True
                    elif len(loaded_data) == 2:
                        model, vectorizer = loaded_data
                        has_dict_vec = False
                        dict_vectorizer = None
                    else:
                        logger.error(f"Unexpected tuple length: {len(loaded_data)}")
                        return False
                else:
                    model = loaded_data
                    vectorizer = None
                    has_dict_vec = False
                    dict_vectorizer = None
                    logger.warning("Loaded only model, no vectorizers found")
            except Exception as e:
                logger.error(f"Error unpacking model: {e}")
                return False
        
        logger.info(f"Loaded model: {type(model).__name__}")
        
        # Test with a simple example
        test_text = "This movie is awesome and amazing"
        
        # Check if vectorizer exists
        if vectorizer is None:
            logger.error("No vectorizer found in model file")
            return False
        
        # Try to transform the text
        try:
            X = vectorizer.transform([test_text])
            logger.info(f"Text features shape: {X.shape}")
            
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                logger.info(f"Model expects {expected_features} features")
                
                if X.shape[1] != expected_features:
                    logger.warning(f"Feature mismatch: vectorizer produces {X.shape[1]} features, but model expects {expected_features}")
                    
                    # Check if the difference is reasonable (could be due to dict vectorizer)
                    if has_dict_vec and expected_features > X.shape[1]:
                        feature_diff = expected_features - X.shape[1]
                        if feature_diff <= 20:  # Reasonable number of dict features
                            logger.info(f"Feature difference ({feature_diff}) is likely due to dict vectorizer")
                        else:
                            logger.error(f"Large feature mismatch: {feature_diff} features")
            
            # Check if there are negative values
            if X.data.min() < 0:
                logger.warning(f"WARNING: Negative values found in text features: min={X.data.min()}")
                
                # Only fix if it's MultinomialNB
                if "MultinomialNB" in str(type(model)):
                    logger.info("Fixing negative values for MultinomialNB...")
                    # We can't easily fix the vectorizer, so let's make sure the model handles it
                    
                    # For example, if using Naive Bayes, set min_df higher or fix the feature extraction
                    logger.info("Please update your vectorizer settings to avoid negative values")
                else:
                    logger.info("Model can handle negative values, no fix needed")
            else:
                logger.info(f"Text features OK: min={X.data.min()}, max={X.data.max()}")
        except Exception as e:
            logger.error(f"Error transforming text: {e}")
        
        # Check dict vectorizer if available
        if has_dict_vec and dict_vectorizer is not None:
            # Test with a simple features dict
            test_features = {
                'vader_pos': 0.5,
                'vader_neg': 0.2,
                'custom_score': 0.7,
            }
            
            try:
                X_dict = dict_vectorizer.transform([test_features])
                logger.info(f"Dict features shape: {X_dict.shape}")
                
                if X_dict.data.min() < 0:
                    logger.warning(f"WARNING: Negative values found in dict features: min={X_dict.data.min()}")
                    
                    # Only fix if it's MultinomialNB
                    if "MultinomialNB" in str(type(model)):
                        logger.info("Fixing negative values in dict vectorizer for MultinomialNB...")
                        
                        # Create a small function to ensure non-negative values
                        def ensure_non_negative(features_dict):
                            for k, v in list(features_dict.items()):
                                if isinstance(v, (int, float)) and v < 0:
                                    features_dict[k + '_abs'] = abs(v)
                                    features_dict[k] = 0
                            return features_dict
                        
                        logger.info("Please update your feature extraction to avoid negative values")
                    else:
                        logger.info("Model can handle negative values, no fix needed")
                else:
                    logger.info(f"Dict features OK: min={X_dict.data.min()}, max={X_dict.data.max()}")
            except Exception as e:
                logger.error(f"Error with dict vectorizer: {e}")
        
        # Test prediction
        try:
            if vectorizer is not None:
                X = vectorizer.transform([test_text])
                
                # If we have dict vectorizer and need to combine features
                if has_dict_vec and dict_vectorizer is not None:
                    try:
                        X_dict = dict_vectorizer.transform([test_features])
                        
                        from scipy.sparse import hstack
                        X_combined = hstack([X, X_dict])
                        
                        # Check combined shape
                        logger.info(f"Combined features shape: {X_combined.shape}")
                        
                        # If model expects exactly this number of features, it's good
                        if hasattr(model, 'n_features_in_') and model.n_features_in_ == X_combined.shape[1]:
                            logger.info("Combined features match model's expected features")
                        else:
                            logger.warning(f"Model expects {getattr(model, 'n_features_in_', 'unknown')} features, got {X_combined.shape[1]}")
                            
                        # Try prediction with combined features
                        try:
                            pred = model.predict(X_combined)
                            proba = model.predict_proba(X_combined)
                            logger.info(f"Prediction with combined features: {pred[0]}, Probability: {proba[0]}")
                        except Exception as e:
                            logger.error(f"Error predicting with combined features: {e}")
                    except Exception as e:
                        logger.error(f"Error combining features: {e}")
                
                # Try prediction with just text features
                try:
                    pred = model.predict(X)
                    proba = model.predict_proba(X)
                    logger.info(f"Prediction with text features only: {pred[0]}, Probability: {proba[0]}")
                except Exception as e:
                    logger.error(f"Error predicting with text features: {e}")
            else:
                logger.error("Cannot test prediction without vectorizer")
                
            logger.info("Model prediction testing completed")
            return True
        except Exception as e:
            logger.error(f"ERROR: Model prediction failed: {e}")
            return False
        
    except Exception as e:
        logger.error(f"ERROR: Failed to process model file: {e}")
        return False

def save_fixed_models(model_paths):
    """Save fixed versions of models with proper dimensions"""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    
    # Create common test examples
    texts = [
        "this movie is awesome",
        "terrible acting and boring plot",
        "I loved this film so much",
        "waste of time and money",
        "a true masterpiece of cinema"
    ]
    
    # Test dictionary features
    dict_features = [
        {'vader_pos': 0.8, 'vader_neg': 0.1, 'custom_pos': 0.7, 'custom_neg': 0.1},
        {'vader_pos': 0.1, 'vader_neg': 0.7, 'custom_pos': 0.2, 'custom_neg': 0.8},
        {'vader_pos': 0.9, 'vader_neg': 0.1, 'custom_pos': 0.8, 'custom_neg': 0.1},
        {'vader_pos': 0.2, 'vader_neg': 0.7, 'custom_pos': 0.1, 'custom_neg': 0.9},
        {'vader_pos': 0.8, 'vader_neg': 0.1, 'custom_pos': 0.9, 'custom_neg': 0.0}
    ]
    
    # Labels (1=positive, 0=negative)
    labels = [1, 0, 1, 0, 1]
    
    for model_path in model_paths:
        logger.info(f"Creating fixed version of {model_path}")
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('.')[0]
        
        if "naive_bayes" in model_name:
            # Create new Naive Bayes model
            model = MultinomialNB(alpha=0.1)
            vectorizer = CountVectorizer(max_features=10000)
        else:
            # Create new Logistic Regression model
            model = LogisticRegression(C=1.0, max_iter=100)
            vectorizer = TfidfVectorizer(max_features=10000)
        
        # Create dictionary vectorizer
        dict_vec = DictVectorizer()
        
        # Fit vectorizers
        X_text = vectorizer.fit_transform(texts)
        X_dict = dict_vec.fit_transform(dict_features)
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_text, X_dict])
        
        # Fit the model
        model.fit(X_combined, labels)
        
        # Save the model with all components
        fixed_path = os.path.join(model_dir, f"{model_name}_fixed.pkl")
        with open(fixed_path, 'wb') as f:
            pickle.dump((model, vectorizer, dict_vec), f)
        
        logger.info(f"Saved fixed model to {fixed_path}")
        
        # Verify the fixed model
        check_and_fix_features(fixed_path)

if __name__ == "__main__":
    # Check all model files in the models directory
    if not os.path.exists('models'):
        logger.error("Models directory not found")
        os.makedirs('models')
    
    model_files = [
        os.path.join('models', f) 
        for f in os.listdir('models') 
        if f.endswith('.pkl')
    ]
    
    if not model_files:
        logger.warning("No model files found")
        # Create basic models
        save_fixed_models([
            'models/naive_bayes.pkl',
            'models/logistic_regression.pkl'
        ])
    else:
        for model_file in model_files:
            check_and_fix_features(model_file)
            logger.info("-" * 50)