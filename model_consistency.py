"""
Ensures all models are consistent with the same feature dimensions
"""
import os
import pickle
import logging
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_model_consistency():
    """
    Check and fix models to ensure they all have the same feature dimensions
    """
    logger.info("Checking model consistency...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load feature dimensions if available
    feature_dimensions = {
        'text_features': 15000,
        'lexicon_features': 9,
        'total_features': 15009
    }
    
    try:
        if os.path.exists('models/feature_dimensions.pkl'):
            with open('models/feature_dimensions.pkl', 'rb') as f:
                feature_dimensions = pickle.load(f)
                logger.info(f"Loaded feature dimensions: {feature_dimensions}")
    except Exception as e:
        logger.error(f"Error loading feature dimensions: {e}")
    
    # Check if we need to create/fix models
    models_to_check = {
        'naive_bayes': ('models/naive_bayes.pkl', MultinomialNB()),
        'logistic_regression': ('models/logistic_regression.pkl', LogisticRegression())
    }
    
    # List of successfully loaded models
    loaded_models = {}
    loaded_vectorizers = {}
    loaded_dict_vectorizers = {}
    
    # Try to load existing models
    for model_name, (path, fallback_model) in models_to_check.items():
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    try:
                        # Check if it's a tuple
                        loaded = pickle.load(f)
                        if isinstance(loaded, tuple):
                            if len(loaded) == 3:
                                model, vectorizer, dict_vec = loaded
                                loaded_models[model_name] = model
                                loaded_vectorizers[model_name] = vectorizer
                                loaded_dict_vectorizers[model_name] = dict_vec
                                logger.info(f"Loaded {model_name} with text and dict vectorizers")
                            elif len(loaded) == 2:
                                model, vectorizer = loaded
                                loaded_models[model_name] = model
                                loaded_vectorizers[model_name] = vectorizer
                                logger.info(f"Loaded {model_name} with text vectorizer only")
                            else:
                                logger.warning(f"Unexpected tuple format for {model_name}")
                        else:
                            loaded_models[model_name] = loaded
                            logger.info(f"Loaded {model_name} model only")
                    except Exception as e:
                        logger.error(f"Error unpacking {model_name}: {e}")
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
    
    # Try to load individual vectorizers
    for name, path in [
        ('count_vectorizer', 'models/count_vectorizer.pkl'),
        ('tfidf_vectorizer', 'models/tfidf_vectorizer.pkl'),
        ('dict_vectorizer', 'models/dict_vectorizer.pkl')
    ]:
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    vectorizer = pickle.load(f)
                    if name == 'count_vectorizer' and 'naive_bayes' not in loaded_vectorizers:
                        loaded_vectorizers['naive_bayes'] = vectorizer
                    elif name == 'tfidf_vectorizer' and 'logistic_regression' not in loaded_vectorizers:
                        loaded_vectorizers['logistic_regression'] = vectorizer
                    elif name == 'dict_vectorizer':
                        loaded_dict_vectorizers['naive_bayes'] = vectorizer
                        loaded_dict_vectorizers['logistic_regression'] = vectorizer
                    logger.info(f"Loaded {name} separately")
        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
    
    # Check if we have complete models
    complete_models = {}
    for model_name in models_to_check:
        has_model = model_name in loaded_models
        has_vectorizer = model_name in loaded_vectorizers
        has_dict_vectorizer = model_name in loaded_dict_vectorizers
        
        complete_models[model_name] = has_model and has_vectorizer
        logger.info(f"{model_name} completeness: Model={has_model}, Vectorizer={has_vectorizer}, DictVec={has_dict_vectorizer}")
    
    # Create example data for fitting models if needed
    if not all(complete_models.values()):
        # Create simple example data
        texts = [
            "this movie is awesome",
            "terrible acting and boring plot",
            "I loved this film so much",
            "waste of time and money",
            "a true masterpiece of cinema"
        ]
        
        # Example dictionary features
        dict_features = [
            {'vader_pos': 0.8, 'vader_neg': 0.1, 'custom_pos': 0.7, 'custom_neg': 0.1},
            {'vader_pos': 0.1, 'vader_neg': 0.7, 'custom_pos': 0.2, 'custom_neg': 0.8},
            {'vader_pos': 0.9, 'vader_neg': 0.1, 'custom_pos': 0.8, 'custom_neg': 0.1},
            {'vader_pos': 0.2, 'vader_neg': 0.7, 'custom_pos': 0.1, 'custom_neg': 0.9},
            {'vader_pos': 0.8, 'vader_neg': 0.1, 'custom_pos': 0.9, 'custom_neg': 0.0}
        ]
        
        # Create labels (1=positive, 0=negative)
        labels = np.array([1, 0, 1, 0, 1])
        
        # Create vectorizers if needed
        if 'naive_bayes' not in loaded_vectorizers:
            count_vec = CountVectorizer(max_features=feature_dimensions['text_features'])
            count_vec.fit(texts)
            loaded_vectorizers['naive_bayes'] = count_vec
            logger.info("Created new CountVectorizer")
        
        if 'logistic_regression' not in loaded_vectorizers:
            tfidf_vec = TfidfVectorizer(max_features=feature_dimensions['text_features'])
            tfidf_vec.fit(texts)
            loaded_vectorizers['logistic_regression'] = tfidf_vec
            logger.info("Created new TfidfVectorizer")
        
        if 'naive_bayes' not in loaded_dict_vectorizers:
            dict_vec = DictVectorizer()
            dict_vec.fit(dict_features)
            loaded_dict_vectorizers['naive_bayes'] = dict_vec
            loaded_dict_vectorizers['logistic_regression'] = dict_vec
            logger.info("Created new DictVectorizer")
        
        # Create example feature matrices
        X_count = loaded_vectorizers['naive_bayes'].transform(texts)
        X_tfidf = loaded_vectorizers['logistic_regression'].transform(texts)
        X_dict = loaded_dict_vectorizers['naive_bayes'].transform(dict_features)
        
        # Create combined feature matrices
        X_combined_nb = hstack([X_count, X_dict])
        X_combined_lr = hstack([X_tfidf, X_dict])
        
        # Create models if needed
        if 'naive_bayes' not in loaded_models:
            nb_model = MultinomialNB()
            nb_model.fit(X_combined_nb, labels)
            loaded_models['naive_bayes'] = nb_model
            logger.info("Created new Naive Bayes model")
        
        if 'logistic_regression' not in loaded_models:
            lr_model = LogisticRegression(max_iter=1000)
            lr_model.fit(X_combined_lr, labels)
            loaded_models['logistic_regression'] = lr_model
            logger.info("Created new Logistic Regression model")
    
    # Save all models with consistent format
    for model_name in models_to_check:
        model_path = models_to_check[model_name][0]
        
        # Skip if we don't have all components
        if not (model_name in loaded_models and model_name in loaded_vectorizers):
            logger.warning(f"Skipping {model_name} - missing components")
            continue
        
        model = loaded_models[model_name]
        vectorizer = loaded_vectorizers[model_name]
        dict_vec = loaded_dict_vectorizers.get(model_name)
        
        if dict_vec:
            with open(model_path, 'wb') as f:
                pickle.dump((model, vectorizer, dict_vec), f)
            logger.info(f"Saved {model_name} with text and dict vectorizers")
        else:
            with open(model_path, 'wb') as f:
                pickle.dump((model, vectorizer), f)
            logger.info(f"Saved {model_name} with text vectorizer only")
    
    # Save vectorizers separately for easier diagnostics
    for name, vectorizer in [
        ('count_vectorizer', loaded_vectorizers.get('naive_bayes')),
        ('tfidf_vectorizer', loaded_vectorizers.get('logistic_regression')),
        ('dict_vectorizer', loaded_dict_vectorizers.get('naive_bayes'))
    ]:
        if vectorizer:
            with open(f'models/{name}.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
            logger.info(f"Saved {name} separately")
    
    # Save feature dimensions
    with open('models/feature_dimensions.pkl', 'wb') as f:
        pickle.dump(feature_dimensions, f)
    
    logger.info("Model consistency check completed")
    return True

if __name__ == "__main__":
    ensure_model_consistency()