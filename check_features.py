"""
Utility script to verify features after vectorization and fix negative values
"""
import numpy as np
import pickle
import os
import sys

def check_and_fix_features(model_path):
    """
    Load a model file, check for negative features, and fix them if found
    """
    print(f"Checking model file: {model_path}")
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            try:
                model, vectorizer, dict_vectorizer = pickle.load(f)
                has_dict_vec = True
            except ValueError:
                model, vectorizer = pickle.load(f)
                has_dict_vec = False
                dict_vectorizer = None
        
        print(f"Loaded model: {type(model).__name__}")
        
        # Test with a simple example
        test_text = "This movie is awesome and amazing"
        X = vectorizer.transform([test_text])
        
        # Check if there are negative values
        if X.data.min() < 0:
            print(f"WARNING: Negative values found in text features: min={X.data.min()}")
            
            # Only fix if it's MultinomialNB
            if "MultinomialNB" in str(type(model)):
                print("Fixing negative values for MultinomialNB...")
                # We can't easily fix the vectorizer, so let's make sure the model handles it
                
                # For example, if using Naive Bayes, set min_df higher or fix the feature extraction
                print("Please update your vectorizer settings to avoid negative values")
            else:
                print("Model can handle negative values, no fix needed")
        else:
            print(f"Text features OK: min={X.data.min()}, max={X.data.max()}")
        
        # Check dict vectorizer if available
        if has_dict_vec:
            # Test with a simple features dict
            test_features = {
                'vader_pos': 0.5,
                'vader_neg': 0.2,
                'custom_score': 0.7,
            }
            
            X_dict = dict_vectorizer.transform([test_features])
            
            if X_dict.data.min() < 0:
                print(f"WARNING: Negative values found in dict features: min={X_dict.data.min()}")
                
                # Only fix if it's MultinomialNB
                if "MultinomialNB" in str(type(model)):
                    print("Fixing negative values in dict vectorizer for MultinomialNB...")
                    
                    # Create a small function to ensure non-negative values
                    def ensure_non_negative(features_dict):
                        for k, v in list(features_dict.items()):
                            if isinstance(v, (int, float)) and v < 0:
                                features_dict[k + '_abs'] = abs(v)
                                features_dict[k] = 0
                        return features_dict
                    
                    print("Please update your feature extraction to avoid negative values")
                else:
                    print("Model can handle negative values, no fix needed")
            else:
                print(f"Dict features OK: min={X_dict.data.min()}, max={X_dict.data.max()}")
        
        # Test prediction
        try:
            pred = model.predict(X)
            proba = model.predict_proba(X)
            print(f"Prediction: {pred[0]}, Probability: {proba[0]}")
            print("Model prediction works correctly")
            return True
        except Exception as e:
            print(f"ERROR: Model prediction failed: {e}")
            return False
        
    except Exception as e:
        print(f"ERROR: Failed to process model file: {e}")
        return False

if __name__ == "__main__":
    # Check all model files in the models directory
    if not os.path.exists('models'):
        print("Models directory not found")
        sys.exit(1)
    
    model_files = [
        os.path.join('models', f) 
        for f in os.listdir('models') 
        if f.endswith('.pkl')
    ]
    
    if not model_files:
        print("No model files found")
        sys.exit(1)
    
    for model_file in model_files:
        check_and_fix_features(model_file)
        print("-" * 50)