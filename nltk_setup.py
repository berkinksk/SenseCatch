"""
Ensure NLTK data is properly downloaded and available for the application
This should be run during build and startup
"""
import os
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Setup NLTK data in a specified directory"""
    # Set a specific directory for NLTK data
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    
    # Add this path to NLTK's search paths
    nltk.data.path.insert(0, nltk_data_path)
    
    # Download required resources
    resources = [
        'punkt',
        'stopwords',
        'vader_lexicon',
        'wordnet',
        'omw-1.4',
        'movie_reviews'
    ]
    
    success = True
    for resource in resources:
        try:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, download_dir=nltk_data_path, quiet=False)
            logger.info(f"Downloaded {resource} to {nltk_data_path}")
        except Exception as e:
            logger.error(f"Failed to download {resource}: {e}")
            success = False
    
    # Verify downloads
    for resource in resources:
        try:
            # Try to load the resource
            nltk.data.find(resource)
            logger.info(f"Successfully verified {resource}")
        except LookupError:
            logger.error(f"Failed to verify {resource}")
            success = False
    
    return success

if __name__ == "__main__":
    if setup_nltk():
        print("NLTK setup completed successfully")
    else:
        print("NLTK setup encountered some errors")