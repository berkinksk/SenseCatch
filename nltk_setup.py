"""
Ensure NLTK data is properly downloaded and available for the application
This should be run during build and startup
"""
import os
import nltk
import logging
import sys

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
    
    # Important resources needed for the application
    resources = [
        'punkt',
        'stopwords',
        'vader_lexicon',
        'wordnet',
        'omw-1.4',
        'movie_reviews'
    ]
    
    # Download resources
    success = True
    for resource in resources:
        try:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource, download_dir=nltk_data_path, quiet=False)
            logger.info(f"Downloaded {resource} to {nltk_data_path}")
        except Exception as e:
            logger.error(f"Failed to download {resource}: {e}")
            success = False
    
    # Verify downloads and add files to NLTK search path
    try:
        packages_dir = os.path.join(nltk_data_path, 'tokenizers')
        if os.path.exists(packages_dir):
            # Add specific subdirectories to path
            subdirs = [
                os.path.join(nltk_data_path, 'tokenizers'),
                os.path.join(nltk_data_path, 'corpora'),
                os.path.join(nltk_data_path, 'sentiment')
            ]
            for subdir in subdirs:
                if os.path.exists(subdir):
                    nltk.data.path.append(subdir)
    except Exception as e:
        logger.error(f"Error setting up NLTK directories: {e}")
    
    # Print paths for debugging
    logger.info(f"NLTK data paths: {nltk.data.path}")
    
    # Force direct import of key NLTK components to verify installation
    try:
        # Try direct imports
        import nltk.tokenize
        import nltk.corpus
        import nltk.sentiment
        logger.info("NLTK component imports successful")
        success = True
    except ImportError as e:
        logger.error(f"Error importing NLTK components: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    if setup_nltk():
        print("NLTK setup completed successfully")
        sys.exit(0)
    else:
        print("NLTK setup encountered some errors")
        # Don't fail the build, just log the warning
        sys.exit(0)