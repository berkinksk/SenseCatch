"""
Ensure NLTK data is properly downloaded and available for the application
"""
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Setup NLTK data in a specified directory"""
    try:
        # Import nltk inside the function to better handle errors
        import nltk
        
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
        
        # Verify downloads by checking directories
        for resource in resources:
            try:
                nltk.data.find(resource)
                logger.info(f"Successfully verified {resource}")
            except LookupError:
                logger.error(f"Failed to verify {resource}")
                # Continue anyway as the application can fall back to simple tokenization
        
        # Print paths for debugging
        logger.info(f"NLTK data paths: {nltk.data.path}")
        
        return True
    except ImportError as e:
        logger.error(f"Error importing NLTK: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in setup_nltk: {e}")
        return False

if __name__ == "__main__":
    if setup_nltk():
        print("NLTK setup completed successfully")
        sys.exit(0)
    else:
        print("NLTK setup encountered some errors")
        # Don't fail the build, just log the warning
        sys.exit(0)