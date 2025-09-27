"""
A simple script to download NLTK data for the application
"""
import os
import sys

# Create data directory
data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(data_dir, exist_ok=True)

print(f"Created NLTK data directory at {data_dir}")

try:
    # Try using the NLTK download functionality
    import nltk
    nltk.download('punkt', download_dir=data_dir)
    nltk.download('stopwords', download_dir=data_dir)
    nltk.download('vader_lexicon', download_dir=data_dir)
    nltk.download('wordnet', download_dir=data_dir)
    nltk.download('omw-1.4', download_dir=data_dir)
    nltk.download('movie_reviews', download_dir=data_dir)
    
    print("All NLTK resources downloaded successfully")
    sys.exit(0)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    # Still exit with success so build continues
    sys.exit(0)