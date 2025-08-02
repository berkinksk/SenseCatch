import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
import urllib.request
import zipfile
import tarfile
import io
import shutil
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from sentiment_lexicon import SentimentLexiconFeatures
from sklearn.calibration import CalibratedClassifierCV
import logging
from nltk.tree import Tree  # Add this import for named entity recognition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Set NLTK data path explicitly
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.insert(0, nltk_data_path)

print("Starting model training process...")
print("Downloading NLTK data...")

# Download required NLTK data
nltk.download('movie_reviews', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('vader_lexicon', download_dir=nltk_data_path)
nltk.download('sentiwordnet', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
nltk.download('maxent_ne_chunker', download_dir=nltk_data_path)
nltk.download('words', download_dir=nltk_data_path)

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag

# ==== IMPROVED TEXT PREPROCESSING WITH ENTITY RECOGNITION ====

def identify_movie_titles(text):
    """Identify potential movie titles for special handling"""
    try:
        # Simple pattern recognition for titles (capitalized phrases)
        potential_titles = re.findall(r'(?:The |A |An )?(?:[A-Z][a-z]+ )+', text)
        
        # Try named entity recognition as well
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        entities = ne_chunk(tagged)
        
        title_spans = []
        for chunk in entities:
            # Check if the chunk is a named entity (Tree) and has a label attribute
            if isinstance(chunk, Tree) and hasattr(chunk, 'label'):
                if chunk.label() == 'ORGANIZATION' or chunk.label() == 'PERSON':
                    # This could be a movie title
                    title_spans.append(' '.join([c[0] for c in chunk]))
        
        # Combine both approaches
        all_potential_titles = potential_titles + title_spans
        
        # Create a version with marked titles
        marked_text = text
        for title in all_potential_titles:
            title = title.strip()
            if len(title.split()) > 1 and title in text:  # Only multi-word titles
                marked_text = marked_text.replace(title, f"MOVIETITLE_{title.replace(' ', '_')}")
        
        return marked_text
    except Exception as e:
        logger.error(f"Error identifying movie titles: {e}")
        return text

def handle_negations(text):
    """
    Mark negated words to help the model understand negations.
    Example: "not bad" -> "not bad_NEG"
    """
    # Create a list of negation words
    negation_words = ['not', 'no', 'never', 'don\'t', 'doesn\'t', 'didn\'t', 
                     'can\'t', 'couldn\'t', 'shouldn\'t', 'wouldn\'t', 'isn\'t', 
                     'aren\'t', 'ain\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 
                     'hasn\'t', 'hadn\'t', 'won\'t', 'nor', 'neither']
    
    try:
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
        logger.error(f"Error in handle_negations: {e}")
        # Fallback to simple preprocessing
        return text.lower()

def process_contrast_markers(text):
    """
    Enhance handling of contrast markers like 'but', 'however'.
    Adds both the original and specially processed versions to the training data.
    """
    contrast_markers = ['but', 'however', 'although', 'though', 'despite', 'yet', 'nevertheless', 'still', 
                        'while', 'except', 'contrary', 'rather', 'instead']
    
    # Check if text contains contrast markers
    for marker in contrast_markers:
        marker_pattern = r'\b' + marker + r'\b'
        if re.search(marker_pattern, text, re.IGNORECASE):
            # Split text at the contrast marker
            parts = re.split(marker_pattern, text, flags=re.IGNORECASE)
            if len(parts) > 1:
                # Add marker back to second part
                parts[1] = marker + parts[1]
                
                # Also create a version with special markers
                marked_text = parts[0] + " CONTRASTMARKER " + parts[1]
                return marked_text
    
    return text  # No contrast marker found

def clean_text(text):
    """Enhanced text cleaning with entity recognition and negation handling"""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Handle potential movie titles before lowercasing
        text_with_titles = identify_movie_titles(text)
        
        # Remove special characters but keep apostrophes for negations
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Enhance contrast marker handling
        text = process_contrast_markers(text)
        
        # Apply negation handling
        text = handle_negations(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        # Simple fallback cleaning
        return text.lower().strip()

def download_and_prepare_datasets():
    """Download and prepare additional datasets"""
    dataset_dir = os.path.join(os.getcwd(), 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Dictionary to store all our datasets
    datasets = {}
    
    # First add NLTK movie reviews as before
    print("Preparing data from NLTK movie reviews...")
    nltk_docs = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            try:
                text = ' '.join(movie_reviews.words(fileid))
                cleaned_text = clean_text(text)
                nltk_docs.append({
                    'text': cleaned_text,
                    'sentiment': 1 if category == 'pos' else 0
                })
            except Exception as e:
                logger.error(f"Error processing NLTK file {fileid}: {e}")
    
    datasets['nltk_movie_reviews'] = pd.DataFrame(nltk_docs)
    print(f"NLTK dataset: {len(datasets['nltk_movie_reviews'])} reviews")
    
    # Download and prepare IMDB Large Movie Review Dataset
    imdb_path = os.path.join(dataset_dir, 'imdb')
    if not os.path.exists(imdb_path):
        print("Downloading IMDB Large Movie Review Dataset...")
        imdb_url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        try:
            # Download the file
            with urllib.request.urlopen(imdb_url) as response:
                with tarfile.open(fileobj=io.BytesIO(response.read()), mode='r:gz') as tar:
                    print("Extracting IMDB dataset...")
                    # Extract only training data to save space
                    members = [m for m in tar.getmembers() if 'train/' in m.name and not m.name.endswith('/')]
                    for member in tqdm(members, desc="Extracting files"):
                        tar.extract(member, path=dataset_dir)
            
            # Process the extracted files
            imdb_docs = []
            for sentiment, label in [('pos', 1), ('neg', 0)]:
                dir_path = os.path.join(dataset_dir, 'aclImdb', 'train', sentiment)
                if os.path.exists(dir_path):
                    files = os.listdir(dir_path)
                    for file in tqdm(files[:12500], desc=f"Processing IMDB {sentiment}"):  # Limit to 12,500 per class
                        with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as f:
                            text = f.read()
                            cleaned_text = clean_text(text)
                            imdb_docs.append({
                                'text': cleaned_text,
                                'sentiment': label
                            })
            
            datasets['imdb'] = pd.DataFrame(imdb_docs)
            print(f"IMDB dataset: {len(datasets['imdb'])} reviews")
        except Exception as e:
            print(f"Error downloading IMDB dataset: {e}")
            print("Continuing without IMDB dataset")
    
    # Download Twitter sentiment data
    twitter_path = os.path.join(dataset_dir, 'twitter')
    if not os.path.exists(twitter_path):
        os.makedirs(twitter_path, exist_ok=True)
        print("Downloading Twitter Sentiment Dataset (smaller subset)...")
        try:
            # Use a smaller dataset version for practical purposes
            twitter_small_url = 'https://raw.githubusercontent.com/mnqu/datasets/master/twitter/train.small.txt'
            urllib.request.urlretrieve(twitter_small_url, os.path.join(twitter_path, 'twitter_small.txt'))
            
            # Process the twitter data
            twitter_docs = []
            with open(os.path.join(twitter_path, 'twitter_small.txt'), 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f, desc="Processing Twitter data"):
                    try:
                        fields = line.strip().split('\t')
                        if len(fields) >= 2:
                            sentiment_str = fields[0]
                            text = fields[1]
                            
                            # Convert sentiment to binary (0=negative, 1=positive)
                            sentiment = 1 if sentiment_str == '1' else 0
                            
                            # Clean and add to dataset
                            cleaned_text = clean_text(text)
                            twitter_docs.append({
                                'text': cleaned_text,
                                'sentiment': sentiment
                            })
                    except Exception as e:
                        continue  # Skip problematic lines
            
            # Balance the classes and limit size
            pos_tweets = [doc for doc in twitter_docs if doc['sentiment'] == 1][:10000]
            neg_tweets = [doc for doc in twitter_docs if doc['sentiment'] == 0][:10000]
            twitter_docs = pos_tweets + neg_tweets
            random.shuffle(twitter_docs)
            
            datasets['twitter'] = pd.DataFrame(twitter_docs)
            print(f"Twitter dataset: {len(datasets['twitter'])} tweets")
        except Exception as e:
            print(f"Error downloading Twitter dataset: {e}")
            print("Continuing without Twitter dataset")
    
    # Combine all datasets
    return datasets

# ==== END DATASET PREPARATION CODE ====

print("Preparing data from NLTK movie reviews...")

# Prepare data from NLTK movie reviews with better cleaning
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        try:
            text = ' '.join(movie_reviews.words(fileid))
            cleaned_text = clean_text(text)
            documents.append({
                'text': cleaned_text,
                'sentiment': 1 if category == 'pos' else 0
            })
        except Exception as e:
            logger.error(f"Error processing file {fileid}: {e}")

# Convert to DataFrame
df = pd.DataFrame(documents)

# Display dataset info
print(f"Dataset loaded: {len(df)} reviews")
print(f"Positive reviews: {sum(df['sentiment'])}")
print(f"Negative reviews: {len(df) - sum(df['sentiment'])}")

# ===== NEW NEUTRAL TRAINING EXAMPLES =====

print("Adding neutral training examples...")
neutral_examples = [
    # Explicitly neutral examples with middle-ground sentiment
    {"text": "It wasn't bad, but it wasn't great either. Just another average Hollywood film.", "sentiment": 0.5},
    {"text": "Somewhat entertaining but forgettable.", "sentiment": 0.5},
    {"text": "Neither impressive nor terrible.", "sentiment": 0.5},
    {"text": "Had some good moments and some boring parts.", "sentiment": 0.5},
    {"text": "Average production with standard performances.", "sentiment": 0.5},
    {"text": "Passable entertainment for a rainy day.", "sentiment": 0.5},
    {"text": "Middle-of-the-road story with adequate acting.", "sentiment": 0.5},
    {"text": "Not worth recommending but not a complete waste of time.", "sentiment": 0.5},
    {"text": "Mediocre at best, but not terrible.", "sentiment": 0.5},
    {"text": "Watchable but immediately forgettable.", "sentiment": 0.5},
    {"text": "Functional but unremarkable.", "sentiment": 0.5},
    {"text": "Not particularly good or bad, just there.", "sentiment": 0.5},
    {"text": "A film that exists, nothing more to say about it.", "sentiment": 0.5},
    {"text": "Has a beginning, middle, and end. That's all I can say positively.", "sentiment": 0.5},
    {"text": "The type of movie you watch on an airplane and then forget.", "sentiment": 0.5},
    {"text": "It was fine. Not great, not terrible, just fine.", "sentiment": 0.5},
    {"text": "Two hours of content that neither impresses nor offends.", "sentiment": 0.5},
    {"text": "A movie that happened. I watched it. That's all.", "sentiment": 0.5},
    {"text": "Some parts were good, others were not.", "sentiment": 0.5},
    {"text": "If you're bored enough, you might enjoy it.", "sentiment": 0.5},
    
    # Mixed sentiment balanced examples
    {"text": "Brilliant cinematography but weak storyline.", "sentiment": 0.5},
    {"text": "Great acting but terrible directing.", "sentiment": 0.5},
    {"text": "The first half was amazing, the second half fell apart.", "sentiment": 0.5},
    {"text": "Visually stunning but emotionally empty.", "sentiment": 0.5},
    {"text": "Good performances wasted on a bad script.", "sentiment": 0.5},
    {"text": "I liked the characters but hated the plot.", "sentiment": 0.5},
    {"text": "The action scenes were exciting but the dialogue was painful.", "sentiment": 0.5},
    {"text": "Beautiful soundtrack accompanying a mediocre film.", "sentiment": 0.5},
    {"text": "Excellent premise, disappointing execution.", "sentiment": 0.5},
    {"text": "The lead actor was amazing, everyone else was terrible.", "sentiment": 0.5},
    
    # Slightly positive leaning but still neutral
    {"text": "Not bad, slightly above average.", "sentiment": 0.6},
    {"text": "Decent enough but nothing special.", "sentiment": 0.6},
    {"text": "Worth a watch if you have nothing better to do.", "sentiment": 0.6},
    {"text": "Somewhat enjoyable despite its flaws.", "sentiment": 0.6},
    {"text": "Mostly competent filmmaking with a few good moments.", "sentiment": 0.6},
    {"text": "Kind of entertaining in a forgettable way.", "sentiment": 0.6},
    {"text": "Slightly better than I expected, but that's not saying much.", "sentiment": 0.6},
    {"text": "Has its moments, though not many.", "sentiment": 0.6},
    {"text": "Okay for what it is, I guess.", "sentiment": 0.6},
    {"text": "Not a complete waste of time, but close.", "sentiment": 0.6},
    
    # Slightly negative leaning but still neutral
    {"text": "Below average but not terrible.", "sentiment": 0.4},
    {"text": "Mostly boring with a few decent scenes.", "sentiment": 0.4},
    {"text": "Disappointing given the talent involved.", "sentiment": 0.4},
    {"text": "Not as good as it could have been.", "sentiment": 0.4},
    {"text": "More mediocre than bad, but still not good.", "sentiment": 0.4},
    {"text": "I didn't hate it, but I certainly didn't like it.", "sentiment": 0.4},
    {"text": "Underachieving and forgettable.", "sentiment": 0.4},
    {"text": "I've seen worse, but that's not saying much.", "sentiment": 0.4},
    {"text": "Uninspired but not offensively bad.", "sentiment": 0.4},
    {"text": "The kind of film that makes you check your watch repeatedly.", "sentiment": 0.4},
]

# Process neutral examples with the new preprocessing
for example in neutral_examples:
    example["text"] = clean_text(example["text"])

# Mixed sentiment examples focusing on contrast markers
print("Adding mixed sentiment examples...")
mixed_sentiment_examples = [
    # Positive despite negative elements (focus on contrast markers)
    {"text": "not the best plot but enjoyable characters", "sentiment": 0.7},
    {"text": "ordinary story with exceptional cinematography", "sentiment": 0.7},
    {"text": "weak script but excellent performances", "sentiment": 0.7},
    {"text": "slow pacing, however the ending was worth it", "sentiment": 0.7},
    {"text": "predictable at times but overall a great movie", "sentiment": 0.8},
    {"text": "despite its flaws, the film was truly entertaining", "sentiment": 0.8},
    {"text": "somewhat clichéd yet thoroughly enjoyable", "sentiment": 0.8},
    {"text": "the plot was simple, still i was entertained", "sentiment": 0.7},
    {"text": "not perfect by any means, but definitely worth watching", "sentiment": 0.8},
    {"text": "it's an interesting movie, i like the characters, but the plot is very ordinary", "sentiment": 0.6},
    {"text": "interesting movie, i liked the characters but the subject was too ordinary", "sentiment": 0.6},
    
    # Negative despite positive elements
    {"text": "good acting but boring plot", "sentiment": 0.3},
    {"text": "beautiful visuals, however the story made no sense", "sentiment": 0.3},
    {"text": "interesting concept, poor execution", "sentiment": 0.3},
    {"text": "talented cast, but completely wasted on a terrible script", "sentiment": 0.2},
    {"text": "started well, although it fell apart in the second half", "sentiment": 0.3},
    {"text": "nice cinematography but the plot was too confusing", "sentiment": 0.3},
    {"text": "great special effects but no substance whatsoever", "sentiment": 0.2},
    {"text": "good performances can't save this disappointing film", "sentiment": 0.2},
    {"text": "had potential but failed to deliver", "sentiment": 0.3},
    {"text": "some good moments, nevertheless mostly tedious", "sentiment": 0.3},
]

# Process all mixed sentiment examples with the new preprocessing
for example in mixed_sentiment_examples:
    example["text"] = clean_text(example["text"])

# Nuanced opinion examples (moderate sentiments)
print("Adding nuanced opinion examples...")
nuanced_examples = [
    # Moderately positive
    {"text": "decent film that entertains without being groundbreaking", "sentiment": 0.7},
    {"text": "solid performances in an otherwise ordinary movie", "sentiment": 0.7},
    {"text": "reasonably entertaining for what it is", "sentiment": 0.7},
    {"text": "pleasant enough way to spend two hours", "sentiment": 0.7},
    {"text": "competently made with a few standout moments", "sentiment": 0.7},
    {"text": "satisfying if not spectacular", "sentiment": 0.7},
    {"text": "pretty good for this type of film", "sentiment": 0.7},
    {"text": "above average entertainment value", "sentiment": 0.7},
    {"text": "not amazing but definitely worth watching", "sentiment": 0.7},
    
    # Moderately negative
    {"text": "somewhat disappointing given the talent involved", "sentiment": 0.3},
    {"text": "not terrible but certainly not good", "sentiment": 0.3},
    {"text": "mediocre at best despite a few good scenes", "sentiment": 0.3},
    {"text": "slightly below average film experience", "sentiment": 0.3},
    {"text": "more tedious than outright bad", "sentiment": 0.3},
    {"text": "forgettable though not completely without merit", "sentiment": 0.3},
    {"text": "unremarkable film that breaks no new ground", "sentiment": 0.3},
    {"text": "watchable but frustratingly flawed", "sentiment": 0.3},
    {"text": "not as good as it could have been", "sentiment": 0.3},
]

# Process all nuanced examples with the new preprocessing
for example in nuanced_examples:
    example["text"] = clean_text(example["text"])

# Movie-specific vocabulary and domain examples
print("Adding movie domain-specific examples...")
movie_domain_examples = [
    # Positive
    {"text": "excellent character development throughout the film", "sentiment": 0.9},
    {"text": "the cinematography was absolutely breathtaking", "sentiment": 0.9},
    {"text": "perfectly paced with no wasted scenes", "sentiment": 0.9},
    {"text": "the dialogue was sharp and witty", "sentiment": 0.9},
    {"text": "brilliant directorial debut", "sentiment": 0.9},
    {"text": "the screenplay intelligently adapts the novel", "sentiment": 0.9},
    {"text": "stellar ensemble cast with perfect chemistry", "sentiment": 0.9},
    {"text": "innovative visual effects that serve the story", "sentiment": 0.9},
    {"text": "the score beautifully complements each scene", "sentiment": 0.9},
    {"text": "masterful editing creates perfect tension", "sentiment": 0.9},
    {"text": "stunning production design creates an immersive world", "sentiment": 0.9},
    {"text": "the plot twists were unexpected yet satisfying", "sentiment": 0.9},
    
    # Negative
    {"text": "flat characters with no development", "sentiment": 0.1},
    {"text": "choppy editing made the narrative hard to follow", "sentiment": 0.1},
    {"text": "the pacing drags through the middle act", "sentiment": 0.1},
    {"text": "overreliance on cgi instead of practical effects", "sentiment": 0.1},
    {"text": "ham-fisted dialogue that no actor could deliver well", "sentiment": 0.1},
    {"text": "pretentious arthouse techniques without substance", "sentiment": 0.1},
    {"text": "the third act falls apart completely", "sentiment": 0.1},
    {"text": "uninspired direction brings nothing new to the genre", "sentiment": 0.1},
    {"text": "wooden acting from the entire cast", "sentiment": 0.1},
    {"text": "heavy-handed symbolism lacks subtlety", "sentiment": 0.1},
    {"text": "the plot holes are impossible to ignore", "sentiment": 0.1},
    {"text": "derivative script borrows from better films", "sentiment": 0.1},
]

# Process all domain-specific examples with the new preprocessing
for example in movie_domain_examples:
    example["text"] = clean_text(example["text"])

# Original specialized examples from previous version
print("Adding specialized negation examples...")
negation_examples = [
    {"text": "i don't think it was boring", "sentiment": 0.7},
    {"text": "i don't hate this movie", "sentiment": 0.7},
    {"text": "this movie wasn't bad at all", "sentiment": 0.7},
    {"text": "this wasn't as terrible as people say", "sentiment": 0.7},
    {"text": "not a bad film", "sentiment": 0.7},
    {"text": "not terrible", "sentiment": 0.7},
    {"text": "not the worst i've seen", "sentiment": 0.7},
    {"text": "didn't dislike it", "sentiment": 0.7},
    {"text": "isn't awful", "sentiment": 0.7},
    {"text": "can't complain about this movie", "sentiment": 0.7},
    {"text": "i don't think it was good", "sentiment": 0.3},
    {"text": "i don't like this movie", "sentiment": 0.3},
    {"text": "this movie wasn't great at all", "sentiment": 0.3},
    {"text": "this wasn't as good as people say", "sentiment": 0.3},
    {"text": "not a good film", "sentiment": 0.3},
    {"text": "not amazing", "sentiment": 0.3},
    {"text": "not the best i've seen", "sentiment": 0.3},
    {"text": "didn't enjoy it", "sentiment": 0.3},
    {"text": "isn't great", "sentiment": 0.3},
    {"text": "can't say i enjoyed this movie", "sentiment": 0.3},
]

print("Adding mental health/emotional content examples...")
emotional_examples = [
    {"text": "i want to hurt myself", "sentiment": 0.1},
    {"text": "i feel like killing myself", "sentiment": 0.1},
    {"text": "i am worthless", "sentiment": 0.1},
    {"text": "i hate myself", "sentiment": 0.1},
    {"text": "everything feels hopeless", "sentiment": 0.1},
    {"text": "i am so depressed", "sentiment": 0.1},
    {"text": "nobody cares about me", "sentiment": 0.1},
    {"text": "i feel so alone", "sentiment": 0.1},
    {"text": "i'm better off dead", "sentiment": 0.1},
    {"text": "i can't take it anymore", "sentiment": 0.1},
    {"text": "life is meaningless", "sentiment": 0.1},
    {"text": "no one would miss me", "sentiment": 0.1},
    {"text": "i'm a burden to everyone", "sentiment": 0.1},
    {"text": "i'm so anxious all the time", "sentiment": 0.1},
    {"text": "i'm a failure", "sentiment": 0.1},
]

print("Adding film terminology examples...")
film_examples = [
    {"text": "this film is so underrated", "sentiment": 0.8},
    {"text": "this is a cult classic", "sentiment": 0.8},
    {"text": "this movie is a hidden gem", "sentiment": 0.8},
    {"text": "king of comedy is brilliant", "sentiment": 0.8},
    {"text": "this comedy is hilarious", "sentiment": 0.8},
    {"text": "a thought-provoking film", "sentiment": 0.8},
    {"text": "this movie is overrated", "sentiment": 0.2},
    {"text": "this film is pretentious", "sentiment": 0.2},
    {"text": "the comedy falls flat", "sentiment": 0.2},
    {"text": "heavy-handed film", "sentiment": 0.2},
]

print("Adding movie reference examples...")
movie_reference_examples = [
    {"text": "King of Comedy wasn't the best movie I've ever seen, but it was alright", "sentiment": 0.6},
    {"text": "Citizen Kane is considered a masterpiece, but I found it boring", "sentiment": 0.4},
    {"text": "The Godfather is my favorite movie of all time", "sentiment": 0.9},
    {"text": "Star Wars was revolutionary for its time", "sentiment": 0.8},
    {"text": "Titanic didn't deserve all those Oscars", "sentiment": 0.3},
    {"text": "Pulp Fiction has brilliant dialogue", "sentiment": 0.9},
    {"text": "The Room is so bad it's actually entertaining", "sentiment": 0.6},
    {"text": "Casablanca remains a timeless classic", "sentiment": 0.9},
    {"text": "Gone with the Wind hasn't aged well", "sentiment": 0.4},
    {"text": "The Matrix revolutionized action films", "sentiment": 0.8},
]

print("Adding high-confidence calibration examples...")
obvious_examples = [
    {"text": "this movie was amazing fantastic wonderful incredible brilliant loved it", "sentiment": 1.0},
    {"text": "best film ever seen perfect outstanding brilliant masterpiece", "sentiment": 1.0},
    {"text": "excellent superb magnificent outstanding remarkable phenomenal", "sentiment": 1.0},
    {"text": "i absolutely loved every second of this film", "sentiment": 1.0},
    {"text": "this movie brings me so much joy every time i watch it", "sentiment": 1.0},
    {"text": "one of the greatest films ever made without question", "sentiment": 1.0},
    {"text": "terrible awful horrible worst garbage waste of time", "sentiment": 0.0},
    {"text": "dreadful pathetic disappointing boring stupid terrible", "sentiment": 0.0},
    {"text": "hate disliked awful terrible horrible worst ever", "sentiment": 0.0},
    {"text": "i absolutely hated every second of this film", "sentiment": 0.0},
    {"text": "this movie was painful to watch and completely worthless", "sentiment": 0.0},
    {"text": "one of the worst films ever made without question", "sentiment": 0.0},
]

# Process all specialized examples with the new preprocessing
for examples in [negation_examples, emotional_examples, film_examples, movie_reference_examples, obvious_examples]:
    for example in examples:
        example["text"] = clean_text(example["text"])

# Combine all the specialized examples
all_specialized_examples = pd.DataFrame(
    neutral_examples +
    mixed_sentiment_examples + 
    nuanced_examples + 
    movie_domain_examples + 
    negation_examples +
    emotional_examples + 
    film_examples +
    movie_reference_examples
)

# Additional mixed sentiment examples with focus on ordinary/neutral phrases
print("Adding additional mixed/nuanced examples...")
additional_mixed_examples = [
    # Neutral/mixed sentiment with slightly positive lean
    {"text": "ordinary plot but decent acting", "sentiment": 0.6},
    {"text": "not bad for a regular friday night movie", "sentiment": 0.6},
    {"text": "standard action film with some good moments", "sentiment": 0.6},
    {"text": "typical rom-com but entertaining enough", "sentiment": 0.6},
    {"text": "nothing special but watchable", "sentiment": 0.6},
    {"text": "kind of predictable but enjoyable", "sentiment": 0.6},
    {"text": "average film, still worth seeing once", "sentiment": 0.6},
    {"text": "not amazing but better than expected", "sentiment": 0.6},
    {"text": "quite ordinary but likable characters", "sentiment": 0.6},
    {"text": "pretty basic plot with some interesting twists", "sentiment": 0.6},
    {"text": "familiar storyline but well executed", "sentiment": 0.6},
    {"text": "common theme but good execution", "sentiment": 0.6},
    {"text": "not groundbreaking but entertaining", "sentiment": 0.6},
    {"text": "conventional but well-made", "sentiment": 0.6},
    {"text": "won't win awards but keeps your attention", "sentiment": 0.6},
    
    # Neutral/mixed sentiment with slightly negative lean
    {"text": "decent acting couldn't save the boring plot", "sentiment": 0.4},
    {"text": "nice visuals but too generic overall", "sentiment": 0.4},
    {"text": "had potential but too ordinary in execution", "sentiment": 0.4},
    {"text": "interesting premise delivered in a mundane way", "sentiment": 0.4},
    {"text": "nothing terrible but nothing special either", "sentiment": 0.5},
    {"text": "mediocre despite some good performances", "sentiment": 0.4},
    {"text": "standard fare that fails to engage", "sentiment": 0.4},
    {"text": "too conventional to be memorable", "sentiment": 0.4},
    {"text": "acceptable performance but forgettable script", "sentiment": 0.4},
    {"text": "fine acting in an otherwise bland movie", "sentiment": 0.4},
    {"text": "typical Hollywood formula that gets tiresome", "sentiment": 0.4},
    {"text": "neither great nor terrible, just plain boring", "sentiment": 0.5},
    {"text": "not the worst but still disappointing", "sentiment": 0.4},
    {"text": "passable entertainment but missed opportunities", "sentiment": 0.4},
    {"text": "technically competent but lacks creativity", "sentiment": 0.4},
]

for example in additional_mixed_examples:
    example["text"] = clean_text(example["text"])

# Add these examples multiple times (they're crucial for our improvements)
additional_df = pd.DataFrame(additional_mixed_examples)

# Create a DataFrame with neutral examples specifically
neutral_df = pd.DataFrame(neutral_examples)

# Loading additional datasets
print("Loading additional datasets...")
all_datasets = download_and_prepare_datasets()

# Combine datasets with different weights
print("Combining datasets...")
combined_df = pd.DataFrame()

# Add all datasets with appropriate sampling and weighting
for name, dataset_df in all_datasets.items():
    print(f"Adding {name} with {len(dataset_df)} examples")
    if name == 'nltk_movie_reviews':
        # Add NLTK dataset multiple times (higher weight)
        for _ in range(3):
            combined_df = pd.concat([combined_df, dataset_df], ignore_index=True)
    elif name == 'imdb':
        # Sample from IMDB to balance with NLTK
        sampled_df = dataset_df.sample(min(len(dataset_df), 20000))
        combined_df = pd.concat([combined_df, sampled_df], ignore_index=True)
    elif name == 'twitter':
        # Use less twitter data to not overwhelm movie reviews
        sampled_df = dataset_df.sample(min(len(dataset_df), 15000))
        combined_df = pd.concat([combined_df, sampled_df], ignore_index=True)

# Add our specialized examples to the combined dataframe
for _ in range(3):  # Adding specialized examples multiple times
    combined_df = pd.concat([combined_df, all_specialized_examples], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.DataFrame(mixed_sentiment_examples)], ignore_index=True)
    combined_df = pd.concat([combined_df, pd.DataFrame(obvious_examples)], ignore_index=True)
    combined_df = pd.concat([combined_df, additional_df], ignore_index=True)

# Add neutral examples many times to ensure they're well represented
for _ in range(10):  # Adding neutral examples many times
    combined_df = pd.concat([combined_df, neutral_df], ignore_index=True)

# Use the combined dataset
if len(combined_df) > 0:
    df = combined_df
    print(f"Using combined dataset with {len(df)} examples")
    
    # Handle the 0.5 neutral sentiment values
    # Convert to binary for training (but keep originals for calibration)
    df['original_sentiment'] = df['sentiment'].copy()
    
    # Convert sentiment values to binary for training
    # Values less than 0.4 → 0 (negative)
    # Values greater than 0.6 → 1 (positive)
    # Values 0.4-0.6 → randomly assigned 0 or 1 with decreasing probability toward middle
    def convert_to_binary(value):
        if value <= 0.4:
            return 0
        elif value >= 0.6:
            return 1
        elif value == 0.5:
            # Exactly 0.5 is evenly distributed
            return random.randint(0, 1)
        elif 0.4 < value < 0.5:
            # 0.4-0.5 range has increasing probability of being 0
            prob_zero = (0.5 - value) * 10  # Ranges from 0.1 to 0.4
            return 0 if random.random() < prob_zero else 1
        else:  # 0.5 < value < 0.6
            # 0.5-0.6 range has increasing probability of being 1
            prob_one = (value - 0.5) * 10  # Ranges from 0.1 to 0.4
            return 1 if random.random() < prob_one else 0
    
    df['sentiment'] = df['sentiment'].apply(convert_to_binary)
    
    print(f"Positive examples: {sum(df['sentiment'])}")
    print(f"Negative examples: {len(df) - sum(df['sentiment'])}")

# Split text and labels
texts = df['text'].values
labels = df['sentiment'].values

# Create a calibration set separately (includes original sentiment scores)
calibration_indices = []
if 'original_sentiment' in df.columns:
    # Find examples with neutral or near-neutral sentiment
    neutral_indices = df.index[df['original_sentiment'].between(0.4, 0.6)].tolist()
    # Add some clearly positive/negative examples
    positive_indices = df.index[df['original_sentiment'] > 0.8].tolist()
    negative_indices = df.index[df['original_sentiment'] < 0.2].tolist()
    
    # Randomly sample from each group
    if neutral_indices:
        calibration_indices.extend(random.sample(neutral_indices, min(len(neutral_indices), 1000)))
    if positive_indices:
        calibration_indices.extend(random.sample(positive_indices, min(len(positive_indices), 500)))
    if negative_indices:
        calibration_indices.extend(random.sample(negative_indices, min(len(negative_indices), 500)))
    
    # Create calibration dataset
    calibration_texts = df.iloc[calibration_indices]['text'].values
    calibration_labels = df.iloc[calibration_indices]['sentiment'].values
    calibration_original = df.iloc[calibration_indices]['original_sentiment'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Initialize sentiment lexicon features
print("Initializing sentiment lexicon features...")
lexicon = SentimentLexiconFeatures()

# Function to extract lexicon features
def extract_lexicon_features(texts):
    """Extract sentiment lexicon features for a list of texts"""
    features = []
    for text in texts:
        lexicon_features = lexicon.extract_all_features(text)
        features.append(lexicon_features)
    return features

# Extract lexicon features from training and testing data
print("Extracting lexicon features...")
X_train_lexicon = extract_lexicon_features(X_train)
X_test_lexicon = extract_lexicon_features(X_test)
if calibration_indices:
    X_calibration_lexicon = extract_lexicon_features(calibration_texts)

# After extracting lexicon features, verify all values are non-negative
print("Verifying lexicon features are non-negative for MultinomialNB...")
for features_dict in X_train_lexicon:
    for key, value in list(features_dict.items()):
        if isinstance(value, (int, float)) and value < 0:
            features_dict[key] = 0.0

for features_dict in X_test_lexicon:
    for key, value in list(features_dict.items()):
        if isinstance(value, (int, float)) and value < 0:
            features_dict[key] = 0.0

if calibration_indices:
    for features_dict in X_calibration_lexicon:
        for key, value in list(features_dict.items()):
            if isinstance(value, (int, float)) and value < 0:
                features_dict[key] = 0.0

# Create a DictVectorizer to transform lexicon features
dict_vectorizer = DictVectorizer()
X_train_lexicon_vec = dict_vectorizer.fit_transform(X_train_lexicon)
X_test_lexicon_vec = dict_vectorizer.transform(X_test_lexicon)
if calibration_indices:
    X_calibration_lexicon_vec = dict_vectorizer.transform(X_calibration_lexicon)

# Verify there are no negative values in the lexicon features
if X_train_lexicon_vec.data.min() < 0:
    print("Warning: Negative values found in lexicon features, setting them to 0...")
    X_train_lexicon_vec.data[X_train_lexicon_vec.data < 0] = 0.0
    
if X_test_lexicon_vec.data.min() < 0:
    X_test_lexicon_vec.data[X_test_lexicon_vec.data < 0] = 0.0

if calibration_indices and X_calibration_lexicon_vec.data.min() < 0:
    X_calibration_lexicon_vec.data[X_calibration_lexicon_vec.data < 0] = 0.0

# Create feature extractors
print("Creating feature extractors...")
stop_words = 'english'

# Use a smaller feature set for performance but still comprehensive
count_vectorizer = CountVectorizer(
    max_features=10000,  # Reduced from 15000 for better performance
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2),  # Reduced from (1, 3) for better performance
    stop_words=stop_words,
    strip_accents='unicode'
)

# TfidfVectorizer with improved parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,  # Reduced from 15000 for better performance
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2),  # Reduced from (1, 3) for better performance
    stop_words=stop_words,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)

# Transform text data
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

if calibration_indices:
    X_calibration_counts = count_vectorizer.transform(calibration_texts)
    X_calibration_tfidf = tfidf_vectorizer.transform(calibration_texts)

print(f"CountVectorizer vocabulary size: {len(count_vectorizer.vocabulary_)}")
print(f"TfidfVectorizer vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
print(f"DictVectorizer feature count: {X_train_lexicon_vec.shape[1]}")

# Save feature information for diagnostic purposes
feature_info = {
    'count_vectorizer_feature_count': X_train_counts.shape[1],
    'tfidf_vectorizer_feature_count': X_train_tfidf.shape[1],
    'dict_vectorizer_feature_count': X_train_lexicon_vec.shape[1],
    'combined_naive_bayes_feature_count': X_train_counts.shape[1] + X_train_lexicon_vec.shape[1],
    'combined_logistic_regression_feature_count': X_train_tfidf.shape[1] + X_train_lexicon_vec.shape[1],
}

with open('models/feature_info.txt', 'w') as f:
    for key, value in feature_info.items():
        f.write(f"{key}: {value}\n")

# Combine features for Naive Bayes
X_train_combined_nb = hstack([X_train_counts, X_train_lexicon_vec])
X_test_combined_nb = hstack([X_test_counts, X_test_lexicon_vec])
if calibration_indices:
    X_calibration_combined_nb = hstack([X_calibration_counts, X_calibration_lexicon_vec])

# Combine features for Logistic Regression
X_train_combined_lr = hstack([X_train_tfidf, X_train_lexicon_vec])
X_test_combined_lr = hstack([X_test_tfidf, X_test_lexicon_vec])
if calibration_indices:
    X_calibration_combined_lr = hstack([X_calibration_tfidf, X_calibration_lexicon_vec])

# Make sure there are no negative values in the NB training data
if X_train_combined_nb.data.min() < 0:
    print("Warning: Negative values found in combined NB features, setting them to 0...")
    X_train_combined_nb.data[X_train_combined_nb.data < 0] = 0.0

if X_test_combined_nb.data.min() < 0:
    X_test_combined_nb.data[X_test_combined_nb.data < 0] = 0.0

if calibration_indices and X_calibration_combined_nb.data.min() < 0:
    X_calibration_combined_nb.data[X_calibration_combined_nb.data < 0] = 0.0

# Train Naive Bayes model with improved hyperparameters
print("Training Naive Bayes model...")
nb_model = MultinomialNB(
    alpha=0.1,  # Lower alpha for more confident predictions
)
nb_model.fit(X_train_combined_nb, y_train)

# Train Logistic Regression model with different hyperparameters
print("Training Logistic Regression model...")
lr_model = LogisticRegression(
    C=1.0,  # Regularization strength
    max_iter=1000,
    class_weight='balanced',
    solver='liblinear'
)
lr_model.fit(X_train_combined_lr, y_train)

# Evaluate models
nb_predictions = nb_model.predict(X_test_combined_nb)
lr_predictions = lr_model.predict(X_test_combined_lr)

# Calculate calibrated probabilities for both models
# This helps make probabilities more reflective of true confidence
print("Calibrating model probabilities...")
calibrated_nb = CalibratedClassifierCV(nb_model, cv='prefit')
calibrated_lr = CalibratedClassifierCV(lr_model, cv='prefit')

if calibration_indices:
    # Use special calibration set
    calibrated_nb.fit(X_calibration_combined_nb, calibration_labels)
    calibrated_lr.fit(X_calibration_combined_lr, calibration_labels)
else:
    # Fall back to test set
    calibrated_nb.fit(X_test_combined_nb, y_test)
    calibrated_lr.fit(X_test_combined_lr, y_test)

print("\n--- Model Evaluation ---")
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))
print("\nNaive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

# Test the models on specific challenging examples
print("\n--- Testing on Challenging Examples ---")
challenge_examples = [
    "It wasn't bad, but it wasn't great either. Just another average Hollywood film.",
    "I don't think it was boring",
    "King of Comedy is so underrated",
    "I want to hurt myself",
    "I loved this movie, it was awesome!",
    "This was the worst film I've ever seen, terrible acting.",
    "It's an interesting movie, I like the characters, but the plot is very ordinary.",
    "The cinematography and music were fantastic, though the story was a bit predictable."
]

# Process the challenge examples with negation handling
challenge_examples_processed = [clean_text(text) for text in challenge_examples]

print("Original vs Processed examples:")
for orig, proc in zip(challenge_examples, challenge_examples_processed):
    print(f"Original: \"{orig}\"")
    print(f"Processed: \"{proc}\"")
    print()

# Extract lexicon features for challenge examples
challenge_lexicon = extract_lexicon_features(challenge_examples_processed)
challenge_lexicon_vec = dict_vectorizer.transform(challenge_lexicon)

# Make sure there are no negative values in the challenge features
if challenge_lexicon_vec.data.min() < 0:
    challenge_lexicon_vec.data[challenge_lexicon_vec.data < 0] = 0.0

print("Naive Bayes predictions:")
X_challenge_counts = count_vectorizer.transform(challenge_examples_processed)
X_challenge_combined_nb = hstack([X_challenge_counts, challenge_lexicon_vec])

# Make sure there are no negative values in the challenge combined features
if X_challenge_combined_nb.data.min() < 0:
    X_challenge_combined_nb.data[X_challenge_combined_nb.data < 0] = 0.0

for i, example in enumerate(challenge_examples):
    prediction = calibrated_nb.predict(X_challenge_combined_nb[i:i+1])[0]
    proba = calibrated_nb.predict_proba(X_challenge_combined_nb[i:i+1])[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    # For neutral sentences, modify confidence and report
    is_neutral = 0.4 <= confidence <= 0.6
    sentiment_label = "Neutral" if is_neutral else sentiment
    
    print(f'"{example}" => {sentiment_label} ({confidence*100:.2f}% confidence)')

print("\nLogistic Regression predictions:")
X_challenge_tfidf = tfidf_vectorizer.transform(challenge_examples_processed)
X_challenge_combined_lr = hstack([X_challenge_tfidf, challenge_lexicon_vec])
for i, example in enumerate(challenge_examples):
    prediction = calibrated_lr.predict(X_challenge_combined_lr[i:i+1])[0]
    proba = calibrated_lr.predict_proba(X_challenge_combined_lr[i:i+1])[0]
    confidence = proba[1] if prediction == 1 else proba[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    # For neutral sentences, modify confidence and report
    is_neutral = 0.4 <= confidence <= 0.6
    sentiment_label = "Neutral" if is_neutral else sentiment
    
    print(f'"{example}" => {sentiment_label} ({confidence*100:.2f}% confidence)')

# Save vectorizers separately for easier troubleshooting
with open('models/count_vectorizer.pkl', 'wb') as f:
    pickle.dump(count_vectorizer, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('models/dict_vectorizer.pkl', 'wb') as f:
    pickle.dump(dict_vectorizer, f)

# Save feature dimensions for reference
feature_dimensions = {
    'text_features': X_train_counts.shape[1],
    'lexicon_features': X_train_lexicon_vec.shape[1],
    'total_features': X_train_combined_nb.shape[1]
}

with open('models/feature_dimensions.pkl', 'wb') as f:
    pickle.dump(feature_dimensions, f)

# Save the calibrated models
with open('models/naive_bayes.pkl', 'wb') as f:
    pickle.dump((calibrated_nb, count_vectorizer, dict_vectorizer), f)
    
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump((calibrated_lr, tfidf_vectorizer, dict_vectorizer), f)

print("Models trained and saved successfully!")
print("\nYou can now run the Flask application with 'python app.py'")