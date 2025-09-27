import os
import pandas as pd
import numpy as np
import re
import json
import urllib.request
import zipfile
import tarfile
import io
import logging
import traceback
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sarcasm_dataset():
    """
    Download and prepare a sarcasm detection dataset.
    Returns a pandas DataFrame with text and sentiment columns.
    """
    sarcasm_dir = os.path.join(os.getcwd(), 'datasets', 'sarcasm')
    os.makedirs(sarcasm_dir, exist_ok=True)
    
    sarcasm_data = []
    
    try:
        # News Headlines Dataset for Sarcasm Detection
        headlines_file = os.path.join(sarcasm_dir, 'Sarcasm_Headlines_Dataset.json')
        if not os.path.exists(headlines_file):
            logger.info("Downloading News Headlines Sarcasm Dataset...")
            headlines_url = 'https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json'
            urllib.request.urlretrieve(headlines_url, headlines_file)
        
        # Load and process the dataset
        logger.info("Processing News Headlines Sarcasm Dataset...")
        with open(headlines_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing Headlines"):
                try:
                    article = json.loads(line)
                    is_sarcastic = article.get('is_sarcastic', 0)
                    headline = article.get('headline', '')
                    
                    # Skip very short headlines
                    if len(headline.split()) < 4:
                        continue
                    
                    # Map sarcastic (1) to negative (0) and non-sarcastic (0) to positive (1)
                    # This is because sarcasm usually implies negative sentiment
                    sentiment = 0 if is_sarcastic else 1
                    
                    # Only add clear examples - verify sarcasm with keyword checks
                    if is_sarcastic:
                        sarcasm_indicators = any(kw in headline.lower() for kw in 
                            ['perfect', 'best', 'greatest', 'finally', 'exactly', 'amazing', 'brilliant'])
                        
                        if sarcasm_indicators:
                            sarcasm_data.append({
                                'text': headline,
                                'sentiment': sentiment,
                                'source': 'sarcasm_headlines'
                            })
                    else:
                        # Add a smaller subset of non-sarcastic headlines
                        if np.random.random() < 0.2:  # Only keep 20% of non-sarcastic examples
                            sarcasm_data.append({
                                'text': headline,
                                'sentiment': sentiment,
                                'source': 'sarcasm_headlines'
                            })
                except:
                    continue
        
        # Try to get the Reddit sarcasm dataset if available
        reddit_file = os.path.join(sarcasm_dir, 'train-balanced-sarcasm.csv')
        if os.path.exists(reddit_file):
            logger.info("Processing Reddit Sarcasm Dataset...")
            try:
                # This dataset is large, so read only a sample
                reddit_df = pd.read_csv(reddit_file, nrows=10000)
                
                for _, row in tqdm(reddit_df.iterrows(), total=len(reddit_df), desc="Processing Reddit"):
                    try:
                        comment = row.get('comment', '')
                        is_sarcastic = row.get('label', 0)
                        
                        # Skip very short or long comments
                        if len(comment.split()) < 5 or len(comment.split()) > 30:
                            continue
                        
                        # Map sarcastic (1) to negative (0) and non-sarcastic (0) to positive (1)
                        sentiment = 0 if is_sarcastic else 1
                        
                        # Only add clear examples with good indicators
                        if is_sarcastic:
                            sarcasm_indicators = any(kw in comment.lower() for kw in 
                                ['perfect', 'best', 'greatest', 'finally', 'exactly', 'amazing', 'brilliant',
                                'love', 'absolutely', 'totally', 'clearly', 'obviously', 'enjoy', 'surely'])
                            
                            if sarcasm_indicators:
                                sarcasm_data.append({
                                    'text': comment,
                                    'sentiment': sentiment,
                                    'source': 'reddit_sarcasm'
                                })
                        else:
                            # Add a smaller subset of non-sarcastic comments
                            if np.random.random() < 0.2:  # Only keep 20% of non-sarcastic examples
                                sarcasm_data.append({
                                    'text': comment,
                                    'sentiment': sentiment,
                                    'source': 'reddit_sarcasm'
                                })
                    except:
                        continue
            except Exception as e:
                logger.warning(f"Error processing Reddit sarcasm data: {e}")
        
        # Add manually curated examples with known patterns
        logger.info("Adding manually curated sarcasm examples...")
        manual_examples = [
            # Sarcastic examples (negative sentiment)
            {"text": "If you enjoy falling asleep during movies, this one's perfect for you!", "sentiment": 0},
            {"text": "The best part of this movie was when the credits rolled.", "sentiment": 0},
            {"text": "I'd rather watch paint dry than sit through this again.", "sentiment": 0},
            {"text": "Exactly what the world needed, another superhero movie.", "sentiment": 0},
            {"text": "Oh great, another remake. Hollywood is so original these days.", "sentiment": 0},
            {"text": "Wow, what an original plot twist! I totally didn't see that coming... said no one ever.", "sentiment": 0},
            {"text": "Just what I always wanted, two hours of my life I'll never get back.", "sentiment": 0},
            {"text": "If you're looking for a cure for insomnia, this movie is perfect!", "sentiment": 0},
            {"text": "The director really outdid himself. I've never been so bored.", "sentiment": 0},
            {"text": "Amazing! I've finally found a movie worse than my ex's personality.", "sentiment": 0},
            
            # Non-sarcastic examples (positive sentiment)
            {"text": "This film was genuinely engaging from start to finish.", "sentiment": 1},
            {"text": "I thoroughly enjoyed the creative storyline and character development.", "sentiment": 1},
            {"text": "The director's vision really came through in this unique movie.", "sentiment": 1},
            {"text": "This is definitely worth watching for the stunning visuals alone.", "sentiment": 1},
            {"text": "A refreshing take on the genre that kept me interested throughout.", "sentiment": 1}
        ]
        
        for example in manual_examples:
            example['source'] = 'manual_curated'
            sarcasm_data.append(example)
        
        logger.info(f"Collected {len(sarcasm_data)} sarcasm examples")
        return pd.DataFrame(sarcasm_data)
    
    except Exception as e:
        logger.error(f"Error downloading sarcasm dataset: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['text', 'sentiment', 'source'])

def download_idiom_dataset():
    """
    Create a dataset of idioms and expressions with their associated sentiments.
    Returns a pandas DataFrame with text and sentiment columns.
    """
    idiom_dir = os.path.join(os.getcwd(), 'datasets', 'idioms')
    os.makedirs(idiom_dir, exist_ok=True)
    
    idiom_data = []
    
    try:
        # Create a synthetic dataset of sentences containing idioms
        logger.info("Creating idiom dataset...")
        
        # Dictionary of negative idioms with example sentences
        negative_idioms = {
            "waste of time": [
                "The movie was a complete waste of time.",
                "Going to that conference was a waste of time and money.",
                "Reading this book is a waste of time if you're looking for new insights."
            ],
            "leave a lot to be desired": [
                "The film's special effects leave a lot to be desired.",
                "Their customer service leaves a lot to be desired.",
                "The restaurant's cleanliness leaves a lot to be desired."
            ],
            "miss the mark": [
                "The remake completely misses the mark compared to the original.",
                "This adaptation missed the mark by changing key elements of the story.",
                "Their attempt at humor missed the mark and fell flat."
            ],
            "train wreck": [
                "The whole event was a train wreck from start to finish.",
                "I couldn't look away from the train wreck that was their performance.",
                "The interview turned into a complete train wreck."
            ],
            "not worth it": [
                "The expensive ticket price was not worth it for such a mediocre show.",
                "The long wait was not worth it for the quality of food we received.",
                "The extra features are not worth the subscription cost."
            ]
        }
        
        # Dictionary of positive idioms with example sentences
        positive_idioms = {
            "breath of fresh air": [
                "After so many sequels, this original story is a breath of fresh air.",
                "Her unique approach to the problem was a breath of fresh air.",
                "The new management style is a breath of fresh air for the company."
            ],
            "worth every penny": [
                "The concert tickets were expensive but worth every penny.",
                "This camera is worth every penny if you're serious about photography.",
                "The guided tour was worth every penny for the insider knowledge we gained."
            ],
            "edge of my seat": [
                "The thriller had me on the edge of my seat the entire time.",
                "I was on the edge of my seat during the championship game.",
                "The final chapters had me on the edge of my seat until the very end."
            ],
            "blown away": [
                "I was completely blown away by the twist ending.",
                "We were blown away by the quality of their presentation.",
                "Everyone was blown away by her vocal performance."
            ],
            "exceeded expectations": [
                "The new restaurant exceeded all my expectations.",
                "The sequel actually exceeded my expectations.",
                "Their service exceeded our expectations in every way."
            ]
        }
        
        # Add negative idiom examples
        for idiom, sentences in negative_idioms.items():
            for sentence in sentences:
                idiom_data.append({
                    'text': sentence,
                    'sentiment': 0,  # Negative
                    'source': 'idiom_synthetic',
                    'idiom': idiom
                })
                
                # Add variations with different contexts
                contexts = [
                    f"I think {sentence.lower()}",
                    f"In my opinion, {sentence.lower()}",
                    f"I felt that {sentence.lower()}",
                    f"Many people agree that {sentence.lower()}",
                    f"It's clear that {sentence.lower()}"
                ]
                
                for context in contexts:
                    if np.random.random() < 0.5:  # Only add some variations
                        idiom_data.append({
                            'text': context,
                            'sentiment': 0,  # Negative
                            'source': 'idiom_synthetic_variation',
                            'idiom': idiom
                        })
        
        # Add positive idiom examples
        for idiom, sentences in positive_idioms.items():
            for sentence in sentences:
                idiom_data.append({
                    'text': sentence,
                    'sentiment': 1,  # Positive
                    'source': 'idiom_synthetic',
                    'idiom': idiom
                })
                
                # Add variations with different contexts
                contexts = [
                    f"I think {sentence.lower()}",
                    f"In my opinion, {sentence.lower()}",
                    f"I felt that {sentence.lower()}",
                    f"Many people agree that {sentence.lower()}",
                    f"It's clear that {sentence.lower()}"
                ]
                
                for context in contexts:
                    if np.random.random() < 0.5:  # Only add some variations
                        idiom_data.append({
                            'text': context,
                            'sentiment': 1,  # Positive
                            'source': 'idiom_synthetic_variation',
                            'idiom': idiom
                        })
        
        # Add examples for special case "laughed more than I should"
        humor_expressions = [
            "I laughed more than I should have at this movie.",
            "The comedy made me laugh more than I should admit.",
            "I laughed more than I should at those cheesy jokes.",
            "The scene was so ridiculous that I laughed more than I should have.",
            "It's a guilty pleasure that made me laugh more than I should.",
            "The dialogue was so bad that I laughed more than I should.",
            "Their attempt at being serious made me laugh more than I should."
        ]
        
        for expression in humor_expressions:
            idiom_data.append({
                'text': expression,
                'sentiment': 1,  # Positive
                'source': 'humor_expression',
                'idiom': 'laugh_more_than_should'
            })
        
        logger.info(f"Created {len(idiom_data)} idiom examples")
        return pd.DataFrame(idiom_data)
    
    except Exception as e:
        logger.error(f"Error creating idiom dataset: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['text', 'sentiment', 'source', 'idiom'])

def download_contrast_dataset():
    """
    Create a dataset focusing on contrast markers like "but", "however", "despite".
    Returns a pandas DataFrame with text and sentiment columns.
    """
    contrast_dir = os.path.join(os.getcwd(), 'datasets', 'contrast')
    os.makedirs(contrast_dir, exist_ok=True)
    
    contrast_data = []
    
    try:
        # Create synthetic examples with contrast markers
        logger.info("Creating contrast marker dataset...")
        
        # Templates for contrast examples
        # Format: (before_text, contrast_marker, after_text, final_sentiment)
        contrast_templates = [
            # Positive despite negative
            ("The movie had poor special effects", "but", "the story was compelling and emotional", 1),
            ("The restaurant was crowded", "but", "the food was absolutely delicious", 1),
            ("The service was slow", "but", "the quality was worth the wait", 1),
            ("The hotel room was small", "but", "the view was breathtaking", 1),
            ("The concert venue was uncomfortable", "but", "the performance was incredible", 1),
            
            # Negative despite positive
            ("The visuals were stunning", "but", "the plot made no sense at all", 0),
            ("The actors did their best", "but", "the script was terrible", 0),
            ("The location was beautiful", "but", "the experience was ruined by poor service", 0),
            ("The idea was innovative", "but", "the execution was disappointingly bad", 0),
            ("The ingredients were high quality", "but", "the dish was bland and overcooked", 0),
            
            # Despite + negative, positive
            ("Despite the rainy weather", "despite", "we had an amazing time at the park", 1),
            ("Despite the bad reviews", "despite", "I really enjoyed the play", 1),
            ("Despite the high price", "despite", "the product exceeded my expectations", 1),
            ("Despite the long line", "despite", "the attraction was worth the wait", 1),
            ("Despite the technical difficulties", "despite", "the presentation was informative", 1),
            
            # Despite + positive, negative
            ("Despite the beautiful setting", "despite", "the event was boring and poorly organized", 0),
            ("Despite the talented cast", "despite", "the movie was a huge disappointment", 0),
            ("Despite the excellent location", "despite", "the hotel was dirty and uncomfortable", 0),
            ("Despite the promising concept", "despite", "the execution was a complete failure", 0),
            ("Despite the friendly staff", "despite", "the service was unacceptably slow", 0),
            
            # However + contrast
            ("The graphics were impressive", "however", "the gameplay was repetitive and dull", 0),
            ("The first half was entertaining", "however", "it fell apart in the second half", 0),
            ("The meal started well", "however", "the main course was undercooked", 0),
            ("The characters were likeable", "however", "the plot had too many holes to enjoy", 0),
            ("The hotel looked great online", "however", "the reality was completely different", 0),
            
            # Nevertheless + contrast
            ("The journey was challenging", "nevertheless", "the destination was worth every hardship", 1),
            ("The exam was difficult", "nevertheless", "I managed to pass with a good grade", 1),
            ("The recipe was complicated", "nevertheless", "the results were delicious", 1),
            ("The training was exhausting", "nevertheless", "I feel much stronger now", 1),
            ("The project had many setbacks", "nevertheless", "we delivered it successfully on time", 1),
            
            # Yet + contrast
            ("The film received poor reviews", "yet", "I found it thoroughly entertaining", 1),
            ("It was a simple design", "yet", "incredibly effective for its purpose", 1),
            ("The solution seemed obvious", "yet", "no one had thought of it before", 1),
            ("They had limited resources", "yet", "created something remarkable", 1),
            ("The book was published decades ago", "yet", "the message is still relevant today", 1),
            
            # Although + contrast
            ("Although the course was challenging", "although", "I learned a tremendous amount", 1),
            ("Although the hike was strenuous", "although", "the view from the top was magnificent", 1),
            ("Although the medicine tasted terrible", "although", "it cured my illness quickly", 1),
            ("Although the museum was crowded", "although", "seeing the famous paintings was worth it", 1),
            ("Although the instructions were confusing", "although", "I managed to assemble it correctly", 1),
            
            # Even though + contrast
            ("Even though the critics loved it", "even though", "I found the movie pretentious and boring", 0),
            ("Even though it won awards", "even though", "the novel was tedious to read", 0),
            ("Even though it was expensive", "even though", "the quality was disappointingly low", 0),
            ("Even though the trailer looked exciting", "even though", "the actual film was dull", 0),
            ("Even though we had reservations", "even though", "we still had to wait an hour", 0)
        ]
        
        # Generate examples from templates
        for before, marker, after, sentiment in contrast_templates:
            # Add the basic example
            full_text = f"{before} {marker} {after}"
            contrast_data.append({
                'text': full_text,
                'sentiment': sentiment,
                'source': 'contrast_template',
                'contrast_marker': marker
            })
            
            # Add variations with different intensifiers
            intensifiers = ["", "very ", "extremely ", "somewhat ", "quite ", "incredibly ", "rather "]
            
            if sentiment == 1:  # Positive sentiment
                positive_adjectives = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "enjoyable", "delightful"]
                for intensifier in intensifiers:
                    for adj in positive_adjectives:
                        if np.random.random() < 0.3:  # Only generate some combinations
                            # Create variation where we strengthen the positive part
                            if "was" in after:
                                enhanced_after = after.replace("was", f"was {intensifier}{adj} and")
                                full_text = f"{before} {marker} {enhanced_after}"
                                contrast_data.append({
                                    'text': full_text,
                                    'sentiment': sentiment,
                                    'source': 'contrast_variation',
                                    'contrast_marker': marker
                                })
            
            elif sentiment == 0:  # Negative sentiment
                negative_adjectives = ["bad", "terrible", "awful", "horrible", "disappointing", "poor", "dreadful", "mediocre"]
                for intensifier in intensifiers:
                    for adj in negative_adjectives:
                        if np.random.random() < 0.3:  # Only generate some combinations
                            # Create variation where we strengthen the negative part
                            if "was" in after:
                                enhanced_after = after.replace("was", f"was {intensifier}{adj} and")
                                full_text = f"{before} {marker} {enhanced_after}"
                                contrast_data.append({
                                    'text': full_text,
                                    'sentiment': sentiment,
                                    'source': 'contrast_variation',
                                    'contrast_marker': marker
                                })
        
        # Add special examples for "The restaurant was crowded, but the food was absolutely delicious"
        restaurant_examples = [
            "The restaurant was crowded, but the food was absolutely delicious and worth the wait.",
            "Despite the noisy atmosphere, the restaurant served amazing dishes.",
            "The service was slow, however the flavors were incredible and memorable.",
            "The restaurant was small and cramped, yet the cuisine was exceptional.",
            "Although we had to wait for a table, the restaurant's signature dish was spectacular.",
            "The restaurant was busy and loud, but their chef's special was mouthwatering.",
            "The dining room was packed with people, but each dish was prepared to perfection.",
            "Getting a reservation was difficult, however their famous dessert alone justified the effort.",
            "The restaurant had a long line, but their award-winning menu lived up to the hype.",
            "The place was too crowded for comfort, nevertheless the food quality was outstanding."
        ]
        
        for example in restaurant_examples:
            contrast_data.append({
                'text': example,
                'sentiment': 1,  # Positive
                'source': 'restaurant_contrast',
                'contrast_marker': 'mixed'
            })
        
        logger.info(f"Created {len(contrast_data)} contrast examples")
        return pd.DataFrame(contrast_data)
    
    except Exception as e:
        logger.error(f"Error creating contrast dataset: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['text', 'sentiment', 'source', 'contrast_marker'])

def combine_datasets():
    """
    Combine all specialized datasets for training.
    """
    try:
        # Create a datasets directory
        dataset_dir = os.path.join(os.getcwd(), 'datasets')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Get all specialized datasets
        logger.info("Downloading and creating specialized datasets...")
        sarcasm_df = download_sarcasm_dataset()
        idiom_df = download_idiom_dataset()
        contrast_df = download_contrast_dataset()
        
        # Combine all datasets
        all_data = pd.concat([sarcasm_df, idiom_df, contrast_df], ignore_index=True)
        
        # Save the combined dataset
        combined_file = os.path.join(dataset_dir, 'augmented_sentiment_dataset.csv')
        all_data.to_csv(combined_file, index=False)
        
        logger.info(f"Combined dataset created with {len(all_data)} examples")
        logger.info(f"Saved to {combined_file}")
        
        return combined_file
    
    except Exception as e:
        logger.error(f"Error combining datasets: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    combine_datasets() 