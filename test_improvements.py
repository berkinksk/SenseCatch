import sys
import logging
import traceback
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ensemble_model import SentimentEnsemble
except ImportError:
    logger.error("Could not import SentimentEnsemble. Make sure your environment is correctly set up.")
    sys.exit(1)

def test_sentiment_analysis():
    """
    Test the sentiment analysis model on a set of challenging test cases.
    """
    try:
        # Initialize the model
        logger.info("Initializing ensemble model...")
        model = SentimentEnsemble()
        logger.info("Model initialized. Starting tests...")
        
        # Define test cases
        test_cases = [
            # Original test cases
            {
                "text": "This movie isn't bad at all, I actually enjoyed most of it.",
                "expected_sentiment": "Positive",
                "description": "Negated negative ('isn't bad')"
            },
            {
                "text": "Despite the beautiful visuals, the plot was confusing and the acting was terrible.",
                "expected_sentiment": "Negative",
                "description": "Contrast marker with negative after"
            },
            {
                "text": "I don't recommend this product because it broke within a week of purchase.",
                "expected_sentiment": "Negative",
                "description": "Negated recommendation"
            },
            {
                "text": "The restaurant was crowded, but the food was absolutely delicious and worth the wait.",
                "expected_sentiment": "Positive",
                "description": "Restaurant contrast case"
            },
            {
                "text": "The performance wasn't great, nor was the dialogue particularly inspiring.",
                "expected_sentiment": "Negative",
                "description": "Double negation with 'nor'"
            },
            
            # Tricky cases we improved
            {
                "text": "If you enjoy falling asleep during movies, this one's perfect for you!",
                "expected_sentiment": "Negative",
                "description": "Sarcasm - sleep pattern"
            },
            {
                "text": "The best part of this movie was when the credits rolled.",
                "expected_sentiment": "Negative",
                "description": "Sarcasm - credits rolled"
            },
            {
                "text": "I'd rather watch paint dry than sit through this again.",
                "expected_sentiment": "Negative",
                "description": "Negative comparison"
            },
            {
                "text": "I thought it would be trash, but it's a guilty pleasure I secretly enjoyed.",
                "expected_sentiment": "Positive",
                "description": "Contrast with positive after"
            },
            {
                "text": "Honestly, I laughed more than I should have. And that's a win in my book.",
                "expected_sentiment": "Positive",
                "description": "Humor appreciation"
            },
            
            # Additional edge cases
            {
                "text": "This is neither good nor bad, just average.",
                "expected_sentiment": "Neutral",
                "description": "True neutral sentiment"
            },
            {
                "text": "Not the best, not the worst.",
                "expected_sentiment": "Neutral",
                "description": "Double negation neutral"
            },
            {
                "text": "This was awful! Just kidding, it was great.",
                "expected_sentiment": "Positive",
                "description": "Contradiction with positive at end"
            },
            
            # NEWLY ADDED CHALLENGING TEST CASES
            
            # Complex negation patterns
            {
                "text": "The product isn't exactly what I wouldn't recommend, but it's not something I'd suggest either.",
                "expected_sentiment": "Negative",
                "description": "Triple negation complexity"
            },
            {
                "text": "It's not that I didn't like it, I just didn't love it as much as I expected.",
                "expected_sentiment": "Neutral",
                "description": "Complex double negation with reservation"
            },
            {
                "text": "Couldn't disagree more with the negative reviews, this product is excellent!",
                "expected_sentiment": "Positive",
                "description": "Negated disagreement (positive reinforcement)"
            },
            
            # Subtle sarcasm cases
            {
                "text": "Sure, it's a masterpiece... if your standards are below ground level.",
                "expected_sentiment": "Negative",
                "description": "Subtle sarcasm with conditional"
            },
            {
                "text": "Wow, they really outdid themselves with how forgettable this was.",
                "expected_sentiment": "Negative",
                "description": "Sarcasm with positive-negative contrast"
            },
            {
                "text": "A cinematic achievement that will be studied for generations... as what not to do.",
                "expected_sentiment": "Negative",
                "description": "Delayed sarcasm reveal"
            },
            
            # Restaurant-specific cases (targeted weakness)
            {
                "text": "The ambiance was terrible but I've never had better pasta in my life.",
                "expected_sentiment": "Positive",
                "description": "Restaurant contrast with strong positive after"
            },
            {
                "text": "The service was impeccable, however the food was completely tasteless.",
                "expected_sentiment": "Negative",
                "description": "Restaurant contrast with strong negative after"
            },
            {
                "text": "While the prices were outrageous, the quality of the meal justified every penny.",
                "expected_sentiment": "Positive",
                "description": "Restaurant value proposition contrast"
            },
            
            # Neutral cases (targeted weakness)
            {
                "text": "It has some good points and some bad points, so it evens out in the end.",
                "expected_sentiment": "Neutral",
                "description": "Explicit balanced opinion"
            },
            {
                "text": "I'm completely on the fence about this one. Can't decide if I like it or not.",
                "expected_sentiment": "Neutral",
                "description": "Explicit indecision"
            },
            {
                "text": "It's exactly what you'd expect, nothing more, nothing less.",
                "expected_sentiment": "Neutral",
                "description": "Met expectations exactly"
            },
            
            # Contextual sentiment
            {
                "text": "For a Monday, the service was surprisingly good.",
                "expected_sentiment": "Positive",
                "description": "Contextual positive with lowered expectations"
            },
            {
                "text": "Given the price, I expected much higher quality.",
                "expected_sentiment": "Negative",
                "description": "Contextual negative with unmet expectations"
            },
            
            # Idiomatic expressions
            {
                "text": "This film is a real diamond in the rough - don't miss it!",
                "expected_sentiment": "Positive",
                "description": "Positive idiom with recommendation"
            },
            {
                "text": "The app runs like a dream on my new phone.",
                "expected_sentiment": "Positive",
                "description": "Positive performance idiom"
            },
            {
                "text": "That movie was a complete train wreck from start to finish.",
                "expected_sentiment": "Negative",
                "description": "Negative disaster idiom"
            },
            
            # Complex mixed sentiment
            {
                "text": "While I loved the graphics, hated the story, and felt neutral about the acting, overall I enjoyed it.",
                "expected_sentiment": "Positive",
                "description": "Mixed sentiment with explicit conclusion"
            },
            {
                "text": "Though some parts were incredible, others were horrible, but ultimately I was disappointed.",
                "expected_sentiment": "Negative",
                "description": "Mixed sentiment with explicit negative conclusion"
            }
        ]
        
        # Run tests with both models
        models_to_test = ["naive_bayes", "logistic_regression"]
        results = {}
        
        for model_name in models_to_test:
            logger.info(f"\nTesting with model: {model_name}")
            model_results = []
            
            for i, test_case in enumerate(test_cases):
                text = test_case["text"]
                expected = test_case["expected_sentiment"]
                description = test_case["description"]
                
                # Get prediction
                logger.info(f"\nTest {i+1}: {description}")
                logger.info(f"Text: '{text}'")
                logger.info(f"Expected: {expected}")
                
                use_sarcasm = "sarcasm" in description.lower()
                use_idiom = "humor" in description.lower() or "idiom" in description.lower() or "guilty pleasure" in text.lower()
                
                result = model.predict(
                    text, 
                    modelname=model_name,
                    use_sarcasm_detection=use_sarcasm,
                    use_idiom_detection=use_idiom
                )
                predicted = result["sentiment"]
                confidence = result["confidence"]
                model_used = result["model_used"]
                
                # Check if prediction matches expectation
                matches = predicted == expected
                logger.info(f"Predicted: {predicted} ({confidence:.2f}%) using {model_used}")
                logger.info(f"Result: {'✓ PASS' if matches else '✗ FAIL'}")
                
                # Store result
                model_results.append({
                    "test_case": description,
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "model_used": model_used,
                    "passed": matches
                })
            
            # Calculate pass rate
            pass_count = sum(1 for r in model_results if r["passed"])
            pass_rate = (pass_count / len(model_results)) * 100
            logger.info(f"\n{model_name} pass rate: {pass_rate:.1f}% ({pass_count}/{len(model_results)})")
            
            # Calculate category-specific pass rates
            categories = {
                "negation": [i for i, tc in enumerate(test_cases) if "negat" in tc["description"].lower()],
                "contrast": [i for i, tc in enumerate(test_cases) if "contrast" in tc["description"].lower()],
                "sarcasm": [i for i, tc in enumerate(test_cases) if "sarcasm" in tc["description"].lower()],
                "neutral": [i for i, tc in enumerate(test_cases) if "neutral" in tc["description"].lower()],
                "idiom": [i for i, tc in enumerate(test_cases) if "idiom" in tc["description"].lower()],
                "restaurant": [i for i, tc in enumerate(test_cases) if "restaurant" in tc["description"].lower()]
            }
            
            category_results = {}
            for category, indices in categories.items():
                if indices:
                    category_pass_count = sum(1 for i in indices if model_results[i]["passed"])
                    category_pass_rate = (category_pass_count / len(indices)) * 100
                    category_results[category] = {
                        "pass_rate": category_pass_rate,
                        "count": f"{category_pass_count}/{len(indices)}"
                    }
                    logger.info(f"{category} pass rate: {category_pass_rate:.1f}% ({category_pass_count}/{len(indices)})")
            
            results[model_name] = {
                "overall_pass_rate": pass_rate,
                "category_pass_rates": category_results,
                "test_results": model_results
            }
        
        # Overall summary
        logger.info("\n----- TEST SUMMARY -----")
        for model_name, model_result in results.items():
            logger.info(f"{model_name}: {model_result['overall_pass_rate']:.1f}% pass rate")
            
            # Compare models on categories
            if model_name == models_to_test[-1]:  # When processing the last model
                logger.info("\n----- CATEGORY COMPARISON -----")
                for category in categories.keys():
                    if all(category in results[m]["category_pass_rates"] for m in models_to_test):
                        rates = [f"{results[m]['category_pass_rates'][category]['pass_rate']:.1f}%" for m in models_to_test]
                        logger.info(f"{category}: {' vs '.join(rates)}")
        
        # Save results to file
        with open("sentiment_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("\nTest results saved to sentiment_test_results.json")
        
        return results
    
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    test_sentiment_analysis() 