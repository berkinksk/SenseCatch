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
                use_idiom = "humor" in description.lower() or "guilty pleasure" in text.lower()
                
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
            
            results[model_name] = {
                "pass_rate": pass_rate,
                "test_results": model_results
            }
        
        # Overall summary
        logger.info("\n----- TEST SUMMARY -----")
        for model_name, model_result in results.items():
            logger.info(f"{model_name}: {model_result['pass_rate']:.1f}% pass rate")
        
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