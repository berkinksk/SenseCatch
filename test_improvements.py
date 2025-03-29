import sys
import logging
import traceback
import json
import time
from datetime import datetime
import os
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple

# Configure enhanced logging with colors and formatting for console output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'      # Reset color
    }
    
    def __init__(self, fmt=None, datefmt=None, style='%', use_color=True):
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color
    
    def format(self, record):
        log_message = super().format(record)
        if not self.use_color:
            return log_message
            
        if hasattr(record, 'highlight') and record.highlight:
            return f"\033[97m\033[1m{log_message}\033[0m"  # Bold white for highlights
        return f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}{log_message}{self.COLORS['RESET']}"

# Set up logging for both console and file
def setup_logging(log_level=logging.INFO, use_color=True):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter('%(levelname)s: %(message)s', use_color=use_color)
    console_handler.setFormatter(console_formatter)
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(f'logs/test_run_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add both handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger

# Set up the logger with default settings - will be reconfigured based on command args
logger = logging.getLogger(__name__)

try:
    from ensemble_model import SentimentEnsemble
except ImportError:
    logger.error("Could not import SentimentEnsemble. Make sure your environment is correctly set up.")
    sys.exit(1)

class DiagnosticEnsemble(SentimentEnsemble):
    """Extended SentimentEnsemble with diagnostic capabilities"""
    
    def __init__(self, *args, **kwargs):
        self.pipeline_trace = []
        self.decision_points = []
        self.state_snapshots = {}
        self.timing_data = {}
        super().__init__(*args, **kwargs)
    
    def _trace(self, stage, message, state=None, decision=None, timing=None):
        """Record a trace point in the pipeline"""
        trace_point = {
            'stage': stage,
            'message': message,
            'timestamp': time.time()
        }
        self.pipeline_trace.append(trace_point)
        
        # Optional detailed data
        if state:
            self.state_snapshots[stage] = state
        
        if decision:
            self.decision_points.append({
                'stage': stage,
                'decision': decision
            })
            
        if timing:
            self.timing_data[stage] = timing
    
    def reset_diagnostics(self):
        """Clear diagnostic data for a new test"""
        self.pipeline_trace = []
        self.decision_points = []
        self.state_snapshots = {}
        self.timing_data = {}
    
    # Override key methods to add tracing
    
    def clean_text(self, text):
        """Enhanced version with tracing"""
        start_time = time.time()
        
        # Record input state
        self._trace('clean_text:start', f"Input text: '{text}'", 
                   state={'original_text': text})
        
        # Call original method
        result, markers, info = super().clean_text(text)
        
        # Record result and timing
        duration = time.time() - start_time
        self._trace('clean_text:end', 
                   f"Output: '{result}', Special phrases: {info.get('special_phrase_detected', False)}", 
                   state={'cleaned_text': result, 'markers': markers, 'info': info},
                   timing=duration)
        
        return result, markers, info
    
    def handle_negations(self, text):
        """Enhanced version with tracing"""
        start_time = time.time()
        
        self._trace('handle_negations:start', f"Processing negations in: '{text}'")
        
        # Call original method
        result, markers, info = super().handle_negations(text)
        
        # Record result and decisions
        duration = time.time() - start_time
        if info.get('special_phrase_detected', False):
            decision = {
                'type': 'special_phrase',
                'phrase': info.get('detected_phrase'),
                'sentiment': info.get('forced_sentiment'),
                'confidence': info.get('forced_confidence')
            }
            self._trace('handle_negations:end', 
                       f"Detected special phrase: '{info.get('detected_phrase')}'", 
                       decision=decision,
                       timing=duration)
        else:
            self._trace('handle_negations:end', 
                       f"No special phrases detected.", 
                       timing=duration)
        
        return result, markers, info
    
    def process_contrast_markers(self, text):
        """Enhanced version with tracing"""
        start_time = time.time()
        
        self._trace('process_contrast_markers:start', f"Checking for contrast markers in: '{text}'")
        
        # Call original method
        result = super().process_contrast_markers(text)
        
        # Record result and decisions
        duration = time.time() - start_time
        if result.get("has_contrast", False):
            decision = {
                'type': 'contrast_marker',
                'marker': result.get("contrast_marker"),
                'before_weight': result.get("before_weight"),
                'after_weight': result.get("after_weight"),
                'forced_positive': result.get("has_forced_positive", False),
                'forced_negative': result.get("has_forced_negative", False)
            }
            self._trace('process_contrast_markers:end', 
                       f"Detected contrast marker: '{result.get('contrast_marker')}'", 
                       decision=decision,
                       timing=duration)
        else:
            self._trace('process_contrast_markers:end', 
                       "No contrast markers detected.", 
                       timing=duration)
        
        return result
    
    def predict(self, text, *args, **kwargs):
        """Enhanced prediction with full diagnostic tracing"""
        self.reset_diagnostics()
        start_time = time.time()
        
        self._trace('predict:start', f"Input text: '{text}'", 
                   state={'original_text': text, 'args': args, 'kwargs': kwargs})
        
        # Call original method
        result = super().predict(text, *args, **kwargs)
        
        # Record final result
        duration = time.time() - start_time
        self._trace('predict:end', 
                   f"Final prediction: {result['sentiment']} with {result['confidence']:.2f}% confidence", 
                   state={'result': result},
                   timing=duration)
        
        return result
    
    def get_diagnostic_summary(self):
        """Generate a readable summary of the diagnostic data"""
        total_duration = sum(self.timing_data.values())
        
        summary = {
            'pipeline_stages': len(self.pipeline_trace),
            'decision_points': len(self.decision_points),
            'total_duration_ms': total_duration * 1000,
            'stages_timing_ms': {k: v * 1000 for k, v in self.timing_data.items()},
            'key_decisions': self.decision_points
        }
        
        return summary

def test_sentiment_analysis(categories: Optional[List[str]] = None,
                           models: Optional[List[str]] = None,
                           enable_debug: bool = False,
                           compare_with: Optional[str] = None) -> Dict[str, Any]:
    """
    Test the sentiment analysis model on a set of challenging test cases.
    Enhanced with diagnostics and performance analysis.
    
    Args:
        categories: Optional list of test categories to run (e.g., ["negation", "sarcasm"])
        models: Optional list of models to test (default: all available models)
        enable_debug: Whether to enable detailed debugging output
        compare_with: Timestamp of previous results to compare with, or "latest"
        
    Returns:
        Dictionary with test results
    """
    try:
        # Initialize the model with diagnostics
        logger.info("Initializing diagnostic ensemble model...")
        model = DiagnosticEnsemble()
        logger.info("Model initialized. Starting tests...")
        
        # Create results directory if it doesn't exist
        if not os.path.exists('test_results'):
            os.makedirs('test_results')
        
        # Define test cases
        test_cases = [
            # Original test cases
            {
                "text": "This movie isn't bad at all, I actually enjoyed most of it.",
                "expected_sentiment": "Positive",
                "description": "Negated negative ('isn't bad')",
                "category": "negation"
            },
            {
                "text": "Despite the beautiful visuals, the plot was confusing and the acting was terrible.",
                "expected_sentiment": "Negative",
                "description": "Contrast marker with negative after",
                "category": "contrast"
            },
            {
                "text": "I don't recommend this product because it broke within a week of purchase.",
                "expected_sentiment": "Negative",
                "description": "Negated recommendation",
                "category": "negation"
            },
            {
                "text": "The restaurant was crowded, but the food was absolutely delicious and worth the wait.",
                "expected_sentiment": "Positive",
                "description": "Restaurant contrast case",
                "category": "restaurant,contrast"
            },
            {
                "text": "The performance wasn't great, nor was the dialogue particularly inspiring.",
                "expected_sentiment": "Negative",
                "description": "Double negation with 'nor'",
                "category": "negation"
            },
            
            # Tricky cases we improved
            {
                "text": "If you enjoy falling asleep during movies, this one's perfect for you!",
                "expected_sentiment": "Negative",
                "description": "Sarcasm - sleep pattern",
                "category": "sarcasm"
            },
            {
                "text": "The best part of this movie was when the credits rolled.",
                "expected_sentiment": "Negative",
                "description": "Sarcasm - credits rolled",
                "category": "sarcasm"
            },
            {
                "text": "I'd rather watch paint dry than sit through this again.",
                "expected_sentiment": "Negative",
                "description": "Negative comparison",
                "category": "sarcasm"
            },
            {
                "text": "I thought it would be trash, but it's a guilty pleasure I secretly enjoyed.",
                "expected_sentiment": "Positive",
                "description": "Contrast with positive after",
                "category": "contrast"
            },
            {
                "text": "Honestly, I laughed more than I should have. And that's a win in my book.",
                "expected_sentiment": "Positive",
                "description": "Humor appreciation",
                "category": "idiom"
            },
            
            # Additional edge cases
            {
                "text": "This is neither good nor bad, just average.",
                "expected_sentiment": "Neutral",
                "description": "True neutral sentiment",
                "category": "neutral"
            },
            {
                "text": "Not the best, not the worst.",
                "expected_sentiment": "Neutral",
                "description": "Double negation neutral",
                "category": "neutral,negation"
            },
            {
                "text": "This was awful! Just kidding, it was great.",
                "expected_sentiment": "Positive",
                "description": "Contradiction with positive at end",
                "category": "contradiction"
            },
            
            # NEWLY ADDED CHALLENGING TEST CASES
            
            # Complex negation patterns
            {
                "text": "The product isn't exactly what I wouldn't recommend, but it's not something I'd suggest either.",
                "expected_sentiment": "Negative",
                "description": "Triple negation complexity",
                "category": "negation"
            },
            {
                "text": "It's not that I didn't like it, I just didn't love it as much as I expected.",
                "expected_sentiment": "Neutral",
                "description": "Complex double negation with reservation",
                "category": "negation"
            },
            {
                "text": "Couldn't disagree more with the negative reviews, this product is excellent!",
                "expected_sentiment": "Positive",
                "description": "Negated disagreement (positive reinforcement)",
                "category": "negation"
            },
            
            # Subtle sarcasm cases
            {
                "text": "Sure, it's a masterpiece... if your standards are below ground level.",
                "expected_sentiment": "Negative",
                "description": "Subtle sarcasm with conditional",
                "category": "sarcasm"
            },
            {
                "text": "Wow, they really outdid themselves with how forgettable this was.",
                "expected_sentiment": "Negative",
                "description": "Sarcasm with positive-negative contrast",
                "category": "sarcasm"
            },
            {
                "text": "A cinematic achievement that will be studied for generations... as what not to do.",
                "expected_sentiment": "Negative",
                "description": "Delayed sarcasm reveal",
                "category": "sarcasm"
            },
            
            # Restaurant-specific cases (targeted weakness)
            {
                "text": "The ambiance was terrible but I've never had better pasta in my life.",
                "expected_sentiment": "Positive",
                "description": "Restaurant contrast with strong positive after",
                "category": "restaurant,contrast"
            },
            {
                "text": "The service was impeccable, however the food was completely tasteless.",
                "expected_sentiment": "Negative",
                "description": "Restaurant contrast with strong negative after",
                "category": "restaurant,contrast"
            },
            {
                "text": "While the prices were outrageous, the quality of the meal justified every penny.",
                "expected_sentiment": "Positive",
                "description": "Restaurant value proposition contrast",
                "category": "restaurant,contrast"
            },
            
            # Neutral cases (targeted weakness)
            {
                "text": "It has some good points and some bad points, so it evens out in the end.",
                "expected_sentiment": "Neutral",
                "description": "Explicit balanced opinion",
                "category": "neutral"
            },
            {
                "text": "I'm completely on the fence about this one. Can't decide if I like it or not.",
                "expected_sentiment": "Neutral",
                "description": "Explicit indecision",
                "category": "neutral"
            },
            {
                "text": "It's exactly what you'd expect, nothing more, nothing less.",
                "expected_sentiment": "Neutral",
                "description": "Met expectations exactly",
                "category": "neutral"
            },
            
            # Contextual sentiment
            {
                "text": "For a Monday, the service was surprisingly good.",
                "expected_sentiment": "Positive",
                "description": "Contextual positive with lowered expectations",
                "category": "contextual"
            },
            {
                "text": "Given the price, I expected much higher quality.",
                "expected_sentiment": "Negative",
                "description": "Contextual negative with unmet expectations",
                "category": "contextual"
            },
            
            # Idiomatic expressions
            {
                "text": "This film is a real diamond in the rough - don't miss it!",
                "expected_sentiment": "Positive",
                "description": "Positive idiom with recommendation",
                "category": "idiom"
            },
            {
                "text": "The app runs like a dream on my new phone.",
                "expected_sentiment": "Positive",
                "description": "Positive performance idiom",
                "category": "idiom"
            },
            {
                "text": "That movie was a complete train wreck from start to finish.",
                "expected_sentiment": "Negative",
                "description": "Negative disaster idiom",
                "category": "idiom"
            },
            
            # Complex mixed sentiment
            {
                "text": "While I loved the graphics, hated the story, and felt neutral about the acting, overall I enjoyed it.",
                "expected_sentiment": "Positive",
                "description": "Mixed sentiment with explicit conclusion",
                "category": "mixed,conclusion"
            },
            {
                "text": "Though some parts were incredible, others were horrible, but ultimately I was disappointed.",
                "expected_sentiment": "Negative",
                "description": "Mixed sentiment with explicit negative conclusion",
                "category": "mixed,conclusion"
            }
        ]
        
        # Filter test cases by category if specified
        if categories:
            filtered_test_cases = []
            for test_case in test_cases:
                test_categories = test_case.get("category", "").split(",")
                if any(cat.strip() in categories for cat in test_categories if cat.strip()):
                    filtered_test_cases.append(test_case)
            
            logger.info(f"Filtering tests by categories: {', '.join(categories)}")
            logger.info(f"Selected {len(filtered_test_cases)} of {len(test_cases)} test cases")
            test_cases = filtered_test_cases
        
        # Select which models to test
        available_models = ["naive_bayes", "logistic_regression"]
        models_to_test = models if models else available_models
        
        # Validate selected models
        for model_name in models_to_test:
            if model_name not in available_models:
                logger.warning(f"Unknown model: {model_name}. Will be skipped.")
        
        # Filter to only valid models
        models_to_test = [m for m in models_to_test if m in available_models]
        
        if not models_to_test:
            logger.error("No valid models selected for testing!")
            return {}
            
        logger.info(f"Testing with models: {', '.join(models_to_test)}")
        
        # Load previous results for comparison if requested
        previous_results = None
        if compare_with:
            previous_results = load_previous_results(compare_with)
            if previous_results:
                logger.info(f"Loaded previous results for comparison: {compare_with}")
            else:
                logger.warning(f"Could not load previous results for comparison: {compare_with}")
        
        results = {}
        diagnostic_data = {}
        
        # Record overall test run timing
        test_start_time = time.time()
        
        for model_name in models_to_test:
            logger.info(f"\nTesting with model: {model_name}")
            model_results = []
            model_diagnostic_data = []
            model_timing = []
            
            # Track confidence bands
            confidence_bands = {
                "50-60": {"count": 0, "correct": 0},
                "60-70": {"count": 0, "correct": 0},
                "70-80": {"count": 0, "correct": 0},
                "80-90": {"count": 0, "correct": 0},
                "90-100": {"count": 0, "correct": 0}
            }
            
            for i, test_case in enumerate(test_cases):
                text = test_case["text"]
                expected = test_case["expected_sentiment"]
                description = test_case["description"]
                categories = test_case.get("category", "").split(",")
                
                # Get prediction with timing
                start_time = time.time()
                
                # Add an extra log highlight for the current test
                test_logger = logging.LoggerAdapter(logger, {"highlight": True})
                test_logger.info(f"\n----- Test {i+1}: {description} -----")
                logger.info(f"Text: '{text}'")
                logger.info(f"Expected: {expected}")
                logger.info(f"Categories: {', '.join(categories)}")
                
                use_sarcasm = "sarcasm" in description.lower() or "sarcasm" in test_case.get("category", "").lower()
                use_idiom = "humor" in description.lower() or "idiom" in test_case.get("category", "").lower() or "guilty pleasure" in text.lower()
                use_contradiction = "contradiction" in test_case.get("category", "").lower()
                
                # Reset diagnostic data
                model.reset_diagnostics()
                
                result = model.predict(
                    text, 
                    modelname=model_name,
                    use_sarcasm_detection=use_sarcasm,
                    use_idiom_detection=use_idiom,
                    use_contradiction_detection=use_contradiction
                )
                
                # Calculate prediction time
                prediction_time = time.time() - start_time
                model_timing.append(prediction_time)
                
                predicted = result["sentiment"]
                confidence = result["confidence"]
                model_used = result["model_used"]
                
                # Get diagnostic data
                diagnostic_summary = model.get_diagnostic_summary()
                model_diagnostic_data.append({
                    "test_case": description,
                    "diagnostic_data": diagnostic_summary
                })
                
                # Update confidence bands statistics
                for band, data in confidence_bands.items():
                    low, high = map(int, band.split("-"))
                    if low <= confidence < high:
                        data["count"] += 1
                        if predicted == expected:
                            data["correct"] += 1
                
                # Check if prediction matches expectation
                matches = predicted == expected
                if matches:
                    logger.info(f"Predicted: {predicted} ({confidence:.2f}%) using {model_used} ✓")
                    logger.info(f"Duration: {prediction_time*1000:.2f}ms")
                else:
                    logger.error(f"Predicted: {predicted} ({confidence:.2f}%) using {model_used} ✗")
                    logger.error(f"Duration: {prediction_time*1000:.2f}ms")
                    
                    # For failures, log detailed diagnostic information
                    logger.error("DIAGNOSTIC INFORMATION:")
                    for trace in model.pipeline_trace:
                        logger.error(f"  {trace['stage']}: {trace['message']}")
                    
                    for decision in model.decision_points:
                        logger.error(f"  Decision ({decision['stage']}): {decision['decision']}")
                
                # Store result
                model_results.append({
                    "test_case": description,
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "model_used": model_used,
                    "passed": matches,
                    "categories": categories,
                    "processing_time_ms": prediction_time * 1000
                })
            
            # Calculate pass rate
            pass_count = sum(1 for r in model_results if r["passed"])
            pass_rate = (pass_count / len(model_results)) * 100
            logger.info(f"\n{model_name} pass rate: {pass_rate:.1f}% ({pass_count}/{len(model_results)})")
            
            # Calculate category-specific pass rates
            # Extract unique categories from all test cases
            all_categories = set()
            for tc in test_cases:
                for cat in tc.get("category", "").split(","):
                    if cat.strip():
                        all_categories.add(cat.strip())
            
            category_results = {}
            for category in all_categories:
                # Find test cases with this category
                category_indices = []
                for i, tc in enumerate(test_cases):
                    if category in tc.get("category", "").split(","):
                        category_indices.append(i)
                
                if category_indices:
                    category_pass_count = sum(1 for i in category_indices if model_results[i]["passed"])
                    category_pass_rate = (category_pass_count / len(category_indices)) * 100
                    category_results[category] = {
                        "pass_rate": category_pass_rate,
                        "count": f"{category_pass_count}/{len(category_indices)}"
                    }
                    logger.info(f"{category} pass rate: {category_pass_rate:.1f}% ({category_pass_count}/{len(category_indices)})")
            
            # Calculate confidence band accuracy
            logger.info("\nConfidence band accuracy:")
            for band, data in confidence_bands.items():
                if data["count"] > 0:
                    accuracy = (data["correct"] / data["count"]) * 100
                    logger.info(f"  {band}%: {accuracy:.1f}% ({data['correct']}/{data['count']})")
            
            # Calculate average processing time
            avg_processing_time = sum(model_timing) / len(model_timing) if model_timing else 0
            logger.info(f"\nAverage processing time: {avg_processing_time*1000:.2f}ms")
            
            results[model_name] = {
                "overall_pass_rate": pass_rate,
                "category_pass_rates": category_results,
                "confidence_bands": confidence_bands,
                "avg_processing_time_ms": avg_processing_time * 1000,
                "test_results": model_results
            }
            
            diagnostic_data[model_name] = model_diagnostic_data
        
        # Calculate total test run time
        total_test_time = time.time() - test_start_time
        logger.info(f"\nTotal test run time: {total_test_time:.2f} seconds")
        
        # Overall summary
        logger.info("\n----- TEST SUMMARY -----")
        for model_name, model_result in results.items():
            logger.info(f"{model_name}: {model_result['overall_pass_rate']:.1f}% pass rate")
            
            # Compare models on categories
            if model_name == models_to_test[-1]:  # When processing the last model
                logger.info("\n----- CATEGORY COMPARISON -----")
                for category in sorted(all_categories):
                    if all(category in results[m]["category_pass_rates"] for m in models_to_test):
                        rates = [f"{results[m]['category_pass_rates'][category]['pass_rate']:.1f}%" for m in models_to_test]
                        logger.info(f"{category}: {' vs '.join(rates)}")
        
        # Create timestamp for this test run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results to file
        results_file = f"test_results/sentiment_test_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nTest results saved to {results_file}")
        
        # Save diagnostic data to file
        diagnostic_file = f"test_results/diagnostic_data_{timestamp}.json"
        with open(diagnostic_file, "w") as f:
            json.dump(diagnostic_data, f, indent=2)
        logger.info(f"Diagnostic data saved to {diagnostic_file}")
        
        # Compare with previous results if requested
        if previous_results:
            compare_results(results, previous_results)
        
        return results
    
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_previous_results(timestamp_or_latest: str) -> Optional[Dict[str, Any]]:
    """Load previous test results for comparison"""
    try:
        if timestamp_or_latest.lower() == "latest":
            # Find the most recent results file (excluding the current run)
            results_files = sorted([f for f in os.listdir("test_results") 
                                  if f.startswith("sentiment_test_results_") and f.endswith(".json")],
                                  key=lambda x: x.split("_")[-1].split(".")[0],
                                  reverse=True)
            
            if not results_files:
                logger.warning("No previous results files found")
                return None
                
            filename = results_files[0]  # Most recent file
        else:
            # Use the specified timestamp
            filename = f"sentiment_test_results_{timestamp_or_latest}.json"
        
        filepath = os.path.join("test_results", filename)
        if not os.path.exists(filepath):
            logger.warning(f"Results file not found: {filepath}")
            return None
            
        with open(filepath, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Error loading previous results: {e}")
        return None

def compare_results(current_results: Dict[str, Any], previous_results: Dict[str, Any]) -> None:
    """Compare current test results with previous results and log the differences"""
    logger.info("\n===== COMPARISON WITH PREVIOUS RESULTS =====")
    
    # Compare overall accuracy
    for model_name in current_results:
        if model_name in previous_results:
            current_acc = current_results[model_name]["overall_pass_rate"]
            previous_acc = previous_results[model_name]["overall_pass_rate"]
            diff = current_acc - previous_acc
            
            if diff > 0:
                logger.info(f"{model_name}: {current_acc:.1f}% (+{diff:.1f}% improvement)")
            elif diff < 0:
                logger.warning(f"{model_name}: {current_acc:.1f}% ({diff:.1f}% regression)")
            else:
                logger.info(f"{model_name}: {current_acc:.1f}% (no change)")
    
    # Compare category-specific results
    for model_name in current_results:
        if model_name in previous_results:
            logger.info(f"\nCategory changes for {model_name}:")
            
            current_cats = current_results[model_name]["category_pass_rates"]
            previous_cats = previous_results[model_name]["category_pass_rates"]
            
            # Find all categories from both results
            all_categories = set(current_cats.keys()) | set(previous_cats.keys())
            
            for category in sorted(all_categories):
                if category in current_cats and category in previous_cats:
                    # Category exists in both results
                    current_rate = current_cats[category]["pass_rate"]
                    previous_rate = previous_cats[category]["pass_rate"]
                    diff = current_rate - previous_rate
                    
                    if diff > 0:
                        logger.info(f"  - {category}: {current_rate:.1f}% (+{diff:.1f}% improvement)")
                    elif diff < 0:
                        logger.warning(f"  - {category}: {current_rate:.1f}% ({diff:.1f}% regression)")
                elif category in current_cats:
                    # New category in current results
                    logger.info(f"  - {category}: {current_cats[category]['pass_rate']:.1f}% (new category)")
                else:
                    # Category only in previous results
                    logger.warning(f"  - {category}: category removed or not tested")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run sentiment analysis tests with enhanced diagnostics')
    
    # Test filtering options
    parser.add_argument('--categories', type=str, help='Comma-separated list of test categories to run')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to test')
    
    # Debug and output options
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    
    # Comparison options
    parser.add_argument('--compare', type=str, help='Compare with previous results (timestamp or "latest")')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging based on arguments
    log_level = logging.DEBUG if args.debug else logging.INFO
    use_color = not args.no_color
    setup_logging(log_level=log_level, use_color=use_color)
    
    # Parse categories and models if provided
    categories = args.categories.split(',') if args.categories else None
    models = args.models.split(',') if args.models else None
    
    # Run tests with specified options
    test_sentiment_analysis(
        categories=categories,
        models=models,
        enable_debug=args.debug,
        compare_with=args.compare
    ) 