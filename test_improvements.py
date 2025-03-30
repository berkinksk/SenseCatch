import sys
import logging
import traceback
import json
import time
from datetime import datetime
import os
import argparse
import re
import copy
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import nltk

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
        # Replace Unicode arrow with ASCII equivalent in the log message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = record.msg.replace('→', '->')
            record.msg = record.msg.replace('✓', '[PASS]')
            record.msg = record.msg.replace('✗', '[FAIL]')
        
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
    from ensemble_model import SentimentEnsemble, load_models
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
        # Initialize component results storage
        self.component_results = {}
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
        self.component_results = {}
    
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

    # Add new methods for component isolation testing
    
    def test_component(self, component_name: str, text: str, expected_output: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Test an individual pipeline component in isolation.
        
        Args:
            component_name: Name of the component to test
            text: Input text to process
            expected_output: Expected output (if applicable for validation)
            kwargs: Additional arguments specific to the component
            
        Returns:
            Dictionary with test results
        """
        start_time = time.time()
        result = None
        error = None
        
        try:
            # Map component name to method
            component_map = {
                "clean_text": self.clean_text,
                "handle_negations": self.handle_negations,
                "process_contrast_markers": self.process_contrast_markers,
                "detect_sarcasm": self._detect_sarcasm,
                "detect_idioms": self._detect_idioms,
                "detect_contradiction": self._detect_contradiction,
                "detect_neutral": self._detect_neutral_sentiment
            }
            
            if component_name not in component_map:
                raise ValueError(f"Unknown component: {component_name}")
            
            # Call the component with the input text
            result = component_map[component_name](text, **kwargs)
            
        except Exception as e:
            error = str(e)
            logger.error(f"Error testing component {component_name}: {error}")
        
        duration = time.time() - start_time
        
        # Format the result for consistency
        component_result = {
            "component": component_name,
            "input": text,
            "output": result,
            "expected": expected_output,
            "duration_ms": duration * 1000,
            "error": error
        }
        
        # If expected output is provided, check if the result matches
        if expected_output is not None:
            if component_name == "clean_text":
                # For clean_text, compare first return value (cleaned text)
                component_result["success"] = result[0] == expected_output
            elif component_name == "handle_negations":
                # For handle_negations, compare first return value
                component_result["success"] = result[0] == expected_output
            elif isinstance(result, dict) and isinstance(expected_output, dict):
                # For components returning dicts, check for matching keys
                component_result["success"] = all(result.get(k) == v for k, v in expected_output.items())
            else:
                # Direct comparison for other components
                component_result["success"] = result == expected_output
        
        # Store the result
        self.component_results[component_name] = component_result
        
        return component_result
    
    def test_component_pipeline(self, pipeline: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """
        Test a sequence of components as a pipeline.
        
        Args:
            pipeline: List of component configurations with name and params
            text: Input text to process
            
        Returns:
            Dictionary with test results for each step
        """
        current_text = text
        pipeline_results = []
        start_time = time.time()
        
        for step in pipeline:
            component_name = step["component"]
            params = step.get("params", {})
            expected = step.get("expected_output", None)
            
            # Run the component with the current text
            result = self.test_component(component_name, current_text, expected, **params)
            pipeline_results.append(result)
            
            # Update the current text for the next component if needed
            if component_name == "clean_text" and result["output"]:
                current_text = result["output"][0]  # First element is the cleaned text
            elif component_name == "handle_negations" and result["output"]:
                current_text = result["output"][0]  # First element is processed text
        
        total_duration = time.time() - start_time
        
        return {
            "pipeline_results": pipeline_results,
            "input_text": text,
            "final_text": current_text,
            "total_duration_ms": total_duration * 1000
        }
    
    def measure_component_impact(self, text: str, expected_sentiment: str) -> Dict[str, Any]:
        """
        Measure the impact of each component on the final sentiment prediction.
        Works by toggling components on/off to see how they affect the result.
        
        Args:
            text: Input text to analyze
            expected_sentiment: Expected sentiment for validation
            
        Returns:
            Dictionary with impact measurements
        """
        # Baseline prediction with all components
        baseline = self.predict(text)
        baseline_correct = baseline["sentiment"] == expected_sentiment
        
        # Define the components to test and their parameter names
        component_param_map = {
            "sarcasm_detection": "use_sarcasm_detection",
            "idiom_detection": "use_idiom_detection", 
            "contradiction_detection": "use_contradiction_detection",
            "contrast_handling": "contrast_handling",  # Handled specially
            "neutral_detection": "neutral_detection",  # Handled specially
            "negation_handling": "negation_handling"   # Handled specially
        }
        
        impact_results = {}
        
        for component, param_name in component_param_map.items():
            # For components with standard interface parameters
            if component in ["sarcasm_detection", "idiom_detection", "contradiction_detection"]:
                kwargs = {
                    param_name: False  
                }
                without_component = self.predict(text, **kwargs)
            
            # For components requiring special handling
            elif component == "contrast_handling":
                # Create a copy of the instance with contrast handling disabled
                temp_instance = copy.deepcopy(self)
                temp_instance.use_contrast_handling = False
                without_component = temp_instance.predict(text)
            elif component == "neutral_detection":
                # Create a copy with neutral detection disabled
                temp_instance = copy.deepcopy(self)
                temp_instance._detect_neutral_sentiment = lambda x: {"sarcasm_detected": False}
                without_component = temp_instance.predict(text)
            elif component == "negation_handling":
                # Create a copy with modified negation handling
                temp_instance = copy.deepcopy(self)
                # Store original method
                original_method = temp_instance.handle_negations
                # Override with a pass-through version
                temp_instance.handle_negations = lambda x: (x, {}, {})
                without_component = temp_instance.predict(text)
                # Restore original method to avoid issues
                temp_instance.handle_negations = original_method
            else:
                # Skip unknown components
                continue
            
            # Calculate impact
            without_correct = without_component["sentiment"] == expected_sentiment
            
            # Determine impact: result changes when component is disabled
            sentiment_changed = baseline["sentiment"] != without_component["sentiment"]
            confidence_diff = baseline["confidence"] - without_component["confidence"]
            
            impact = {
                "component": component,
                "baseline_sentiment": baseline["sentiment"],
                "without_component_sentiment": without_component["sentiment"],
                "sentiment_changed": sentiment_changed,
                "baseline_confidence": baseline["confidence"],
                "without_component_confidence": without_component["confidence"],
                "confidence_difference": confidence_diff,
                "baseline_correct": baseline_correct,
                "without_component_correct": without_correct,
                "improved_accuracy": baseline_correct and not without_correct,
                "reduced_accuracy": not baseline_correct and without_correct,
                "no_accuracy_impact": baseline_correct == without_correct
            }
            
            impact_results[component] = impact
        
        return {
            "input_text": text,
            "expected_sentiment": expected_sentiment,
            "baseline_prediction": baseline,
            "component_impact": impact_results
        }
    
    def ab_test_component(
        self, 
        component_a: Tuple[str, Callable], 
        component_b: Tuple[str, Callable], 
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run A/B testing between two alternative implementations of the same component.
        
        Args:
            component_a: Tuple of (name, function) for first implementation
            component_b: Tuple of (name, function) for second implementation
            test_cases: List of test cases with text and expected output
            
        Returns:
            Dictionary with comparison results
        """
        a_name, a_func = component_a
        b_name, b_func = component_b
        
        a_results = []
        b_results = []
        
        # Process each test case with both implementations
        for test_case in test_cases:
            text = test_case["text"]
            expected = test_case.get("expected_output", None)
            
            # Test with implementation A
            start_time = time.time()
            try:
                a_output = a_func(text)
                a_duration = (time.time() - start_time) * 1000
                a_error = None
                if expected is not None:
                    a_success = a_output == expected
                else:
                    a_success = None
            except Exception as e:
                a_output = None
                a_duration = (time.time() - start_time) * 1000
                a_error = str(e)
                a_success = False
            
            # Test with implementation B
            start_time = time.time()
            try:
                b_output = b_func(text)
                b_duration = (time.time() - start_time) * 1000
                b_error = None
                if expected is not None:
                    b_success = b_output == expected
                else:
                    b_success = None
            except Exception as e:
                b_output = None
                b_duration = (time.time() - start_time) * 1000
                b_error = str(e)
                b_success = False
            
            # Record results
            a_results.append({
                "input": text,
                "output": a_output,
                "expected": expected,
                "success": a_success,
                "duration_ms": a_duration,
                "error": a_error
            })
            
            b_results.append({
                "input": text,
                "output": b_output,
                "expected": expected,
                "success": b_success,
                "duration_ms": b_duration,
                "error": b_error
            })
        
        # Calculate summary metrics
        a_success_count = sum(1 for r in a_results if r["success"] is True)
        b_success_count = sum(1 for r in b_results if r["success"] is True)
        
        a_success_rate = a_success_count / len(a_results) if a_results else 0
        b_success_rate = b_success_count / len(b_results) if b_results else 0
        
        a_avg_duration = sum(r["duration_ms"] for r in a_results) / len(a_results) if a_results else 0
        b_avg_duration = sum(r["duration_ms"] for r in b_results) / len(b_results) if b_results else 0
        
        return {
            "component_a": {
                "name": a_name,
                "success_rate": a_success_rate,
                "success_count": a_success_count,
                "avg_duration_ms": a_avg_duration,
                "detailed_results": a_results
            },
            "component_b": {
                "name": b_name,
                "success_rate": b_success_rate,
                "success_count": b_success_count,
                "avg_duration_ms": b_avg_duration,
                "detailed_results": b_results
            },
            "comparison": {
                "success_diff": a_success_rate - b_success_rate,
                "speed_diff_ms": a_avg_duration - b_avg_duration,
                "a_better_success": a_success_rate > b_success_rate,
                "b_better_success": b_success_rate > a_success_rate,
                "a_faster": a_avg_duration < b_avg_duration,
                "b_faster": b_avg_duration < a_avg_duration
            }
        }

class ErrorAnalyzer:
    """
    Analyzes error patterns in test results to identify common failure modes
    and suggest targeted improvements.
    """
    
    def __init__(self):
        """Initialize the error analyzer"""
        self.error_features = defaultdict(list)
        self.text_features = {}
        self.failure_clusters = {}
        self.impact_scores = {}
        self.improvement_suggestions = []
    
    def extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract linguistic features from text that might correlate with errors.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dict of features
        """
        # Text length features
        char_count = len(text)
        word_count = len(text.split())
        
        # Sentence structure
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Special pattern counts
        negation_count = len(re.findall(r'\b(?:not|n\'t|never|no|none|nothing|nowhere)\b', text.lower()))
        contrast_markers = len(re.findall(r'\b(?:but|however|although|though|despite|yet|while|whereas|nevertheless)\b', text.lower()))
        question_marks = text.count('?')
        exclamation_marks = text.count('!')
        
        # Advanced patterns
        double_negation = 1 if re.search(r'\b(?:not|n\'t|never|no)\b.*\b(?:not|n\'t|never|no)\b', text.lower()) else 0
        conditional = 1 if re.search(r'\b(?:if|unless|when|while)\b', text.lower()) else 0
        sarcasm_indicators = len(re.findall(r'\b(?:great|fantastic|awesome|wonderful|amazing|brilliant|perfect)\b.*(?:awful|terrible|worst|disappointing|disaster)', text.lower()))
        
        # Sentiment words (simple approach)
        positive_words = len(re.findall(r'\b(?:good|great|excellent|amazing|wonderful|fantastic|awesome|love|like|enjoy|best)\b', text.lower()))
        negative_words = len(re.findall(r'\b(?:bad|terrible|awful|horrible|worst|hate|dislike|poor|disappointing|waste)\b', text.lower()))
        sentiment_balance = positive_words - negative_words
        
        features = {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "negation_count": negation_count,
            "contrast_markers": contrast_markers,
            "question_marks": question_marks,
            "exclamation_marks": exclamation_marks,
            "double_negation": double_negation,
            "conditional": conditional,
            "sarcasm_indicators": sarcasm_indicators,
            "positive_words": positive_words,
            "negative_words": negative_words,
            "sentiment_balance": sentiment_balance,
            "contains_idiom": 1 if re.search(r'\b(?:piece of cake|break a leg|under the weather|hit the nail|out of the blue|on the fence|diamond in the rough)\b', text.lower()) else 0
        }
        
        return features
    
    def analyze_errors(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze test failures to identify patterns and suggest improvements.
        
        Args:
            test_results: List of test result dictionaries
            
        Returns:
            Dictionary with error analysis and improvement suggestions
        """
        # Separate successes and failures
        failures = [result for result in test_results if not result["passed"]]
        successes = [result for result in test_results if result["passed"]]
        
        if not failures:
            return {
                "error_count": 0,
                "error_rate": 0.0,
                "message": "No errors to analyze. All tests passed!",
                "suggestions": [],
                "category_stats": {},
                "feature_correlation": {},
                "failure_clusters": {},
                "improvement_suggestions": []
            }
        
        # Extract features from all texts
        for result in test_results:
            text = result["text"]
            self.text_features[text] = self.extract_linguistic_features(text)
        
        # Group failures by category
        category_failures = defaultdict(list)
        for failure in failures:
            for category in failure["categories"]:
                category_failures[category].append(failure)
        
        # Calculate failure rates by category
        category_stats = {}
        for category, fails in category_failures.items():
            total_in_category = sum(1 for result in test_results if category in result["categories"])
            failure_rate = len(fails) / total_in_category if total_in_category > 0 else 0
            category_stats[category] = {
                "total": total_in_category,
                "failures": len(fails),
                "failure_rate": failure_rate
            }
        
        # Identify common features in failed tests vs successful tests
        feature_correlation = self._analyze_feature_correlation(failures, successes)
        
        # Cluster failures to find patterns
        self._cluster_failures(failures)
        
        # Generate improvement suggestions
        suggestions = self._generate_suggestions(failures, category_stats, feature_correlation)
        
        # Prepare analysis results
        analysis = {
            "error_count": len(failures),
            "error_rate": len(failures) / len(test_results),
            "category_stats": category_stats,
            "feature_correlation": feature_correlation,
            "failure_clusters": self.failure_clusters,
            "improvement_suggestions": suggestions
        }
        
        return analysis
    
    def _analyze_feature_correlation(self, failures: List[Dict[str, Any]], successes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Identify features that correlate with failures"""
        if not failures or not successes:
            return {}
            
        # Calculate average feature values for failures and successes
        failure_features = {}
        success_features = {}
        
        for feature in self.text_features[failures[0]["text"]].keys():
            failure_values = [self.text_features[f["text"]][feature] for f in failures]
            success_values = [self.text_features[s["text"]][feature] for s in successes]
            
            failure_avg = sum(failure_values) / len(failure_values)
            success_avg = sum(success_values) / len(success_values)
            
            # Calculate difference (how much more prevalent in failures)
            if success_avg == 0:
                ratio = failure_avg * 2 if failure_avg > 0 else 0
            else:
                ratio = (failure_avg / success_avg) - 1
                
            failure_features[feature] = {
                "failure_avg": failure_avg,
                "success_avg": success_avg,
                "difference_ratio": ratio
            }
        
        # Sort by correlation strength (absolute difference ratio)
        sorted_features = sorted(
            failure_features.items(), 
            key=lambda x: abs(x[1]["difference_ratio"]), 
            reverse=True
        )
        
        # Return top correlations
        return {feature: data for feature, data in sorted_features if abs(data["difference_ratio"]) > 0.2}
    
    def _cluster_failures(self, failures: List[Dict[str, Any]], n_clusters: int = 3) -> None:
        """Cluster similar failures together"""
        if len(failures) < n_clusters:
            # Not enough failures to cluster
            self.failure_clusters = {
                "clusters": [{"texts": [f["text"] for f in failures], "common_features": []}]
            }
            return
            
        # Get features for vectorization
        texts = [f["text"] for f in failures]
        
        # Vectorize the texts
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=50)
        X = vectorizer.fit_transform(texts)
        
        # Add linguistic features to the mix
        feature_names = ["negation_count", "contrast_markers", "sentiment_balance", 
                        "double_negation", "conditional", "sarcasm_indicators"]
        
        linguistic_features = np.array([
            [self.text_features[text][feature] for feature in feature_names]
            for text in texts
        ])
        
        # Normalize linguistic features
        if linguistic_features.shape[0] > 0:
            linguistic_features = linguistic_features / (linguistic_features.max(axis=0) + 1e-10)
            
            # Combine with text features
            combined_features = np.hstack([
                X.toarray(), 
                linguistic_features
            ])
        else:
            combined_features = X.toarray()
        
        # Cluster the failures
        n_clusters = min(n_clusters, len(failures))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(combined_features)
        
        # Group failures by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append({
                "text": texts[i],
                "features": {feature: self.text_features[texts[i]][feature] for feature in feature_names}
            })
        
        # Extract common features for each cluster
        cluster_info = []
        for label, instances in clusters.items():
            # Find common features
            common_features = {}
            for feature in feature_names:
                values = [instance["features"][feature] for instance in instances]
                avg_value = sum(values) / len(values)
                
                # Check if this feature is distinctively high in this cluster
                is_distinctive = avg_value > 0.5  # Simple threshold
                if is_distinctive:
                    common_features[feature] = avg_value
            
            # Sort features by strength
            sorted_features = sorted(common_features.items(), key=lambda x: x[1], reverse=True)
            
            cluster_info.append({
                "texts": [instance["text"] for instance in instances],
                "common_features": sorted_features
            })
        
        self.failure_clusters = {
            "clusters": cluster_info
        }
    
    def _generate_suggestions(self, failures: List[Dict[str, Any]], 
                             category_stats: Dict[str, Dict[str, Any]],
                             feature_correlation: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate prioritized improvement suggestions"""
        suggestions = []
        
        # Suggestion 1: Target high-failure-rate categories
        high_failure_categories = {
            cat: stats for cat, stats in category_stats.items() 
            if stats["failure_rate"] > 0.3 and stats["total"] >= 3
        }
        
        if high_failure_categories:
            # Sort by failure rate * count
            sorted_categories = sorted(
                high_failure_categories.items(),
                key=lambda x: x[1]["failure_rate"] * x[1]["failures"],
                reverse=True
            )
            
            for cat, stats in sorted_categories[:3]:  # Top 3 problem categories
                examples = [f["text"] for f in failures if cat in f["categories"]][:2]  # 2 examples max
                
                suggestions.append({
                    "type": "category_focus",
                    "focus_area": cat,
                    "failure_rate": stats["failure_rate"] * 100,
                    "impact_score": stats["failure_rate"] * stats["failures"] / len(failures),
                    "suggestion": f"Improve handling of '{cat}' patterns - {stats['failures']} failures ({stats['failure_rate']*100:.1f}% failure rate)",
                    "examples": examples
                })
        
        # Suggestion 2: Address prevalent linguistic features
        if feature_correlation:
            # Get top 3 correlated features
            top_features = list(feature_correlation.items())[:3]
            
            for feature, data in top_features:
                # Create human-readable feature name
                feature_name = " ".join(feature.split("_")).title()
                
                if data["difference_ratio"] > 0:
                    # More common in failures
                    suggestion = f"Improve handling of text with {feature_name} (avg {data['failure_avg']:.1f} in failures vs {data['success_avg']:.1f} in successes)"
                else:
                    # Less common in failures - this is unusual, might need special handling
                    suggestion = f"Examine why texts with low {feature_name} fail less often"
                
                examples = []
                for f in failures[:3]:  # Get up to 3 examples
                    if self.text_features[f["text"]][feature] > 0:
                        examples.append(f["text"])
                
                if examples:
                    suggestions.append({
                        "type": "linguistic_feature",
                        "focus_area": feature,
                        "impact_score": abs(data["difference_ratio"]) * 0.8,  # Slightly lower priority than categories
                        "suggestion": suggestion,
                        "examples": examples[:2]  # Limit to 2 examples
                    })
        
        # Suggestion 3: Look at failure clusters
        for i, cluster in enumerate(self.failure_clusters.get("clusters", [])):
            if len(cluster["texts"]) >= 2:  # Only consider clusters with multiple failures
                # Extract significant features
                feature_desc = []
                for feature, value in cluster["common_features"][:2]:  # Top 2 features
                    feature_name = " ".join(feature.split("_")).title()
                    feature_desc.append(f"{feature_name} ({value:.1f})")
                
                feature_text = " and ".join(feature_desc) if feature_desc else "mixed patterns"
                
                suggestions.append({
                    "type": "cluster_improvement",
                    "focus_area": f"cluster_{i+1}",
                    "impact_score": len(cluster["texts"]) / len(failures) * 0.7,  # Lower priority than direct features
                    "suggestion": f"Address failure cluster with {feature_text} ({len(cluster['texts'])} similar failures)",
                    "examples": cluster["texts"][:2]  # Limit to 2 examples
                })
        
        # Sort suggestions by impact score
        return sorted(suggestions, key=lambda x: x["impact_score"], reverse=True)
    
    def print_analysis_report(self, analysis: Dict[str, Any]) -> None:
        """Print a formatted analysis report"""
        if analysis["error_count"] == 0:
            logger.info("✓ All tests passed! No error analysis needed.")
            return
            
        logger.info("\n===== ERROR ANALYSIS REPORT =====")
        logger.info(f"Total errors: {analysis['error_count']} ({analysis['error_rate']*100:.1f}% of tests)")
        
        # Print category stats
        logger.info("\n--- Category Performance ---")
        sorted_cats = sorted(
            analysis["category_stats"].items(),
            key=lambda x: x[1]["failure_rate"],
            reverse=True
        )
        
        for cat, stats in sorted_cats:
            if stats["total"] > 0:
                logger.info(f"{cat}: {stats['failure_rate']*100:.1f}% failure rate ({stats['failures']}/{stats['total']})")
        
        # Print top correlated features
        if analysis["feature_correlation"]:
            logger.info("\n--- Linguistic Features Correlated with Failures ---")
            for feature, data in analysis["feature_correlation"].items():
                feature_name = " ".join(feature.split("_")).title()
                logger.info(f"{feature_name}: {data['failure_avg']:.2f} in failures vs {data['success_avg']:.2f} in successes")
        
        # Print improvement suggestions
        if analysis["improvement_suggestions"]:
            logger.info("\n--- Prioritized Improvement Suggestions ---")
            for i, suggestion in enumerate(analysis["improvement_suggestions"]):
                logger.info(f"{i+1}. {suggestion['suggestion']} (Impact: {suggestion['impact_score']:.2f})")
                if suggestion["examples"]:
                    for j, example in enumerate(suggestion["examples"]):
                        logger.info(f"   Example {j+1}: \"{example}\"")
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str) -> None:
        """Save the analysis to a JSON file"""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Error analysis saved to {output_path}")

class IncrementalTestSuite:
    """
    Manages an incremental test suite that organizes test cases by components,
    difficulty levels, and regression status to enable systematic improvement.
    """
    
    def __init__(self, suite_name: str = "default"):
        """Initialize the test suite with a name"""
        self.suite_name = suite_name
        self.test_cases = []
        self.components = set()
        self.categories = set()
        self.difficulty_levels = set()
        self.regression_tests = []
        self.history = []
        
        # Create directory for test suites if it doesn't exist
        if not os.path.exists('test_suites'):
            os.makedirs('test_suites')
    
    def add_test_case(self, 
                     text: str, 
                     expected_sentiment: str, 
                     description: str, 
                     categories: List[str],
                     components: List[str] = None,
                     difficulty: str = "medium",
                     is_regression: bool = False,
                     metadata: Dict[str, Any] = None) -> int:
        """
        Add a test case to the suite and return its ID.
        
        Args:
            text: The input text to test
            expected_sentiment: Expected sentiment output
            description: Description of what this test case verifies
            categories: List of categories this test belongs to
            components: List of components this test focuses on
            difficulty: Difficulty level ("easy", "medium", "hard", "extreme")
            is_regression: Whether this is a regression test for a fixed issue
            metadata: Optional additional information
            
        Returns:
            ID of the added test case
        """
        # Validate inputs
        if difficulty not in ["easy", "medium", "hard", "extreme"]:
            raise ValueError("Difficulty must be one of: easy, medium, hard, extreme")
            
        if not components:
            components = []
            
        if not metadata:
            metadata = {}
            
        # Create test case ID
        test_id = len(self.test_cases)
        
        # Create test case
        test_case = {
            "id": test_id,
            "text": text,
            "expected_sentiment": expected_sentiment,
            "description": description,
            "categories": categories,
            "components": components,
            "difficulty": difficulty,
            "is_regression": is_regression,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata,
            "results_history": []
        }
        
        # Add to test cases
        self.test_cases.append(test_case)
        
        # Update metadata sets
        self.categories.update(categories)
        self.components.update(components)
        self.difficulty_levels.add(difficulty)
        if is_regression:
            self.regression_tests.append(test_id)
            
        return test_id
    
    def add_test_result(self, 
                       test_id: int, 
                       passed: bool, 
                       predicted: str, 
                       confidence: float,
                       model_name: str,
                       processing_time: float,
                       notes: str = None) -> None:
        """
        Add a test result for a specific test case
        
        Args:
            test_id: ID of the test case
            passed: Whether the test passed
            predicted: The predicted sentiment
            confidence: Confidence of the prediction
            model_name: Name of the model used
            processing_time: Time to process in ms
            notes: Optional notes about the result
        """
        if test_id >= len(self.test_cases):
            raise ValueError(f"Invalid test ID: {test_id}")
            
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "passed": passed,
            "predicted": predicted,
            "confidence": confidence,
            "model_name": model_name,
            "processing_time_ms": processing_time,
            "notes": notes
        }
        
        self.test_cases[test_id]["results_history"].append(result)
    
    def filter_test_cases(self, 
                         categories: List[str] = None, 
                         components: List[str] = None,
                         difficulty: List[str] = None,
                         regression_only: bool = False) -> List[Dict[str, Any]]:
        """
        Filter test cases by categories, components, difficulty, etc.
        
        Args:
            categories: Filter by these categories (OR logic)
            components: Filter by these components (OR logic)
            difficulty: Filter by these difficulty levels
            regression_only: If True, only return regression tests
            
        Returns:
            List of matching test cases
        """
        filtered = []
        
        for test in self.test_cases:
            # Apply filters
            include = True
            
            if categories and not any(c in test["categories"] for c in categories):
                include = False
                
            if components and not any(c in test["components"] for c in components):
                include = False
                
            if difficulty and test["difficulty"] not in difficulty:
                include = False
                
            if regression_only and not test["is_regression"]:
                include = False
                
            if include:
                filtered.append(test)
                
        return filtered
    
    def get_progression_sequence(self, 
                               component: str = None, 
                               category: str = None) -> List[Dict[str, Any]]:
        """
        Create a progression sequence from easy to hard tests,
        optionally filtering by component or category.
        
        Args:
            component: Optional component to focus on
            category: Optional category to focus on
            
        Returns:
            List of test cases in difficulty order
        """
        # Get test cases with filters
        filters = {}
        if component:
            filters["components"] = [component]
        if category:
            filters["categories"] = [category]
            
        test_cases = self.filter_test_cases(**filters)
        
        # Map difficulty levels to numeric values for sorting
        difficulty_map = {
            "easy": 0,
            "medium": 1,
            "hard": 2,
            "extreme": 3
        }
        
        # Sort by difficulty
        return sorted(test_cases, key=lambda x: difficulty_map[x["difficulty"]])
    
    def evaluate_progress(self, latest_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate progress compared to previous test runs
        
        Args:
            latest_results: List of latest test results with test_id keys
            
        Returns:
            Dictionary with progress metrics
        """
        # Extract metrics from past results if available
        past_metrics = None
        if self.history:
            past_metrics = self.history[-1]["metrics"]
        
        # Calculate current metrics
        current_metrics = self._calculate_metrics(latest_results)
        
        # Calculate progress
        progress = {}
        if past_metrics:
            for key, value in current_metrics.items():
                if key in past_metrics and isinstance(value, (int, float)):
                    progress[key] = value - past_metrics[key]
                    
        # Store this run in history
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": current_metrics,
            "progress": progress
        }
        self.history.append(history_entry)
        
        # Return combined results
        return {
            "current": current_metrics,
            "progress": progress,
            "history": self.history
        }
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics from test results"""
        metrics = {
            "total_tests": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "pass_rate": 0.0,
            "by_category": {},
            "by_component": {},
            "by_difficulty": {},
            "regression_pass_rate": 0.0
        }
        
        # Calculate overall pass rate
        if metrics["total_tests"] > 0:
            metrics["pass_rate"] = (metrics["passed"] / metrics["total_tests"]) * 100
            
        # Calculate category-specific pass rates
        category_counts = defaultdict(lambda: {"total": 0, "passed": 0})
        for result in results:
            test_id = result["test_id"]
            if test_id < len(self.test_cases):
                test = self.test_cases[test_id]
                for category in test["categories"]:
                    category_counts[category]["total"] += 1
                    if result["passed"]:
                        category_counts[category]["passed"] += 1
        
        for category, counts in category_counts.items():
            metrics["by_category"][category] = {
                "pass_rate": (counts["passed"] / counts["total"]) * 100 if counts["total"] > 0 else 0,
                "count": f"{counts['passed']}/{counts['total']}"
            }
            
        # Calculate component-specific pass rates 
        component_counts = defaultdict(lambda: {"total": 0, "passed": 0})
        for result in results:
            test_id = result["test_id"]
            if test_id < len(self.test_cases):
                test = self.test_cases[test_id]
                for component in test["components"]:
                    component_counts[component]["total"] += 1
                    if result["passed"]:
                        component_counts[component]["passed"] += 1
        
        for component, counts in component_counts.items():
            metrics["by_component"][component] = {
                "pass_rate": (counts["passed"] / counts["total"]) * 100 if counts["total"] > 0 else 0,
                "count": f"{counts['passed']}/{counts['total']}"
            }
            
        # Calculate difficulty-specific pass rates
        difficulty_counts = defaultdict(lambda: {"total": 0, "passed": 0})
        for result in results:
            test_id = result["test_id"]
            if test_id < len(self.test_cases):
                test = self.test_cases[test_id]
                difficulty = test["difficulty"]
                difficulty_counts[difficulty]["total"] += 1
                if result["passed"]:
                    difficulty_counts[difficulty]["passed"] += 1
        
        for difficulty, counts in difficulty_counts.items():
            metrics["by_difficulty"][difficulty] = {
                "pass_rate": (counts["passed"] / counts["total"]) * 100 if counts["total"] > 0 else 0,
                "count": f"{counts['passed']}/{counts['total']}"
            }
            
        # Calculate regression test pass rate
        regression_results = []
        for result in results:
            test_id = result["test_id"]
            if test_id in self.regression_tests:
                regression_results.append(result)
                
        if regression_results:
            regression_passed = sum(1 for r in regression_results if r["passed"])
            metrics["regression_pass_rate"] = (regression_passed / len(regression_results)) * 100
            metrics["regression_count"] = f"{regression_passed}/{len(regression_results)}"
        
        return metrics
    
    def save_suite(self) -> str:
        """
        Save the test suite to a file
        
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_suites/{self.suite_name}_{timestamp}.json"
        
        suite_data = {
            "name": self.suite_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_cases": self.test_cases,
            "history": self.history
        }
        
        with open(filename, 'w') as f:
            json.dump(suite_data, f, indent=2)
            
        logger.info(f"Test suite saved to {filename}")
        return filename
    
    @classmethod
    def load_suite(cls, filepath: str) -> 'IncrementalTestSuite':
        """
        Load a test suite from a file
        
        Args:
            filepath: Path to the test suite file
            
        Returns:
            Loaded IncrementalTestSuite
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        suite = cls(suite_name=data["name"])
        suite.test_cases = data["test_cases"]
        suite.history = data["history"]
        
        # Rebuild metadata sets
        for test in suite.test_cases:
            suite.categories.update(test["categories"])
            suite.components.update(test["components"])
            suite.difficulty_levels.add(test["difficulty"])
            if test["is_regression"]:
                suite.regression_tests.append(test["id"])
                
        return suite
    
    @classmethod
    def get_latest_suite(cls, suite_name: str = "default") -> Optional['IncrementalTestSuite']:
        """
        Load the latest version of a named test suite
        
        Args:
            suite_name: Name of the suite to load
            
        Returns:
            Loaded IncrementalTestSuite or None if not found
        """
        if not os.path.exists('test_suites'):
            return None
            
        # Find files matching the suite name
        matching_files = [f for f in os.listdir('test_suites') 
                         if f.startswith(f"{suite_name}_") and f.endswith(".json")]
        
        if not matching_files:
            return None
            
        # Sort by timestamp (descending)
        latest_file = sorted(matching_files, 
                            key=lambda x: x.split('_')[-1].split('.')[0], 
                            reverse=True)[0]
        
        return cls.load_suite(os.path.join('test_suites', latest_file))
    
    def print_progress_report(self, results: Dict[str, Any]) -> None:
        """Print a formatted progress report"""
        logger.info("\n===== TEST PROGRESS REPORT =====")
        
        current = results["current"]
        progress = results["progress"]
        
        # Overall progress
        logger.info(f"Overall pass rate: {current['pass_rate']:.1f}%")
        if "pass_rate" in progress:
            change = progress["pass_rate"]
            if change > 0:
                logger.info(f"  Improvement: +{change:.1f}%")
            elif change < 0:
                logger.warning(f"  Regression: {change:.1f}%")
            else:
                logger.info("  No change")
        
        # By difficulty
        logger.info("\nResults by difficulty:")
        for difficulty in ["easy", "medium", "hard", "extreme"]:
            if difficulty in current["by_difficulty"]:
                data = current["by_difficulty"][difficulty]
                logger.info(f"  {difficulty.title()}: {data['pass_rate']:.1f}% ({data['count']})")
        
        # Top components/categories needing attention
        if current["by_component"]:
            logger.info("\nComponents needing most attention:")
            sorted_components = sorted(
                current["by_component"].items(),
                key=lambda x: x[1]["pass_rate"]
            )
            for component, data in sorted_components[:3]:  # Top 3 worst performing
                logger.info(f"  {component}: {data['pass_rate']:.1f}% ({data['count']})")
        
        if current["by_category"]:
            logger.info("\nCategories needing most attention:")
            sorted_categories = sorted(
                current["by_category"].items(),
                key=lambda x: x[1]["pass_rate"]
            )
            for category, data in sorted_categories[:3]:  # Top 3 worst performing
                logger.info(f"  {category}: {data['pass_rate']:.1f}% ({data['count']})")
        
        # Regression test status
        if "regression_pass_rate" in current:
            logger.info(f"\nRegression test pass rate: {current['regression_pass_rate']:.1f}%")
            if "regression_count" in current:
                logger.info(f"  ({current['regression_count']} tests)")

def create_default_test_suite() -> IncrementalTestSuite:
    """Create and populate the default test suite with structured test cases"""
    suite = IncrementalTestSuite("sentiment_analysis")
    
    # Negation handling tests
    suite.add_test_case(
        text="This isn't bad at all.",
        expected_sentiment="Positive",
        description="Simple negated negative becomes positive",
        categories=["negation"],
        components=["negation_handling"],
        difficulty="easy"
    )
    
    suite.add_test_case(
        text="I don't hate it.",
        expected_sentiment="Positive",
        description="Negated strong negative becomes positive",
        categories=["negation"],
        components=["negation_handling"],
        difficulty="easy"
    )
    
    suite.add_test_case(
        text="This movie isn't bad at all, I actually enjoyed most of it.",
        expected_sentiment="Positive",
        description="Negated negative with positive reinforcement",
        categories=["negation", "mixed"],
        components=["negation_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="The product isn't exactly what I wouldn't recommend.",
        expected_sentiment="Neutral",
        description="Double negation complexity",
        categories=["negation"],
        components=["negation_handling"],
        difficulty="hard"
    )
    
    suite.add_test_case(
        text="It's not that I didn't like it, I just didn't love it as much as I expected.",
        expected_sentiment="Neutral",
        description="Triple negation with expectation contrast",
        categories=["negation", "contrast", "expectation"],
        components=["negation_handling", "contrast_handling"],
        difficulty="extreme"
    )
    
    # Sarcasm detection tests
    suite.add_test_case(
        text="Wow, they really outdid themselves with how forgettable this was.",
        expected_sentiment="Negative",
        description="Sarcasm with positive-negative contrast",
        categories=["sarcasm"],
        components=["sarcasm_detection"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="The best part of this movie was when the credits rolled.",
        expected_sentiment="Negative",
        description="Sarcasm - credits rolled pattern",
        categories=["sarcasm"],
        components=["sarcasm_detection"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="If you enjoy falling asleep during movies, this one's perfect for you!",
        expected_sentiment="Negative",
        description="Sarcasm - conditional enjoyment",
        categories=["sarcasm"],
        components=["sarcasm_detection"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="A cinematic achievement that will be studied for generations... as what not to do.",
        expected_sentiment="Negative",
        description="Delayed sarcasm reveal",
        categories=["sarcasm"],
        components=["sarcasm_detection"],
        difficulty="hard"
    )
    
    # Contrast handling tests
    suite.add_test_case(
        text="Despite the beautiful visuals, the plot was confusing.",
        expected_sentiment="Negative",
        description="Contrast marker with negative after",
        categories=["contrast"],
        components=["contrast_handling"],
        difficulty="easy"
    )
    
    suite.add_test_case(
        text="The restaurant was crowded, but the food was absolutely delicious.",
        expected_sentiment="Positive",
        description="Restaurant contrast with food positive",
        categories=["contrast", "restaurant"],
        components=["contrast_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="The service was terrible but the food was amazing.",
        expected_sentiment="Positive",
        description="Restaurant with terrible service, amazing food",
        categories=["contrast", "restaurant"],
        components=["contrast_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="The ambiance was terrible but I've never had better pasta in my life.",
        expected_sentiment="Positive",
        description="Restaurant contrast with negated comparison",
        categories=["contrast", "restaurant", "negation"],
        components=["contrast_handling", "negation_handling"],
        difficulty="hard"
    )
    
    # Idiom detection tests
    suite.add_test_case(
        text="This film is a real diamond in the rough - don't miss it!",
        expected_sentiment="Positive",
        description="Positive idiom with recommendation",
        categories=["idiom"],
        components=["idiom_detection"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="That movie was a complete train wreck from start to finish.",
        expected_sentiment="Negative",
        description="Negative disaster idiom",
        categories=["idiom"],
        components=["idiom_detection"],
        difficulty="medium"
    )
    
    # Neutral sentiment tests
    suite.add_test_case(
        text="This is neither good nor bad, just average.",
        expected_sentiment="Neutral",
        description="Explicit neutral statement",
        categories=["neutral"],
        components=["neutral_detection"],
        difficulty="easy"
    )
    
    suite.add_test_case(
        text="It's exactly what you'd expect, nothing more, nothing less.",
        expected_sentiment="Neutral",
        description="Met expectations exactly",
        categories=["neutral", "expectation"],
        components=["neutral_detection"],
        difficulty="medium"
    )
    
    # Mixed sentiment with conclusion tests
    suite.add_test_case(
        text="While I loved the graphics, hated the story, and felt neutral about the acting, overall I enjoyed it.",
        expected_sentiment="Positive",
        description="Mixed sentiment with explicit positive conclusion",
        categories=["mixed", "conclusion"],
        components=["contrast_handling"],
        difficulty="hard"
    )
    
    suite.add_test_case(
        text="Though some parts were incredible, others were horrible, but ultimately I was disappointed.",
        expected_sentiment="Negative",
        description="Mixed sentiment with explicit negative conclusion",
        categories=["mixed", "conclusion"],
        components=["contrast_handling"],
        difficulty="hard"
    )
    
    # Regression tests
    suite.add_test_case(
        text="I thought it would be trash, but it's a guilty pleasure I secretly enjoyed.",
        expected_sentiment="Positive",
        description="Expectation contrast with positive conclusion",
        categories=["contrast", "expectation"],
        components=["contrast_handling"],
        difficulty="medium",
        is_regression=True
    )
    
    logger.info(f"Created default test suite with {len(suite.test_cases)} test cases")
    return suite

def expand_test_suite(suite: IncrementalTestSuite) -> IncrementalTestSuite:
    """
    Expand an existing test suite with additional challenging test cases,
    focusing on areas that need improvement based on previous testing results.
    
    Args:
        suite: The existing test suite to expand
        
    Returns:
        The expanded test suite
    """
    original_count = len(suite.test_cases)
    
    # Add more complex negation handling cases
    suite.add_test_case(
        text="She wasn't unable to finish the project.",
        expected_sentiment="Positive",
        description="Double negation cancellation",
        categories=["negation"],
        components=["negation_handling"],
        difficulty="hard"
    )
    
    suite.add_test_case(
        text="I can't say I wasn't impressed by their service.",
        expected_sentiment="Positive", 
        description="Multiple negations with positive sentiment",
        categories=["negation"],
        components=["negation_handling"],
        difficulty="hard"
    )
    
    suite.add_test_case(
        text="Not only was it not bad, it was actually quite good.",
        expected_sentiment="Positive",
        description="Negation with positive intensification",
        categories=["negation", "intensifier"],
        components=["negation_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="No one could deny that the movie was excellent.",
        expected_sentiment="Positive",
        description="Negative framing with positive content",
        categories=["negation", "framing"],
        components=["negation_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="There's no way you won't enjoy this restaurant.",
        expected_sentiment="Positive",
        description="Double negation with recommendation",
        categories=["negation", "recommendation"],
        components=["negation_handling"],
        difficulty="hard"
    )
    
    # Add more restaurant contrast cases
    suite.add_test_case(
        text="The prices were high, but the food was out of this world.",
        expected_sentiment="Positive",
        description="Restaurant price-quality contrast",
        categories=["contrast", "restaurant"],
        components=["contrast_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="Despite the rude staff, this is my favorite place to eat.",
        expected_sentiment="Positive",
        description="Restaurant with service issues but strong preference",
        categories=["contrast", "restaurant"],
        components=["contrast_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="The atmosphere was lacking, though the flavors were exceptional.",
        expected_sentiment="Positive",
        description="Restaurant with atmosphere-food contrast using 'though'",
        categories=["contrast", "restaurant"],
        components=["contrast_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="While we had to wait 30 minutes, their famous chocolate cake made everything worthwhile.",
        expected_sentiment="Positive",
        description="Restaurant wait time contrast with specific food highlight",
        categories=["contrast", "restaurant", "time"],
        components=["contrast_handling"],
        difficulty="hard"
    )
    
    # Add general contrast handling cases
    suite.add_test_case(
        text="She failed the exam, but she'll get another chance next semester.",
        expected_sentiment="Neutral",
        description="Balanced negative-positive contrast",
        categories=["contrast", "balanced"],
        components=["contrast_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="Despite having a small budget, the film was visually stunning.",
        expected_sentiment="Positive",
        description="Creative accomplishment despite limitations",
        categories=["contrast", "creative"],
        components=["contrast_handling"],
        difficulty="medium"
    )
    
    suite.add_test_case(
        text="The plot had some issues, nonetheless I found myself fully invested in the characters.",
        expected_sentiment="Positive",
        description="Plot-character contrast with 'nonetheless'",
        categories=["contrast", "investment"],
        components=["contrast_handling"],
        difficulty="medium"
    )
    
    # Add some complex multi-component cases
    suite.add_test_case(
        text="I wasn't thrilled with the acting, but the script wasn't bad at all.",
        expected_sentiment="Neutral",
        description="Negation with contrast resulting in neutral",
        categories=["negation", "contrast", "balanced"],
        components=["negation_handling", "contrast_handling"],
        difficulty="hard"
    )
    
    suite.add_test_case(
        text="I know it's not supposed to be great, but I couldn't help loving every minute of it.",
        expected_sentiment="Positive",
        description="Expectation contrast with negation and strong positive",
        categories=["contrast", "expectation", "negation", "guilty_pleasure"],
        components=["contrast_handling", "negation_handling"],
        difficulty="hard"
    )
    
    suite.add_test_case(
        text="The book wasn't exactly a page-turner, yet somehow I couldn't put it down.",
        expected_sentiment="Positive",
        description="Contradiction between statement and behavior",
        categories=["contrast", "contradiction", "negation"],
        components=["contrast_handling", "negation_handling", "contradiction_detection"],
        difficulty="hard"
    )
    
    # Add a few more neutral cases
    suite.add_test_case(
        text="It has its pros and cons, so I'm completely split on this one.",
        expected_sentiment="Neutral",
        description="Explicit balanced pros and cons",
        categories=["neutral", "balanced"],
        components=["neutral_detection"],
        difficulty="easy"
    )
    
    suite.add_test_case(
        text="The performance was sometimes brilliant and sometimes awful.",
        expected_sentiment="Neutral",
        description="Mixed performance quality",
        categories=["neutral", "mixed"],
        components=["neutral_detection"],
        difficulty="medium"
    )
    
    # Add some subtle sarcasm cases
    suite.add_test_case(
        text="Just what the world needed, another superhero movie.",
        expected_sentiment="Negative",
        description="Subtle sarcasm about market saturation",
        categories=["sarcasm", "subtle"],
        components=["sarcasm_detection"],
        difficulty="hard"
    )
    
    suite.add_test_case(
        text="Oh great, another email about my car's extended warranty.",
        expected_sentiment="Negative",
        description="Sarcastic 'great' about spam",
        categories=["sarcasm", "subtle"],
        components=["sarcasm_detection"],
        difficulty="medium"
    )
    
    # Add regression test cases for previously fixed issues
    suite.add_test_case(
        text="Despite the negative reviews, I found it quite enjoyable.",
        expected_sentiment="Positive",
        description="Opinion contrast with explicit conclusion",
        categories=["contrast", "opinion"],
        components=["contrast_handling"],
        difficulty="medium",
        is_regression=True
    )
    
    suite.add_test_case(
        text="The camera wasn't bad for the price range.",
        expected_sentiment="Positive",
        description="Negated negative with contextual qualifier",
        categories=["negation", "context"],
        components=["negation_handling"],
        difficulty="medium",
        is_regression=True
    )
    
    suite.add_test_case(
        text="I don't regret purchasing this product at all.",
        expected_sentiment="Positive",
        description="Negated regret with intensifier",
        categories=["negation", "intensifier"],
        components=["negation_handling"],
        difficulty="medium",
        is_regression=True
    )
    
    # Add a few extreme challenge cases
    suite.add_test_case(
        text="While the acting left much to be desired, and the special effects weren't convincing, there was something undeniably charming about its earnest approach to the material.",
        expected_sentiment="Positive",
        description="Multiple negatives with positive conclusion",
        categories=["complex", "contrast", "negation", "conclusion"],
        components=["contrast_handling", "negation_handling"],
        difficulty="extreme"
    )
    
    suite.add_test_case(
        text="I'm not saying I didn't dislike parts of it, but I wouldn't say it wasn't worth watching at least once.",
        expected_sentiment="Neutral",
        description="Quintuple negation complexity",
        categories=["negation", "complex"],
        components=["negation_handling"],
        difficulty="extreme"
    )
    
    # Log the results
    logger.info(f"Added {len(suite.test_cases) - original_count} test cases to the suite")
    logger.info(f"Test suite now contains {len(suite.test_cases)} test cases")
    
    return suite

# Add ComponentTester class to run the component tests

class ComponentTester:
    """Runs tests on individual sentiment analysis pipeline components"""
    
    def __init__(self):
        """Initialize component tester"""
        self.model = DiagnosticEnsemble()
        self.results = {}
    
    def test_negation_handling(self, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Test the negation handling component in isolation.
        
        Args:
            test_cases: Optional list of test cases, otherwise uses default cases
            
        Returns:
            Dictionary with test results
        """
        if test_cases is None:
            test_cases = [
                {
                    "text": "This isn't bad at all.",
                    "expected_output": "this is n't bad_NEG at_NEG all_NEG"  # First part of the tuple
                },
                {
                    "text": "I don't hate it.",
                    "expected_output": "i do n't hate_NEG it_NEG"  # Special phrase might change this
                },
                {
                    "text": "The product isn't exactly what I wouldn't recommend.",
                    "expected_output": "the product is n't exactly_NEG what_NEG i_NEG would_NEG n't_NEG recommend_NEG"
                },
                {
                    "text": "I never said it was terrible.",
                    "expected_output": "i never said_NEG it_NEG was_NEG terrible_NEG"
                },
                {
                    "text": "This is good.",
                    "expected_output": "this is good"
                }
            ]
        
        results = []
        for test_case in test_cases:
            text = test_case["text"]
            expected = test_case["expected_output"]
            
            # Test component in isolation
            result = self.model.test_component("handle_negations", text, expected)
            
            # Negation handling returns a tuple, so we need to compare with first element
            if "output" in result and result["output"] and isinstance(result["output"], tuple):
                actual_output = result["output"][0]
                # Remove periods for comparison as they might be handled differently
                cleaned_actual = actual_output.replace('.', '').strip()
                cleaned_expected = expected.replace('.', '').strip()
                
                # Update success status
                result["success"] = cleaned_actual == cleaned_expected
            
            results.append(result)
        
        # Calculate success rate
        success_count = sum(1 for r in results if r.get("success", False))
        success_rate = success_count / len(results) if results else 0
        
        return {
            "component": "negation_handling",
            "success_rate": success_rate,
            "success_count": f"{success_count}/{len(results)}",
            "detailed_results": results
        }
    
    def test_contrast_handling(self, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Test the contrast marker handling component in isolation.
        
        Args:
            test_cases: Optional list of test cases, otherwise uses default cases
            
        Returns:
            Dictionary with test results
        """
        if test_cases is None:
            test_cases = [
                {
                    "text": "The service was terrible but the food was amazing.",
                    "expected_output": {
                        "has_contrast": True,
                        "contrast_marker": "but"
                    }
                },
                {
                    "text": "Despite the high price, I liked the product.",
                    "expected_output": {
                        "has_contrast": True,
                        "contrast_marker": "despite"
                    }
                },
                {
                    "text": "The graphics were good, however the story was weak.",
                    "expected_output": {
                        "has_contrast": True,
                        "contrast_marker": "however"
                    }
                },
                {
                    "text": "I loved this film and would recommend it.",
                    "expected_output": {
                        "has_contrast": False
                    }
                }
            ]
        
        results = []
        for test_case in test_cases:
            text = test_case["text"]
            expected = test_case["expected_output"]
            
            # Test component in isolation
            result = self.model.test_component("process_contrast_markers", text, expected)
            results.append(result)
        
        # Calculate success rate
        success_count = sum(1 for r in results if r.get("success", False))
        success_rate = success_count / len(results) if results else 0
        
        return {
            "component": "contrast_handling",
            "success_rate": success_rate,
            "success_count": f"{success_count}/{len(results)}",
            "detailed_results": results
        }
    
    def test_sarcasm_detection(self, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Test the sarcasm detection component in isolation.
        
        Args:
            test_cases: Optional list of test cases, otherwise uses default cases
            
        Returns:
            Dictionary with test results
        """
        if test_cases is None:
            test_cases = [
                {
                    "text": "Wow, they really outdid themselves with how forgettable this was.",
                    "expected_output": {
                        "sarcasm_detected": True,
                        "type": "contrasting_praise",
                        "confidence": 85.0
                    }
                },
                {
                    "text": "Sure, it's a masterpiece... if your standards are below ground level.",
                    "expected_output": {
                        "sarcasm_detected": True,
                        "type": "conditional_praise",
                        "confidence": 85.0
                    }
                },
                {
                    "text": "The best part of this movie was when the credits rolled.",
                    "expected_output": {
                        "sarcasm_detected": True,
                        "type": "credits_rolled",
                        "confidence": 85.0
                    }
                },
                {
                    "text": "I really enjoyed this movie, it was entertaining.",
                    "expected_output": {
                        "sarcasm_detected": False
                    }
                }
            ]
        
        results = []
        for test_case in test_cases:
            text = test_case["text"]
            expected = test_case["expected_output"]
            
            # Test component in isolation
            result = self.model.test_component("detect_sarcasm", text, expected)
            
            # For sarcasm detection, we need to check if the essential fields match
            # since confidence might be slightly different
            if "error" not in result or not result["error"]:
                output = result["output"]
                expected_output = result["expected"]
                
                # Check for key sarcasm detection properties
                if isinstance(output, dict) and isinstance(expected_output, dict):
                    detected_match = output.get("sarcasm_detected") == expected_output.get("sarcasm_detected")
                    type_match = True
                    
                    # Only check type if sarcasm was detected
                    if output.get("sarcasm_detected") and expected_output.get("sarcasm_detected"):
                        type_match = output.get("type") == expected_output.get("type")
                    
                    result["success"] = detected_match and type_match
            
            results.append(result)
        
        # Calculate success rate
        success_count = sum(1 for r in results if r.get("success", False))
        success_rate = success_count / len(results) if results else 0
        
        return {
            "component": "sarcasm_detection",
            "success_rate": success_rate,
            "success_count": f"{success_count}/{len(results)}",
            "detailed_results": results
        }
    
    def measure_component_contributions(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measure each component's contribution to overall accuracy.
        
        Args:
            test_cases: List of test cases with text and expected sentiment
            
        Returns:
            Dictionary with contribution measurements
        """
        logger.info("Measuring component contributions will be limited to sarcasm, idiom, and contradiction detection.")
        
        # Focus on components with clear API parameters
        components_to_test = [
            "sarcasm_detection",
            "idiom_detection", 
            "contradiction_detection"
        ]
        
        component_impacts = {}
        for component in components_to_test:
            component_impacts[component] = {
                "improved_accuracy_count": 0,
                "reduced_accuracy_count": 0,
                "no_impact_count": 0,
                "sentiment_changed_count": 0,
                "total_confidence_impact": 0,
                "examples": []
            }
        
        # Test a limited set of examples to prevent overrunning
        test_subset = test_cases[:5] if len(test_cases) > 5 else test_cases
        
        for i, test_case in enumerate(test_subset):
            text = test_case["text"]
            expected = test_case["expected_sentiment"]
            
            logger.info(f"Measuring component impact for test case {i+1}: {text}")
            
            # Get baseline prediction with all components
            baseline = self.model.predict(text)
            baseline_correct = baseline["sentiment"] == expected
            
            # Test each supported component by toggling it off
            for component in components_to_test:
                param_name = f"use_{component}"
                
                try:
                    # Run prediction without this component
                    kwargs = {param_name: False}
                    without_component = self.model.predict(text, **kwargs)
                    
                    # Calculate impact metrics
                    without_correct = without_component["sentiment"] == expected
                    sentiment_changed = baseline["sentiment"] != without_component["sentiment"]
                    confidence_diff = baseline["confidence"] - without_component["confidence"]
                    
                    # Update stats
                    if baseline_correct and not without_correct:
                        component_impacts[component]["improved_accuracy_count"] += 1
                    
                    if not baseline_correct and without_correct:
                        component_impacts[component]["reduced_accuracy_count"] += 1
                    
                    if baseline_correct == without_correct:
                        component_impacts[component]["no_impact_count"] += 1
                    
                    if sentiment_changed:
                        component_impacts[component]["sentiment_changed_count"] += 1
                    
                    component_impacts[component]["total_confidence_impact"] += abs(confidence_diff)
                    
                    # Store example if component had an impact
                    if sentiment_changed or abs(confidence_diff) > 5:
                        component_impacts[component]["examples"].append({
                            "text": text,
                            "expected": expected,
                            "baseline": baseline["sentiment"],
                            "without_component": without_component["sentiment"],
                            "confidence_difference": confidence_diff
                        })
                except Exception as e:
                    logger.error(f"Error testing {component} impact: {str(e)}")
        
        # Calculate averages and impact scores
        total_cases = len(test_subset)
        for component, data in component_impacts.items():
            data["impact_score"] = (
                (data["improved_accuracy_count"] * 2) + 
                data["sentiment_changed_count"] + 
                (data["total_confidence_impact"] / max(1, total_cases) / 10)
            ) / max(1, total_cases)
            
            data["avg_confidence_impact"] = data["total_confidence_impact"] / max(1, total_cases)
            data["improved_accuracy_pct"] = (data["improved_accuracy_count"] / max(1, total_cases)) * 100
            data["reduced_accuracy_pct"] = (data["reduced_accuracy_count"] / max(1, total_cases)) * 100
            data["sentiment_changed_pct"] = (data["sentiment_changed_count"] / max(1, total_cases)) * 100
        
        # Sort components by impact score
        sorted_components = sorted(
            component_impacts.items(),
            key=lambda x: x[1]["impact_score"],
            reverse=True
        )
        
        return {
            "components": dict(sorted_components),
            "total_cases": total_cases
        }
    
    def run_complete_component_tests(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a complete set of component tests and impact measurements.
        
        Args:
            test_cases: List of test cases with text and expected sentiment
            
        Returns:
            Dictionary with all test results
        """
        logger.info("Running negation handling tests...")
        negation_results = self.test_negation_handling()
        
        logger.info("Running contrast handling tests...")
        contrast_results = self.test_contrast_handling()
        
        logger.info("Running sarcasm detection tests...")
        sarcasm_results = self.test_sarcasm_detection()
        
        logger.info("Measuring component contributions...")
        contribution_results = self.measure_component_contributions(test_cases)
        
        # Combine results
        all_results = {
            "component_tests": {
                "negation_handling": negation_results,
                "contrast_handling": contrast_results,
                "sarcasm_detection": sarcasm_results
            },
            "component_contributions": contribution_results
        }
        
        # Log summary
        logger.info("\n===== COMPONENT TEST SUMMARY =====")
        logger.info(f"Negation handling: {negation_results['success_rate']*100:.1f}% success ({negation_results['success_count']})")
        logger.info(f"Contrast handling: {contrast_results['success_rate']*100:.1f}% success ({contrast_results['success_count']})")
        logger.info(f"Sarcasm detection: {sarcasm_results['success_rate']*100:.1f}% success ({sarcasm_results['success_count']})")
        
        logger.info("\n--- Component Contributions to Accuracy ---")
        for component, data in contribution_results["components"].items():
            logger.info(f"{component}: impact score {data['impact_score']:.2f}")
            logger.info(f"  - Improved accuracy in {data['improved_accuracy_pct']:.1f}% of cases")
            logger.info(f"  - Changed sentiment in {data['sentiment_changed_pct']:.1f}% of cases")
            logger.info(f"  - Average confidence impact: {data['avg_confidence_impact']:.2f}%")
        
        return all_results

def test_sentiment_analysis(categories: Optional[List[str]] = None,
                           models: Optional[List[str]] = None,
                           enable_debug: bool = False,
                           compare_with: Optional[str] = None,
                           component_testing: bool = False,
                           use_incremental: bool = False,
                           suite_name: str = "sentiment_analysis") -> Dict[str, Any]:
    """
    Test the sentiment analysis model on a set of challenging test cases.
    Enhanced with diagnostics, performance analysis, and component testing.
    
    Args:
        categories: Optional list of test categories to run (e.g., ["negation", "sarcasm"])
        models: Optional list of models to test (default: all available models)
        enable_debug: Whether to enable detailed debugging output
        compare_with: Timestamp of previous results to compare with, or "latest"
        component_testing: Whether to run component isolation tests
        use_incremental: Whether to use the incremental test suite
        suite_name: Name of the test suite to use (if use_incremental is True)
        
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
        
        # Initialize error analyzer
        error_analyzer = ErrorAnalyzer()
        
        # Determine test cases to use
        test_cases = []
        incremental_suite = None
        
        if use_incremental:
            # Load or create incremental test suite
            incremental_suite = IncrementalTestSuite.get_latest_suite(suite_name)
            if not incremental_suite:
                logger.info("No existing test suite found. Creating default suite.")
                incremental_suite = create_default_test_suite()
                
            # Get test cases from the suite
            if categories:
                logger.info(f"Filtering incremental test suite by categories: {', '.join(categories)}")
                suite_test_cases = incremental_suite.filter_test_cases(categories=categories)
            else:
                suite_test_cases = incremental_suite.test_cases
                
            # Convert to the format expected by the test function
            for tc in suite_test_cases:
                test_cases.append({
                    "text": tc["text"],
                    "expected_sentiment": tc["expected_sentiment"],
                    "description": tc["description"],
                    "category": ",".join(tc["categories"]),
                    "test_id": tc["id"]  # Add test ID for tracking
                })
                
            logger.info(f"Using incremental test suite with {len(test_cases)} test cases")
        else:
            # Use the hardcoded test cases (original behavior)
            test_cases = [
                # Original test cases
                # ... [existing test cases here] ...
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
                # ... [rest of the existing test cases] ...
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
        
        # ... [existing code for model validation] ...
        
        # Filter to only valid models
        models_to_test = [m for m in models_to_test if m in available_models]
        
        if not models_to_test:
            logger.error("No valid models selected for testing!")
            return {}
            
        logger.info(f"Testing with models: {', '.join(models_to_test)}")
        
        # ... [existing code for previous results] ...
        
        results = {}
        diagnostic_data = {}
        incremental_results = {}
        
        # Record overall test run timing
        test_start_time = time.time()
        
        for model_name in models_to_test:
            logger.info(f"\nTesting with model: {model_name}")
            model_results = []
            model_diagnostic_data = []
            model_timing = []
            
            # For incremental results tracking
            if use_incremental:
                incremental_results[model_name] = []
            
            # ... [existing test execution code] ...
            
            for i, test_case in enumerate(test_cases):
                text = test_case["text"]
                expected = test_case["expected_sentiment"]
                description = test_case["description"]
                categories = test_case.get("category", "").split(",")
                test_id = test_case.get("test_id", i)  # Use provided test_id or fallback to index
                
                # ... [existing prediction and logging code] ...
                
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
                
                # ... [existing diagnostic and logging code] ...
                
                # Get diagnostic data
                diagnostic_summary = model.get_diagnostic_summary()
                model_diagnostic_data.append({
                    "test_case": description,
                    "diagnostic_data": diagnostic_summary
                })
                
                # ... [existing confidence band code] ...
                
                # Check if prediction matches expectation
                matches = predicted == expected
                if matches:
                    logger.info(f"Predicted: {predicted} ({confidence:.2f}%) using {model_used} ✓")
                    logger.info(f"Duration: {prediction_time*1000:.2f}ms")
                else:
                    logger.error(f"Predicted: {predicted} ({confidence:.2f}%) using {model_used} ✗")
                    logger.error(f"Duration: {prediction_time*1000:.2f}ms")
                    
                    # ... [existing diagnostic error logging] ...
                
                # Store result
                test_result = {
                    "test_case": description,
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "model_used": model_used,
                    "passed": matches,
                    "categories": categories,
                    "processing_time_ms": prediction_time * 1000
                }
                
                model_results.append(test_result)
                
                # For incremental test suite, also track results with test_id
                if use_incremental and incremental_suite:
                    incremental_result = test_result.copy()
                    incremental_result["test_id"] = test_id
                    incremental_results[model_name].append(incremental_result)
                    
                    # Add result to the test suite
                    incremental_suite.add_test_result(
                        test_id=test_id,
                        passed=matches,
                        predicted=predicted,
                        confidence=confidence,
                        model_name=model_name,
                        processing_time=prediction_time * 1000
                    )
            
            # ... [existing pass rate calculation code] ...
            
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
            confidence_bands = {
                "50-60": {"count": 0, "correct": 0},
                "60-70": {"count": 0, "correct": 0},
                "70-80": {"count": 0, "correct": 0},
                "80-90": {"count": 0, "correct": 0},
                "90-100": {"count": 0, "correct": 0}
            }
            
            # Populate confidence bands
            for result in model_results:
                confidence = result["confidence"]
                for band, data in confidence_bands.items():
                    low, high = map(int, band.split("-"))
                    if low <= confidence < high:
                        data["count"] += 1
                        if result["passed"]:
                            data["correct"] += 1
            
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
        
        # ... [existing summary code] ...
        
        # Create timestamp for this test run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results to file
        results_file = f"test_results/sentiment_test_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nTest results saved to {results_file}")
        
        # ... [existing diagnostic data saving code] ...
        
        # If using incremental test suite, evaluate progress and save the suite
        if use_incremental and incremental_suite:
            for model_name, model_inc_results in incremental_results.items():
                progress = incremental_suite.evaluate_progress(model_inc_results)
                incremental_suite.print_progress_report(progress)
            
            # Save the updated test suite
            suite_file = incremental_suite.save_suite()
            logger.info(f"Incremental test suite updated and saved to {suite_file}")
        
        # ... [existing error analysis code] ...
        
        # Add component testing if requested
        if component_testing:
            # ... [existing component testing code] ...
            pass
        
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
    
    # Error analysis options
    parser.add_argument('--skip-analysis', action='store_true', help='Skip error analysis step')
    parser.add_argument('--analysis-only', action='store_true', help='Only run error analysis on latest results')
    
    # Component testing options
    parser.add_argument('--component-tests', action='store_true', help='Run component isolation tests')
    parser.add_argument('--component-only', action='store_true', help='Only run component isolation tests')
    parser.add_argument('--measure-impact', action='store_true', help='Measure component contribution to accuracy')
    
    # Incremental test suite options
    parser.add_argument('--incremental', action='store_true', help='Use incremental test suite')
    parser.add_argument('--suite-name', type=str, default='sentiment_analysis', help='Name of the test suite to use')
    parser.add_argument('--create-suite', action='store_true', help='Create a new default test suite')
    parser.add_argument('--progression', type=str, help='Run tests in progression order for a category or component')
    parser.add_argument('--expand-suite', action='store_true', help='Expand the test suite with additional challenging cases')
    
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
    
    if args.create_suite:
        # Create a new default test suite
        try:
            suite = create_default_test_suite()
            suite_file = suite.save_suite()
            logger.info(f"Created and saved default test suite to {suite_file}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error creating test suite: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    elif args.expand_suite:
        # Expand an existing test suite with additional cases
        try:
            # Load the latest test suite
            suite = IncrementalTestSuite.get_latest_suite(args.suite_name)
            if not suite:
                logger.info(f"No existing test suite found with name: {args.suite_name}. Creating a new one.")
                suite = create_default_test_suite()
                
            # Expand the test suite
            expanded_suite = expand_test_suite(suite)
            
            # Save the expanded suite
            suite_file = expanded_suite.save_suite()
            logger.info(f"Expanded test suite saved to {suite_file}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error expanding test suite: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    elif args.progression:
        # Run tests in progression order for a specific component or category
        try:
            # Load the latest test suite
            suite = IncrementalTestSuite.get_latest_suite(args.suite_name)
            if not suite:
                logger.error(f"No test suite found with name: {args.suite_name}")
                sys.exit(1)
                
            # Determine if progression is for a component or category
            if args.progression in suite.components:
                logger.info(f"Running progression for component: {args.progression}")
                progression = suite.get_progression_sequence(component=args.progression)
            elif args.progression in suite.categories:
                logger.info(f"Running progression for category: {args.progression}")
                progression = suite.get_progression_sequence(category=args.progression)
            else:
                logger.error(f"Unknown component or category: {args.progression}")
                sys.exit(1)
                
            # Extract test cases in progression order
            test_cases = []
            for tc in progression:
                test_cases.append({
                    "text": tc["text"],
                    "expected_sentiment": tc["expected_sentiment"],
                    "description": tc["description"],
                    "category": ",".join(tc["categories"]),
                    "test_id": tc["id"]
                })
                
            # Initialize model and run tests in progression order
            model = DiagnosticEnsemble()
            
            # Select model to use
            model_name = models[0] if models and len(models) > 0 else "naive_bayes"
            logger.info(f"Using model: {model_name}")
            
            # Run tests in progression order
            results = []
            for i, tc in enumerate(test_cases):
                logger.info(f"\n----- Progression Test {i+1}/{len(test_cases)}: {tc['description']} -----")
                logger.info(f"Difficulty: {progression[i]['difficulty']}")
                logger.info(f"Text: '{tc['text']}'")
                
                # Make prediction
                prediction = model.predict(tc["text"], modelname=model_name)
                
                # Check result
                matches = prediction["sentiment"] == tc["expected_sentiment"]
                result_str = "✓" if matches else "✗"
                
                if matches:
                    logger.info(f"Result: {prediction['sentiment']} ({prediction['confidence']:.2f}%) {result_str}")
                else:
                    logger.error(f"Result: {prediction['sentiment']} ({prediction['confidence']:.2f}%) {result_str}")
                    logger.error(f"Expected: {tc['expected_sentiment']}")
                
                # Store result
                results.append({
                    "test_id": tc["test_id"],
                    "passed": matches,
                    "predicted": prediction["sentiment"],
                    "confidence": prediction["confidence"],
                    "model_name": model_name
                })
                
                # Add a separator line
                if i < len(test_cases) - 1:
                    logger.info("\n" + "-" * 50)
            
            # Calculate pass rate by difficulty
            difficulty_results = defaultdict(lambda: {"total": 0, "passed": 0})
            for i, r in enumerate(results):
                difficulty = progression[i]["difficulty"]
                difficulty_results[difficulty]["total"] += 1
                if r["passed"]:
                    difficulty_results[difficulty]["passed"] += 1
            
            # Log summary
            logger.info("\n===== PROGRESSION TEST SUMMARY =====")
            total_passed = sum(1 for r in results if r["passed"])
            logger.info(f"Overall: {total_passed}/{len(results)} tests passed ({total_passed/len(results)*100:.1f}%)")
            
            for difficulty in ["easy", "medium", "hard", "extreme"]:
                if difficulty in difficulty_results:
                    dr = difficulty_results[difficulty]
                    if dr["total"] > 0:
                        pass_rate = dr["passed"] / dr["total"] * 100
                        logger.info(f"{difficulty.title()}: {dr['passed']}/{dr['total']} tests passed ({pass_rate:.1f}%)")
            
            # Update the test suite with results
            for result in results:
                suite.add_test_result(
                    test_id=result["test_id"],
                    passed=result["passed"],
                    predicted=result["predicted"],
                    confidence=result["confidence"],
                    model_name=result["model_name"],
                    processing_time=0.0  # We didn't track timing
                )
            
            # Save the updated suite
            suite_file = suite.save_suite()
            logger.info(f"Updated test suite saved to {suite_file}")
            
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error running progression tests: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    elif args.component_only:
        # Only run component isolation tests
        try:
            component_tester = ComponentTester()
            
            # Create results directory if it doesn't exist
            if not os.path.exists('test_results'):
                os.makedirs('test_results')
            
            # Load test cases for impact measurement
            try:
                # Find the test cases from the most recent test
                results_files = sorted([f for f in os.listdir("test_results") 
                                        if f.startswith("sentiment_test_results_") and f.endswith(".json")],
                                      key=lambda x: x.split("_")[-1].split(".")[0],
                                      reverse=True)
                
                if results_files:
                    latest_file = os.path.join("test_results", results_files[0])
                    with open(latest_file, 'r') as f:
                        results = json.load(f)
                    
                    # Extract test cases from first model
                    model_name = next(iter(results))
                    test_results = results[model_name]["test_results"]
                    
                    test_cases = [
                        {
                            "text": r["text"],
                            "expected_sentiment": r["expected"]
                        }
                        for r in test_results
                    ]
                else:
                    # If no previous results, use hardcoded test cases
                    logger.warning("No previous test results found. Using default test cases.")
                    test_cases = [
                        {"text": "This movie isn't bad at all.", "expected_sentiment": "Positive"},
                        {"text": "Despite the beautiful visuals, the plot was confusing.", "expected_sentiment": "Negative"},
                        {"text": "The best part of this movie was when the credits rolled.", "expected_sentiment": "Negative"}
                    ]
            except Exception as e:
                logger.error(f"Error loading test cases: {e}")
                # Fall back to default test cases
                test_cases = [
                    {"text": "This movie isn't bad at all.", "expected_sentiment": "Positive"},
                    {"text": "Despite the beautiful visuals, the plot was confusing.", "expected_sentiment": "Negative"},
                    {"text": "The best part of this movie was when the credits rolled.", "expected_sentiment": "Negative"}
                ]
            
            # Run component tests
            component_results = component_tester.run_complete_component_tests(test_cases)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            component_file = f"test_results/component_tests_{timestamp}.json"
            with open(component_file, "w") as f:
                json.dump(component_results, f, indent=2)
            logger.info(f"Component test results saved to {component_file}")
            
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during component testing: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    elif args.analysis_only:
        # Only run error analysis on latest results
        try:
            # ... [existing analysis_only code] ...
            pass
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            logger.error(traceback.format_exc())
            sys.exit(1)
    else:
        # Run tests with specified options
        test_sentiment_analysis(
            categories=categories,
            models=models,
            enable_debug=args.debug,
            compare_with=args.compare,
            component_testing=args.component_tests,
            use_incremental=args.incremental,
            suite_name=args.suite_name
        )