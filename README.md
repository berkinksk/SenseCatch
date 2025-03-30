# SenseCatch - Advanced Sentiment Analysis

SenseCatch is a sophisticated sentiment analysis system designed to understand and analyze text sentiment with high accuracy, especially for challenging linguistic patterns like sarcasm, negation, and mixed sentiments.

## Features

- **Ensemble Learning Architecture**: Combines multiple machine learning models (Naive Bayes, Logistic Regression) with independent pipelines and confidence calibration
- **Advanced Text Understanding**: Handles complex language patterns including:
  - Negation handling (e.g., "This isn't bad")
  - Sarcasm detection using lexical incongruity patterns
  - Contrast markers (e.g., "despite", "however", "but")
  - Idiom recognition with semantic override capabilities
  - Mixed sentiment analysis
- **Rule-Based Enhancements**: Augments statistical models with explicit linguistic rules for improved accuracy on edge cases
- **Comprehensive Testing Framework**: Rigorous validation with extensive test cases and diagnostic tools
- **Performance Visualization**: Detailed graphical reports on model performance

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (install via pip):
  ```
  pip install numpy pandas matplotlib scikit-learn nltk vader-sentiment
  ```

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/berkinksk/SenseCatch.git
   cd SenseCatch
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Basic Sentiment Analysis

```python
from ensemble_model import SentimentEnsemble

# Initialize the model
model = SentimentEnsemble()

# Predict sentiment
result = model.predict("This movie isn't bad at all, I actually enjoyed most of it.")
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']}%")
```

#### Running Tests

The project includes a comprehensive testing framework to evaluate model performance:

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --categories negation,sarcasm

# Test specific models
python run_tests.py --models logistic_regression

# Generate visualizations
python run_tests.py --visualize

# Compare with previous results
python run_tests.py --compare latest
```

## Testing Framework

SenseCatch includes an advanced testing and visualization framework:

### Test Components

- **test_improvements.py**: Main testing script with diagnostics
- **test_visualizer.py**: Generates visual performance reports
- **run_tests.py**: Command-line interface for running tests

### Key Testing Features

- **Category-based Testing**: Test specific types of language patterns
- **Diagnostic Tracing**: Track text processing through the pipeline
- **Performance Metrics**: Accuracy by category, confidence analysis
- **Result Visualization**: Charts and reports for easy analysis

### Visualization Reports

The testing framework generates comprehensive HTML reports with:

- Overall accuracy charts
- Category-specific performance
- Confidence-accuracy relationship
- Confusion matrices
- Pipeline timing analysis

## Technical Implementation

### Advanced ML Components

- **Dual-Model Architecture**:
  - Probabilistic generative model (Naive Bayes) with Laplace smoothing
  - Discriminative linear classifier (Logistic Regression) with L2 regularization
  - Independent feature extraction pipelines for each model type
  - Custom calibration techniques for confidence estimation

- **Linguistic Pattern Processing**:
  - Advanced negation scope detection with context window analysis
  - Double and triple negation resolution algorithms
  - Sarcasm detection with pattern recognition
  - Idiom recognition with semantic override capabilities

## Empirical Evaluation

Our recent test results from the diagnostic framework show:

| Model | Overall Accuracy | Challenging Cases |
|-------|-----------------|-------------------|
| Naive Bayes | 81.25% | Variable performance on negation (57%) |
| Logistic Regression | 81.25% | Strong on sarcasm (100%), weaker on contradiction |
| Combined Analysis | - | Perfect performance on idioms and conclusion markers |

Performance varies significantly by linguistic category:
- Sarcasm detection: ~100% accuracy
- Idiom recognition: ~100% accuracy
- Negation handling: ~57% accuracy
- Contrast handling: ~67% accuracy
- Contradiction cases: Needs improvement

## Project Structure

- `ensemble_model.py`: Core sentiment analysis engine with `SentimentEnsemble` class
- `train_models.py`: Model training scripts
- `test_improvements.py`: Testing framework with diagnostics (`DiagnosticEnsemble` class)
- `test_visualizer.py`: Visualization generator
- `run_tests.py`: Command-line test runner
- `data/`: Training and validation datasets
- `test_results/`: Test outputs and reports

## Contributing

Contributions are welcome! Key areas for improvement:

1. Adding more test cases for challenging linguistic patterns
2. Enhancing the diagnostic capabilities
3. Improving the visualization features
4. Adding support for additional languages
5. Extending model capabilities for domain-specific sentiment analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK and scikit-learn libraries
- VADER sentiment analysis tool
- All contributors who helped improve the model accuracy

## Author

Berkin Kaynar