# SenseCatch.ai - Advanced Sentiment Analysis Platform

SenseCatch is a sophisticated sentiment analysis platform that uses ensemble machine learning techniques to analyze the emotional tone of text with greater nuance than typical sentiment analysis tools.

## Features

- **Ensemble Model Approach**: Combines multiple sentiment classifiers (Naive Bayes, Logistic Regression) for improved accuracy
- **Neutrality Detection**: Identifies neutral sentiment with dedicated handling for the 40-60% confidence range
- **Named Entity Recognition**: Identifies movie titles and other entities to avoid misclassifying them as sentiment indicators
- **Contrast Marker Handling**: Special processing for sentences with contrast markers like "but" and "however"
- **Negation Handling**: Properly processes negated expressions (e.g., "not bad" → positive)
- **Influential Word Extraction**: Identifies and displays the words that most influenced the sentiment prediction

## Implementation Plan

We're implementing several significant improvements to the sentiment analysis models:

1. ✅ **Enhanced Neutrality Detection**:
   - Added dedicated neutral sentiment handling (confidence range of 40-60%)
   - Created extensive training examples specifically for neutral sentiment
   - Improved handling of mixed sentiment texts with contrast markers

2. ✅ **Improved Named Entity Recognition**:
   - Implemented movie title recognition to avoid misclassifying film names as sentiment indicators
   - Using a dynamic approach that can fetch movie titles from public sources
   - Special handling for sentences containing movie titles

3. 🔄 **Fixing Influential Word Extraction**:
   - Enhanced algorithm to select and display accurate sentiment-driving words
   - Fixed color coding logic for the word chips in the UI
   - Improved negation handling in feature extraction and display

4. 🔄 **Performance Optimization**:
   - Adding caching mechanisms to improve response times
   - Reducing model size by optimizing feature dimensions
   - Improving preprocessing speed with more efficient text handling

5. 🔄 **Better Confidence Calibration**:
   - Implementing confidence adjustments to better reflect uncertainty
   - Different calibration approaches for Naive Bayes vs. Logistic Regression
   - Special handling for contrast markers like "but" and "however"

## Technical Details

- **Backend**: Python with Flask
- **NLP**: NLTK for natural language processing
- **Machine Learning**: scikit-learn for model implementation
- **Frontend**: HTML/CSS/JavaScript

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/berkinksk/SenseCatch.git
   cd SenseCatch
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```
   python nltk_downloader.py
   ```

4. Train the models:
   ```
   python train_models.py
   ```

5. Run the application:
   ```
   python app.py
   ```

## Usage

1. Access the web interface at `http://localhost:5000`
2. Enter text in the input field
3. Select the model type (or use the ensemble)
4. Click "Analyze Sentiment" to see the results

## Deployment

The application is deployed on render.com:
- Main branch: [sensecatch.ai](https://sensecatch.ai)
- Model improvements branch: [SenseCatch-(model-improvements branch testing)](https://sensecatch-model-improvements-branch-testing.onrender.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Berkin Keske - Software Engineering Student at Eastern Mediterranean University
