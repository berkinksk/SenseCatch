# SenseCatch

A sentiment analysis engine that combines an ensemble of ML classifiers with rule-based linguistic pattern detection to handle challenging cases like sarcasm, negation, and mixed sentiment.

**Live Demo:** [sensecatch.ai](https://sensecatch.ai)

## How It Works

SenseCatch goes beyond bag-of-words classification. It runs text through a multi-stage prediction pipeline where each stage can short-circuit with a high-confidence result, and only ambiguous cases reach the full ensemble inference:

```
Input Text
  │
  ├─ 1. Simple Case Detection ──────────────── (50+ phrase patterns, immediate return)
  │
  ├─ 2. Text Preprocessing
  │     ├─ Named entity recognition (movie titles, products)
  │     ├─ Negation scope detection (5-word context window)
  │     └─ Contrast marker identification
  │
  ├─ 3. Linguistic Pattern Detection
  │     ├─ Sarcasm detection (6 pattern categories)
  │     ├─ Idiom recognition (21 idiom patterns)
  │     ├─ Contradiction resolution
  │     └─ Neutral sentiment detection
  │
  ├─ 4. Dual-Model Ensemble Inference
  │     ├─ Naive Bayes (CountVectorizer) ──── weight: 0.6
  │     ├─ Logistic Regression (TF-IDF) ──── weight: 0.4
  │     └─ Weighted probability averaging + confidence calibration
  │
  └─ Output: { sentiment, confidence, model_used, influential_words }
```

The early-exit design keeps latency low for obvious cases while reserving the full pipeline for genuinely ambiguous text.

## Key Technical Details

### Ensemble Architecture

Two classifiers with independent feature extraction pipelines:

- **Naive Bayes** with CountVectorizer and Laplace smoothing — good at capturing word presence patterns
- **Logistic Regression** with TF-IDF and L2 regularization — better at weighting term importance

Predictions are combined via weighted probability averaging (60/40 split), producing calibrated confidence scores between 50–95%.

### Feature Engineering

Each input produces a ~10,000-dimensional feature vector built from multiple sources:

| Source | Dimensions | What It Captures |
|--------|-----------|-----------------|
| TF-IDF / Count vectorization | ~10,000 | Word importance and frequency patterns |
| VADER sentiment scores | 5 | Lexicon-based sentiment (handles emoticons, intensifiers) |
| SentiWordNet scores | 2 | WordNet synset-level sentiment averaging |
| Custom domain lexicon | 2 | Domain-specific terms (67 movie/review terms with manual scores) |

Text and lexicon features are fused via `scipy.sparse.hstack` to keep memory usage efficient at scale.

### Linguistic Pattern Detection

Beyond statistical classification, the system applies rule-based detection for patterns that classifiers typically struggle with:

- **Negation handling** — Identifies negation triggers ("not", "never", "don't") and marks affected words within a 5-word scope. Handles sentence boundaries, double negations, and 20+ special phrases like "not bad at all"
- **Sarcasm detection** — Six pattern categories covering sleep/boredom sarcasm ("perfect if you enjoy falling asleep"), mocking praise, negative comparisons, and more
- **Contrast markers** — Splits text at markers ("but", "however", "despite") with weighted analysis (45% before, 55% after the marker). Dynamically adjusts weights based on term strength
- **Idiom recognition** — 21 idiom patterns ("guilty pleasure", "waste of time", "worth every penny") that override model output with known sentiment
- **Domain adaptation** — Detects restaurant reviews via food/service term counting and adjusts analysis accordingly. Recognizes movie titles via NER and a cached title database

### Training Data

Models are trained on a combined corpus from three sources:

- **NLTK Movie Reviews** — ~2,000 labeled reviews
- **IMDB Large Movie Review Dataset** — up to 25,000 reviews
- **Twitter Sentiment Dataset** — 20,000 tweets for short-form text coverage

Training includes negation-aware preprocessing (appending `_NEG` tokens to negated words) and contrast marker tokenization.

## Evaluation

Results from the internal diagnostic test suite (32 hand-crafted examples targeting specific linguistic phenomena):

| Model | Accuracy | Strengths | Weaknesses |
|-------|----------|-----------|------------|
| Naive Bayes | 81.25% | Simple sentiment, idioms | Negation (57%) |
| Logistic Regression | 81.25% | Sarcasm, conclusion markers | Contradiction |

| Category | Accuracy |
|----------|----------|
| Sarcasm detection | Strong |
| Idiom recognition | Strong |
| Contrast handling | ~67% |
| Negation handling | ~57% |

> These results are from a targeted internal test suite, not standard benchmarks. The test cases are intentionally difficult — they focus on edge cases like sarcasm, double negation, and mixed sentiment that most sentiment tools get wrong.

## Deployment

SenseCatch runs a split architecture designed to keep the frontend fast and the backend focused on ML inference:

- **Frontend** — Static HTML/CSS/JS served via Vercel CDN (zero cold start)
- **Backend API** — Flask + Gunicorn on Render, serving the `/analyze` endpoint
- **Automatic environment detection** — Production traffic hits `api.sensecatch.ai`, preview deployments route to a staging backend

The frontend and backend deploy independently; pushing to `main` triggers both.

## Getting Started

```bash
# Clone and set up
git clone https://github.com/berkinksk/SenseCatch.git
cd SenseCatch
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download NLP resources
python src/sensecatch/utils/nltk_downloader.py

# Generate models if not present
python src/sensecatch/utils/model_consistency.py

# Run the app locally
python src/sensecatch/app.py
```

**Run tests:**
```bash
python tests/run_tests.py
python tests/run_tests.py --categories negation,sarcasm
python tests/run_tests.py --visualize
```

**Basic usage in Python:**
```python
from src.sensecatch.ensemble_model import SentimentEnsemble

model = SentimentEnsemble()
result = model.predict("The acting was terrible but the storyline kept me hooked.")
print(result)  # {'sentiment': 'Positive', 'confidence': 72.5, 'model_used': '...', ...}
```

## Project Structure

```
src/sensecatch/
├── app.py                  # Flask API (routes, CORS, error handling)
├── ensemble_model.py       # Core SentimentEnsemble class (2,200+ lines)
├── sentiment_lexicon.py    # VADER, SentiWordNet, and custom lexicon features
└── utils/
    ├── model_consistency.py    # Fallback model generation
    ├── nltk_downloader.py      # NLTK resource setup
    └── ...
src/training/
├── train_models.py         # Multi-dataset training pipeline
└── dataset_augmentation.py # Data augmentation utilities
tests/
├── test_improvements.py    # Diagnostic test suite with tracing
├── test_visualizer.py      # HTML report generation
└── run_tests.py            # CLI test runner
```

## Tech Stack

Python 3.9 | Flask | scikit-learn | NLTK | NumPy | SciPy | VADER | SentiWordNet

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Berkin Kaynar**
