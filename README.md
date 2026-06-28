# SenseCatch

*Sentiment analysis with classical models, a fine-tuned DistilBERT, and a stacked ensemble, benchmarked on three datasets.*

SenseCatch is a sentiment analysis web app that labels reviews as Positive or Negative. Live demo: [sensecatch.ai](https://sensecatch.ai).

The aim is a small, interpretable system that runs cheaply on CPU, with each design choice checked against a fine-tuned transformer, a zero-shot model, and an ensemble.

It has seven selectable models: four classical machine learning models, a DistilBERT transformer fine-tuned for this project, a stacked ensemble that combines them, and the original rule-based system. The stacked ensemble reaches **92.98% accuracy on the full IMDB test set** and is the most accurate of the seven. On IMDB its lead over every other model is statistically significant, not measurement noise.

Every number below comes from a seeded pipeline that can be re-run. The write-up also shows where the system falls short, for example the optional rule-based mode that lowers accuracy on long reviews.

## Results

### Retrained classical models

The original models scored around 80% on IMDB because they were trained on a diluted mixed corpus with untuned features. Retraining on a cleaner IMDB-tuned setup raised them to around 90%, measured on the full 25,000-review IMDB test set.

| Model | Before | After |
|---|---|---|
| Naive Bayes | 78.0 | 85.5 |
| Logistic Regression | 80.2 | 90.1 |

The gain came from IMDB-tuned features, a larger vocabulary, no stopword removal, proper probability calibration, no train/test leakage, and dropping a dead Twitter data source.

### Two stronger classical models added

The rework also added two stronger classical models, trained the same way:

| Model | IMDB accuracy |
|---|---|
| LinearSVC | 90.1 |
| NBSVM | 90.8 |

### Benchmark across three datasets

The table reports accuracy on three held-out test sets. IMDB (25,000 reviews) is in-domain, since the models are trained on IMDB. SST-2 (872) and Yelp (2,000) are different domains and measure generalization.

| Model | IMDB | SST-2 | Yelp |
|---|---|---|---|
| VADER (lexicon floor) | 70.0 | 66.2 | 72.0 |
| NBSVM (best classical) | 90.8 | 79.1 | 86.0 |
| DistilBERT (fine-tuned) | 91.2 | 84.4 | 91.3 |
| **Stacked ensemble** | **93.0** | **85.7** | **92.0** |
| DistilBERT (off-the-shelf, reference) | 89.1 | 91.1* | 91.5 |

\* SST-2 is the off-the-shelf model's own fine-tuning data, so that cell is in-domain and not a fair generalization number.

The stacked ensemble is the best model on all three datasets. On IMDB it scores 92.98% (95% Wilson interval 92.66 to 93.29) and beats the fine-tuned DistilBERT by 1.8 points (McNemar p = 2.71e-33). On the smaller SST-2 and Yelp sets it is still best, but the margin is not statistically significant (p = 0.21 and p = 0.26), so the result does not support a real-gain claim there. For reference, the state of the art on IMDB is around 96%, which this project does not try to reach.

One result worth calling out: the strongest classical models (Logistic Regression, LinearSVC, NBSVM) beat the off-the-shelf DistilBERT on IMDB (90.8 vs 89.1). The off-the-shelf model wins on the cross-domain sets. The stacked ensemble is best on all three datasets.

### Fine-tuning vs zero-shot

A zero-shot bart-large-mnli (407M parameters, no task training) scores 87.7% on the 2,000-review IMDB sample. The fine-tuned DistilBERT (66M parameters) scores 89.6% on the same sample. So a model about six times smaller, fine-tuned on the task, beats the larger zero-shot model, and it is also faster on CPU and gives calibrated confidence. The gap is modest, so this is an efficiency point rather than a large accuracy win. (The 91.2% DistilBERT figure in the table is the full 25,000-review test set; the 89.6% here is the 2,000-review sample used for this comparison.)

## Why the ensemble wins

The classical models and the transformer make different mistakes, and the gap is clearest on long reviews. On reviews longer than 150 words (1,210 of a 2,000-review held-out sample):

| Model | Long-review accuracy |
|---|---|
| DistilBERT (fine-tuned) | 87.5 |
| NBSVM | 88.6 |
| Stacked ensemble | 89.8 |

DistilBERT truncates input at 256 tokens, so it loses the end of long reviews. The bag-of-words classical models read the whole text. The stacked ensemble learns when to trust each one, so it is strongest across all review lengths.

### Different models catch different things

Take the sentence "This movie was not bad at all." The raw classical models predict Negative, because the word "bad" pulls them down. The fine-tuned DistilBERT predicts Positive, because it handles the negation. No single model is best on every input, which is the reason to combine them.

## The seven models

- **Naive Bayes, Logistic Regression, LinearSVC, NBSVM** are the four classical models. Each is calibrated, so its confidence scores mean something.
- **DistilBERT (fine-tuned)** is a transformer fine-tuned on the IMDB training set.
- **Stacked ensemble** is a logistic-regression meta-learner trained on the five base models' predictions. A learned combiner can match or beat its best member, and here it pairs the full-text classical models with the context-aware transformer. It is the most accurate option.
- **Rule-based (optional)** is the original linguistic system (negation, sarcasm, idioms, contrast). It helps on specific cases like the negation example above, but on average it lowers accuracy on real reviews (the full rule system scores 70.8% on IMDB versus about 90% for the raw models). It is kept as a selectable comparison point and a record of the original approach, not a recommended default.

At serve time the app runs the text through the selected model and returns the label, a confidence score, and the model name.

## Engineering analysis

Cost on CPU (macOS arm64, single process). Latency is the mean per review through the full serving path.

| Option | Model size | CPU latency |
|---|---|---|
| Classical (NB / LR / LinearSVC / NBSVM) | 2.4 to 12.2 MB | 24 to 28 ms |
| DistilBERT (fine-tuned) | 268.5 MB | 20.4 ms |
| Stacked ensemble | loads all five, about 290 MB | 39.3 ms |
| Rule-based | reuses the four classical, 22.6 MB | 10.9 ms |

Rule-based looks fastest only because it short-circuits most inputs with early rules, before any model runs.

The classical models are 20 to 100 times smaller than DistilBERT and are interpretable, but on CPU they are not faster, because the NLTK preprocessing dominates their latency. A GPU barely changes single-review latency (DistilBERT 20 to 13 ms) but roughly doubles batch throughput (82 to 186 reviews per second), which is why training and benchmarking used the GPU while serving stays on CPU.

Confidence was checked on the 2,000-review held-out sample with Expected Calibration Error (lower is better). The calibrated classical models give trustworthy confidence (Logistic Regression 0.020, LinearSVC 0.021, NBSVM 0.021). DistilBERT is the most overconfident (0.032, raw softmax) and the stacked ensemble sits at 0.031. Temperature scaling DistilBERT is the next step.

## Methodology and reproducibility

- One seeded split (seed 42) into train, dev, and test. All tuning uses dev only. The test set is touched once, for the final numbers.
- No train/test leakage, checked by hashing the review text across splits.
- Every accuracy carries a Wilson 95% confidence interval. Every before/after or model-vs-model claim uses a McNemar significance test. The benchmark also reports macro-F1 alongside accuracy.
- Base transformer: distilbert-base-uncased. Datasets: IMDB (aclImdb), SST-2 (from GLUE), and Yelp polarity.
- Two number sources, labeled throughout: the headline accuracies are the full 25,000-review IMDB test set; the calibration and error-analysis numbers are a separate 2,000-review held-out sample.

The trained models (the .pkl files and the 268 MB DistilBERT directory) are not in git. They are large and fully regenerable, and the training and evaluation scripts are committed. To reproduce from scratch:

```bash
python src/training/data_split.py          # seeded train/dev/test split
python src/training/retrain.py             # the four classical models
python src/training/finetune_distilbert.py # fine-tune DistilBERT (GPU recommended)
python src/training/stack_ensemble.py      # build and evaluate the stacked ensemble
python benchmark.py                        # all 7 options + baselines on IMDB, SST-2, Yelp
```

The classical pipeline is exactly reproducible. DistilBERT fine-tuning is approximately reproducible, with small variation from GPU nondeterminism.

## Limitations and next steps

- The stacked ensemble is below the IMDB state of the art (around 96%). The goal was a measured, well-understood system, not a leaderboard score.
- The ensemble's gains on SST-2 and Yelp are within noise, so the strong claim holds only for IMDB.
- DistilBERT's confidence is overconfident. Temperature scaling is the planned fix.
- The fine-tuned DistilBERT and the stacked ensemble need PyTorch and 268 MB, so the live demo serves only the four classical models and the rule-based mode. A heavier host for the transformer options is still open.

## Live demo and deployment

The frontend is static HTML, CSS, and JavaScript on Vercel. The backend is Flask and Gunicorn on Render, serving the `/analyze` endpoint. The demo at [sensecatch.ai](https://sensecatch.ai) runs the four classical models and the rule-based mode. The transformer and the stacked ensemble run locally, because the production server does not include PyTorch.

## Getting started

```bash
git clone https://github.com/berkinksk/SenseCatch.git
cd SenseCatch
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/sensecatch/app.py
```

A fresh clone has no model files, since they are gitignored. Regenerate them with the commands in the reproducibility section. The four classical models train in a few minutes on CPU. The DistilBERT and stacked ensemble options also need PyTorch.

Basic usage in Python:

```python
from src.sensecatch.ensemble_model import SentimentEnsemble

model = SentimentEnsemble()
result = model.predict("An absolute masterpiece, beautifully acted and deeply moving.", specific_model="logistic_regression")
print(result)
# {'text': 'An absolute masterpiece, beautifully acted and deeply moving.', 'sentiment': 'Positive', 'confidence': 99.99, 'model_used': 'logistic_regression'}
```

## Project structure

```
src/sensecatch/
  app.py                  Flask API (/analyze, /healthz)
  ensemble_model.py       serving for all 7 model options
  sentiment_lexicon.py    lexicon features
  nbsvm_transformer.py    NBSVM feature transform
src/training/
  data_split.py           seeded train/dev/test split
  retrain.py              trains the 4 classical models
  finetune_distilbert.py  fine-tunes DistilBERT
  stack_ensemble.py       builds and evaluates the stacked ensemble
  benchmark_datasets.py   SST-2 and Yelp loaders
  cost_table.py           size and latency measurements
  calibration.py          Expected Calibration Error
  error_analysis.py       error breakdown by length and negation
  zero_shot_baseline.py   zero-shot bart-large-mnli baseline
evaluate.py               IMDB evaluation harness
benchmark.py              multi-dataset benchmark, all 7 options
tests/                    test suite
templates/, static/, vercel_static/   frontend
```

## Tech stack

Python, Flask, scikit-learn, NLTK, NumPy, and SciPy for the app and the classical models. PyTorch and Hugging Face Transformers for the fine-tuned DistilBERT and the stacked ensemble.

## License

MIT License. See [LICENSE](LICENSE).

## Author

Berkin Kaynar
