# SenseCatch Model Card

SenseCatch is a sentiment analysis web app that labels English reviews as Positive or Negative. It offers seven selectable models: four classical machine learning models (Naive Bayes, Logistic Regression, LinearSVC, NBSVM), a DistilBERT transformer fine-tuned on IMDB, a stacked ensemble that combines them, and an optional rule-based mode.

## Intended use

Binary sentiment of English review text (Positive or Negative), for demos, teaching, and as a study of how strong small classical models can be next to a transformer.

Out of scope:
- Medical, legal, financial, or safety decisions.
- Non-English text. The models are trained on English only.
- Aspect-level or fine-grained emotion analysis. The output is a single binary label.

## The seven models

Every option returns a label, a confidence score, and the model name.

- **Naive Bayes**: MultinomialNB over count features plus a sentiment lexicon, probability-calibrated. Served raw.
- **Logistic Regression**: TF-IDF features plus the lexicon, calibrated. Served raw.
- **LinearSVC**: TF-IDF features plus the lexicon, calibrated. Served raw.
- **NBSVM**: Naive Bayes log-count-ratio features with a linear SVM, calibrated. Served raw.
- **DistilBERT (fine-tuned)**: distilbert-base-uncased fine-tuned on the IMDB training set. Served raw, binary, with no rule layer.
- **Stacked ensemble**: a logistic-regression meta-learner over the five base models' probabilities. Served raw. The most accurate option.
- **Rule-based (optional)**: the original linguistic system (negation, sarcasm, idioms, contrast). Runs the full rule cascade.

The four classical models and the two heavy models (DistilBERT and the stacked ensemble) are served raw, so the app reproduces the benchmark numbers. The rule-based option runs the full linguistic system.

## Training data

- Deployed classical models: NLTK movie_reviews (2,000) plus the IMDB training split (22,500), a balanced positive/negative corpus, with a fixed seed of 42.
- Fine-tuned DistilBERT and the stacked ensemble: the IMDB training split.
- Stack meta-learner: a logistic regression trained on held-out dev-set probabilities from the five base models.
- No Twitter data is used.

## Metrics

Accuracy on three held-out test sets. IMDB is in-domain. SST-2 and Yelp are different domains and measure generalization.

| Model | IMDB | SST-2 | Yelp |
|---|---|---|---|
| VADER (lexicon floor) | 70.0 | 66.2 | 72.0 |
| NBSVM (best classical) | 90.8 | 79.1 | 86.0 |
| DistilBERT (fine-tuned) | 91.2 | 84.4 | 91.3 |
| Stacked ensemble | 93.0 | 85.7 | 92.0 |
| DistilBERT (off-the-shelf, reference) | 89.1 | 91.1* | 91.5 |

\* SST-2 is the off-the-shelf model's own fine-tuning data, so that cell is in-domain and not a fair generalization number.

The stacked ensemble reaches 92.98 on IMDB (Wilson 95% interval 92.66 to 93.29, McNemar p = 2.71e-33), a statistically significant lead over every other model. On SST-2 and Yelp it is still the best model, but the margins are not statistically significant (p = 0.21 and p = 0.26).

Calibration as Expected Calibration Error on a 2,000-review held-out sample (lower is better): Logistic Regression 0.020, LinearSVC 0.021, NBSVM 0.021, DistilBERT 0.032, stacked ensemble 0.031.

## Evaluation methodology

- One seeded split (seed 42) into train, dev, and test. All tuning uses dev only. The test set is touched once, for the final numbers.
- No train/test leakage, checked by hashing the review text across splits.
- A Wilson 95% confidence interval on every accuracy.
- A McNemar test for every model-vs-model and before/after claim.
- Macro-F1 reported alongside accuracy.

## Limitations

- The rule-based mode lowers accuracy on real reviews (the full rule system scores about 70.8 on IMDB) and is kept only as an optional comparison, not a default.
- The fine-tuned DistilBERT truncates input at 256 tokens, so it is weakest on long reviews.
- Cross-domain accuracy drops, since the models are tuned on IMDB and the other test sets come from different domains.
- The published state of the art on IMDB is about 96 percent, which this project does not chase.
- DistilBERT is the least calibrated of the models.

## Safety filter

The rule-based path includes a safety check that flags self-harm or otherwise harmful content and returns a guarded result. It affects only the rule-based option, not the raw model options.

## Reproducibility

- `python reproduce.py` regenerates every artifact the metrics come from, in order.
- `python verify_readme_numbers.py` confirms that every reported number matches the regenerated artifacts.
