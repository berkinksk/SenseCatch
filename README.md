# SenseCatch

*Sentiment analysis with classical models, a fine-tuned DistilBERT, and a stacked ensemble, benchmarked on three datasets.*

SenseCatch is a deployable sentiment web app (live at [sensecatch.ai](https://sensecatch.ai)) that labels reviews as Positive or Negative. The project builds small, interpretable models that run cheaply on CPU, and tests whether that is enough by benchmarking them against a fine-tuned DistilBERT, a zero-shot classifier, and a stacked ensemble.

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

The table reports accuracy on three held-out test sets. IMDB (25,000 reviews) [1] is in-domain, since the models are trained on IMDB. SST-2 (872) [2], from the GLUE benchmark [3], and Yelp (2,000) [4] are different domains and measure generalization.

| Model | IMDB | SST-2 | Yelp |
|---|---|---|---|
| VADER (lexicon floor) [5] | 70.0 | 66.2 | 72.0 |
| NBSVM (best classical) | 90.8 | 79.1 | 86.0 |
| DistilBERT (fine-tuned) | 91.2 | 84.4 | 91.3 |
| **Stacked ensemble** | **93.0** | **85.7** | **92.0** |
| DistilBERT (off-the-shelf, reference) | 89.1 | 91.1* | 91.5 |

\* SST-2 is the off-the-shelf model's own fine-tuning data, so that cell is in-domain and not a fair generalization number.

The stacked ensemble is the best model on all three datasets. On IMDB it scores 92.98% (95% Wilson interval 92.66 to 93.29) and beats the fine-tuned DistilBERT by 1.8 points (McNemar p = 2.71e-33). On the smaller SST-2 and Yelp sets it is still best, but the margin is not statistically significant (p = 0.21 and p = 0.26), so the result does not support a real-gain claim there. For reference, the state of the art on IMDB is around 96%, which this project does not try to reach.

One result worth calling out: the strongest classical models (Logistic Regression, LinearSVC, NBSVM) beat the off-the-shelf DistilBERT on IMDB (90.8 vs 89.1). The off-the-shelf model wins on the cross-domain sets.

### Fine-tuning vs zero-shot

Two questions: does fine-tuning help, and is a small fine-tuned model worth more than a large off-the-shelf one. Both use the same 2,000-review IMDB sample. The zero-shot models classify by entailment [6], with no task-specific training.

| Model | Size | Setup | Accuracy |
|---|---|---|---|
| DistilBERT, zero-shot (MNLI) | 66M | no task training | 71.8 |
| bart-large-mnli, zero-shot | 407M | no task training | 87.7 |
| DistilBERT, fine-tuned | 66M | fine-tuned on IMDB | 89.6 |

Same architecture, fine-tuning lifts the 66M DistilBERT [7] from 71.8 to 89.6 (the value of fine-tuning). The fine-tuned 66M model also beats the 407M bart-large-mnli [8] (the value of a small, task-specific model), and it is faster on CPU and better calibrated. The 89.6 here is on the 2,000-review sample; the 91.2 in the main table is the full 25,000-review test set.

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

- **Naive Bayes, Logistic Regression, LinearSVC, NBSVM [9]** are the four classical models. Each is calibrated, so a stated confidence is close to how often the model is actually right.
- **DistilBERT (fine-tuned)** is a transformer fine-tuned on the IMDB training set.
- **Stacked ensemble** is a logistic-regression meta-learner [10] trained on the five base models' predictions. A learned combiner can match or beat its best member [11], and here it pairs the full-text classical models with the context-aware transformer. It is the most accurate option.
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

Confidence was checked on the 2,000-review held-out sample with Expected Calibration Error [12] (lower is better). The calibrated classical models give trustworthy confidence (Logistic Regression 0.020, LinearSVC 0.021, NBSVM 0.021). DistilBERT is the most overconfident (0.032, raw softmax) and the stacked ensemble sits at 0.031.

## Methodology and reproducibility

- One seeded split (seed 42) into train, dev, and test. All tuning uses dev only. The test set is touched once, for the final numbers.
- No train/test leakage, checked by hashing the review text across splits.
- Every accuracy carries a Wilson 95% confidence interval [13]. Every before/after or model-vs-model claim uses a McNemar significance test [14]. The benchmark also reports macro-F1 alongside accuracy.
- Base transformer: distilbert-base-uncased. Datasets: IMDB (aclImdb), SST-2 (from GLUE), and Yelp polarity.
- Two number sources, labeled throughout: the headline accuracies are the full 25,000-review IMDB test set; the calibration and error-analysis numbers are a separate 2,000-review held-out sample.

The trained models (the .pkl files and the 268 MB DistilBERT directory) are not in git. They are large and fully regenerable, and the training and evaluation scripts are committed. To reproduce from scratch:

```bash
python src/training/data_split.py          # seeded train/dev/test split
python src/training/retrain.py --train-deployed  # the four classical models
python src/training/finetune_distilbert.py # fine-tune DistilBERT (GPU recommended)
python src/training/stack_ensemble.py      # build and evaluate the stacked ensemble
python benchmark.py                        # all 7 options + baselines on IMDB, SST-2, Yelp
```

Running `python reproduce.py` regenerates every artifact above in order, and `python verify_readme_numbers.py` then checks that every reported number matches the regenerated artifacts.

The classical pipeline is exactly reproducible. DistilBERT fine-tuning is approximately reproducible, with small variation from GPU nondeterminism.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, training data, per-dataset metrics, and limitations.

## Live demo and deployment

The frontend is static HTML, CSS, and JavaScript on Vercel. The backend runs as a Docker container on Hugging Face Spaces, serving the `/analyze` endpoint with all seven models, including the fine-tuned DistilBERT and the stacked ensemble. The demo at [sensecatch.ai](https://sensecatch.ai) calls that backend. The free Space sleeps after a period of inactivity, so the first request after idle takes a few seconds to wake it.

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
  zero_shot_baseline.py   zero-shot bart and distilbert baselines
evaluate.py               IMDB evaluation harness
benchmark.py              multi-dataset benchmark, all 7 options
reproduce.py              regenerates every benchmark artifact in order
verify_readme_numbers.py  checks the README numbers against the saved artifacts
tests/                    test suite
templates/, static/, vercel_static/   frontend
```

## Tech stack

Python, Flask, scikit-learn [15], NLTK [16], NumPy, and SciPy for the app and the classical models. PyTorch and Hugging Face Transformers [17] for the fine-tuned DistilBERT and the stacked ensemble.

## References

[1] A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts, "Learning Word Vectors for Sentiment Analysis," in Proc. ACL, 2011, pp. 142-150. https://aclanthology.org/P11-1015/

[2] R. Socher et al., "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank," in Proc. EMNLP, 2013, pp. 1631-1642. https://aclanthology.org/D13-1170/

[3] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding," arXiv:1804.07461, 2018. https://arxiv.org/abs/1804.07461

[4] X. Zhang, J. Zhao, and Y. LeCun, "Character-level Convolutional Networks for Text Classification," in Proc. NeurIPS, 2015, pp. 649-657. https://arxiv.org/abs/1509.01626

[5] C. J. Hutto and E. Gilbert, "VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text," in Proc. ICWSM, 2014, pp. 216-225. https://ojs.aaai.org/index.php/ICWSM/article/view/14550

[6] W. Yin, J. Hay, and D. Roth, "Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach," in Proc. EMNLP-IJCNLP, 2019. https://arxiv.org/abs/1909.00161

[7] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019. https://arxiv.org/abs/1910.01108

[8] M. Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension," in Proc. ACL, 2020, pp. 7871-7880. https://aclanthology.org/2020.acl-main.703/

[9] S. Wang and C. D. Manning, "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification," in Proc. ACL, 2012, pp. 90-94. https://aclanthology.org/P12-2018/

[10] D. H. Wolpert, "Stacked Generalization," Neural Networks, vol. 5, no. 2, pp. 241-259, 1992. https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231

[11] M. J. van der Laan, E. C. Polley, and A. E. Hubbard, "Super Learner," Statistical Applications in Genetics and Molecular Biology, vol. 6, no. 1, 2007. https://biostats.bepress.com/ucbbiostat/paper222/

[12] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proc. ICML, 2017. https://arxiv.org/abs/1706.04599

[13] E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference," J. Amer. Stat. Assoc., vol. 22, no. 158, pp. 209-212, 1927. https://www.tandfonline.com/doi/abs/10.1080/01621459.1927.10502953

[14] T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," Neural Computation, vol. 10, no. 7, pp. 1895-1923, 1998. https://direct.mit.edu/neco/article-abstract/10/7/1895/6224

[15] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011. https://jmlr.org/papers/v12/pedregosa11a.html

[16] S. Bird and E. Loper, "NLTK: The Natural Language Toolkit," in Proc. ACL Interactive Poster and Demonstration Sessions, 2004. https://aclanthology.org/P04-3031/

[17] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in Proc. EMNLP System Demonstrations, 2020, pp. 38-45. https://aclanthology.org/2020.emnlp-demos.6/

## License

MIT License. See [LICENSE](LICENSE).

## Author

Berkin Kaynar
