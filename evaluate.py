#!/usr/bin/env python3
"""
Standard benchmark evaluation for SenseCatch sentiment analysis system.

Evaluates on the IMDB test set (25,000 reviews) using proper ML evaluation
methodology: accuracy, precision, recall, F1-score, and confusion matrix.

Includes:
  - Per-model evaluation (Naive Bayes, Logistic Regression)
  - Ablation study (VADER -> NB -> LR -> Ensemble -> Full System)
  - Optional DistilBERT comparison baseline

Usage:
    python evaluate.py                 # Full evaluation (25K reviews)
    python evaluate.py --quick         # Quick evaluation (2000 samples)
    python evaluate.py --no-distilbert # Skip DistilBERT comparison
"""

import os
import sys
import json
import random
import time
import shutil
import tarfile
import logging
import argparse
import urllib.request
from datetime import datetime

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_DIR = os.path.join(PROJECT_ROOT, "datasets", "aclImdb")
IMDB_TAR = os.path.join(PROJECT_ROOT, "datasets", "aclImdb_v1.tar.gz")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "evaluation_results.json")
PREPROCESS_VERSION = "v1"  # bump when ensemble._preprocess_text changes -> invalidates the preprocess cache

# ---------------------------------------------------------------------------
# IMDB dataset
# ---------------------------------------------------------------------------


def download_imdb():
    """Download and extract the IMDB dataset if not already present."""
    if os.path.isdir(IMDB_DIR):
        return

    os.makedirs(os.path.join(PROJECT_ROOT, "datasets"), exist_ok=True)

    if not os.path.isfile(IMDB_TAR):
        print("Downloading IMDB dataset (84 MB)...")
        import ssl

        ctx = ssl.create_default_context()
        try:
            import certifi

            ctx.load_verify_locations(certifi.where())
        except (ImportError, Exception):
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(IMDB_URL)
        with urllib.request.urlopen(req, context=ctx) as resp, open(
            IMDB_TAR, "wb"
        ) as out:
            shutil.copyfileobj(resp, out)
        print("Download complete.")

    print("Extracting IMDB dataset...")
    with tarfile.open(IMDB_TAR, "r:gz") as tar:
        tar.extractall(path=os.path.join(PROJECT_ROOT, "datasets"))
    print("Extraction complete.")


def load_imdb_test(max_per_class=None, seed=42):
    """Load the IMDB test split. Returns (texts, labels) where labels are 0/1.

    When max_per_class is set (quick mode), draw a seeded, class-balanced
    random sample per class, not the first-N-sorted slice. The sorted slice
    was about 2 to 4 points optimistic; a random sample is an unbiased,
    reproducible preview of the full test set.
    """
    texts, labels = [], []
    rng = random.Random(seed)

    for sentiment, label in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(IMDB_DIR, "test", sentiment)
        filenames = sorted(os.listdir(folder))
        if max_per_class and max_per_class < len(filenames):
            filenames = sorted(rng.sample(filenames, max_per_class))
        for fname in filenames:
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(label)

    return texts, np.array(labels)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def ensure_models():
    """Make sure model .pkl files are available in models/ directory."""
    models_dir = os.path.join(PROJECT_ROOT, "models")
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts", "models")

    needed = ["naive_bayes.pkl", "logistic_regression.pkl"]
    missing = [f for f in needed if not os.path.isfile(os.path.join(models_dir, f))]

    if not missing:
        return

    if os.path.isdir(artifacts_dir):
        os.makedirs(models_dir, exist_ok=True)
        for f in missing:
            src = os.path.join(artifacts_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(models_dir, f))
                print(f"  Copied {f} from artifacts/models/ to models/")

    still_missing = [
        f for f in needed if not os.path.isfile(os.path.join(models_dir, f))
    ]
    if still_missing:
        print("  Model files not found. Generating fallback models...")
        sys.path.insert(0, PROJECT_ROOT)
        from src.sensecatch.utils.model_consistency import ensure_model_consistency

        ensure_model_consistency()


def load_ensemble():
    """Load and return a SentimentEnsemble instance."""
    sys.path.insert(0, PROJECT_ROOT)
    from src.sensecatch.ensemble_model import SentimentEnsemble

    return SentimentEnsemble()


# ---------------------------------------------------------------------------
# Feature extraction (cached)
# ---------------------------------------------------------------------------


def preprocess_texts(ensemble, texts):
    """Preprocess all texts once using the ensemble's pipeline. Returns list of strings.

    Caches the preprocessed TEXT to disk (model-independent) keyed by a hash of
    (PREPROCESS_VERSION + the input texts). The NLTK pos_tag/ne_chunk pass
    dominates runtime (~0.1s/review -> ~41 min on 25K), so a warm run skips it.
    Feature matrices are deliberately not cached; they change on every retrain.
    """
    import hashlib

    cache_dir = os.path.join(PROJECT_ROOT, "artifacts", "cache")
    key = hashlib.sha256(
        (PREPROCESS_VERSION + "\x00".join(texts)).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"preproc_{key}.json")

    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) == len(texts):
            print(f"  Loaded {len(cached)} preprocessed texts from cache.")
            return cached

    total = len(texts)
    step = max(1, total // 10)  # report every 10%
    print(f"  Preprocessing {total} texts...")
    processed = []
    for i, text in enumerate(texts):
        processed.append(ensemble._preprocess_text(text))
        if (i + 1) % step == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"    {i + 1}/{total} ({pct:.0f}%) preprocessed", flush=True)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(processed, f)
    return processed


def build_features(ensemble, model_name, processed_texts):
    """Build the feature matrix for a specific model from preprocessed texts."""
    vectorizer = ensemble.vectorizers.get(model_name)
    dict_vec = ensemble.dict_vectorizers.get(model_name)
    model = ensemble.models[model_name]

    text_features = vectorizer.transform(processed_texts)

    if dict_vec and ensemble.lexicon:
        lexicon_feats = [
            ensemble.lexicon.extract_all_features(t) for t in processed_texts
        ]
        dict_features = dict_vec.transform(lexicon_feats)
        features = hstack([text_features, dict_features])
    else:
        features = text_features

    # Handle dimension mismatch (pad or trim)
    expected_dim = getattr(model, "n_features_in_", None)
    if expected_dim is None and hasattr(model, "coef_"):
        expected_dim = model.coef_.shape[1]
    if expected_dim and features.shape[1] != expected_dim:
        diff = expected_dim - features.shape[1]
        if diff > 0:
            padding = csr_matrix((features.shape[0], diff))
            features = hstack([features, padding])
        else:
            features = features[:, :expected_dim]

    return features


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(name, y_true, y_pred, neutral_count=0):
    """Compute and print classification metrics. Returns dict of metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n  {'=' * 56}")
    print(f"  {name}")
    print(f"  {'=' * 56}")
    print(f"  Accuracy:  {acc:.4f}  ({int(acc * len(y_true))}/{len(y_true)})")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    if neutral_count > 0:
        print(f"  Neutral predictions (counted as incorrect): {neutral_count}")
    print()
    print(f"  Confusion Matrix:")
    print(f"                   Predicted Neg  Predicted Pos")
    print(f"    Actual Neg     {cm[0][0]:>10}     {cm[0][1]:>10}")
    print(f"    Actual Pos     {cm[1][0]:>10}     {cm[1][1]:>10}")
    print()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "neutral_count": neutral_count,
        "total_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------


def evaluate_vader(texts, labels):
    """VADER lexicon baseline (no ML, no rules)."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()
    total = len(texts)
    step = max(1, total // 10)
    preds = []
    for i, t in enumerate(texts):
        preds.append(1 if vader.polarity_scores(t)["compound"] >= 0.05 else 0)
        if (i + 1) % step == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"    {i + 1}/{total} ({pct:.0f}%) scored", flush=True)
    return compute_metrics("VADER Lexicon Baseline", labels, np.array(preds))


def evaluate_raw_model(ensemble, model_name, features, labels):
    """Evaluate a single sklearn model (no rule-based layer)."""
    model = ensemble.models[model_name]
    preds = model.predict(features)
    display = "Naive Bayes" if model_name == "naive_bayes" else "Logistic Regression"
    return compute_metrics(f"{display} (raw model, no rules)", labels, preds)


def evaluate_raw_ensemble(ensemble, features_nb, features_lr, labels):
    """Evaluate weighted ensemble of NB + LR (no rule-based layer)."""
    nb_probs = ensemble.models["naive_bayes"].predict_proba(features_nb)
    lr_probs = ensemble.models["logistic_regression"].predict_proba(features_lr)

    w_nb = ensemble.model_weights.get("naive_bayes", 0.6)
    w_lr = ensemble.model_weights.get("logistic_regression", 0.4)
    combined = w_nb * nb_probs + w_lr * lr_probs

    preds = np.argmax(combined, axis=1)
    return compute_metrics("Weighted Ensemble (NB + LR, no rules)", labels, preds)


def evaluate_full_system(ensemble, texts, labels):
    """Full system including all rule-based layers."""
    preds = []
    neutral_count = 0

    total = len(texts)
    step = max(1, total // 20)  # report every 5%
    for i, text in enumerate(texts):
        if (i + 1) % step == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"    {i + 1}/{total} ({pct:.0f}%) reviews processed", flush=True)

        result = ensemble.predict(text)
        sent = result["sentiment"]

        if sent == "Positive":
            preds.append(1)
        elif sent == "Negative":
            preds.append(0)
        else:
            # Neutral on binary data, so count it as incorrect
            preds.append(1 - labels[i])
            neutral_count += 1

    return compute_metrics(
        "Full System (ensemble + rule-based layer)",
        labels,
        np.array(preds),
        neutral_count=neutral_count,
    )


def evaluate_distilbert(texts, labels):
    """DistilBERT (SST-2 fine-tuned) as a SOTA reference."""
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        print(
            "\n  [Skipped] DistilBERT requires: pip install transformers torch"
        )
        return None

    # Auto-select the fastest available device: CUDA GPU, then Apple
    # Silicon GPU (MPS), else CPU. Predictions are device-independent.
    if torch.cuda.is_available():
        device = 0
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1

    print(f"  Loading DistilBERT model... (device={device})")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True,
        max_length=512,
    )

    preds = []
    batch_size = 64

    total = len(texts)
    for i in range(0, total, batch_size):
        batch = [t[:2000] for t in texts[i : i + batch_size]]
        results = classifier(batch)
        for r in results:
            preds.append(1 if r["label"] == "POSITIVE" else 0)
        done = min(i + batch_size, total)
        pct = done / total * 100
        if done % max(1, (total // 10)) < batch_size or done == total:
            print(f"    {done}/{total} ({pct:.0f}%) classified", flush=True)

    return compute_metrics(
        "DistilBERT (SST-2 fine-tuned, SOTA reference)", labels, np.array(preds)
    )


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------


def print_ablation_table(rows, distilbert_metrics=None):
    """Print formatted ablation study summary."""
    print(f"\n{'=' * 62}")
    print(f"  ABLATION STUDY: IMDB Test Set")
    print(f"{'=' * 62}")
    print(f"  {'Component':<42} {'Acc':>7}  {'F1':>7}")
    print(f"  {'-' * 42} {'-' * 7}  {'-' * 7}")

    for name, m in rows:
        if m:
            print(f"  {name:<42} {m['accuracy']:>7.4f}  {m['f1']:>7.4f}")
        else:
            print(f"  {name:<42} {'N/A':>7}  {'N/A':>7}")

    if distilbert_metrics:
        print(f"  {'-' * 42} {'-' * 7}  {'-' * 7}")
        print(
            f"  {'DistilBERT (SST-2, SOTA reference)':<42} "
            f"{distilbert_metrics['accuracy']:>7.4f}  "
            f"{distilbert_metrics['f1']:>7.4f}"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Force line-buffered stdout so progress is visible even when piped
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        description="Evaluate SenseCatch on IMDB test set"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation on 1000 samples per class (2000 total)",
    )
    parser.add_argument(
        "--no-distilbert",
        action="store_true",
        help="Skip DistilBERT comparison",
    )
    args = parser.parse_args()

    max_per_class = 1000 if args.quick else None
    total_samples = (max_per_class * 2) if max_per_class else 25000

    print(f"\n SenseCatch Evaluation: IMDB Test Set ({total_samples:,} reviews)")
    print("=" * 62)

    # --- Load data --------------------------------------------------------
    download_imdb()
    print(f"\nLoading IMDB test set...")
    texts, labels = load_imdb_test(max_per_class)
    print(
        f"  {len(texts)} reviews loaded "
        f"({sum(labels == 1)} positive, {sum(labels == 0)} negative)"
    )

    # --- Load models ------------------------------------------------------
    ensure_models()
    print("\nLoading SentimentEnsemble...")
    ensemble = load_ensemble()
    print(f"  Models: {', '.join(ensemble.models.keys())}")

    if not ensemble.models:
        print("ERROR: No models loaded.")
        sys.exit(1)

    # --- Preprocess once (shared across raw model evaluations) ------------
    print("\n[Step 1] Preprocessing texts (shared across all raw model evals)...")
    t0 = time.time()
    processed = preprocess_texts(ensemble, texts)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Build feature matrices once per model ----------------------------
    print("\n[Step 2] Building feature matrices...")
    t0 = time.time()
    features = {}
    for model_name in ensemble.models:
        if ensemble.vectorizers.get(model_name):
            features[model_name] = build_features(ensemble, model_name, processed)
            print(f"  {model_name}: {features[model_name].shape}")
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Evaluations ------------------------------------------------------
    all_results = {}
    ablation = []

    # VADER baseline
    print("\n[3/7] VADER baseline...")
    t0 = time.time()
    m = evaluate_vader(texts, labels)
    all_results["vader"] = m
    ablation.append(("VADER (lexicon-only baseline)", m))
    print(f"  Time: {time.time() - t0:.1f}s")

    # Naive Bayes (raw)
    if "naive_bayes" in features:
        print("\n[4/7] Naive Bayes (raw model)...")
        t0 = time.time()
        m = evaluate_raw_model(ensemble, "naive_bayes", features["naive_bayes"], labels)
        all_results["naive_bayes_raw"] = m
        ablation.append(("+ Naive Bayes (CountVectorizer)", m))
        print(f"  Time: {time.time() - t0:.1f}s")

    # Logistic Regression (raw)
    if "logistic_regression" in features:
        print("\n[5/7] Logistic Regression (raw model)...")
        t0 = time.time()
        m = evaluate_raw_model(
            ensemble, "logistic_regression", features["logistic_regression"], labels
        )
        all_results["logistic_regression_raw"] = m
        ablation.append(("+ Logistic Regression (TF-IDF)", m))
        print(f"  Time: {time.time() - t0:.1f}s")

    # Weighted ensemble (raw)
    if "naive_bayes" in features and "logistic_regression" in features:
        print("\n[6/7] Weighted ensemble (NB + LR)...")
        t0 = time.time()
        m = evaluate_raw_ensemble(
            ensemble, features["naive_bayes"], features["logistic_regression"], labels
        )
        all_results["ensemble_raw"] = m
        ablation.append(("+ Weighted Ensemble (0.6 NB + 0.4 LR)", m))
        print(f"  Time: {time.time() - t0:.1f}s")

    # Full system
    print("\n[7/7] Full system (ensemble + rules)...")
    t0 = time.time()
    m = evaluate_full_system(ensemble, texts, labels)
    all_results["full_system"] = m
    ablation.append(("+ Rule-based layer (full system)", m))
    print(f"  Time: {time.time() - t0:.1f}s")

    # DistilBERT (optional)
    distilbert_metrics = None
    if not args.no_distilbert:
        print("\n[Bonus] DistilBERT (SOTA reference)...")
        t0 = time.time()
        distilbert_metrics = evaluate_distilbert(texts, labels)
        all_results["distilbert"] = distilbert_metrics
        if distilbert_metrics:
            print(f"  Time: {time.time() - t0:.1f}s")

    # --- Summary ----------------------------------------------------------
    print_ablation_table(ablation, distilbert_metrics)

    # --- Save results -----------------------------------------------------
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "IMDB",
        "split": "test",
        "total_samples": len(texts),
        "quick_mode": args.quick,
        "results": all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
