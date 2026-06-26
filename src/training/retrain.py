#!/usr/bin/env python3
"""
Clean retraining module for SenseCatch — Step 7 (branch: evaluation-upgrade).

Replaces the messy src/training/train_models.py for the Step-7 retrain
(train_models.py stays as historical; the Mar-2025 models are backed up).

Status:
  7.5 (THIS commit) — corpus-loading FOUNDATION only:
    * GENERAL training corpus = NLTK movie_reviews (all) + IMDB-train (the
      seeded 'train' split = 22,500 from data_split.py). NO Twitter (dead
      source).
    * 'dev' (2,500) is reserved for tuning; 'test' (25,000) is NEVER
      touched here (touched once, at final eval).
    * Preprocessing uses the SERVING path (ensemble._preprocess_text via
      evaluate.preprocess_texts, which is cached) so train == serve.
    * FAILS LOUDLY if any declared corpus is missing/empty (the old code
      silently trained without a missing corpus — review C6).
  7.6 (next) — IMDB-ONLY benchmark versions (LinearSVC + faithful NBSVM).
  7.7 (next) — DEPLOYED GENERAL versions: NB, LR, LinearSVC, NBSVM — each
      CalibratedClassifierCV(cv=5) on TRAIN, saved in the app-compatible
      3-tuple format per the MODEL/FEATURE CONTRACT in project-plans.md.
  7.8 — wire LinearSVC + NBSVM into the app + serving smoke test.
  7.9 — training manifest.

Run:
    venv/bin/python src/training/retrain.py    # loads + reports corpus (no training)
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/training
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))      # repo root
for _p in (PROJECT_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_split  # noqa: E402  (src/training/data_split.py — the seeded split, 7.2)


class CorpusError(RuntimeError):
    """A declared training corpus is missing or empty. Fail loudly — never silently skip."""


def load_nltk_movie_reviews():
    """Full NLTK movie_reviews corpus -> (texts, labels); label 1=pos / 0=neg. FAIL LOUDLY if missing."""
    import nltk
    for cand in (os.path.join(PROJECT_ROOT, "nltk_data"),
                 os.path.join(PROJECT_ROOT, "artifacts", "nltk_data")):
        if os.path.isdir(cand) and cand not in nltk.data.path:
            nltk.data.path.insert(0, cand)
    try:
        from nltk.corpus import movie_reviews
        fileids = movie_reviews.fileids()
    except LookupError as e:
        raise CorpusError(
            "NLTK 'movie_reviews' corpus is unavailable. Run "
            "`python src/sensecatch/utils/nltk_downloader.py` or fix the nltk_data path. "
            f"Original: {e}"
        ) from e
    if not fileids:
        raise CorpusError("NLTK 'movie_reviews' returned 0 fileids — corpus is empty.")
    texts, labels = [], []
    for cat in movie_reviews.categories():
        label = 1 if cat == "pos" else 0
        for fid in movie_reviews.fileids(cat):
            texts.append(" ".join(movie_reviews.words(fid)))
            labels.append(label)
    return texts, labels


def _require_imdb_dir():
    """Raise CorpusError unless datasets/aclImdb/train/{pos,neg}/ exist and are non-empty."""
    for sub in ("pos", "neg"):
        d = os.path.join(data_split.IMDB_DIR, "train", sub)
        if not os.path.isdir(d) or not os.listdir(d):
            raise CorpusError(
                f"IMDB training data missing/empty: {d}. "
                "Expected datasets/aclImdb/train/{pos,neg}/ — download/extract the IMDB dataset."
            )


def load_imdb_train():
    """IMDB-train from the seeded 'train' split (22,500) -> (texts, labels). FAIL LOUDLY if missing."""
    _require_imdb_dir()
    items = data_split.get_split()["train"]
    if not items:
        raise CorpusError("IMDB 'train' split is empty.")
    return data_split.load_texts(items)


def load_imdb_dev():
    """Held-out IMDB 'dev' set (2,500) for tuning. (TEST is never loaded here.)"""
    _require_imdb_dir()
    return data_split.load_texts(data_split.get_split()["dev"])


def load_imdb_only_train():
    """IMDB-train ONLY -> (texts, labels). For the Step-7.6 benchmark track."""
    return load_imdb_train()


def load_training_corpus():
    """GENERAL training corpus = NLTK movie_reviews + IMDB-train. NO Twitter. FAIL LOUDLY.

    Returns {'texts','labels','sources','nltk_n','imdb_n','n'}.
    """
    nltk_texts, nltk_labels = load_nltk_movie_reviews()
    imdb_texts, imdb_labels = load_imdb_train()
    texts = nltk_texts + imdb_texts
    labels = nltk_labels + imdb_labels
    sources = ["nltk"] * len(nltk_texts) + ["imdb"] * len(imdb_texts)
    if not texts:
        raise CorpusError("Combined training corpus is empty.")
    return {
        "texts": texts,
        "labels": labels,
        "sources": sources,
        "nltk_n": len(nltk_texts),
        "imdb_n": len(imdb_texts),
        "n": len(texts),
    }


def preprocess(texts, ensemble=None):
    """Preprocess via the SERVING path (ensemble._preprocess_text), cached (7.4), so train == serve.

    Reuses evaluate.preprocess_texts, sharing the on-disk cache with evaluation.
    Heavy on a COLD cache (NLTK pos_tag/ne_chunk per text); 7.6 calls this once.
    """
    import evaluate as ev
    if ensemble is None:
        ensemble = ev.load_ensemble()
    return ev.preprocess_texts(ensemble, texts)


if __name__ == "__main__":
    print("=== SenseCatch retrain.py — 7.5 corpus foundation (no training) ===")
    corpus = load_training_corpus()
    npos = sum(corpus["labels"])
    print(f"GENERAL training corpus: {corpus['n']} docs ({npos} pos / {corpus['n'] - npos} neg)")
    print(f"  - NLTK movie_reviews : {corpus['nltk_n']}")
    print(f"  - IMDB-train (seeded): {corpus['imdb_n']}")
    dev_t, dev_l = load_imdb_dev()
    print(f"DEV (held-out, for tuning): {len(dev_l)} ({sum(dev_l)} pos / {len(dev_l) - sum(dev_l)} neg)")
    print("TEST split is NOT loaded here (touched once, at final eval).")
    print("NO Twitter. Preprocessing = ensemble._preprocess_text (cached, train==serve).")
