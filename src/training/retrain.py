#!/usr/bin/env python3
"""Train and benchmark the SenseCatch sentiment models.

Trains the four deployed models (Naive Bayes, Logistic Regression, LinearSVC,
NBSVM) on the NLTK movie reviews plus the IMDB training split. Can also train
IMDB-only models used as benchmark references. Tuning uses a held-out dev set;
the IMDB test set is only touched for final scoring. Preprocessing reuses the
serving path so training matches serving.

Usage:
    python src/training/retrain.py                  # report the corpus
    python src/training/retrain.py --imdb-benchmark # IMDB-only benchmark models
    python src/training/retrain.py --train-deployed # the four deployed models
    python src/training/retrain.py --manifest       # write the training manifest
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
    """Raised when a declared training corpus is missing or empty."""


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
        raise CorpusError("NLTK 'movie_reviews' returned 0 fileids; the corpus is empty.")
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
                "Expected datasets/aclImdb/train/{pos,neg}/. Download and extract the IMDB dataset."
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
    """IMDB training split only, as (texts, labels). Used for the benchmark models."""
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
    """Preprocess texts with the serving path (ensemble._preprocess_text), cached on disk.

    Reuses evaluate.preprocess_texts so training and evaluation share one cache.
    Slow on a cold cache (NLTK pos_tag and ne_chunk run per text).
    """
    import evaluate as ev
    if ensemble is None:
        ensemble = ev.load_ensemble()
    return ev.preprocess_texts(ensemble, texts)


# IMDB-only benchmark models (reference numbers, not the deployed models).
# Trained on the IMDB training split, tuned on the dev set, scored once on the
# IMDB test set. Plain sklearn vectorizers on raw text, so it runs fast.
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "artifacts", "benchmark_imdb_only.json")
_C_GRID = [0.1, 1.0, 10.0]


def _wilson_ci(acc, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = (z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def _mcnemar(y_true, pred_a, pred_b):
    """McNemar's test, model_b vs model_a. Returns (a_only_correct, b_only_correct, chi2, p)."""
    import numpy as np
    from scipy.stats import chi2 as _chi2
    yt, pa, pb = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    ca, cb = (pa == yt), (pb == yt)
    b = int(np.sum(ca & ~cb))   # a correct, b wrong
    c = int(np.sum(~ca & cb))   # a wrong, b correct
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    return b, c, float(stat), float(_chi2.sf(stat, 1))


def _metrics(name, y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    acc = float(accuracy_score(y_true, y_pred))
    lo, hi = _wilson_ci(acc, len(y_true))
    return {"model": name, "accuracy": round(acc, 4),
            "acc_ci95": [round(lo, 4), round(hi, 4)],
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
            "n_test": len(y_true)}


def _tune_C_on_dev(make_model, Xtr, ytr, Xdev, ydev, grid):
    """Fit make_model(C) on TRAIN, pick the C with best DEV accuracy (no test peeking)."""
    from sklearn.metrics import accuracy_score
    best_C, best_acc = grid[0], -1.0
    for C in grid:
        m = make_model(C); m.fit(Xtr, ytr)
        a = accuracy_score(ydev, m.predict(Xdev))
        if a > best_acc:
            best_acc, best_C = a, C
    return best_C, best_acc


def benchmark_tfidf_linear(kind, tr_texts, ytr, dev_texts, ydev, te_texts, yte):
    """IMDB-only tuned TF-IDF linear model. kind in {'logreg','linsvc'}.
    Levers vs the OLD pipeline: max_features=50000 (was 10000) and NO stopwords (was 'english')."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3,
                          sublinear_tf=True, stop_words=None, strip_accents="unicode")
    Xtr = vec.fit_transform(tr_texts); Xdev = vec.transform(dev_texts); Xte = vec.transform(te_texts)
    if kind == "logreg":
        make = lambda C: LogisticRegression(C=C, max_iter=1000, solver="liblinear")
        name = "IMDB-only TF-IDF LogisticRegression (tuned, 50k feats, no stopwords)"
    else:
        make = lambda C: LinearSVC(C=C, max_iter=4000)
        name = "IMDB-only TF-IDF LinearSVC (tuned, 50k feats, no stopwords)"
    bestC, devacc = _tune_C_on_dev(make, Xtr, ytr, Xdev, ydev, _C_GRID)
    final = make(bestC); final.fit(Xtr, ytr)
    pred = final.predict(Xte)
    m = _metrics(name, yte, pred)
    m.update({"tuned_C": bestC, "dev_acc": round(devacc, 4), "n_features": Xtr.shape[1]})
    return m, pred


def benchmark_nbsvm(tr_texts, ytr, dev_texts, ydev, te_texts, yte, beta=0.25, use_interpolation=False):
    """Faithful NBSVM (Wang & Manning 2012), IMDB-only: binarized bigrams, NO stopword removal,
    punctuation kept (token_pattern=[^\\s]+); NB log-count-ratio r from TRAIN counts only,
    applied to test; LinearSVC on r-weighted features. (beta interpolation optional, off by default.)"""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    vec = CountVectorizer(binary=True, ngram_range=(1, 2), lowercase=True,
                          token_pattern=r"[^\s]+", min_df=5)
    Xtr = vec.fit_transform(tr_texts); Xdev = vec.transform(dev_texts); Xte = vec.transform(te_texts)
    y = np.asarray(ytr)
    p = 1.0 + np.asarray(Xtr[y == 1].sum(axis=0)).ravel()   # smoothed positive feature counts (TRAIN only)
    q = 1.0 + np.asarray(Xtr[y == 0].sum(axis=0)).ravel()   # smoothed negative feature counts (TRAIN only)
    r = np.log((p / p.sum()) / (q / q.sum())).reshape(1, -1)
    nb = lambda X: X.multiply(r).tocsr()                    # binarized presence * r
    Xtr_nb, Xdev_nb, Xte_nb = nb(Xtr), nb(Xdev), nb(Xte)
    devacc, bestC_val = -1.0, 1.0
    for C in [0.5, 1.0, 5.0]:
        s = LinearSVC(C=C, max_iter=4000); s.fit(Xtr_nb, y)
        a = accuracy_score(ydev, s.predict(Xdev_nb))
        if a > devacc:
            devacc, bestC_val = a, C
    svm = LinearSVC(C=bestC_val, max_iter=4000); svm.fit(Xtr_nb, y)
    pred = svm.predict(Xte_nb)
    name = "IMDB-only NBSVM (faithful: binarized bigrams, no stopwords, r from train; LinearSVC)"
    m = _metrics(name, yte, pred)
    m.update({"tuned_C": bestC_val, "dev_acc": round(devacc, 4), "n_features": Xtr.shape[1],
              "beta_interpolation": (beta if use_interpolation else None)})
    return m, pred


def run_imdb_only_benchmark():
    """Train + evaluate the IMDB-only BENCHMARK models on IMDB TEST once. Saves JSON. REFERENCE only.
    These are NOT the deployed app models (that is 7.7); the app .pkl contract is untouched here."""
    import json
    import time
    print("=== IMDB-only benchmark (reference numbers, not deployed) ===")
    t0 = time.time()
    tr_texts, ytr = load_imdb_only_train()
    dev_texts, ydev = load_imdb_dev()
    te_texts, yte = data_split.load_texts(data_split.get_split()["test"])
    print(f"train {len(ytr)} / dev {len(ydev)} / test {len(yte)} (raw text; no _preprocess_text)")
    results, preds = [], {}
    for kind in ("logreg", "linsvc"):
        m, pred = benchmark_tfidf_linear(kind, tr_texts, ytr, dev_texts, ydev, te_texts, yte)
        results.append(m); preds[m["model"]] = pred
        print(f"  {m['accuracy']}  CI{m['acc_ci95']}  F1 {m['macro_f1']}  C={m['tuned_C']}  | {m['model']}")
    m, pred = benchmark_nbsvm(tr_texts, ytr, dev_texts, ydev, te_texts, yte)
    results.append(m); preds[m["model"]] = pred
    print(f"  {m['accuracy']}  CI{m['acc_ci95']}  F1 {m['macro_f1']}  C={m['tuned_C']}  | {m['model']}")
    comparisons = []
    ordered = sorted(results, key=lambda r: r["accuracy"])  # McNemar each vs the next-best below it
    for i in range(1, len(ordered)):
        a, b = ordered[i - 1]["model"], ordered[i]["model"]
        bb, cc, chi2, pval = _mcnemar(yte, preds[a], preds[b])
        comparisons.append({"comparison": f"{b}  vs  {a}", "b_only_correct": cc, "a_only_correct": bb,
                            "chi2": round(chi2, 2), "p_value": float(f"{pval:.3g}"),
                            "b_significantly_better": bool(pval < 0.05 and cc > bb)})
    best = max(results, key=lambda r: r["accuracy"])
    out = {"track": "IMDB-only benchmark (reference, NOT deployed)", "test_n": len(yte),
           "frontier_target": "~88-91% (Wang & Manning 2012 NBSVM ~91.2%)",
           "models": results, "mcnemar": comparisons,
           "best_model": best["model"], "best_accuracy": best["accuracy"],
           "best_acc_ci95": best["acc_ci95"], "runtime_sec": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)
    with open(BENCHMARK_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"BEST IMDB-only: {best['accuracy']} (CI {best['acc_ci95']}) -> {best['model']}")
    print(f"Saved -> {BENCHMARK_PATH}  | runtime {out['runtime_sec']}s")
    return out


# The four deployed models shown in the app dropdown. Each is wrapped in
# CalibratedClassifierCV (cv=5) so it can output probabilities. Trained on the
# NLTK plus IMDB corpus with the serving preprocessing. Saved as a 3-tuple
# (model, text_vectorizer, dict_vectorizer) that the app loads.
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEPLOY_BACKUP = os.path.join(PROJECT_ROOT, "artifacts", "models_backup_pre-step7-deploy")
DEPLOY_RESULTS = os.path.join(PROJECT_ROOT, "artifacts", "deploy_results.json")
_STALE_VECTORIZER_FILES = (
    "count_vectorizer.pkl", "tfidf_vectorizer.pkl", "dict_vectorizer.pkl",
    "feature_dimensions.pkl", "feature_info.txt",
)


def backup_current_models():
    """Copy the current model files to the backup folder, with their SHA-256 hashes."""
    import shutil, hashlib, glob, json
    os.makedirs(DEPLOY_BACKUP, exist_ok=True)
    hashes = {}
    files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl"))) + sorted(glob.glob(os.path.join(MODELS_DIR, "*.txt")))
    for f in files:
        shutil.copy2(f, DEPLOY_BACKUP)
        with open(f, "rb") as fh:
            hashes[os.path.basename(f)] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(DEPLOY_BACKUP, "HASHES.json"), "w") as fh:
        json.dump(hashes, fh, indent=2)
    return DEPLOY_BACKUP, hashes


def _remove_stale_vectorizer_files():
    """Remove old standalone vectorizer files if present. They would override the
    vectorizers saved with each model and cause silent feature-size mismatches."""
    removed = []
    for name in _STALE_VECTORIZER_FILES:
        p = os.path.join(MODELS_DIR, name)
        if os.path.exists(p):
            os.remove(p); removed.append(name)
    return removed


def _lexicon_dicts(lexicon, processed):
    """9-dim lexicon dict features; negatives clamped to 0 (MNB needs non-negative)."""
    out = []
    for t in processed:
        d = lexicon.extract_all_features(t)
        for k, v in list(d.items()):
            if isinstance(v, (int, float)) and v < 0:
                d[k] = 0.0
        out.append(d)
    return out


def train_deployed_models(n_sample=None):
    """Train + save the FOUR deployed GENERAL models (NB, LR, LinearSVC, NBSVM), all calibrated.
    n_sample != None => DRY-RUN on a tiny subset (small vocab) to validate the pipeline fast.
    Writes artifacts/deploy_results.json and returns the results dict."""
    import json, time, pickle, random
    import numpy as np
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score
    import evaluate as ev
    from src.sensecatch.nbsvm_transformer import NBLogCountRatio

    t0 = time.time()
    backup_current_models()

    corpus = load_training_corpus()
    tr_texts, ytr = corpus["texts"], list(corpus["labels"])
    dev_texts, ydev = load_imdb_dev()
    if n_sample:
        idx = list(range(len(tr_texts))); random.Random(0).shuffle(idx); idx = idx[:n_sample]
        tr_texts = [tr_texts[i] for i in idx]; ytr = [ytr[i] for i in idx]
        k = max(60, n_sample // 4); dev_texts, ydev = dev_texts[:k], ydev[:k]
    ytr = np.asarray(ytr); ydev = np.asarray(ydev)

    ensemble = ev.load_ensemble()
    print(f"[deploy] preprocessing {len(tr_texts)} train + {len(dev_texts)} dev via serving path "
          f"({'DRY-RUN' if n_sample else 'FULL; cold cache ~40min'})...", flush=True)
    tr_proc = ev.preprocess_texts(ensemble, tr_texts)
    dev_proc = ev.preprocess_texts(ensemble, dev_texts)
    lex = ensemble.lexicon

    dv = DictVectorizer()
    Ltr = dv.fit_transform(_lexicon_dicts(lex, tr_proc))
    Ldev = dv.transform(_lexicon_dicts(lex, dev_proc))

    mf = 2000 if n_sample else 30000
    mindf_nbsvm = 2 if n_sample else 5
    specs = [
        ("naive_bayes", CountVectorizer(max_features=mf, ngram_range=(1, 2), min_df=2, stop_words=None),
         (lambda: MultinomialNB(alpha=0.1)), "naive_bayes.pkl"),
        ("logistic_regression", TfidfVectorizer(max_features=mf, ngram_range=(1, 2), min_df=2,
                                                sublinear_tf=True, stop_words=None),
         (lambda: LogisticRegression(C=10.0, max_iter=1000, solver="liblinear")), "logistic_regression.pkl"),
        ("linear_svc", TfidfVectorizer(max_features=mf, ngram_range=(1, 2), min_df=2,
                                       sublinear_tf=True, stop_words=None),
         (lambda: LinearSVC(C=1.0, max_iter=4000)), "linear_svc.pkl"),
        ("nbsvm", Pipeline([("cv", CountVectorizer(binary=True, ngram_range=(1, 2),
                                                   token_pattern=r"[^\s]+", min_df=mindf_nbsvm)),
                            ("nb", NBLogCountRatio())]),
         (lambda: LinearSVC(C=0.5, max_iter=4000)), "nbsvm.pkl"),
    ]

    results, dev_probas = {}, {}
    for name, tv, make_base, fname in specs:
        Xtr_t = tv.fit_transform(tr_proc, ytr) if isinstance(tv, Pipeline) else tv.fit_transform(tr_proc)
        Xdev_t = tv.transform(dev_proc)
        Xtr = hstack([Xtr_t, Ltr]).tocsr(); Xdev = hstack([Xdev_t, Ldev]).tocsr()
        clf = CalibratedClassifierCV(make_base(), cv=5)
        clf.fit(Xtr, ytr)
        dev_acc = float(accuracy_score(ydev, clf.predict(Xdev)))
        with open(os.path.join(MODELS_DIR, fname), "wb") as f:
            pickle.dump((clf, tv, dv), f)
        results[name] = {"dev_accuracy": round(dev_acc, 4), "n_features": int(Xtr.shape[1]), "file": fname}
        if name in ("naive_bayes", "logistic_regression"):
            dev_probas[name] = clf.predict_proba(Xdev)
        print(f"[deploy] {name}: dev_acc {dev_acc:.4f}  ({Xtr.shape[1]} feats)  -> models/{fname}", flush=True)

    # re-tune NB+LR ensemble weights on DEV only
    best_w, best_acc = 0.6, -1.0
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        comb = w * dev_probas["naive_bayes"] + (1 - w) * dev_probas["logistic_regression"]
        a = float(accuracy_score(ydev, np.argmax(comb, axis=1)))
        if a > best_acc:
            best_acc, best_w = a, w

    removed = _remove_stale_vectorizer_files()
    out = {"track": "deployed general (4 calibrated models)", "dry_run": bool(n_sample),
           "n_train": int(len(ytr)), "n_dev": int(len(ydev)), "models": results,
           "ensemble_weights": {"naive_bayes": best_w, "logistic_regression": round(1 - best_w, 2)},
           "ensemble_dev_acc": round(best_acc, 4), "backup_dir": DEPLOY_BACKUP,
           "stale_files_removed": removed, "runtime_sec": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(DEPLOY_RESULTS), exist_ok=True)
    with open(DEPLOY_RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[deploy] ensemble weights (dev-tuned): NB {best_w} / LR {round(1-best_w,2)} (dev acc {best_acc:.4f})")
    print(f"[deploy] saved 4 models; stale vectorizer files removed: {removed or 'none'}; "
          f"results -> {DEPLOY_RESULTS}; runtime {out['runtime_sec']}s")
    return out


def write_training_manifest(path=None):
    """Write a manifest describing how the four deployed models were trained.
    Reads the existing result files and hashes the model files. Does not retrain.
    Output: artifacts/training_manifest.json."""
    import json, hashlib, sys
    from datetime import datetime
    from importlib.metadata import version as _ver

    path = path or os.path.join(PROJECT_ROOT, "artifacts", "training_manifest.json")
    deploy = json.load(open(DEPLOY_RESULTS))
    split = json.load(open(os.path.join(PROJECT_ROOT, "artifacts", "data_split_manifest.json")))

    try:
        import evaluate as _ev
        preprocess_version = getattr(_ev, "PREPROCESS_VERSION", "v1")
    except Exception:
        preprocess_version = "v1"

    def _sha256(fp):
        h = hashlib.sha256()
        with open(fp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _v(pkg):
        try:
            return _ver(pkg)
        except Exception:
            return "unknown"

    imdb_train = split["splits"]["train"]["n"]      # 22500 (seeded IMDB-train split)
    train_total = deploy["n_train"]                 # 24500 = NLTK + IMDB-train

    recipes = {
        "naive_bayes": "CountVectorizer(max_features=30000, ngram(1,2), min_df=2, no stopwords) + lexicon DictVectorizer(9) -> MultinomialNB(alpha=0.1), CalibratedClassifierCV(cv=5)",
        "logistic_regression": "TfidfVectorizer(max_features=30000, ngram(1,2), min_df=2, sublinear_tf, no stopwords) + lexicon DictVectorizer(9) -> LogisticRegression(C=10.0, liblinear), CalibratedClassifierCV(cv=5)",
        "linear_svc": "TfidfVectorizer(max_features=30000, ngram(1,2), min_df=2, sublinear_tf, no stopwords) + lexicon DictVectorizer(9) -> LinearSVC(C=1.0), CalibratedClassifierCV(cv=5)",
        "nbsvm": "Pipeline(CountVectorizer(binary, ngram(1,2), token=non-whitespace runs, min_df=5) -> NBLogCountRatio) + lexicon DictVectorizer(9) -> LinearSVC(C=0.5), CalibratedClassifierCV(cv=5)",
    }

    models = {}
    for name, info in deploy["models"].items():
        fp = os.path.join(MODELS_DIR, info["file"])
        models[name] = {
            "file": "models/" + info["file"],
            "sha256": _sha256(fp) if os.path.exists(fp) else "missing",
            "n_features": info["n_features"],
            "dev_accuracy": info["dev_accuracy"],
            "recipe": recipes.get(name, ""),
        }

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": 42,
        "dev_fraction": 0.10,
        "preprocess_version": preprocess_version,
        "corpus": {
            "train_total": train_total,
            "dev_total": deploy["n_dev"],
            "sources": {
                "nltk_movie_reviews": train_total - imdb_train,
                "imdb_train": imdb_train,
            },
            "balance": "50/50 pos/neg",
            "twitter": False,
        },
        "data_split": {
            "seed": split["seed"],
            "dev_fraction": split["dev_fraction"],
            "splits": split["splits"],
        },
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit-learn": _v("scikit-learn"),
            "numpy": _v("numpy"),
            "scipy": _v("scipy"),
            "nltk": _v("nltk"),
        },
        "models": models,
        "ensemble_weights": deploy.get("ensemble_weights"),
        "ensemble_dev_acc": deploy.get("ensemble_dev_acc"),
        "notes": "DEV accuracies are on the held-out IMDB dev set, not the final test set.",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {path}")
    return manifest


if __name__ == "__main__":
    if "--imdb-benchmark" in sys.argv:
        run_imdb_only_benchmark(); sys.exit(0)
    if "--dry-run-deploy" in sys.argv:
        print("DRY-RUN:", train_deployed_models(n_sample=300)); sys.exit(0)
    if "--train-deployed" in sys.argv:
        print("DEPLOY:", train_deployed_models()); sys.exit(0)
    if "--manifest" in sys.argv:
        write_training_manifest(); sys.exit(0)
    print("=== SenseCatch retrain.py - corpus foundation (no training) ===")
    corpus = load_training_corpus()
    npos = sum(corpus["labels"])
    print(f"GENERAL training corpus: {corpus['n']} docs ({npos} pos / {corpus['n'] - npos} neg)")
    print(f"  - NLTK movie_reviews : {corpus['nltk_n']}")
    print(f"  - IMDB-train (seeded): {corpus['imdb_n']}")
    dev_t, dev_l = load_imdb_dev()
    print(f"DEV (held-out, for tuning): {len(dev_l)} ({sum(dev_l)} pos / {len(dev_l) - sum(dev_l)} neg)")
    print("TEST split is NOT loaded here (touched once, at final eval).")
    print("NO Twitter. Preprocessing = ensemble._preprocess_text (cached, train==serve).")
