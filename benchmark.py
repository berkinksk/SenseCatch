"""Multi-dataset sentiment benchmark.

Scores each component on IMDB, SST-2, and Yelp. For every component we report
accuracy, a Wilson 95% confidence interval, macro F1, and a confusion matrix.
The full system can abstain (Neutral), so it is reported two ways: full
coverage (abstentions resolved by the NB+LR vote) and selective (only the
items it commits to, plus a coverage fraction).

Usage:
    python benchmark.py                 # full sets
    python benchmark.py --quick 40      # tiny smoke, 40 items per dataset
    python benchmark.py --no-distilbert # skip the transformer baseline
"""
import os
import sys
import json
import math
import argparse

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

import evaluate as ev

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "training"))
import benchmark_datasets as bd

RESULTS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "benchmark_multi.json")
RAW_MODELS = ["naive_bayes", "logistic_regression", "linear_svc", "nbsvm"]


def wilson_ci(acc, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def _stats(name, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"component": name, "accuracy": 0.0, "acc_ci95": [0.0, 0.0],
                "macro_f1": 0.0, "confusion_matrix": [[0, 0], [0, 0]], "n": 0}
    acc = float((y_true == y_pred).mean())
    return {
        "component": name,
        "accuracy": round(acc, 4),
        "acc_ci95": wilson_ci(acc, n),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": n,
    }


def vader_preds(texts):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    return [1 if vader.polarity_scores(t)["compound"] >= 0.05 else 0 for t in texts]


def model_preds(ensemble, name, processed):
    feats = ev.build_features(ensemble, name, processed)
    return ensemble.models[name].predict(feats).tolist()


def nb_lr_combined(ensemble, processed):
    nb = ensemble.models["naive_bayes"].predict_proba(
        ev.build_features(ensemble, "naive_bayes", processed))
    lr = ensemble.models["logistic_regression"].predict_proba(
        ev.build_features(ensemble, "logistic_regression", processed))
    w_nb = ensemble.model_weights.get("naive_bayes", 0.6)
    w_lr = ensemble.model_weights.get("logistic_regression", 0.4)
    return w_nb * nb + w_lr * lr, w_nb, w_lr


def distilbert_preds(texts):
    from transformers import pipeline
    import torch
    if torch.cuda.is_available():
        device = 0
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1
    clf = pipeline("sentiment-analysis",
                   model="distilbert-base-uncased-finetuned-sst-2-english",
                   device=device, truncation=True, max_length=512)
    preds = []
    for i in range(0, len(texts), 64):
        batch = [t[:2000] for t in texts[i:i + 64]]
        preds.extend(1 if r["label"] == "POSITIVE" else 0 for r in clf(batch))
    return preds


def score_dataset(ensemble, texts, labels, distilbert_in_domain, no_distilbert):
    labels = [int(x) for x in labels]
    processed = ev.preprocess_texts(ensemble, texts)
    comps = {}

    maj = max(set(labels), key=labels.count)
    comps["majority_baseline"] = _stats("majority_baseline", labels, [maj] * len(labels))
    comps["vader"] = _stats("vader", labels, vader_preds(texts))

    for name in RAW_MODELS:
        comps[name] = _stats(name, labels, model_preds(ensemble, name, processed))

    combined, w_nb, w_lr = nb_lr_combined(ensemble, processed)
    ens_pred = np.argmax(combined, axis=1).tolist()
    comps["ensemble_nb_lr"] = _stats("ensemble_nb_lr", labels, ens_pred)
    comps["ensemble_nb_lr"]["weights"] = {"naive_bayes": w_nb, "logistic_regression": w_lr}

    # Full system. Neutral predictions are abstentions.
    full_pred, sel_true, sel_pred, neutral = [], [], [], 0
    for i, t in enumerate(texts):
        s = ensemble.predict(t)["sentiment"]
        if s == "Positive":
            p = 1
        elif s == "Negative":
            p = 0
        else:
            p = None
        if p is None:
            neutral += 1
            full_pred.append(ens_pred[i])  # resolve abstention with the NB+LR vote
        else:
            full_pred.append(p)
            sel_true.append(labels[i])
            sel_pred.append(p)
    fc = _stats("full_system_full_coverage", labels, full_pred)
    sel = _stats("full_system_selective", sel_true, sel_pred)
    sel["coverage"] = round((len(labels) - neutral) / len(labels), 4) if labels else 0.0
    comps["full_system"] = {"full_coverage": fc, "selective": sel}

    if not no_distilbert:
        d = _stats("distilbert", labels, distilbert_preds(texts))
        d["in_domain"] = bool(distilbert_in_domain)
        comps["distilbert"] = d

    return comps


def main():
    ap = argparse.ArgumentParser(description="Multi-dataset sentiment benchmark")
    ap.add_argument("--quick", type=int, default=0, help="tiny smoke: N items per dataset")
    ap.add_argument("--no-distilbert", action="store_true", help="skip the transformer baseline")
    args = ap.parse_args()

    ensemble = ev.load_ensemble()

    if args.quick:
        imdb = ev.load_imdb_test(max_per_class=max(1, args.quick // 2), seed=42)
        st, sl = bd.load_sst2()
        st, sl = st[:args.quick], sl[:args.quick]
        yelp = bd.load_yelp(args.quick, 42)
    else:
        imdb = ev.load_imdb_test()
        st, sl = bd.load_sst2()
        yelp = bd.load_yelp(2000, 42)

    datasets = {
        "imdb": (imdb[0], list(imdb[1]), "in", False),
        "sst2": (st, sl, "cross", True),
        "yelp": (yelp[0], yelp[1], "cross", False),
    }

    out = {}
    for dname, (texts, labels, domain, db_in) in datasets.items():
        print(f"\n=== {dname} (n={len(texts)}, domain={domain}) ===", flush=True)
        comps = score_dataset(ensemble, texts, labels, db_in, args.no_distilbert)
        out[dname] = {"n": len(texts), "domain": domain, "components": comps}
        for cname, c in comps.items():
            if cname == "full_system":
                print(f"  {cname:20s} full_cov={c['full_coverage']['accuracy']} "
                      f"selective={c['selective']['accuracy']} coverage={c['selective']['coverage']}")
            else:
                extra = " in_domain" if c.get("in_domain") else ""
                print(f"  {cname:20s} acc={c['accuracy']} f1={c['macro_f1']}{extra}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
