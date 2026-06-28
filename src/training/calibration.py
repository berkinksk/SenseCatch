"""Confidence calibration (ECE) on the seeded quick IMDB subset.

Measures the Expected Calibration Error for each probability model: when a model
says it is X percent confident, is it right about X percent of the time. The
classical models were calibrated on TRAIN (CalibratedClassifierCV), so the TEST
set is held out from that fit and these numbers are honest. DistilBERT softmax is
not temperature-scaled, so it is expected to be overconfident. Saves
artifacts/calibration.json.
"""
import os
import sys
import json

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
for _p in (PROJECT_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate as ev
import stack_ensemble as se
import finetune_distilbert as fd

CLASSICAL = ["naive_bayes", "logistic_regression", "linear_svc", "nbsvm"]
N_BINS = 15


def ece_bins(p, labels, n_bins=N_BINS):
    """Expected Calibration Error and the per-bin reliability table (Guo et al.)."""
    p = np.asarray(p, dtype=float)
    labels = np.asarray(labels)
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    width = 0.5 / n_bins
    idx = np.minimum(((conf - 0.5) / width).astype(int), n_bins - 1)
    n = len(p)
    ece = 0.0
    table = []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        c = float(conf[m].mean())
        a = float(correct[m].mean())
        ece += (cnt / n) * abs(c - a)
        table.append({"bin": [round(float(edges[b]), 3), round(float(edges[b + 1]), 3)],
                      "conf": round(c, 4), "acc": round(a, 4), "count": cnt})
    return round(ece, 4), table


def main():
    texts, labels = ev.load_imdb_test(max_per_class=1000, seed=42)
    labels = np.asarray(labels)
    print(f"quick subset: {len(texts)} reviews")

    ensemble = ev.load_ensemble()
    processed = ev.preprocess_texts(ensemble, texts)

    ppos = {}
    for m in CLASSICAL:
        X = ev.build_features(ensemble, m, processed)
        ppos[m] = ensemble.models[m].predict_proba(X)[:, 1]
        print(f"{m} done")
    ppos["distilbert"] = fd.predict_proba(texts)[:, 1]
    print("distilbert done")
    ppos["stack"] = se.predict_proba(texts)[:, 1]
    print("stack done")

    order = CLASSICAL + ["distilbert", "stack"]
    results = {m: dict(zip(("ece", "reliability"), ece_bins(ppos[m], labels))) for m in order}

    ranked = sorted(order, key=lambda m: results[m]["ece"])
    best, worst = ranked[0], ranked[-1]

    out = {"subset": {"name": "imdb_quick_seeded", "seed": 42, "n": len(texts)},
           "n_bins": N_BINS, "bin_range": [0.5, 1.0],
           "definition": "ECE (Guo et al.): confidence = max(p, 1-p); 15 equal-width bins on "
                         "[0.5, 1.0]; ECE = sum (bin_count / N) * |mean_conf - acc|",
           "ece": {m: results[m]["ece"] for m in order},
           "reliability_best": {"model": best, "bins": results[best]["reliability"]},
           "reliability_worst": {"model": worst, "bins": results[worst]["reliability"]},
           "models": results}
    out_path = os.path.join(PROJECT_ROOT, "artifacts", "calibration.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\nECE per model (best to worst, lower = better calibrated):")
    for m in ranked:
        print(f"  {m:20s} ECE {results[m]['ece']:.4f}")
    for tag, m in [("BEST", best), ("WORST", worst)]:
        print(f"\nReliability ({tag} = {m}):")
        for row in results[m]["reliability"]:
            print(f"  [{row['bin'][0]:.3f}, {row['bin'][1]:.3f}]  conf {row['conf']:.3f}  "
                  f"acc {row['acc']:.3f}  n {row['count']}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
