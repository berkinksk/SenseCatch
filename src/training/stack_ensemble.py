"""Stacked ensemble for sentiment.

A logistic-regression meta-learner is trained on held-out dev probabilities from
the base models, then evaluated on the held-out test sets. Base models, in fixed
column order: naive_bayes, logistic_regression, linear_svc, nbsvm, distilbert.
"""
import os
import sys
import json
import math
import pickle
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
for _p in (PROJECT_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate as ev
import benchmark_datasets as bd
import data_split
import finetune_distilbert as fd

BASE_NAMES = ["naive_bayes", "logistic_regression", "linear_svc", "nbsvm", "distilbert"]
CLASSICAL = ["naive_bayes", "logistic_regression", "linear_svc", "nbsvm"]
STACK_PATH = os.path.join(PROJECT_ROOT, "models", "stack_ensemble.pkl")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "stack_results.json")


def wilson_ci(acc, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def base_ppos(ensemble, texts, distilbert_dir=None):
    """Return an (n x 5) array of P(positive), one column per base model."""
    proc = ev.preprocess_texts(ensemble, texts)
    cols = []
    for name in CLASSICAL:
        X = ev.build_features(ensemble, name, proc)
        cols.append(ensemble.models[name].predict_proba(X)[:, 1])
    cols.append(fd.predict_proba(texts, model_dir=distilbert_dir or fd.OUTPUT_DIR)[:, 1])
    return np.column_stack(cols)


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "acc_ci95": [0.0, 0.0], "macro_f1": 0.0, "n": 0}
    acc = float((y_true == y_pred).mean())
    return {"accuracy": round(acc, 4), "acc_ci95": wilson_ci(acc, n),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4), "n": n}


def mcnemar(y_true, pred_a, pred_b):
    """McNemar for pred_a vs pred_b. b = a right and b wrong, c = a wrong and b right."""
    from scipy.stats import chi2 as chi2_dist
    yt, pa, pb = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    ca, cb = (pa == yt), (pb == yt)
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    return {"b": b, "c": c, "chi2": round(float(chi2), 3),
            "p_value": float(f"{float(chi2_dist.sf(chi2, 1)):.3g}")}


def _fit_meta(Xdev, ydev, col_idx):
    """Fit a logistic-regression meta-learner on the chosen columns. Pick C by dev cross-val."""
    Xsub = Xdev[:, col_idx]
    best_C, best_cv = 1.0, -1.0
    for C in [0.1, 1.0, 10.0]:
        cv = cross_val_score(LogisticRegression(C=C, max_iter=1000), Xsub, ydev, cv=5).mean()
        if cv > best_cv:
            best_cv, best_C = cv, C
    clf = LogisticRegression(C=best_C, max_iter=1000).fit(Xsub, ydev)
    return clf, float(best_cv), best_C


def build_stack(Xdev, ydev):
    """Try the full base set and pruned subsets. Keep the one with the best dev cross-val."""
    idx = {n: i for i, n in enumerate(BASE_NAMES)}

    best_classical, best_cv = CLASSICAL[0], -1.0
    for name in CLASSICAL:
        _, cv, _ = _fit_meta(Xdev, ydev, [idx[name]])
        if cv > best_cv:
            best_cv, best_classical = cv, name

    candidates = {
        "all5": list(range(5)),
        "distilbert+nbsvm": [idx["distilbert"], idx["nbsvm"]],
        "distilbert+" + best_classical: [idx["distilbert"], idx[best_classical]],
    }

    chosen = (-1.0, None, None, None, 1.0)
    cv_by_set = {}
    for label, cols in candidates.items():
        clf, cv, C = _fit_meta(Xdev, ydev, cols)
        cv_by_set[label] = round(cv, 4)
        if cv > chosen[0]:
            chosen = (cv, label, cols, clf, C)
    cv, label, cols, clf, C = chosen
    return {"clf": clf, "cols": cols, "col_names": [BASE_NAMES[i] for i in cols],
            "label": label, "dev_cv_acc": round(cv, 4), "C": C, "cv_by_set": cv_by_set}


def evaluate_all(ensemble, stack, datasets, distilbert_dir=None):
    out = {}
    for dname, (texts, labels) in datasets.items():
        labels = [int(x) for x in labels]
        P = base_ppos(ensemble, texts, distilbert_dir)
        comp, base_preds = {}, {}
        for i, name in enumerate(BASE_NAMES):
            pred = (P[:, i] >= 0.5).astype(int).tolist()
            base_preds[name] = pred
            comp[name] = _metrics(labels, pred)
        stack_pred = stack["clf"].predict_proba(P[:, stack["cols"]]).argmax(1).tolist()
        comp["stack"] = _metrics(labels, stack_pred)

        best_name = max(BASE_NAMES, key=lambda n: comp[n]["accuracy"])
        mc = mcnemar(labels, base_preds[best_name], stack_pred)
        beats_all = all(comp["stack"]["accuracy"] > comp[n]["accuracy"] for n in BASE_NAMES)
        significant = mc["p_value"] < 0.05 and mc["c"] > mc["b"]
        gate = "PASS" if beats_all and significant else "FAIL"
        out[dname] = {"n": len(labels), "components": comp, "best_individual": best_name,
                      "mcnemar_stack_vs_best": mc, "gate": gate}
        print(f"  {dname}: stack={comp['stack']['accuracy']} "
              f"best={best_name}({comp[best_name]['accuracy']}) "
              f"mcnemar b={mc['b']} c={mc['c']} p={mc['p_value']} gate={gate}", flush=True)
    return out


def load_stack(path=STACK_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba(texts, stack_path=STACK_PATH, distilbert_dir=None):
    stack = load_stack(stack_path)
    ensemble = ev.load_ensemble()
    P = base_ppos(ensemble, texts, distilbert_dir)
    return stack["clf"].predict_proba(P[:, stack["cols"]])


def predict(texts, stack_path=STACK_PATH, distilbert_dir=None):
    return [int(x) for x in predict_proba(texts, stack_path, distilbert_dir).argmax(1)]


DEV_NOTE = ("Meta-learner trained on dev base-probabilities. Dev was also used for base-model "
            "selection, so dev is a mild-optimism blend set. The test numbers are the clean truth.")


def main():
    ap = argparse.ArgumentParser(description="Stacked sentiment ensemble")
    ap.add_argument("--smoke", action="store_true", help="tiny end to end check")
    args = ap.parse_args()

    ensemble = ev.load_ensemble()

    if args.smoke:
        import tempfile, shutil
        dev_texts, dev_labels = data_split.load_texts(data_split.get_split()["dev"])
        dev_texts, dev_labels = dev_texts[:150], dev_labels[:150]
        db_dir = tempfile.mkdtemp(prefix="db_smoke_")
        try:
            fd.finetune(output_dir=db_dir, smoke=True)
            Xdev = base_ppos(ensemble, dev_texts, distilbert_dir=db_dir)
            stack = build_stack(Xdev, np.array(dev_labels))
            print("chosen base set:", stack["label"], "dev_cv", stack["dev_cv_acc"],
                  "cv_by_set", stack["cv_by_set"])
            imdb_t, imdb_l = ev.load_imdb_test(max_per_class=30, seed=42)
            st, sl = bd.load_sst2()
            yt, yl = bd.load_yelp(60, 42)
            datasets = {"imdb": (imdb_t, list(imdb_l)), "sst2": (st[:60], sl[:60]), "yelp": (yt, yl)}
            evaluate_all(ensemble, stack, datasets, distilbert_dir=db_dir)
        finally:
            shutil.rmtree(db_dir, ignore_errors=True)
        print("SMOKE OK")
        return

    dev_texts, dev_labels = data_split.load_texts(data_split.get_split()["dev"])
    Xdev = base_ppos(ensemble, dev_texts)
    stack = build_stack(Xdev, np.array(dev_labels))
    os.makedirs(os.path.dirname(STACK_PATH), exist_ok=True)
    with open(STACK_PATH, "wb") as f:
        pickle.dump({"clf": stack["clf"], "cols": stack["cols"], "col_names": stack["col_names"]}, f)

    datasets = {
        "imdb": ev.load_imdb_test(),
        "sst2": bd.load_sst2(),
        "yelp": bd.load_yelp(),
    }
    datasets = {k: (v[0], list(v[1])) for k, v in datasets.items()}
    res = evaluate_all(ensemble, stack, datasets)
    res["_meta"] = {"chosen_base_set": stack["label"], "col_names": stack["col_names"],
                    "dev_cv_acc": stack["dev_cv_acc"], "C": stack["C"],
                    "cv_by_set": stack["cv_by_set"], "note": DEV_NOTE}
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", STACK_PATH, "and", RESULTS_PATH)


if __name__ == "__main__":
    main()
