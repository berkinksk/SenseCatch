"""Error analysis on the seeded quick IMDB subset for stack, distilbert, nbsvm.

Reports overall accuracy with Wilson CIs, a negation split, length buckets, and a
few concrete stack misclassifications. Saves artifacts/error_analysis.json.
"""
import os
import sys
import re
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

NEG_RE = re.compile(r"\bnot\b|n't|\bnever\b|\bno\b|\bcannot\b|\bnor\b", re.IGNORECASE)


def acc_ci(preds, labels, mask=None):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if mask is not None:
        preds, labels = preds[mask], labels[mask]
    n = len(labels)
    if n == 0:
        return {"acc": None, "ci": [None, None], "n": 0}
    acc = float((preds == labels).mean())
    return {"acc": round(acc, 4), "ci": se.wilson_ci(acc, n), "n": int(n)}


def analyze(preds, labels, is_neg, wc):
    short = wc < 50
    medium = (wc >= 50) & (wc <= 150)
    long_ = wc > 150
    return {
        "overall": acc_ci(preds, labels),
        "negation": {"neg": acc_ci(preds, labels, is_neg),
                     "non_neg": acc_ci(preds, labels, ~is_neg)},
        "length": {"short(<50w)": acc_ci(preds, labels, short),
                   "medium(50-150w)": acc_ci(preds, labels, medium),
                   "long(>150w)": acc_ci(preds, labels, long_)},
    }


def clean_snippet(text, n=200):
    t = re.sub(r"<br\s*/?>", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:n]


def main():
    texts, labels = ev.load_imdb_test(max_per_class=1000, seed=42)
    labels = np.asarray(labels)
    n = len(texts)
    print(f"quick subset: {n} reviews")

    ensemble = ev.load_ensemble()
    is_neg = np.array([bool(NEG_RE.search(t)) for t in texts])
    wc = np.array([len(t.split()) for t in texts])

    processed = ev.preprocess_texts(ensemble, texts)
    X = ev.build_features(ensemble, "nbsvm", processed)
    nbsvm_preds = np.asarray(ensemble.models["nbsvm"].predict(X))
    print("nbsvm done")

    distilbert_preds = np.asarray(fd.predict(texts))
    print("distilbert done")

    stack_preds = np.asarray(se.predict(texts))
    print("stack done")

    models = {"stack": stack_preds, "distilbert": distilbert_preds, "nbsvm": nbsvm_preds}
    results = {name: analyze(p, labels, is_neg, wc) for name, p in models.items()}

    fp = [i for i in range(n) if stack_preds[i] == 1 and labels[i] == 0]
    fn = [i for i in range(n) if stack_preds[i] == 0 and labels[i] == 1]
    examples = []
    for i in fp[:4] + fn[:4]:
        examples.append({
            "true": "Positive" if labels[i] == 1 else "Negative",
            "stack_pred": "Positive" if stack_preds[i] == 1 else "Negative",
            "text": clean_snippet(texts[i]),
        })

    out = {"subset": {"name": "imdb_quick_seeded", "seed": 42, "n": n},
           "negation_rule": "not | n't | never | no | cannot | nor (word-boundary, case-insensitive)",
           "models": results, "stack_misclassifications": examples}
    out_path = os.path.join(PROJECT_ROOT, "artifacts", "error_analysis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    for name in ["stack", "distilbert", "nbsvm"]:
        r = results[name]
        o = r["overall"]
        print(f"\n=== {name} ===  overall {o['acc']} CI{o['ci']} (n={o['n']})")
        ng, nn = r["negation"]["neg"], r["negation"]["non_neg"]
        print(f"  negation     acc {ng['acc']} CI{ng['ci']} (n={ng['n']})")
        print(f"  non-negation acc {nn['acc']} CI{nn['ci']} (n={nn['n']})")
        for b, d in r["length"].items():
            print(f"  {b:18s} acc {d['acc']} (n={d['n']})")
    print("\nStack misclassifications (sample):")
    for e in examples:
        print(f"  true={e['true']:8s} pred={e['stack_pred']:8s} | {e['text'][:120]}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
