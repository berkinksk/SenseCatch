"""Diagnose where the rule layer flips predictions and how often the flip is wrong.

Runs on the seeded quick IMDB subset. Compares the raw NB+LR ensemble (no rules)
to the full system (all rules), attributes each flip to the first rule layer that
fires in cascade order, and splits the result by review length. Writes
artifacts/rule_diagnosis.json.
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

LAYERS = ["simple_case", "neutral", "special_phrase", "safety",
          "sarcasm", "idiom", "contradiction", "none"]


def first_layer(ensemble, text):
    """Return the first rule layer that fires for this text, in cascade order."""
    simple = ensemble._handle_simple_cases(text)
    if simple and simple[0]:
        return "simple_case"
    neutral = ensemble._detect_neutral_sentiment(text)
    if neutral and neutral[0]:
        return "neutral"
    special = ensemble.clean_text(text)[2]
    if special and special.get("special_phrase_detected"):
        return "special_phrase"
    if ensemble.safety_check(text) is True:
        return "safety"
    if ensemble._detect_sarcasm(text):
        return "sarcasm"
    if ensemble._detect_idioms(text):
        return "idiom"
    if ensemble._detect_contradiction(text):
        return "contradiction"
    return "none"


def mcnemar(b, c):
    """Continuity-corrected McNemar chi-square and p-value for discordant counts."""
    from scipy.stats import chi2
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stat), float(chi2.sf(stat, 1))


def raw_ensemble_preds(ensemble, texts):
    """Raw NB+LR weighted-probability argmax, the same definition as evaluate.py."""
    processed = ev.preprocess_texts(ensemble, texts)
    f_nb = ev.build_features(ensemble, "naive_bayes", processed)
    f_lr = ev.build_features(ensemble, "logistic_regression", processed)
    nb = ensemble.models["naive_bayes"].predict_proba(f_nb)
    lr = ensemble.models["logistic_regression"].predict_proba(f_lr)
    w_nb = ensemble.model_weights.get("naive_bayes", 0.6)
    w_lr = ensemble.model_weights.get("logistic_regression", 0.4)
    return np.argmax(w_nb * nb + w_lr * lr, axis=1)


def full_system_decisions(ensemble, texts):
    """Full-system label per text as Positive / Negative / Neutral."""
    out = []
    n = len(texts)
    for i, t in enumerate(texts):
        out.append(ensemble.predict(t)["sentiment"])
        if (i + 1) % 250 == 0 or (i + 1) == n:
            print(f"  full-system {i + 1}/{n}", flush=True)
    return out


def main():
    max_per_class = 1000
    ensemble = ev.load_ensemble()
    texts, labels = ev.load_imdb_test(max_per_class=max_per_class, seed=42)
    labels = np.asarray(labels)
    n = len(texts)
    print(f"quick subset: {n} reviews")

    raw = raw_ensemble_preds(ensemble, texts)
    decisions = full_system_decisions(ensemble, texts)

    # Full-system binary label. A Neutral on binary data counts as wrong.
    full = np.array([1 if d == "Positive" else (0 if d == "Negative" else 1 - labels[i])
                     for i, d in enumerate(decisions)])

    raw_correct = (raw == labels)
    full_correct = (full == labels)
    raw_acc = float(raw_correct.mean())
    full_acc = float(full_correct.mean())
    helped = int((full_correct & ~raw_correct).sum())
    hurt = int((~full_correct & raw_correct).sum())
    stat, p = mcnemar(hurt, helped)

    raw_dec = ["Positive" if r == 1 else "Negative" for r in raw]
    per = {L: {"fires": 0, "flips": 0, "wrong_flips": 0, "right_flips": 0} for L in LAYERS}
    for i, t in enumerate(texts):
        layer = first_layer(ensemble, t)
        per[layer]["fires"] += 1
        if decisions[i] != raw_dec[i]:
            per[layer]["flips"] += 1
            if raw_correct[i] and not full_correct[i]:
                per[layer]["wrong_flips"] += 1
            elif (not raw_correct[i]) and full_correct[i]:
                per[layer]["right_flips"] += 1

    wc = np.array([len(t.split()) for t in texts])
    short = wc < 50
    buckets = {}
    for name, mask in [("short(<50w)", short), ("long(>=50w)", ~short)]:
        k = int(mask.sum())
        buckets[name] = {
            "n": k,
            "raw_acc": round(float(raw_correct[mask].mean()), 4) if k else None,
            "full_acc": round(float(full_correct[mask].mean()), 4) if k else None,
        }

    result = {
        "subset": {"name": "imdb_quick_seeded", "seed": 42, "n": n,
                   "max_per_class": max_per_class},
        "raw_vs_full": {"raw_acc": round(raw_acc, 4), "full_acc": round(full_acc, 4),
                        "delta": round(full_acc - raw_acc, 4),
                        "helped_full_right_raw_wrong": helped,
                        "hurt_full_wrong_raw_right": hurt,
                        "mcnemar_chi2": round(stat, 3), "mcnemar_p": float(f"{p:.3g}")},
        "per_layer": per,
        "by_length": buckets,
    }

    out_path = os.path.join(PROJECT_ROOT, "artifacts", "rule_diagnosis.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nRAW vs FULL")
    print(f"  raw {raw_acc:.4f}  full {full_acc:.4f}  delta {full_acc - raw_acc:+.4f}")
    print(f"  helped {helped}  hurt {hurt}  McNemar chi2 {stat:.2f} p {p:.3g}")
    print("\nPER-LAYER (first firing layer in cascade order)")
    print(f"  {'layer':16s} {'fires':>6s} {'flips':>6s} {'wrong':>6s} {'right':>6s}")
    for L in LAYERS:
        d = per[L]
        print(f"  {L:16s} {d['fires']:6d} {d['flips']:6d} {d['wrong_flips']:6d} {d['right_flips']:6d}")
    print("\nBY LENGTH")
    for name, d in buckets.items():
        if d["raw_acc"] is not None:
            print(f"  {name:12s} n={d['n']:5d}  raw {d['raw_acc']:.4f}  full {d['full_acc']:.4f}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
