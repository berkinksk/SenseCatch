"""Measure CPU latency and model size for each app model option.

Times the real serving path (SentimentEnsemble.predict) on a fixed set of reviews,
on CPU for every option so the comparison is same-device, with warm-up excluded.
Saves artifacts/cost_table.json.
"""
import os
import sys
import json
import time
import platform
import statistics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
for _p in (PROJECT_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import finetune_distilbert as fd
# Force CPU so distilbert and the stack are measured on the same device as the rest.
fd.pick_device = lambda: "cpu"

import evaluate as ev
from src.sensecatch.ensemble_model import SentimentEnsemble

OPTIONS = ["naive_bayes", "logistic_regression", "linear_svc", "nbsvm",
           "distilbert", "stack", "rule_based"]
CLASSICAL = ["naive_bayes", "logistic_regression", "linear_svc", "nbsvm"]
PKL = {m: os.path.join(PROJECT_ROOT, "models", f"{m}.pkl") for m in CLASSICAL}
PKL["stack"] = os.path.join(PROJECT_ROOT, "models", "stack_ensemble.pkl")


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def size_mb(option):
    if option == "distilbert":
        return round(dir_size(os.path.join(PROJECT_ROOT, "models", "distilbert_imdb")) / 1e6, 1)
    if option == "rule_based":
        return round(sum(os.path.getsize(PKL[m]) for m in CLASSICAL) / 1e6, 3)
    return round(os.path.getsize(PKL[option]) / 1e6, 3)


def time_option(ensemble, option, texts, warmup=3):
    for t in texts[:warmup]:
        ensemble.predict(t, specific_model=option)
    lat = []
    for t in texts[warmup:]:
        t0 = time.perf_counter()
        ensemble.predict(t, specific_model=option)
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat.sort()
    n = len(lat)
    return {"mean_ms": round(statistics.mean(lat), 1),
            "median_ms": round(statistics.median(lat), 1),
            "p95_ms": round(lat[min(n - 1, int(0.95 * n))], 1),
            "n_timed": n}


def main():
    import torch
    texts, _ = ev.load_imdb_test(max_per_class=1000, seed=42)
    texts = texts[:80]
    ensemble = SentimentEnsemble()

    hardware = {"platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor() or platform.machine(),
                "python": platform.python_version(),
                "torch_threads": torch.get_num_threads(), "device": "cpu"}

    results = {}
    for opt in OPTIONS:
        r = time_option(ensemble, opt, texts)
        r["size_mb"] = size_mb(opt)
        results[opt] = r
        print(f"  timed {opt}: mean {r['mean_ms']} ms", flush=True)

    db_device = str(fd._LOADED[fd.OUTPUT_DIR][2]) if fd.OUTPUT_DIR in fd._LOADED else "not_loaded"

    out = {"hardware": hardware, "n_texts": len(texts), "warmup": 3,
           "distilbert_device": db_device,
           "notes": {"stack": "size is the meta-learner pickle; at serve time it also loads "
                              "the 4 classical models and distilbert",
                     "rule_based": "reuses the 4 classical pkls"},
           "options": results}
    out_path = os.path.join(PROJECT_ROOT, "artifacts", "cost_table.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nHardware: {hardware['platform']} | {hardware['processor']} | "
          f"python {hardware['python']} | torch threads {hardware['torch_threads']} | "
          f"distilbert device {db_device}")
    print(f"\n{'option':22s} {'mean ms':>9s} {'median ms':>10s} {'p95 ms':>8s} {'size MB':>9s}")
    for opt in OPTIONS:
        r = results[opt]
        print(f"{opt:22s} {r['mean_ms']:9.1f} {r['median_ms']:10.1f} {r['p95_ms']:8.1f} {r['size_mb']:>9}")
    print("\nNote: NLTK preprocessing dominates the classical latency; the transformer "
          "forward pass dominates distilbert and stack.")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
