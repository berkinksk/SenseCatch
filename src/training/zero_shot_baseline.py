"""Zero-shot sentiment baselines on the seeded IMDB sample.

Runs each zero-shot model on the same 2,000-review IMDB sample and saves accuracy
plus a Wilson 95% CI, as comparison points for the fine-tuned models.
Saves artifacts/zero_shot_baseline.json.
"""
import os
import sys
import json
import math

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
for _p in (PROJECT_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate as ev

# Same architecture as our fine-tuned model first, then a larger zero-shot model.
MODELS = [
    ("typeform/distilbert-base-uncased-mnli", "distilbert-mnli (same architecture)"),
    ("facebook/bart-large-mnli", "bart-large-mnli (407M)"),
]


def wilson_ci(acc, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def measure(model_id, texts, labels, device):
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=model_id, device=device)
    candidate = ["positive", "negative"]
    capped = [t[:4000] for t in texts]
    preds = []
    chunk = 100
    for i in range(0, len(texts), chunk):
        out = clf(capped[i:i + chunk], candidate_labels=candidate, batch_size=16)
        if isinstance(out, dict):
            out = [out]
        for o in out:
            preds.append(1 if o["labels"][0] == "positive" else 0)
        print(f"  {min(i + chunk, len(texts))}/{len(texts)}", flush=True)
    acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
    return {"accuracy": round(acc, 4), "ci": wilson_ci(acc, len(labels)), "n": len(labels)}


def main():
    import torch
    texts, labels = ev.load_imdb_test(max_per_class=1000, seed=42)
    labels = [int(x) for x in labels]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out_path = os.path.join(PROJECT_ROOT, "artifacts", "zero_shot_baseline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = {"sample": "imdb seeded sample, max_per_class=1000, seed=42",
               "device": device, "models": {}}
    for model_id, label in MODELS:
        print(f"=== {label} ({model_id}) on {len(texts)} reviews, device {device} ===", flush=True)
        r = measure(model_id, texts, labels, device)
        r["label"] = label
        results["models"][model_id] = r
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"RESULT {label}: acc {r['accuracy']} CI {r['ci']} (n={r['n']})", flush=True)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
