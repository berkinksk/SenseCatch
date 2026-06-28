"""Zero-shot sentiment baseline with bart-large-mnli on the seeded IMDB sample.

Classifies each review as positive or negative with no task training, as a
comparison point for the fine-tuned models. Saves artifacts/zero_shot_baseline.json.
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


def wilson_ci(acc, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def main():
    import torch
    from transformers import pipeline
    texts, labels = ev.load_imdb_test(max_per_class=1000, seed=42)
    labels = [int(x) for x in labels]
    n = len(texts)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"zero-shot bart-large-mnli on {n} reviews, device {device}", flush=True)

    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    candidate = ["positive", "negative"]
    # Cap length to stay within bart's context and avoid tokenizer overflow.
    capped = [t[:4000] for t in texts]

    preds = []
    chunk = 100
    for i in range(0, n, chunk):
        out = clf(capped[i:i + chunk], candidate_labels=candidate, batch_size=16)
        if isinstance(out, dict):
            out = [out]
        for o in out:
            preds.append(1 if o["labels"][0] == "positive" else 0)
        print(f"  {min(i + chunk, n)}/{n}", flush=True)

    acc = sum(int(p == l) for p, l in zip(preds, labels)) / n
    ci = wilson_ci(acc, n)
    result = {"model": "facebook/bart-large-mnli", "n": n, "accuracy": round(acc, 4),
              "ci": ci, "sample": "imdb seeded sample, max_per_class=1000, seed=42"}
    out_path = os.path.join(PROJECT_ROOT, "artifacts", "zero_shot_baseline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"RESULT zero-shot accuracy {acc:.4f} CI {ci} (n={n}, device {device})")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
