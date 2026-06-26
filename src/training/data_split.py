#!/usr/bin/env python3
"""
Canonical seeded data split for SenseCatch (Step 7.2).

Provides ONE fixed train/dev/test split so every tuning decision
(model choice, hyperparameters, ensemble weights, gate threshold,
calibration) is made on DEV, and the IMDB official TEST set is touched
only once, for final reporting.

  - IMDB official split: train/ (25k) and test/ (25k) are kept separate.
  - DEV is carved from IMDB train/ (seeded, stratified) — default 10%.
  - A manifest (sizes + a hash of each split's file ids) is saved for
    reproducibility and as a no-leakage proof artifact.

Usage:
    python src/training/data_split.py        # writes/print the manifest
    from data_split import get_split, load_texts
"""
import os
import json
import random
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMDB_DIR = os.path.join(PROJECT_ROOT, "datasets", "aclImdb")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "artifacts", "data_split_manifest.json")

SEED = 42
DEV_FRACTION = 0.10


def _list_imdb(split):
    """Return [(relpath, label)] for an IMDB split ('train'/'test'); label 1=pos, 0=neg."""
    items = []
    for sentiment, label in (("pos", 1), ("neg", 0)):
        folder = os.path.join(IMDB_DIR, split, sentiment)
        for fn in sorted(os.listdir(folder)):
            items.append((os.path.join(split, sentiment, fn), label))
    return items


def get_split(dev_fraction=DEV_FRACTION, seed=SEED):
    """Deterministic train/dev/test split.

    Returns {'train': [(relpath, label)], 'dev': [...], 'test': [...]}.
    DEV is a stratified random sample of IMDB-train; the remainder of
    train is 'train'; IMDB-test is 'test' (held out until final eval).
    """
    rng = random.Random(seed)
    train_all = _list_imdb("train")
    test = _list_imdb("test")

    pos = [x for x in train_all if x[1] == 1]
    neg = [x for x in train_all if x[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_dev_pos = int(len(pos) * dev_fraction)
    n_dev_neg = int(len(neg) * dev_fraction)
    dev = pos[:n_dev_pos] + neg[:n_dev_neg]
    train = pos[n_dev_pos:] + neg[n_dev_neg:]
    rng.shuffle(train)
    rng.shuffle(dev)
    return {"train": train, "dev": dev, "test": test}


def load_texts(items):
    """Given [(relpath, label)], read files -> (texts, labels)."""
    texts, labels = [], []
    for rel, lab in items:
        with open(os.path.join(IMDB_DIR, rel), encoding="utf-8") as f:
            texts.append(f.read())
        labels.append(lab)
    return texts, labels


def save_manifest(path=MANIFEST_PATH):
    """Write a manifest: per-split sizes + SHA-256 of each split's sorted file ids."""
    sp = get_split()
    manifest = {"seed": SEED, "dev_fraction": DEV_FRACTION, "splits": {}}
    for name, items in sp.items():
        ids = sorted(rel for rel, _ in items)
        h = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        n_pos = sum(1 for _, l in items if l == 1)
        manifest["splits"][name] = {
            "n": len(items),
            "n_pos": n_pos,
            "n_neg": len(items) - n_pos,
            "ids_sha256": h,
        }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    print(json.dumps(save_manifest(), indent=2))
