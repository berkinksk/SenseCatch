"""Benchmark dataset loaders for sentiment evaluation.

SST-2: glue/sst2, validation split (the test split has hidden labels).
Yelp polarity: yelp_polarity, test split.
Labels are 0 for negative and 1 for positive.
"""
import random

from datasets import load_dataset


def load_sst2():
    """Return (texts, labels) from the SST-2 validation split."""
    ds = load_dataset("glue", "sst2")["validation"]
    texts = [r["sentence"] for r in ds]
    labels = [int(r["label"]) for r in ds]
    return texts, labels


def load_yelp(n=2000, seed=42):
    """Return a seeded, class-balanced subset of the Yelp test split.

    Picks n/2 positive and n/2 negative reviews, then shuffles.
    """
    ds = load_dataset("yelp_polarity")["test"]
    per_class = n // 2
    pos, neg = [], []
    for r in ds:
        (pos if int(r["label"]) == 1 else neg).append(r["text"])
    rng = random.Random(seed)
    texts = rng.sample(pos, per_class) + rng.sample(neg, per_class)
    labels = [1] * per_class + [0] * per_class
    order = list(range(n))
    rng.shuffle(order)
    return [texts[i] for i in order], [labels[i] for i in order]
