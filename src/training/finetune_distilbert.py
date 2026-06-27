"""Fine-tune DistilBERT for binary sentiment on the IMDB training split.

Trains distilbert-base-uncased, keeps the best checkpoint by dev accuracy, and
saves it to models/distilbert_imdb. Also exposes predict_proba and predict that
load from that directory, for reuse by the benchmark and the app.

Labels: 0 = negative, 1 = positive.
"""
import os
import sys
import argparse

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
for _p in (PROJECT_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_split

BASE_MODEL = "distilbert-base-uncased"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert_imdb")
MAX_LEN = 256


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode(tok, texts, labels):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN)
    feats = []
    for i in range(len(texts)):
        feats.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": int(labels[i]),
        })
    return feats


def _dev_accuracy(model, loader, device):
    import torch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].numel()
    return correct / total if total else 0.0


def finetune(output_dir=OUTPUT_DIR, epochs=2, lr=2e-5, train_batch=16, smoke=False):
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding)

    torch.manual_seed(42)
    device = pick_device()
    print(f"device: {device}")

    train_texts, train_labels = data_split.load_texts(data_split.get_split()["train"])
    dev_texts, dev_labels = data_split.load_texts(data_split.get_split()["dev"])
    if smoke:
        train_texts, train_labels = train_texts[:200], train_labels[:200]
        dev_texts, dev_labels = dev_texts[:100], dev_labels[:100]
    print(f"train {len(train_texts)} / dev {len(dev_texts)}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(device)
    collator = DataCollatorWithPadding(tokenizer=tok)

    train_loader = DataLoader(_encode(tok, train_texts, train_labels),
                              batch_size=train_batch, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(_encode(tok, dev_texts, dev_labels),
                            batch_size=64, shuffle=False, collate_fn=collator)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    max_steps = 20 if smoke else None
    n_epochs = 1 if smoke else epochs

    best_acc, step = -1.0, 0
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0 or (smoke and step % 5 == 0):
                print(f"  step {step} loss {loss.item():.4f}", flush=True)
            if max_steps and step >= max_steps:
                break
        acc = _dev_accuracy(model, dev_loader, device)
        print(f"  epoch {epoch + 1} dev_acc {acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tok.save_pretrained(output_dir)
        if max_steps and step >= max_steps:
            break

    print(f"best dev_acc {best_acc:.4f} saved to {output_dir}")
    return output_dir, best_acc


_LOADED = {}


def _load_model(model_dir):
    """Load and cache the tokenizer, model, and device for a directory."""
    if model_dir not in _LOADED:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = pick_device()
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()
        _LOADED[model_dir] = (tok, model, device)
    return _LOADED[model_dir]


def predict_proba(texts, batch_size=64, model_dir=OUTPUT_DIR):
    """Return an array of [p_neg, p_pos] for each text."""
    import torch
    tok, model, device = _load_model(model_dir)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)


def predict(texts, batch_size=64, model_dir=OUTPUT_DIR):
    """Return a list of 0/1 labels."""
    return [int(x) for x in predict_proba(texts, batch_size, model_dir).argmax(axis=1)]


def main():
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT on IMDB train")
    ap.add_argument("--smoke", action="store_true", help="tiny end to end check")
    args = ap.parse_args()

    if args.smoke:
        out_dir = os.path.join(PROJECT_ROOT, "models", "distilbert_imdb_smoke")
        path, acc = finetune(output_dir=out_dir, smoke=True)
        pos = "An absolute masterpiece, beautifully acted and deeply moving."
        neg = "A boring, poorly written mess that I deeply regret watching."
        proba = predict_proba([pos, neg], model_dir=path)
        labels = predict([pos, neg], model_dir=path)
        print("SMOKE train+save+reload+predict OK")
        print(f"  positive sample -> label {labels[0]} p_pos {proba[0][1]:.3f}")
        print(f"  negative sample -> label {labels[1]} p_pos {proba[1][1]:.3f}")
        print("SMOKE OK")
    else:
        finetune()


if __name__ == "__main__":
    main()
