"""
Fine-tunes a BioBERT model for clinical PHI NER using BIO-tagged data.

Usage:
    python3 scripts/train.py
    python3 scripts/train.py --config models/configs/ner_config.yaml
    python3 scripts/train.py --epochs 10 --batch_size 32
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ── Data ────────────────────────────────────────────────────────────────────

def load_bio_file(path: str):
    """Loads BIO file into list of (tokens, labels) sentence pairs."""
    sentences, tokens, labels = [], [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
            else:
                parts = line.split("\t")
                tokens.append(parts[0])
                labels.append(parts[1] if len(parts) > 1 else "O")
    if tokens:
        sentences.append((tokens, labels))
    return sentences


class NERDataset(Dataset):
    def __init__(self, sentences, tokenizer, label2id, max_len):
        self.samples = []
        for tokens, labels in sentences:
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            word_ids = encoding.word_ids()
            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    label_ids.append(label2id.get(labels[word_id], 0))
                else:
                    # subword tokens get I- version of the label or -100
                    lbl = labels[word_id]
                    if lbl.startswith("B-"):
                        label_ids.append(label2id.get("I-" + lbl[2:], -100))
                    else:
                        label_ids.append(label2id.get(lbl, -100))
                prev_word_id = word_id

            self.samples.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_f1(preds, targets, id2label):
    tp = fp = fn = 0
    for pred_seq, target_seq in zip(preds, targets):
        pred_entities = extract_entities(pred_seq, id2label)
        true_entities = extract_entities(target_seq, id2label)
        tp += len(pred_entities & true_entities)
        fp += len(pred_entities - true_entities)
        fn += len(true_entities - pred_entities)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def extract_entities(label_seq, id2label):
    entities = set()
    start, current = None, None
    for i, label_id in enumerate(label_seq):
        label = id2label.get(label_id, "O")
        if label.startswith("B-"):
            if current:
                entities.add((start, i - 1, current))
            start, current = i, label[2:]
        elif label.startswith("I-") and current == label[2:]:
            pass
        else:
            if current:
                entities.add((start, i - 1, current))
            start, current = None, None
    if current:
        entities.add((start, len(label_seq) - 1, current))
    return entities


# ── Training ────────────────────────────────────────────────────────────────

def train(cfg):
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    labels = cfg["model"]["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    base_model = cfg["model"]["base_model"]
    max_len = cfg["model"]["max_seq_length"]

    print(f"\nLoading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Loading data ...")
    train_data = load_bio_file(cfg["data"]["train"])
    dev_data = load_bio_file(cfg["data"]["dev"])
    print(f"  Train: {len(train_data)} sentences")
    print(f"  Dev  : {len(dev_data)} sentences")

    train_ds = NERDataset(train_data, tokenizer, label2id, max_len)
    dev_ds = NERDataset(dev_data, tokenizer, label2id, max_len)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size)

    print(f"\nLoading model: {base_model}")
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    print(f"\nTraining for {epochs} epochs ...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % cfg["logging"]["log_every_steps"] == 0:
                avg = total_loss / step
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {avg:.4f}")

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0

        # Evaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                targets = batch["labels"].cpu().numpy()
                for pred_seq, target_seq in zip(preds, targets):
                    mask = target_seq != -100
                    all_preds.append(pred_seq[mask].tolist())
                    all_targets.append(target_seq[mask].tolist())

        precision, recall, f1 = compute_f1(all_preds, all_targets, id2label)
        print(f"\nEpoch {epoch}/{epochs} | Loss {avg_loss:.4f} | "
              f"P {precision:.4f} | R {recall:.4f} | F1 {f1:.4f} | "
              f"Time {elapsed:.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            best_path = ckpt_dir / cfg["output"]["best_model_name"]
            torch.save(model.state_dict(), best_path)
            tokenizer.save_pretrained(ckpt_dir / "tokenizer")
            print(f"  ✓ New best F1 {best_f1:.4f} — saved to {best_path}\n")
        else:
            print()

    final_path = ckpt_dir / cfg["output"]["final_model_name"]
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"Best checkpoint : {ckpt_dir / cfg['output']['best_model_name']}")
    print(f"Final checkpoint: {final_path}")


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/configs/ner_config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr

    train(cfg)


if __name__ == "__main__":
    main()
