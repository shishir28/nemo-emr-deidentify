"""
Evaluates the trained NER model on the test split.
Reports per-label and overall precision, recall, F1.

Usage:
    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --config models/configs/ner_config.yaml
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from train import NERDataset, load_bio_file, extract_entities


def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = cfg["model"]["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")

    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model"]["base_model"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(ckpt_dir / cfg["output"]["best_model_name"], map_location=device))
    model.to(device).eval()

    test_data = load_bio_file(cfg["data"]["test"])
    test_ds = NERDataset(test_data, tokenizer, label2id, cfg["model"]["max_seq_length"])
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"])

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = batch["labels"].cpu().numpy()
            for pred_seq, target_seq in zip(preds, targets):
                mask = target_seq != -100
                all_preds.append(pred_seq[mask].tolist())
                all_targets.append(target_seq[mask].tolist())

    # Per-entity-type breakdown
    entity_tp = defaultdict(int)
    entity_fp = defaultdict(int)
    entity_fn = defaultdict(int)

    for pred_seq, target_seq in zip(all_preds, all_targets):
        pred_entities = extract_entities(pred_seq, id2label)
        true_entities = extract_entities(target_seq, id2label)
        for e in pred_entities & true_entities:
            entity_tp[e[2]] += 1
        for e in pred_entities - true_entities:
            entity_fp[e[2]] += 1
        for e in true_entities - pred_entities:
            entity_fn[e[2]] += 1

    print("\n=== Evaluation Results (Test Set) ===\n")
    print(f"{'Entity':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 58)

    all_tp = all_fp = all_fn = 0
    entity_types = sorted(set(list(entity_tp) + list(entity_fn)))
    for etype in entity_types:
        tp = entity_tp[etype]
        fp = entity_fp[etype]
        fn = entity_fn[etype]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp + fn
        print(f"{etype:<15} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10}")
        all_tp += tp; all_fp += fp; all_fn += fn

    print("-" * 58)
    p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    print(f"{'OVERALL':<15} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {all_tp + all_fn:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/configs/ner_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg)


if __name__ == "__main__":
    main()
