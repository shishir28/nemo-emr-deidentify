"""
Converts annotated notes (JSONL span format) to NeMo BIO token classification format.

Input:  data/synthetic/notes.jsonl  (or data/processed/notes.jsonl for i2b2)
Output: data/processed/train.txt, dev.txt, test.txt

BIO format (one token per line, blank line between documents):
  John    B-NAME
  Smith   I-NAME
  was     O
  seen    O
  ...
"""

import json
import random
from pathlib import Path


TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
# remaining 10% → test


def simple_tokenize(text):
    """Returns list of (token, start, end) tuples using whitespace + punctuation splitting."""
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        if text[j] in ".,;:!?()[]{}\"'":
            tokens.append((text[j], j, j + 1))
            i = j + 1
        else:
            while j < len(text) and not text[j].isspace() and text[j] not in ".,;:!?()[]{}\"'":
                j += 1
            tokens.append((text[i:j], i, j))
            i = j
    return tokens


def assign_bio_labels(tokens, spans):
    """Assigns BIO label to each token based on span annotations."""
    labels = ["O"] * len(tokens)

    for span in spans:
        s_start, s_end, label = span["start"], span["end"], span["label"]
        first = True
        for idx, (_, t_start, t_end) in enumerate(tokens):
            if t_end <= s_start or t_start >= s_end:
                continue
            if labels[idx] != "O":
                # skip tokens already claimed by a higher-priority span
                first = False
                continue
            if first:
                labels[idx] = f"B-{label}"
                first = False
            else:
                labels[idx] = f"I-{label}"

    return labels


def convert(input_path: Path, output_dir: Path):
    records = []
    with input_path.open() as f:
        for line in f:
            records.append(json.loads(line.strip()))

    random.shuffle(records)
    n = len(records)
    train_end = int(n * TRAIN_RATIO)
    dev_end = train_end + int(n * DEV_RATIO)

    splits = {
        "train": records[:train_end],
        "dev": records[train_end:dev_end],
        "test": records[dev_end:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records in splits.items():
        out_file = output_dir / f"{split_name}.txt"
        with out_file.open("w") as f:
            for record in split_records:
                tokens = simple_tokenize(record["text"])
                labels = assign_bio_labels(tokens, record["spans"])
                for (token, _, _), label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")
        print(f"  {split_name}: {len(split_records)} notes → {out_file}")


def main():
    sources = [
        Path("data/synthetic/notes.jsonl"),
        Path("data/processed/notes.jsonl"),  # i2b2 converted output goes here
    ]

    output_dir = Path("data/processed")

    input_path = None
    for src in sources:
        if src.exists():
            input_path = src
            break

    if not input_path:
        print("No input data found. Run generate_synthetic.py first, or place i2b2 data at data/processed/notes.jsonl")
        return

    print(f"Converting {input_path} ...")
    convert(input_path, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
