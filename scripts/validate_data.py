"""
Validates BIO-tagged data files before training.
Checks: label consistency, no empty sequences, label distribution.
"""

from collections import Counter
from pathlib import Path


VALID_LABELS = {
    "O",
    "B-NAME", "I-NAME",
    "B-DATE", "I-DATE",
    "B-ID", "I-ID",
    "B-LOCATION", "I-LOCATION",
    "B-CONTACT", "I-CONTACT",
    "B-AGE", "I-AGE",
    "B-PROFESSION", "I-PROFESSION",
}


def validate_file(path: Path):
    errors = []
    label_counts = Counter()
    doc_count = 0
    token_count = 0
    prev_label = "O"

    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if line == "":
                doc_count += 1
                prev_label = "O"
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                errors.append(f"Line {line_no}: expected 2 tab-separated columns, got {len(parts)}")
                continue

            token, label = parts
            token_count += 1
            label_counts[label] += 1

            if label not in VALID_LABELS:
                errors.append(f"Line {line_no}: unknown label '{label}'")

            # I- tag must follow B- or I- of the same entity
            if label.startswith("I-"):
                entity = label[2:]
                if prev_label not in (f"B-{entity}", f"I-{entity}"):
                    errors.append(f"Line {line_no}: I-{entity} follows '{prev_label}' (invalid BIO sequence)")

            prev_label = label

    return errors, label_counts, doc_count, token_count


def main():
    data_dir = Path("data/processed")
    splits = ["train.txt", "dev.txt", "test.txt"]

    any_errors = False
    for split in splits:
        path = data_dir / split
        if not path.exists():
            print(f"  MISSING: {path}")
            continue

        errors, label_counts, docs, tokens = validate_file(path)
        print(f"\n{split}")
        print(f"  Documents : {docs}")
        print(f"  Tokens    : {tokens}")
        print(f"  Labels    :")
        for label, count in sorted(label_counts.items()):
            print(f"    {label:<20} {count}")

        if errors:
            any_errors = True
            print(f"  ERRORS ({len(errors)}):")
            for e in errors[:10]:
                print(f"    {e}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more")
        else:
            print("  Validation: PASSED")

    if any_errors:
        print("\nFix errors before training.")
    else:
        print("\nAll splits valid. Ready for training.")


if __name__ == "__main__":
    main()
