"""
Core de-identification pipeline.
Loads the fine-tuned BioBERT model and runs NER + regex safety pass.
"""

import yaml
import torch
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pipeline.regex_pass import Span, run_regex_pass


class Redactor:
    def __init__(self, config_path: str = "models/configs/ner_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        labels = cfg["model"]["labels"]
        self.label2id = {l: i for i, l in enumerate(labels)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
        tokenizer_dir = ckpt_dir / "tokenizer"

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

        model = AutoModelForTokenClassification.from_pretrained(
            cfg["model"]["base_model"],
            num_labels=len(labels),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        model.load_state_dict(
            torch.load(ckpt_dir / cfg["output"]["best_model_name"], map_location=self.device)
        )
        model.to(self.device).eval()
        self.model = model

    def _ner_spans(self, text: str) -> list[Span]:
        """Runs the NER model and returns detected PHI spans."""
        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.cfg["model"]["max_seq_length"],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoding).logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        word_ids = encoding.word_ids()

        # Map token-level predictions back to word-level (first subword wins)
        word_labels = {}
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in word_labels:
                continue
            word_labels[word_id] = self.id2label.get(predictions[token_idx], "O")

        # Reconstruct char-level spans from word labels
        spans = []
        char_pos = 0
        current_label = None
        span_start = None

        for word_idx, word in enumerate(words):
            word_start = text.find(word, char_pos)
            word_end = word_start + len(word)
            label = word_labels.get(word_idx, "O")

            if label.startswith("B-"):
                if current_label:
                    spans.append(Span(span_start, char_pos - 1, current_label,
                                      text[span_start:char_pos - 1]))
                current_label = label[2:]
                span_start = word_start
            elif label.startswith("I-") and current_label == label[2:]:
                pass  # extend current span
            else:
                if current_label:
                    spans.append(Span(span_start, word_start - 1, current_label,
                                      text[span_start:word_start - 1].strip()))
                    current_label = None
                if label not in ("O", ""):
                    current_label = label[2:] if "-" in label else label
                    span_start = word_start

            char_pos = word_end + 1  # +1 for the space

        if current_label:
            spans.append(Span(span_start, len(text), current_label,
                              text[span_start:].strip()))
        return spans

    def deidentify(self, text: str) -> dict:
        """
        Runs full de-identification pipeline on text.
        Returns redacted text and list of detected PHI spans.
        """
        ner_spans = self._ner_spans(text)
        regex_spans = run_regex_pass(text, ner_spans)
        all_spans = sorted(ner_spans + regex_spans, key=lambda s: s.start)

        # Build redacted text by replacing spans with [LABEL] placeholders
        redacted = []
        cursor = 0
        for span in all_spans:
            if span.start < cursor:
                continue  # skip overlapping spans
            redacted.append(text[cursor:span.start])
            redacted.append(f"[{span.label}]")
            cursor = span.end
        redacted.append(text[cursor:])

        return {
            "redacted_text": "".join(redacted),
            "phi_spans": [
                {
                    "start": s.start,
                    "end": s.end,
                    "label": s.label,
                    "text": s.text,
                    "source": s.source,
                    "confidence": round(s.confidence, 4),
                }
                for s in all_spans
            ],
        }
