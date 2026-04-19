"""
Audit logger — records every de-identification request without storing PHI.

Logged per request:
  - timestamp, doc_id (SHA-256 hash of text — not reversible), char_length,
    phi_count, detection sources, latency_ms
"""

import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

_handler = logging.FileHandler("logs/audit.log")
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger = logging.getLogger("audit")
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False


def log_request(text: str, phi_spans: list, latency_ms: float):
    doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
    sources = dict(Counter(s["source"] for s in phi_spans))
    labels = dict(Counter(s["label"] for s in phi_spans))

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "doc_id": doc_id,
        "char_length": len(text),
        "phi_count": len(phi_spans),
        "sources": sources,
        "labels": labels,
        "latency_ms": round(latency_ms, 2),
    }
    _logger.info(json.dumps(entry))
