# Architecture

## Overview

The pipeline processes raw clinical text through four sequential stages: ingestion, NER-based PHI detection, redaction, and a rule-based safety pass. The FastAPI service wraps the entire pipeline behind a single HTTP endpoint.

## System Diagram

```
  ╔══════════════════════════════════════════════════════════════════╗
  ║                    DGX Spark (on-prem)                           ║
  ║                                                                  ║
  ║  ┌─────────────── DATA LAYER (Phase 2 ✓) ──────────────────┐    ║
  ║  │                                                           │    ║
  ║  │  generate_synthetic.py ──► data/synthetic/notes.jsonl    │    ║
  ║  │                                    │                      │    ║
  ║  │  i2b2 XML (optional) ──────────────┤                      │    ║
  ║  │                                    ▼                      │    ║
  ║  │                       convert_to_bio.py                   │    ║
  ║  │                                    │                      │    ║
  ║  │                       validate_data.py                    │    ║
  ║  │                                    │                      │    ║
  ║  │              data/processed/ train│dev│test.txt           │    ║
  ║  └───────────────────────────────────┼───────────────────────┘    ║
  ║                                      │                            ║
  ║  ┌─────────────── MODEL LAYER (Phase 3 ✓) ──────────────────┐     ║
  ║  │                                   ▼                      │     ║
  ║  │              scripts/train.py (BioBERT fine-tune)        │     ║
  ║  │              scripts/evaluate.py (per-label F1)          │     ║
  ║  │                                   │                      │     ║
  ║  │              models/checkpoints/phi_ner_best.pt          │     ║
  ║  └───────────────────────────────────┼───────────────────────┘    ║
  ║                                      │                            ║
  ║  ┌─────────────── SERVING LAYER (Phase 4 ✓) ────────────────┐     ║
  ║  │                                                           │     ║
  ║  │   Clinical Note ──► FastAPI :8000 ──► pipeline/          │     ║
  ║  │   (raw text)         api/main.py      redactor.py        │     ║
  ║  │                                           │              │     ║
  ║  │                                  ┌────────┴────────┐     │     ║
  ║  │                                  ▼                 ▼     │     ║
  ║  │                           BioBERT NER       Regex Safety │     ║
  ║  │                           (NER spans)       Pass         │     ║
  ║  │                                  └────────┬────────┘     │     ║
  ║  │                                           ▼              │     ║
  ║  │                                      Redactor            │     ║
  ║  │                                  (text replacement)      │     ║
  ║  │                                           │              │     ║
  ║  │   Redacted Note ◄──────────── PHI Span JSON              │     ║
  ║  │                              + phi_count + sources        │     ║
  ║  └──────────────────────────────────────────────────────────┘     ║
  ╚══════════════════════════════════════════════════════════════════╝
```

## Component Breakdown

### 1. FastAPI Service (`api/`)

- `GET  /health` — liveness check, returns model name and device
- `POST /deidentify` — de-identify a single clinical note
- `POST /deidentify/batch` — de-identify up to 100 notes in one request
- `GET  /docs` — interactive Swagger UI
- Stateless — no patient data stored at rest

### 2. NER Model (`models/`)

- Base model: **BioBERT** (`dmis-lab/biobert-base-cased-v1.2`) — BERT pretrained on PubMed abstracts and PMC full-text
- Task: token classification (BIO tagging)
- Fine-tuned on: synthetic data (Phase 2); drop-in replacement with i2b2 2014 when available
- PHI labels: `NAME`, `DATE`, `ID`, `LOCATION`, `CONTACT`, `AGE`, `PROFESSION`
- Training: PyTorch 2.7+cu128, ~38s/epoch on GB10 Blackwell, best checkpoint saved by dev F1
- Test F1: 1.0 on synthetic data; expect ~0.85–0.93 on real i2b2

### 3. Redactor Module (`pipeline/`)

- Takes original text + NER span predictions
- Replaces PHI spans with `[LABEL]` placeholders
- Preserves document structure (line breaks, whitespace)

### 4. Regex Safety Pass (`pipeline/`)

- Secondary rule-based layer to catch PHI the NER model may miss
- Rules cover: dates (various formats), phone numbers, MRNs, emails, SSNs, postcodes
- Runs after NER — adds coverage, not a replacement

## Data Flow

```
Raw Note
   │
   ▼
Tokenisation (WordPiece / SentencePiece)
   │
   ▼
BioBERT Token Classification
   │   → outputs: [B-NAME, I-NAME, O, B-DATE, ...]
   ▼
Span Extraction (BIO → entity spans)
   │
   ▼
Regex Safety Pass
   │   → adds any missed PHI spans
   ▼
Redactor
   │   → replaces spans in original text
   ▼
Redacted Note + PHI Span JSON
```

## Training Pipeline

```
i2b2 2014 XML files                    data/synthetic/notes.jsonl
(real PHI data, requires access)  OR   (generated by generate_synthetic.py)
              │                                      │
              └──────────────┬───────────────────────┘
                             ▼
              scripts/convert_to_bio.py
              (span JSONL → BIO token format)
              (word-boundary matching, overlap protection)
                             │
                             ▼
              data/processed/
                ├── train.txt  (80% — 400 notes)
                ├── dev.txt    (10% —  50 notes)
                └── test.txt   (10% —  50 notes)
                             │
                             ▼
              scripts/validate_data.py
              (BIO sequence checks, label distribution)
                             │
                             ▼
              scripts/train.py          ← Phase 3 ✓
              (BioBERT fine-tuning, AdamW + linear warmup, DGX Spark)
                             │
              scripts/evaluate.py       ← Phase 3 ✓
              (per-label precision / recall / F1 on test set)
                             │
                             ▼
              models/checkpoints/
                ├── phi_ner_best.pt   (best dev F1 checkpoint)
                ├── phi_ner_final.pt  (final epoch checkpoint)
                └── tokenizer/        (saved tokenizer files)
                             │
                             ▼
              api/main.py               ← Phase 4 ✓
              pipeline/redactor.py      (NER inference + char span extraction)
              pipeline/regex_pass.py    (rule-based safety pass)
```

## API Contract

**Request**
```json
POST /deidentify
{
  "text": "Patient John Smith, DOB 12/03/1965, MRN 4821903..."
}
```

**Response**
```json
{
  "redacted_text": "Patient [NAME], DOB [DATE], MRN [ID]...",
  "phi_spans": [
    { "start": 8,  "end": 18, "label": "NAME", "text": "John Smith",  "source": "ner",   "confidence": 1.0 },
    { "start": 25, "end": 35, "label": "DATE", "text": "12/03/1965", "source": "ner",   "confidence": 1.0 },
    { "start": 41, "end": 48, "label": "ID",   "text": "4821903",    "source": "regex", "confidence": 0.85 }
  ],
  "phi_count": 3,
  "sources": { "ner": 2, "regex": 1 }
}
```

## Infrastructure

| Component | Detail |
|---|---|
| Hardware | DGX Spark — GB10 Grace Blackwell, 121.7 GB unified memory |
| GPU arch | sm_121 (Blackwell) |
| CUDA | 12.8 |
| PyTorch | 2.7.0+cu128 |
| NeMo | 2.7.2 |
| Python | 3.12 |
| Serving | FastAPI + Uvicorn |
| Serving container | `docker/Dockerfile` (based on `nvcr.io/nvidia/nemo:24.09`) |
| Training container | `nvcr.io/nvidia/nemo:24.09` directly via `docker compose --profile training` |

## Docker Architecture

```
docker-compose.yml
  ├── api (serving)
  │     image: nemo-emr-deidentify:latest
  │     build: docker/Dockerfile
  │     port: 8000
  │     volume: ./models/checkpoints → /app/models/checkpoints (read-only)
  │     gpu: all
  │
  └── trainer (profile: training — run once)
        image: nvcr.io/nvidia/nemo:24.09
        volume: . → /app (full repo mount)
        runs: generate_synthetic → convert_to_bio → validate → train
        gpu: all
```

## Security Considerations

- No data egress — all compute is on-prem
- API runs on localhost by default; not exposed externally without explicit config
- `data/` and `models/checkpoints/` are git-ignored — no PHI in version control
- Audit log (`logs/audit.log`) records every request without storing PHI text — only SHA-256 hash, char length, phi count, labels, and latency
- Null bytes stripped from input before processing
- Batch endpoint capped at 100 notes per request

## Test Coverage (Phase 5 ✓)

41 tests across 3 files:

| File | Scope |
|---|---|
| `tests/test_regex_pass.py` | Unit — all 14 regex patterns, overlap protection, edge cases |
| `tests/test_api.py` | Integration — all endpoints, input validation, batch behaviour |
| `tests/conftest.py` | Shared test client (model loaded once per session) |

**Benchmark (GB10 Blackwell):** 143 notes/sec · p50 6.9ms · p95 7.6ms · p99 8.2ms
