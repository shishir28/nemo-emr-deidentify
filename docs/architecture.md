# Architecture

## Overview

The pipeline processes raw clinical text through four sequential stages: ingestion, NER-based PHI detection, redaction, and a rule-based safety pass. The FastAPI service wraps the entire pipeline behind a single HTTP endpoint.

## System Diagram

```
                        ┌─────────────────────────────────────────┐
                        │              DGX Spark (on-prem)         │
                        │                                          │
  Clinical Note         │   ┌──────────┐      ┌────────────────┐  │
  (raw text)  ─────────►│   │  FastAPI  │─────►│  NeMo NER Model│  │
                        │   │  Service  │      │  (BioMegatron) │  │
  Redacted Note◄─────── │   │  :8000   │◄─────│  Fine-tuned on │  │
                        │   └──────────┘      │  i2b2 PHI data │  │
                        │         │           └────────────────┘  │
                        │         │                   │            │
                        │         ▼                   ▼            │
                        │   ┌──────────┐      ┌────────────────┐  │
                        │   │  Redactor │      │  Regex Safety  │  │
                        │   │  Module   │      │  Pass          │  │
                        │   └──────────┘      └────────────────┘  │
                        └─────────────────────────────────────────┘
```

## Component Breakdown

### 1. FastAPI Service (`api/`)

- Single `POST /deidentify` endpoint
- Accepts raw clinical note text
- Returns: redacted text + JSON list of detected PHI spans with labels and confidence scores
- Stateless — no patient data stored

### 2. NeMo NER Model (`models/`)

- Base model: **BioMegatron** (BERT pretrained on PubMed + clinical notes)
- Task: token classification (BIO tagging)
- Fine-tuned on: i2b2 2014 de-identification dataset
- PHI labels: `NAME`, `DATE`, `ID`, `LOCATION`, `CONTACT`, `AGE`, `PROFESSION`
- Training framework: NeMo 2.7.2 + PyTorch 2.7+cu128 on GB10 Blackwell GPU

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
BioMegatron Token Classification
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
i2b2 2014 XML files
   │
   ▼
scripts/prepare_data.py   → converts to NeMo BIO NER format
   │
   ▼
data/processed/           → train.txt / dev.txt / test.txt
   │
   ▼
scripts/train.py          → NeMo token classification training
   │   (BioMegatron + LoRA fine-tuning on DGX Spark)
   ▼
models/checkpoints/       → saved .nemo model file
   │
   ▼
api/main.py               → loads checkpoint, serves inference
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
    { "start": 8,  "end": 18, "label": "NAME", "text": "John Smith",  "confidence": 0.98 },
    { "start": 25, "end": 35, "label": "DATE", "text": "12/03/1965", "confidence": 0.97 },
    { "start": 41, "end": 48, "label": "ID",   "text": "4821903",    "confidence": 0.95 }
  ]
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
| Container | `nvcr.io/nvidia/nemo:24.09` (production) |

## Security Considerations

- No data egress — all compute is on-prem
- API runs on localhost by default; not exposed externally without explicit config
- `data/` and `models/checkpoints/` are git-ignored — no PHI in version control
- Audit log of every de-identification request planned for Phase 5
