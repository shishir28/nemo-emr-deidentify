# NeMo EMR De-identification

A clinical de-identification pipeline built with NVIDIA NeMo on DGX Spark. Detects and redacts Protected Health Information (PHI) from unstructured clinical notes — entirely on-premise, no patient data leaves the machine.

## What It Does

Takes raw clinical text like this:

```
Patient John Smith, DOB 12/03/1965, MRN 4821903, was seen on 04/10/2026
at Melbourne General Hospital by Dr. Emily Tan.
```

And returns:

```
Patient [NAME], DOB [DATE], MRN [ID], was seen on [DATE]
at [LOCATION] by Dr. [NAME].
```

## PHI Categories Detected

| Label | Examples |
|---|---|
| `NAME` | Patient names, doctor names |
| `DATE` | DOB, visit dates, discharge dates |
| `ID` | MRN, SSN, account numbers |
| `LOCATION` | Hospital names, addresses, cities |
| `CONTACT` | Phone numbers, email addresses |
| `AGE` | Age when over 89 |
| `PROFESSION` | Job titles that could identify a patient |

## Project Structure

```
nemo-emr-deidentify/
├── data/
│   ├── raw/          # Original i2b2 / MIMIC downloads (never committed)
│   ├── processed/    # NeMo BIO-tagged NER format
│   └── synthetic/    # Generated test notes for evaluation
├── models/
│   ├── checkpoints/  # Fine-tuned model weights (never committed)
│   └── configs/      # NeMo YAML training configs
├── pipeline/         # Core de-identification logic
├── api/              # FastAPI service
├── notebooks/        # Exploration and evaluation notebooks
├── scripts/          # Setup, data prep, and training scripts
├── docker/           # Dockerfile using official NeMo base image
└── docs/             # Architecture and design docs
```

## Prerequisites

- DGX Spark (or any machine with NVIDIA GPU, CUDA 12.8+)
- Python 3.12
- Docker (optional, recommended for production)

## Setup

```bash
git clone <repo-url>
cd nemo-emr-deidentify
bash scripts/setup_env.sh
source venv/bin/activate
python3 scripts/verify_gpu.py
```

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 | Done | Environment setup — NeMo 2.7.2, PyTorch 2.7+cu128 |
| 2 | Done | Data preparation — synthetic generator, BIO converter, validation |
| 3 | Pending | Model fine-tuning — BioMegatron NER |
| 4 | Pending | FastAPI pipeline service |
| 5 | Pending | Validation and hardening |

## Data Scripts

| Script | Purpose |
|---|---|
| `scripts/generate_synthetic.py` | Generates 500 synthetic clinical notes with annotated PHI spans |
| `scripts/convert_to_bio.py` | Converts span-annotated JSONL to NeMo BIO format (80/10/10 split) |
| `scripts/validate_data.py` | Validates BIO files — checks label consistency and reports distribution |

### Run the data pipeline

```bash
source venv/bin/activate
python3 scripts/generate_synthetic.py   # → data/synthetic/notes.jsonl
python3 scripts/convert_to_bio.py       # → data/processed/train|dev|test.txt
python3 scripts/validate_data.py        # confirms all splits pass
```

### Using real i2b2 data

Once you have the i2b2 2014 XML files, convert them to the span JSONL format and place at `data/processed/notes.jsonl`. The converter will prefer that over the synthetic file automatically.

## Key Design Decisions

- **On-premise only** — PHI never leaves the machine; all inference and training runs locally on DGX Spark
- **NeMo base toolkit** — `nemo_toolkit[nlp]` extras skipped due to `mamba-ssm` build constraints on aarch64; NLP deps installed individually
- **PyTorch 2.7+cu128** — required for GB10 Blackwell (sm_121) GPU compatibility
- **Secondary rule-based pass** — regex fallback for dates, phone numbers, MRNs to catch any NER misses

## Data Sources

Training data requires free registration:
- [i2b2 2014 De-identification Dataset](https://www.i2b2.org/NLP/DataSets/) — gold standard for PHI detection
- [MIMIC-IV](https://physionet.org/content/mimiciv/) — large-scale clinical notes (PhysioNet credentialed access)

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system design.
