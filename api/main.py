"""
FastAPI de-identification service.

Endpoints:
  GET  /health        — liveness check + model info
  POST /deidentify    — de-identify a single clinical note
  POST /deidentify/batch — de-identify a list of notes
"""

import time
from collections import Counter
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.audit import log_request
from api.models import DeidentifyRequest, DeidentifyResponse, HealthResponse, PHISpan
from pipeline.redactor import Redactor

redactor: Redactor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redactor
    print("Loading de-identification model ...")
    redactor = Redactor()
    print("Model ready.")
    yield
    redactor = None


app = FastAPI(
    title="Clinical PHI De-identification API",
    description="Detects and redacts Protected Health Information from clinical notes using BioBERT + regex.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    if redactor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model=redactor.cfg["model"]["base_model"],
        device=str(redactor.device),
    )


def _sanitize(text: str) -> str:
    return text.replace("\x00", "")


def _build_response(text: str) -> DeidentifyResponse:
    t0 = time.monotonic()
    result = redactor.deidentify(text)
    latency_ms = (time.monotonic() - t0) * 1000
    log_request(text, result["phi_spans"], latency_ms)
    return DeidentifyResponse(
        redacted_text=result["redacted_text"],
        phi_spans=[PHISpan(**s) for s in result["phi_spans"]],
        phi_count=len(result["phi_spans"]),
        sources=dict(Counter(s["source"] for s in result["phi_spans"])),
    )


@app.post("/deidentify", response_model=DeidentifyResponse)
def deidentify(request: DeidentifyRequest):
    if redactor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _build_response(_sanitize(request.text))


@app.post("/deidentify/batch", response_model=list[DeidentifyResponse])
def deidentify_batch(requests: list[DeidentifyRequest]):
    if redactor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100 notes")
    return [_build_response(_sanitize(req.text)) for req in requests]
