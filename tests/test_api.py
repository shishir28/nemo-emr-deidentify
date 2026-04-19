"""Integration tests for the FastAPI de-identification service."""

import pytest


SAMPLE_NOTE = (
    "Patient John Smith, DOB 12/03/1965, MRN 4821903, was reviewed on "
    "15/04/2026 at Melbourne General Hospital by Dr. Emily Tan. "
    "Contact: 0412 345 678."
)


# ── Health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_fields(self, client):
        r = client.get("/health").json()
        assert r["status"] == "ok"
        assert "model" in r
        assert "device" in r


# ── Deidentify ───────────────────────────────────────────────────────────────

class TestDeidentify:
    def test_returns_200(self, client):
        r = client.post("/deidentify", json={"text": SAMPLE_NOTE})
        assert r.status_code == 200

    def test_response_fields(self, client):
        r = client.post("/deidentify", json={"text": SAMPLE_NOTE}).json()
        assert "redacted_text" in r
        assert "phi_spans" in r
        assert "phi_count" in r
        assert "sources" in r

    def test_phi_detected(self, client):
        r = client.post("/deidentify", json={"text": SAMPLE_NOTE}).json()
        assert r["phi_count"] > 0

    def test_redacted_text_has_placeholders(self, client):
        r = client.post("/deidentify", json={"text": SAMPLE_NOTE}).json()
        assert "[NAME]" in r["redacted_text"] or "[DATE]" in r["redacted_text"]

    def test_original_phi_not_in_redacted_text(self, client):
        r = client.post("/deidentify", json={"text": SAMPLE_NOTE}).json()
        assert "John Smith" not in r["redacted_text"]
        assert "4821903" not in r["redacted_text"]

    def test_span_fields(self, client):
        spans = client.post("/deidentify", json={"text": SAMPLE_NOTE}).json()["phi_spans"]
        for span in spans:
            assert "start" in span
            assert "end" in span
            assert "label" in span
            assert "text" in span
            assert "source" in span
            assert span["source"] in ("ner", "regex")
            assert 0.0 <= span["confidence"] <= 1.0

    def test_span_positions_valid(self, client):
        body = {"text": SAMPLE_NOTE}
        r = client.post("/deidentify", json=body).json()
        for span in r["phi_spans"]:
            assert span["start"] >= 0
            assert span["end"] <= len(SAMPLE_NOTE)
            assert span["start"] < span["end"]

    def test_clean_note_has_no_phi(self, client):
        r = client.post("/deidentify", json={"text": "The medication is effective."}).json()
        assert r["phi_count"] == 0
        assert r["redacted_text"] == "The medication is effective."


# ── Validation ───────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_empty_text_rejected(self, client):
        r = client.post("/deidentify", json={"text": ""})
        assert r.status_code == 422

    def test_missing_text_field_rejected(self, client):
        r = client.post("/deidentify", json={})
        assert r.status_code == 422

    def test_text_over_limit_rejected(self, client):
        r = client.post("/deidentify", json={"text": "x" * 50001})
        assert r.status_code == 422

    def test_null_bytes_handled(self, client):
        r = client.post("/deidentify", json={"text": "Patient\x00Jane Doe"})
        assert r.status_code in (200, 422)  # either sanitised or rejected — not 500


# ── Batch ────────────────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_returns_list(self, client):
        payload = [{"text": SAMPLE_NOTE}, {"text": "Follow up in 6 weeks."}]
        r = client.post("/deidentify/batch", json=payload)
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) == 2

    def test_batch_limit_enforced(self, client):
        payload = [{"text": "note"} for _ in range(101)]
        r = client.post("/deidentify/batch", json=payload)
        assert r.status_code == 400

    def test_batch_individual_results_correct(self, client):
        payload = [{"text": SAMPLE_NOTE}, {"text": "The patient is responding well to treatment."}]
        results = client.post("/deidentify/batch", json=payload).json()
        assert results[0]["phi_count"] > 0
        assert results[1]["phi_count"] == 0
