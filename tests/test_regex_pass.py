"""Unit tests for the regex safety pass — tests each pattern independently."""

import pytest
from pipeline.regex_pass import run_regex_pass, Span


def spans_for(text):
    return run_regex_pass(text, existing_spans=[])


def labels_found(text):
    return {s.label for s in spans_for(text)}


# ── DATE patterns ────────────────────────────────────────────────────────────

class TestDatePatterns:
    def test_dd_mm_yyyy(self):
        assert "DATE" in labels_found("Seen on 12/03/1965.")

    def test_dd_mm_yy(self):
        assert "DATE" in labels_found("Admitted 01/06/24.")

    def test_dd_dash_mm_dash_yyyy(self):
        assert "DATE" in labels_found("DOB: 15-04-1980")

    def test_dd_month_yyyy(self):
        assert "DATE" in labels_found("Discharged 15 March 2024.")

    def test_month_dd_yyyy(self):
        assert "DATE" in labels_found("Visit on April 3, 2024.")

    def test_abbreviated_month(self):
        assert "DATE" in labels_found("Reviewed Jan 12, 2023.")


# ── CONTACT patterns ─────────────────────────────────────────────────────────

class TestContactPatterns:
    def test_australian_mobile(self):
        assert "CONTACT" in labels_found("Call 0412 345 678.")

    def test_australian_mobile_no_spaces(self):
        assert "CONTACT" in labels_found("Phone: 0423456789")

    def test_australian_landline(self):
        assert "CONTACT" in labels_found("Office: 03 9876 5432")

    def test_email(self):
        assert "CONTACT" in labels_found("Email: john.smith@health.vic.gov.au")

    def test_us_phone(self):
        assert "CONTACT" in labels_found("Contact (555) 867-5309")


# ── ID patterns ──────────────────────────────────────────────────────────────

class TestIDPatterns:
    def test_mrn_with_prefix(self):
        assert "ID" in labels_found("MRN: 4821903")

    def test_mrn_with_hash(self):
        assert "ID" in labels_found("MRN# 1234567")

    def test_ssn(self):
        assert "ID" in labels_found("SSN 123-45-6789")


# ── AGE patterns ─────────────────────────────────────────────────────────────

class TestAgePatterns:
    def test_age_keyword(self):
        assert "AGE" in labels_found("The patient is aged 72.")

    def test_year_old_hyphen(self):
        assert "AGE" in labels_found("A 45-year-old female.")

    def test_year_old_space(self):
        assert "AGE" in labels_found("A 65 year old male.")


# ── Overlap protection ───────────────────────────────────────────────────────

class TestOverlapProtection:
    def test_regex_does_not_overwrite_ner_span(self):
        existing = [Span(start=8, end=18, label="NAME", text="John Smith", source="ner")]
        text = "Patient John Smith called 0412 345 678."
        new_spans = run_regex_pass(text, existing_spans=existing)
        labels = {s.label for s in new_spans}
        assert "NAME" not in labels
        assert "CONTACT" in labels

    def test_no_duplicate_spans(self):
        text = "DOB: 12/03/1965"
        spans = spans_for(text)
        date_spans = [s for s in spans if s.label == "DATE"]
        starts = [s.start for s in date_spans]
        assert len(starts) == len(set(starts)), "Duplicate spans detected"


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_text(self):
        assert spans_for("") == []

    def test_no_phi(self):
        assert spans_for("The weather is nice today.") == []

    def test_phi_at_start_of_text(self):
        spans = spans_for("12/03/1965 was the patient's DOB.")
        assert any(s.label == "DATE" and s.start == 0 for s in spans)

    def test_phi_at_end_of_text(self):
        spans = spans_for("Contact number: 0412 345 678")
        assert any(s.label == "CONTACT" for s in spans)

    def test_multiple_phi_types(self):
        text = "DOB 01/01/1970, MRN: 1234567, email: a@b.com"
        found = labels_found(text)
        assert "DATE" in found
        assert "ID" in found
        assert "CONTACT" in found
