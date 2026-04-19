"""
Rule-based regex safety pass — catches PHI the NER model may miss.
Runs after NER and fills gaps only (does not overwrite NER predictions).
"""

import re
from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int
    label: str
    text: str
    source: str = "ner"  # "ner" or "regex"
    confidence: float = 1.0


PATTERNS = [
    # Dates
    ("DATE", r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                          # 12/03/1965
    ("DATE", r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),                          # 12-03-1965
    ("DATE", r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
              r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
              r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
              r"\s+\d{4}\b", re.IGNORECASE),                            # 12 March 2024
    ("DATE", r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
              r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
              r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
              re.IGNORECASE),                                            # March 12, 2024

    # Phone numbers
    ("CONTACT", r"\b04\d{2}\s?\d{3}\s?\d{3}\b"),                       # Australian mobile
    ("CONTACT", r"\b0[2-9]\s?\d{4}\s?\d{4}\b"),                        # Australian landline
    ("CONTACT", r"\b\+?1?\s?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"),  # US format
    ("CONTACT", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  # Email

    # IDs
    ("ID", r"\bMRN\s*[:#]?\s*\d{5,10}\b", re.IGNORECASE),              # MRN 4821903
    ("ID", r"\b\d{3}-\d{2}-\d{4}\b"),                                   # SSN
    ("ID", r"\b[A-Z]{1,3}\d{6,10}\b"),                                  # Alphanumeric IDs

    # Age
    ("AGE", r"\b(?:aged?|age:)\s*\d{1,3}\b", re.IGNORECASE),           # age 45, aged 45
    ("AGE", r"\b\d{1,3}[\s-]year[\s-]old\b", re.IGNORECASE),           # 45-year-old

    # Postcodes (Australian)
    ("LOCATION", r"\b\d{4}\b(?=\s*(?:VIC|NSW|QLD|WA|SA|TAS|ACT|NT)\b)", re.IGNORECASE),
]


def run_regex_pass(text: str, existing_spans: list[Span]) -> list[Span]:
    """Returns regex-detected spans that don't overlap existing NER spans."""
    occupied = set()
    for s in existing_spans:
        for i in range(s.start, s.end):
            occupied.add(i)

    new_spans = []
    for pattern_args in PATTERNS:
        label = pattern_args[0]
        regex = pattern_args[1]
        flags = pattern_args[2] if len(pattern_args) > 2 else 0

        for m in re.finditer(regex, text, flags):
            overlap = any(i in occupied for i in range(m.start(), m.end()))
            if not overlap:
                span = Span(
                    start=m.start(),
                    end=m.end(),
                    label=label,
                    text=m.group(),
                    source="regex",
                    confidence=0.85,
                )
                new_spans.append(span)
                for i in range(m.start(), m.end()):
                    occupied.add(i)

    return new_spans
