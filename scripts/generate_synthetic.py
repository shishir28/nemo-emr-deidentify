"""
Generates synthetic clinical notes with annotated PHI spans.
Output: data/synthetic/notes.jsonl — one JSON object per line.

Each record:
  { "text": "...", "spans": [{"start": 0, "end": 10, "label": "NAME"}, ...] }
"""

import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "Michael", "Jennifer",
    "William", "Linda", "David", "Barbara", "Sarah", "Emily",
    "Ravi", "Priya", "Chen", "Wei", "Fatima", "Omar",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Wilson", "Taylor", "Kumar", "Patel",
    "Zhang", "Nguyen", "Hassan", "Ali",
]
DOCTOR_TITLES = ["Dr.", "Dr"]
HOSPITALS = [
    "Melbourne General Hospital", "St Vincent's Medical Centre",
    "Royal Brisbane Hospital", "Sydney Central Clinic",
    "Westmead Medical Centre", "Alfred Health",
]
SUBURBS = [
    "Fitzroy", "Newtown", "Paddington", "Northcote",
    "South Yarra", "Glebe", "Hawthorn",
]
CITIES = ["Melbourne", "Sydney", "Brisbane", "Perth", "Adelaide"]
DIAGNOSES = [
    "hypertension", "type 2 diabetes mellitus", "chronic kidney disease",
    "heart failure", "asthma", "COPD", "atrial fibrillation",
    "osteoarthritis", "depression", "hypothyroidism",
]
MEDICATIONS = [
    "metformin 500mg twice daily", "lisinopril 10mg once daily",
    "atorvastatin 40mg at night", "salbutamol inhaler PRN",
    "levothyroxine 50mcg once daily", "ramipril 5mg once daily",
]
PROFESSIONS = [
    "nurse", "teacher", "engineer", "accountant",
    "retired police officer", "bus driver",
]


def random_date(start_year=1940, end_year=2005):
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def random_visit_date():
    base = date(2024, 1, 1)
    return base + timedelta(days=random.randint(0, 480))


def random_mrn():
    return str(random.randint(1000000, 9999999))


def random_phone():
    return f"04{random.randint(10,99)} {random.randint(100,999)} {random.randint(100,999)}"


def build_note():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    patient_name = f"{first} {last}"

    doc_first = random.choice(FIRST_NAMES)
    doc_last = random.choice(LAST_NAMES)
    doc_title = random.choice(DOCTOR_TITLES)
    doctor_name = f"{doc_title} {doc_first} {doc_last}"

    dob = random_date()
    visit = random_visit_date()
    mrn = random_mrn()
    hospital = random.choice(HOSPITALS)
    suburb = random.choice(SUBURBS)
    city = random.choice(CITIES)
    address = f"{random.randint(1, 200)} {last} Street, {suburb}, {city}"
    phone = random_phone()
    diagnosis = random.choice(DIAGNOSES)
    medication = random.choice(MEDICATIONS)
    profession = random.choice(PROFESSIONS)
    age = visit.year - dob.year

    template = (
        f"Patient {patient_name}, DOB {dob.strftime('%d/%m/%Y')}, MRN {mrn}, "
        f"was reviewed on {visit.strftime('%d/%m/%Y')} at {hospital} "
        f"by {doctor_name}.\n\n"
        f"The patient is a {age}-year-old {profession} residing at {address}. "
        f"Contact number: {phone}.\n\n"
        f"Presenting complaint: follow-up for {diagnosis}.\n"
        f"Current medications include {medication}.\n\n"
        f"Assessment and plan documented by {doctor_name} on "
        f"{visit.strftime('%d %B %Y')}."
    )

    spans = []

    def find_and_tag(text, pattern, label, word_boundary=False):
        regex = (r'\b' + re.escape(pattern) + r'\b') if word_boundary else re.escape(pattern)
        for m in re.finditer(regex, text):
            spans.append({"start": m.start(), "end": m.end(), "label": label})

    find_and_tag(template, patient_name, "NAME")
    find_and_tag(template, dob.strftime('%d/%m/%Y'), "DATE")
    find_and_tag(template, mrn, "ID")
    find_and_tag(template, visit.strftime('%d/%m/%Y'), "DATE")
    find_and_tag(template, visit.strftime('%d %B %Y'), "DATE")
    find_and_tag(template, hospital, "LOCATION")
    find_and_tag(template, f"{doc_title} {doc_first} {doc_last}", "NAME")
    find_and_tag(template, address, "LOCATION")
    find_and_tag(template, phone, "CONTACT")
    find_and_tag(template, profession, "PROFESSION", word_boundary=True)
    find_and_tag(template, str(age), "AGE", word_boundary=True)

    spans.sort(key=lambda s: s["start"])
    return {"text": template, "spans": spans}


def main():
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "notes.jsonl"

    n = 500
    with out_file.open("w") as f:
        for _ in range(n):
            record = build_note()
            f.write(json.dumps(record) + "\n")

    print(f"Generated {n} synthetic notes → {out_file}")


if __name__ == "__main__":
    main()
