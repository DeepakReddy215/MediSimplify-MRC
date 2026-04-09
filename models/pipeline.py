from __future__ import annotations

from typing import Any

from models.disease_classifier import classify_diseases
from models.ner_model import extract_entities
from models.simplifier import simplify


def run_pipeline(raw_text: str, reading_level: str = "standard", use_zero_shot: bool = True) -> dict[str, Any]:
    entities = extract_entities(raw_text)
    patient_text = simplify(raw_text, mode="patient", reading_level=reading_level)
    caregiver_text = simplify(raw_text, mode="caregiver", reading_level=reading_level)
    diseases = classify_diseases(raw_text, use_zero_shot=use_zero_shot)

    return {
        "raw_text": raw_text,
        "simplified_text": patient_text,
        "caregiver_text": caregiver_text,
        "entities": entities,
        "diseases": diseases,
    }
