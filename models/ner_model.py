from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_NER_DIR = ROOT_DIR / "model_cache" / "ner_finetuned"
BASE_NER_MODEL = "d4data/biomedical-ner-all"
FALLBACK_NER_MODEL = "dmis-lab/biobert-base-cased-v1.2"

_NER_PIPELINE = None


def _normalize_entity_label(raw_label: str) -> str:
    label = (raw_label or "").upper().replace("_", "-")
    if not label:
        return "O"

    if label.startswith("B-") or label.startswith("I-"):
        prefix = label[:2]
        entity = label[2:]
    else:
        prefix = "B-"
        entity = label

    if "CHEM" in entity or "DRUG" in entity or "MED" in entity:
        mapped = "DRUG"
    elif "DIS" in entity or "DISEASE" in entity:
        mapped = "DISEASE"
    elif "DOSAGE" in entity or "DOSE" in entity:
        mapped = "DOSAGE"
    elif "DURATION" in entity:
        mapped = "DURATION"
    elif "TEST" in entity or "LAB" in entity or "ASSAY" in entity:
        mapped = "TEST"
    elif "SYMPTOM" in entity or "SIGN" in entity:
        mapped = "SYMPTOM"
    elif "ANATOM" in entity or "CELL" in entity or "TISSUE" in entity or "ORGAN" in entity:
        mapped = "ANATOMY"
    elif "DNA" in entity or "RNA" in entity or "PROTEIN" in entity or "GENE" in entity:
        mapped = "ANATOMY"
    else:
        return "O"

    return f"{prefix}{mapped}"


def _load_ner_pipeline():
    global _NER_PIPELINE
    if _NER_PIPELINE is not None:
        return _NER_PIPELINE

    transformers = importlib.import_module("transformers")
    pipeline_fn = getattr(transformers, "pipeline")

    model_candidates = [str(LOCAL_NER_DIR), BASE_NER_MODEL, FALLBACK_NER_MODEL]
    for model_name in model_candidates:
        if model_name == str(LOCAL_NER_DIR) and not LOCAL_NER_DIR.exists():
            continue
        try:
            _NER_PIPELINE = pipeline_fn("token-classification", model=model_name, aggregation_strategy="simple")
            return _NER_PIPELINE
        except Exception:
            continue

    raise RuntimeError("Failed to load NER model from local cache and fallback models.")


def _token_spans(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    spans: list[tuple[int, int]] = []
    parts: list[str] = []
    cursor = 0
    for idx, token in enumerate(tokens):
        if idx > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        end = cursor
        spans.append((start, end))
    return "".join(parts), spans


def _labels_from_entities(tokens: list[str], spans: list[tuple[int, int]], entities: list[dict[str, Any]]) -> list[str]:
    labels = ["O"] * len(tokens)
    for ent in entities:
        label = _normalize_entity_label(str(ent.get("entity_group") or ent.get("entity") or ""))
        if label == "O":
            continue

        start = int(ent.get("start", -1))
        end = int(ent.get("end", -1))
        if start < 0 or end < 0:
            continue

        entity_type = label.split("-", 1)[1]
        first = True
        for idx, (tok_start, tok_end) in enumerate(spans):
            overlap = not (tok_end <= start or tok_start >= end)
            if not overlap:
                continue
            labels[idx] = f"{'B' if first else 'I'}-{entity_type}"
            first = False
    return labels


def extract_entities(text_or_tokens: str | list[str]) -> Any:
    ner_pipe = _load_ner_pipeline()

    if isinstance(text_or_tokens, list):
        tokens = [str(t) for t in text_or_tokens]
        text, spans = _token_spans(tokens)
        predictions = ner_pipe(text)
        labels = _labels_from_entities(tokens, spans, predictions)
        entities = []
        for pred in predictions:
            label = _normalize_entity_label(str(pred.get("entity_group") or pred.get("entity") or ""))
            if label == "O":
                continue
            entities.append(
                {
                    "text": str(pred.get("word", "")).strip(),
                    "label": label.split("-", 1)[1],
                    "score": float(pred.get("score", 0.0)),
                    "start": int(pred.get("start", -1)),
                    "end": int(pred.get("end", -1)),
                }
            )
        return {"tokens": tokens, "labels": labels, "entities": entities}

    text = str(text_or_tokens)
    predictions = ner_pipe(text)
    entities = []
    for pred in predictions:
        label = _normalize_entity_label(str(pred.get("entity_group") or pred.get("entity") or ""))
        if label == "O":
            continue
        entities.append(
            {
                "text": str(pred.get("word", "")).strip(),
                "label": label.split("-", 1)[1],
                "score": float(pred.get("score", 0.0)),
                "start": int(pred.get("start", -1)),
                "end": int(pred.get("end", -1)),
            }
        )
    return entities


@dataclass
class BioBERTNER:
    model_name: str = BASE_NER_MODEL

    def load(self) -> Any:
        return _load_ner_pipeline()

    def predict(self, text: str) -> list[dict[str, Any]]:
        result = extract_entities(text)
        return result if isinstance(result, list) else result.get("entities", [])
