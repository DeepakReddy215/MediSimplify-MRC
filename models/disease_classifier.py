from __future__ import annotations

import importlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
LOCAL_CLASSIFIER_DIR = ROOT_DIR / "model_cache" / "disease_classifier_finetuned"
BASE_CLASSIFIER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
FALLBACK_CLASSIFIER_MODEL = "dmis-lab/biobert-base-cased-v1.2"

_FINETUNED = None
_ZERO_SHOT = None
_DISEASE_TO_ICD = None

_RULE_KEYWORDS = {
    "hypertension": ("Hypertension", 0.9),
    "high blood pressure": ("Hypertension", 0.85),
    "diabetes": ("Type 2 Diabetes", 0.9),
    "asthma": ("Asthma", 0.9),
    "anemia": ("Anemia", 0.9),
    "hypothyroid": ("Hypothyroidism", 0.85),
    "thyroid": ("Hypothyroidism", 0.6),
    "ckd": ("Chronic Kidney Disease", 0.9),
    "kidney disease": ("Chronic Kidney Disease", 0.8),
    "gerd": ("Gastroesophageal Reflux Disease", 0.9),
    "reflux": ("Gastroesophageal Reflux Disease", 0.75),
    "hyperlipidemia": ("Hyperlipidemia", 0.9),
    "high cholesterol": ("Hyperlipidemia", 0.8),
}


def _load_disease_to_icd_map() -> dict[str, str]:
    global _DISEASE_TO_ICD
    if _DISEASE_TO_ICD is not None:
        return _DISEASE_TO_ICD

    mapping: dict[str, str] = {}
    file_path = DATA_DIR / "disease" / "classification_train.json"
    if file_path.exists():
        rows = json.loads(file_path.read_text(encoding="utf-8"))
        counter: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            disease = str(row.get("disease", "")).strip()
            icd10 = str(row.get("icd10", "")).strip()
            if not disease or not icd10:
                continue
            counter[disease][icd10] += 1

        for disease, counts in counter.items():
            mapping[disease] = counts.most_common(1)[0][0]

    defaults = {
        "Hypertension": "I10",
        "Type 2 Diabetes": "E11",
        "Hypothyroidism": "E03.9",
        "Hyperlipidemia": "E78.5",
        "Gastroesophageal Reflux Disease": "K21.9",
        "Asthma": "J45.909",
        "Chronic Kidney Disease": "N18.9",
        "Anemia": "D64.9",
    }
    for k, v in defaults.items():
        mapping.setdefault(k, v)

    _DISEASE_TO_ICD = mapping
    return mapping


def _load_finetuned_classifier() -> tuple[Any, Any, dict[str, int]] | None:
    global _FINETUNED
    if _FINETUNED is not None:
        return _FINETUNED

    if not LOCAL_CLASSIFIER_DIR.exists():
        return None

    label_path = LOCAL_CLASSIFIER_DIR / "label2id.json"
    if not label_path.exists():
        return None

    transformers = importlib.import_module("transformers")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AutoModelForSequenceClassification = getattr(transformers, "AutoModelForSequenceClassification")

    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_CLASSIFIER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(LOCAL_CLASSIFIER_DIR))
    label2id = json.loads(label_path.read_text(encoding="utf-8"))

    _FINETUNED = (tokenizer, model, label2id)
    return _FINETUNED


def _load_zero_shot_pipeline():
    global _ZERO_SHOT
    if _ZERO_SHOT is not None:
        return _ZERO_SHOT

    transformers = importlib.import_module("transformers")
    pipeline_fn = getattr(transformers, "pipeline")
    try:
        _ZERO_SHOT = pipeline_fn("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception:
        _ZERO_SHOT = None
    return _ZERO_SHOT


def _rule_based_scores(text: str) -> dict[str, float]:
    lower = text.lower()
    scores: dict[str, float] = {}
    for keyword, (disease, score) in _RULE_KEYWORDS.items():
        if keyword in lower:
            scores[disease] = max(scores.get(disease, 0.0), score)
    return scores


def _zero_shot_scores(text: str, candidates: list[str]) -> dict[str, float]:
    if not candidates:
        return {}

    zero_shot = _load_zero_shot_pipeline()
    if zero_shot is None:
        return {}

    try:
        output = zero_shot(text, candidate_labels=candidates, multi_label=True)
    except Exception:
        return {}

    labels = output.get("labels", [])
    scores = output.get("scores", [])
    return {str(label): float(score) for label, score in zip(labels, scores)}


def _finetuned_scores(text: str) -> dict[str, float]:
    scores_batch = _finetuned_scores_batch([text], batch_size=1)
    return scores_batch[0] if scores_batch else {}


def _finetuned_scores_batch(texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
    loaded = _load_finetuned_classifier()
    if loaded is None:
        return [{} for _ in texts]

    tokenizer, model, label2id = loaded
    torch = importlib.import_module("torch")

    id2label = {idx: label for label, idx in label2id.items()}
    all_scores: list[dict[str, float]] = []

    for start in range(0, len(texts), max(1, batch_size)):
        chunk = texts[start:start + max(1, batch_size)]
        encoded = tokenizer(chunk, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

        for row in probs:
            scores: dict[str, float] = {}
            for idx, prob in enumerate(row):
                disease = id2label.get(idx)
                if disease is None:
                    continue
                scores[disease] = float(prob)
            all_scores.append(scores)

    return all_scores


def classify_diseases(
    text: str | list[str],
    top_k: int = 5,
    use_zero_shot: bool = True,
    batch_size: int = 16,
) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
    disease_to_icd = _load_disease_to_icd_map()

    if isinstance(text, list):
        texts = [str(t) for t in text]
        finetuned_batch = _finetuned_scores_batch(texts, batch_size=batch_size)
        outputs: list[list[dict[str, Any]]] = []

        for one_text, finetuned in zip(texts, finetuned_batch):
            rule_scores = _rule_based_scores(one_text)
            candidates = sorted(set(disease_to_icd.keys()) | set(rule_scores.keys()) | set(finetuned.keys()))
            zero_shot = _zero_shot_scores(one_text, candidates) if use_zero_shot and candidates else {}

            fused: dict[str, float] = defaultdict(float)
            for disease, score in finetuned.items():
                fused[disease] += 0.5 * score
            for disease, score in rule_scores.items():
                fused[disease] += 0.3 * score
            for disease, score in zero_shot.items():
                fused[disease] += 0.2 * score

            if not fused:
                outputs.append([])
                continue

            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results: list[dict[str, Any]] = []
            for disease, score in ranked:
                results.append(
                    {
                        "disease": disease,
                        "icd10": disease_to_icd.get(disease, "UNKNOWN"),
                        "confidence": round(float(score), 4),
                        "sources": {
                            "finetuned": round(float(finetuned.get(disease, 0.0)), 4),
                            "rule_based": round(float(rule_scores.get(disease, 0.0)), 4),
                            "zero_shot": round(float(zero_shot.get(disease, 0.0)), 4),
                        },
                    }
                )
            outputs.append(results)

        return outputs

    finetuned = _finetuned_scores(text)
    rule_scores = _rule_based_scores(text)

    candidates = sorted(set(disease_to_icd.keys()) | set(rule_scores.keys()) | set(finetuned.keys()))
    zero_shot = _zero_shot_scores(text, candidates) if use_zero_shot and candidates else {}

    fused: dict[str, float] = defaultdict(float)
    for disease, score in finetuned.items():
        fused[disease] += 0.5 * score
    for disease, score in rule_scores.items():
        fused[disease] += 0.3 * score
    for disease, score in zero_shot.items():
        fused[disease] += 0.2 * score

    if not fused:
        return []

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results: list[dict[str, Any]] = []
    for disease, score in ranked:
        results.append(
            {
                "disease": disease,
                "icd10": disease_to_icd.get(disease, "UNKNOWN"),
                "confidence": round(float(score), 4),
                "sources": {
                    "finetuned": round(float(finetuned.get(disease, 0.0)), 4),
                    "rule_based": round(float(rule_scores.get(disease, 0.0)), 4),
                    "zero_shot": round(float(zero_shot.get(disease, 0.0)), 4),
                },
            }
        )

    return results


@dataclass
class DiseaseClassifier:
    model_name: str = BASE_CLASSIFIER_MODEL

    def load(self) -> Any:
        transformers = importlib.import_module("transformers")
        AutoModelForSequenceClassification = getattr(transformers, "AutoModelForSequenceClassification")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")

        for candidate in (BASE_CLASSIFIER_MODEL, FALLBACK_CLASSIFIER_MODEL):
            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate)
                model = AutoModelForSequenceClassification.from_pretrained(candidate)
                return tokenizer, model
            except Exception:
                continue

        raise RuntimeError("Failed to load disease classifier base/fallback model.")

    def classify(self, text: str) -> list[dict[str, Any]]:
        return classify_diseases(text)
