from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

RNG = random.Random(42)


NER_LABEL_MAP = {
    "chemical": "DRUG",
    "drug": "DRUG",
    "medication": "DRUG",
    "disease": "DISEASE",
    "disorder": "DISEASE",
    "symptom": "DISEASE",
    "dna": "GENE",
    "rna": "GENE",
    "protein": "GENE",
    "gene": "GENE",
    "cellline": "ANATOMY",
    "cell_line": "ANATOMY",
    "celltype": "ANATOMY",
    "cell_type": "ANATOMY",
    "tissue": "ANATOMY",
    "anatomy": "ANATOMY",
    "organ": "ANATOMY",
    "test": "TEST",
    "lab": "TEST",
    "assay": "TEST",
    "procedure": "TEST",
}


CONDITION_TO_DRUGS = {
    "Hypertension": ["Amlodipine 5mg OD", "Losartan 50mg OD", "Telmisartan 40mg OD"],
    "Diabetes T2": ["Metformin 500mg BID", "Glimepiride 1mg OD", "Sitagliptin 100mg OD"],
    "Hypothyroidism": ["Levothyroxine 50mcg OD"],
    "Hyperlipidemia": ["Atorvastatin 20mg HS", "Rosuvastatin 10mg HS"],
    "GERD": ["Omeprazole 20mg AC", "Pantoprazole 40mg AC"],
    "Asthma": ["Salbutamol inhaler PRN", "Budesonide inhaler BID"],
    "CKD": ["Sodium bicarbonate 500mg TID", "Calcium carbonate 500mg TID"],
    "Anemia": ["Ferrous sulphate 200mg BID", "Folic acid 5mg OD"],
}

CONDITION_TO_ICD10 = {
    "Hypertension": "I10",
    "Diabetes T2": "E11",
    "Hypothyroidism": "E03.9",
    "Hyperlipidemia": "E78.5",
    "GERD": "K21.9",
    "Asthma": "J45.909",
    "CKD": "N18.9",
    "Anemia": "D64.9",
}

CONDITION_TO_DISEASE = {
    "Hypertension": "Hypertension",
    "Diabetes T2": "Type 2 Diabetes",
    "Hypothyroidism": "Hypothyroidism",
    "Hyperlipidemia": "Hyperlipidemia",
    "GERD": "Gastroesophageal Reflux Disease",
    "Asthma": "Asthma",
    "CKD": "Chronic Kidney Disease",
    "Anemia": "Anemia",
}

INVESTIGATIONS = [
    "CBC",
    "KFT",
    "LFT",
    "HbA1c",
    "Lipid Profile",
    "Blood glucose fasting",
    "TSH",
    "Urine routine",
    "ECG",
    "Chest X-ray",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_compatible(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _save_hf_dataset_splits(dataset: Any, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    for split_name, split_data in dataset.items():
        rows = [_to_json_compatible(row) for row in split_data]
        _write_json(out_dir / f"{split_name}.json", rows)


def _load_hf_dataset(name: str):
    import importlib

    datasets_mod = importlib.import_module("datasets")
    load_dataset = getattr(datasets_mod, "load_dataset")
    return load_dataset(name, trust_remote_code=True)


def _load_first_available_dataset(candidates: list[str]):
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return _load_hf_dataset(candidate)
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("No dataset candidates provided")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def _guess_token_field(example: dict[str, Any]) -> str:
    for key in ("tokens", "words"):
        if key in example:
            return key
    raise KeyError("No token field found in example")


def _guess_tag_field(example: dict[str, Any]) -> str:
    for key in ("ner_tags", "tags", "labels"):
        if key in example:
            return key
    raise KeyError("No NER tag field found in example")


def _decode_labels(raw_tags: list[Any], label_names: list[str] | None) -> list[str]:
    if raw_tags and isinstance(raw_tags[0], int) and label_names:
        return [label_names[idx] if 0 <= idx < len(label_names) else "O" for idx in raw_tags]
    return [str(tag) for tag in raw_tags]


def _normalize_ner_label(label: str) -> str:
    cleaned = str(label or "O").strip()
    if not cleaned or cleaned.upper() == "O":
        return "O"

    if "-" in cleaned:
        prefix, entity = cleaned.split("-", 1)
    elif cleaned.startswith(("B", "I")) and len(cleaned) > 1:
        prefix, entity = cleaned[0], cleaned[1:]
    else:
        prefix, entity = "B", cleaned

    prefix = prefix.upper()
    if prefix not in {"B", "I"}:
        prefix = "B"

    entity_norm = entity.lower().replace("-", "_").strip("_")
    mapped = None
    for raw_key, target in NER_LABEL_MAP.items():
        if raw_key in entity_norm:
            mapped = target
            break

    if not mapped:
        return "O"

    return f"{prefix}-{mapped}"


def _build_ner_examples(dataset: Any, source_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for split_name, split_data in dataset.items():
        if len(split_data) == 0:
            continue

        example0 = split_data[0]
        token_field = _guess_token_field(example0)
        tag_field = _guess_tag_field(example0)

        label_names: list[str] | None = None
        features = split_data.features.get(tag_field)
        if features is not None and hasattr(features, "feature") and hasattr(features.feature, "names"):
            label_names = list(features.feature.names)
        elif features is not None and hasattr(features, "names"):
            label_names = list(features.names)

        for row in split_data:
            tokens = [str(t) for t in row.get(token_field, [])]
            raw_tags = row.get(tag_field, [])
            labels = _decode_labels(raw_tags, label_names)
            normalized = [_normalize_ner_label(label) for label in labels]
            if len(tokens) != len(normalized):
                continue
            out.append(
                {
                    "tokens": tokens,
                    "labels": normalized,
                    "source": source_name,
                    "split": split_name,
                }
            )

    return out


def prepare_ner_datasets() -> list[dict[str, Any]]:
    ner_dir = DATA_DIR / "ner"
    _ensure_dir(ner_dir)

    bc5cdr = _load_hf_dataset("tner/bc5cdr")
    ncbi = _load_hf_dataset("ncbi_disease")
    jnlpba = _load_hf_dataset("jnlpba")

    _save_hf_dataset_splits(bc5cdr, ner_dir / "bc5cdr")
    _save_hf_dataset_splits(ncbi, ner_dir / "ncbi_disease")
    _save_hf_dataset_splits(jnlpba, ner_dir / "jnlpba")

    merged = []
    merged.extend(_build_ner_examples(bc5cdr, "bc5cdr"))
    merged.extend(_build_ner_examples(ncbi, "ncbi_disease"))
    merged.extend(_build_ner_examples(jnlpba, "jnlpba"))

    _write_json(ner_dir / "merged_ner.json", merged)
    return merged


def _extract_text_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    complex_candidates = [
        "complex_text",
        "Expert",
        "original",
        "source",
        "input",
        "article",
        "complex",
        "text",
    ]
    simple_candidates = [
        "simple_text",
        "Simple",
        "simplified",
        "target",
        "output",
        "summary",
        "plain",
    ]

    complex_text = ""
    simple_text = ""

    for key in complex_candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            complex_text = value.strip()
            break

    for key in simple_candidates:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            simple_text = value.strip()
            break

    if complex_text and simple_text:
        return complex_text, simple_text
    return None


def _split_80_10_10(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    items = list(rows)
    RNG.shuffle(items)
    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def prepare_simplification_datasets() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    simp_dir = DATA_DIR / "simplification"
    t5_dir = simp_dir / "t5_ready"

    cochrane = _load_first_available_dataset([
        "MaximeGervais/cochrane-simplification",
        "GEM/cochrane-simplification",
    ])
    medeasi = _load_first_available_dataset([
        "cbasu/Med-EASi",
        "shtosti/Med-EASi",
        "Sumit-SaH/Med-EASi",
    ])

    _save_hf_dataset_splits(cochrane, simp_dir / "cochrane")
    _save_hf_dataset_splits(medeasi, simp_dir / "medeasi")

    pairs: list[dict[str, Any]] = []
    for source_name, dataset in (("cochrane", cochrane), ("medeasi", medeasi)):
        for _, split_data in dataset.items():
            for row in split_data:
                pair = _extract_text_pair(row)
                if not pair:
                    continue
                complex_text, simple_text = pair
                pairs.append(
                    {
                        "input": f"Simplify the following medical text for a patient: {complex_text}",
                        "target": simple_text,
                        "source": source_name,
                    }
                )

    train, val, test = _split_80_10_10(pairs)
    _write_json(t5_dir / "train.json", train)
    _write_json(t5_dir / "val.json", val)
    _write_json(t5_dir / "test.json", test)

    return pairs, {"train": train, "val": val, "test": test}


def _download_icd10_csv(url: str, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    urllib.request.urlretrieve(url, out_path)


def _load_icd10_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = str(row.get("code") or row.get("Code") or row.get("ICD10") or "").strip()
            description = str(
                row.get("description")
                or row.get("Description")
                or row.get("diagnosis")
                or row.get("name")
                or ""
            ).strip()
            if code and description:
                rows.append({"code": code, "description": description, "normalized": _normalize_text(description)})
    return rows


def _map_disease_to_icd10(term: str, icd_rows: list[dict[str, str]]) -> tuple[str, str, float]:
    normalized_term = _normalize_text(term)
    if not normalized_term:
        return "UNKNOWN", "Unknown disease", 0.0

    for row in icd_rows:
        if normalized_term == row["normalized"]:
            return row["code"], row["description"], 1.0

    best: tuple[str, str, float] | None = None
    term_tokens = set(normalized_term.split())
    for row in icd_rows:
        desc = row["normalized"]
        if normalized_term in desc or desc in normalized_term:
            score = 0.9
        else:
            desc_tokens = set(desc.split())
            overlap = len(term_tokens & desc_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(term_tokens), 1)
            score = min(0.8, max(0.4, score))

        if best is None or score > best[2]:
            best = (row["code"], row["description"], score)

    if best:
        return best

    keyword_fallback = {
        "diabetes": ("E11", "Type 2 diabetes mellitus", 0.5),
        "hypertension": ("I10", "Essential (primary) hypertension", 0.5),
        "asthma": ("J45", "Asthma", 0.5),
        "anemia": ("D64", "Other anemia", 0.5),
    }
    for keyword, output in keyword_fallback.items():
        if keyword in normalized_term:
            return output

    return "UNKNOWN", term, 0.0


def _extract_disease_spans(tokens: list[str], labels: list[str]) -> list[str]:
    entities: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            entities.append(" ".join(current).strip())
            current.clear()

    for tok, label in zip(tokens, labels):
        norm = str(label)
        label_l = norm.lower()
        is_disease = "disease" in label_l

        if not is_disease or norm.upper() == "O":
            flush()
            continue

        if label_l.startswith("b-"):
            flush()
            current.append(tok)
        elif label_l.startswith("i-"):
            if not current:
                current.append(tok)
            else:
                current.append(tok)
        else:
            current.append(tok)

    flush()
    return [e for e in entities if e]


def prepare_disease_classification_dataset() -> list[dict[str, Any]]:
    disease_dir = DATA_DIR / "disease"
    _ensure_dir(disease_dir)

    icd_path = disease_dir / "icd10_codes.csv"
    _download_icd10_csv(
        "https://raw.githubusercontent.com/kamillamagna/ICD-10-CSV/master/codes.csv",
        icd_path,
    )
    icd_rows = _load_icd10_rows(icd_path)

    dataset = _load_hf_dataset("ncbi_disease")

    records: list[dict[str, Any]] = []
    for split_data in dataset.values():
        if len(split_data) == 0:
            continue

        example0 = split_data[0]
        token_field = _guess_token_field(example0)
        tag_field = _guess_tag_field(example0)

        label_names: list[str] | None = None
        features = split_data.features.get(tag_field)
        if features is not None and hasattr(features, "feature") and hasattr(features.feature, "names"):
            label_names = list(features.feature.names)

        for row in split_data:
            tokens = [str(t) for t in row.get(token_field, [])]
            raw_tags = row.get(tag_field, [])
            labels = _decode_labels(raw_tags, label_names)
            text = " ".join(tokens).strip()
            if not text:
                continue

            disease_terms = _extract_disease_spans(tokens, labels)
            if disease_terms:
                for term in disease_terms:
                    icd10, disease_name, confidence = _map_disease_to_icd10(term, icd_rows)
                    records.append(
                        {
                            "text": text,
                            "icd10": icd10,
                            "disease": disease_name,
                            "confidence": round(float(confidence), 3),
                            "source": "ncbi_disease",
                        }
                    )
            else:
                records.append(
                    {
                        "text": text,
                        "icd10": "NONE",
                        "disease": "No disease mention",
                        "confidence": 1.0,
                        "source": "ncbi_disease",
                    }
                )

    _write_json(disease_dir / "classification_train.json", records)
    return records


def _primary_drug_name(drug: str) -> str:
    drug = drug.strip()
    name = re.sub(r"\s+\d.*$", "", drug).strip()
    return name or drug


def generate_synthetic_prescriptions(n_samples: int = 500) -> list[dict[str, Any]]:
    synth_dir = DATA_DIR / "synthetic"
    _ensure_dir(synth_dir)

    condition_pool = list(CONDITION_TO_DRUGS.keys())
    samples: list[dict[str, Any]] = []

    for i in range(1, n_samples + 1):
        n_conditions = RNG.randint(1, 3)
        conditions = RNG.sample(condition_pool, n_conditions)

        chosen_drugs: list[tuple[str, str]] = []
        for condition in conditions:
            drug = RNG.choice(CONDITION_TO_DRUGS[condition])
            chosen_drugs.append((condition, drug))

        investigations = RNG.sample(INVESTIGATIONS, RNG.randint(2, 4))

        systolic = RNG.randint(110, 160)
        diastolic = RNG.randint(70, 100)
        hr = RNG.randint(60, 100)
        spo2 = RNG.randint(95, 99)
        weight = RNG.randint(50, 95)
        age = RNG.randint(18, 85)

        med_text = "; ".join(drug for _, drug in chosen_drugs)
        condition_text = ", ".join(conditions)
        investigation_text = ", ".join(investigations)

        raw_text = (
            f"Rx note: Pt age {age}y. Dx: {condition_text}. "
            f"Vitals: BP {systolic}/{diastolic}, HR {hr}, SpO2 {spo2}%, Wt {weight}kg. "
            f"Meds: {med_text}. Ix: {investigation_text}."
        )

        simplified_text = (
            f"You are {age} years old. Your current conditions are {', '.join(CONDITION_TO_DISEASE[c] for c in conditions)}. "
            f"Your blood pressure is {systolic}/{diastolic}, heart rate is {hr}, oxygen is {spo2}%, and weight is {weight} kg. "
            f"Medicines prescribed: {', '.join(drug for _, drug in chosen_drugs)}. "
            f"Recommended tests: {investigation_text}."
        )

        entities = [
            {
                "text": _primary_drug_name(drug),
                "label": "DRUG",
                "preferred": f"{_primary_drug_name(drug)} ({CONDITION_TO_DISEASE[condition].lower()} medicine)",
            }
            for condition, drug in chosen_drugs
        ]

        diseases = [
            {
                "disease": CONDITION_TO_DISEASE[condition],
                "icd10": CONDITION_TO_ICD10[condition],
                "confidence": 1.0,
            }
            for condition in conditions
        ]

        samples.append(
            {
                "id": f"synth_{i:03d}",
                "source": "synthetic",
                "raw_text": raw_text,
                "simplified_text": simplified_text,
                "entities": entities,
                "diseases": diseases,
            }
        )

    _write_json(synth_dir / "prescriptions_500.json", samples)
    return samples


def _build_synthetic_classification_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        text = str(sample.get("raw_text", "")).strip()
        for disease_entry in sample.get("diseases", []):
            rows.append(
                {
                    "text": text,
                    "icd10": str(disease_entry.get("icd10", "UNKNOWN")),
                    "disease": str(disease_entry.get("disease", "Unknown disease")),
                    "confidence": float(disease_entry.get("confidence", 1.0)),
                    "source": "synthetic",
                    "synthetic_id": sample_id,
                }
            )
    return rows


def create_final_splits(
    merged_ner: list[dict[str, Any]],
    simplification_pairs: list[dict[str, Any]],
    disease_rows: list[dict[str, Any]],
    synthetic_samples: list[dict[str, Any]],
) -> None:
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    ner_train, ner_val, ner_test = _split_80_10_10(merged_ner)
    _write_json(train_dir / "ner_train.json", ner_train)
    _write_json(val_dir / "ner_val.json", ner_val)
    _write_json(test_dir / "ner_test.json", ner_test)

    simp_train, simp_val, simp_test = _split_80_10_10(simplification_pairs)
    _write_json(train_dir / "simplification_train.json", simp_train)
    _write_json(val_dir / "simplification_val.json", simp_val)
    _write_json(test_dir / "simplification_test.json", simp_test)

    synthetic_rows = _build_synthetic_classification_rows(synthetic_samples)

    sample_ids = [str(s.get("id", "")) for s in synthetic_samples if str(s.get("id", ""))]
    RNG.shuffle(sample_ids)
    heldout_ids = set(sample_ids[:100])

    syn_test = [row for row in synthetic_rows if row.get("synthetic_id") in heldout_ids]
    syn_rest = [row for row in synthetic_rows if row.get("synthetic_id") not in heldout_ids]

    cls_train, cls_val, cls_test = _split_80_10_10(disease_rows)
    syn_train, syn_val, _ = _split_80_10_10(syn_rest)

    disease_train = cls_train + syn_train
    disease_val = cls_val + syn_val
    disease_test = cls_test + syn_test

    RNG.shuffle(disease_train)
    RNG.shuffle(disease_val)
    RNG.shuffle(disease_test)

    _write_json(train_dir / "disease_train.json", disease_train)
    _write_json(val_dir / "disease_val.json", disease_val)
    _write_json(test_dir / "disease_test.json", disease_test)


def load_ner_dataset(split: str = "train") -> list[dict]:
    file_path = DATA_DIR / split / f"ner_{split}.json"
    if not file_path.exists():
        return []
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_simplification_dataset(split: str = "train") -> list[dict]:
    file_path = DATA_DIR / split / f"simplification_{split}.json"
    if not file_path.exists():
        return []
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_disease_dataset(split: str = "train") -> list[dict]:
    file_path = DATA_DIR / split / f"disease_{split}.json"
    if not file_path.exists():
        return []
    return json.loads(file_path.read_text(encoding="utf-8"))


def get_dataset_stats() -> dict:
    stats: dict[str, Any] = {"splits": {}}

    for split in ("train", "val", "test"):
        ner = load_ner_dataset(split)
        simp = load_simplification_dataset(split)
        disease = load_disease_dataset(split)

        ner_labels = Counter()
        ner_text_lens = []
        for row in ner:
            tokens = row.get("tokens", [])
            ner_text_lens.append(len(tokens))
            for label in row.get("labels", []):
                ner_labels[str(label)] += 1

        simp_input_lens = [len(str(row.get("input", "")).split()) for row in simp]
        simp_target_lens = [len(str(row.get("target", "")).split()) for row in simp]

        icd_dist = Counter(str(row.get("icd10", "UNKNOWN")) for row in disease)
        disease_text_lens = [len(str(row.get("text", "")).split()) for row in disease]

        stats["splits"][split] = {
            "ner": {
                "count": len(ner),
                "label_distribution": dict(ner_labels),
                "avg_tokens": round(sum(ner_text_lens) / len(ner_text_lens), 2) if ner_text_lens else 0.0,
            },
            "simplification": {
                "count": len(simp),
                "avg_input_tokens": round(sum(simp_input_lens) / len(simp_input_lens), 2) if simp_input_lens else 0.0,
                "avg_target_tokens": round(sum(simp_target_lens) / len(simp_target_lens), 2) if simp_target_lens else 0.0,
            },
            "disease": {
                "count": len(disease),
                "icd10_distribution": dict(icd_dist),
                "avg_text_tokens": round(sum(disease_text_lens) / len(disease_text_lens), 2) if disease_text_lens else 0.0,
            },
        }

    return stats


def setup_all() -> dict:
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "utils/data_loader.py --setup requires Python 3.11 or 3.12 for datasets compatibility."
        )

    print("[1/6] Preparing NER datasets...")
    merged_ner = prepare_ner_datasets()

    print("[2/6] Preparing simplification datasets...")
    simplification_pairs, _ = prepare_simplification_datasets()

    print("[3/6] Preparing disease classification dataset + ICD10 mapping...")
    disease_rows = prepare_disease_classification_dataset()

    print("[4/6] Generating synthetic prescriptions...")
    synthetic_samples = generate_synthetic_prescriptions(n_samples=500)

    print("[5/6] Creating train/val/test splits...")
    create_final_splits(
        merged_ner=merged_ner,
        simplification_pairs=simplification_pairs,
        disease_rows=disease_rows,
        synthetic_samples=synthetic_samples,
    )

    print("[6/6] Computing dataset stats...")
    stats = get_dataset_stats()
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical NLP data pipeline setup")
    parser.add_argument("--setup", action="store_true", help="Download and prepare all datasets")
    args = parser.parse_args()

    if args.setup:
        setup_all()
    else:
        print(json.dumps(get_dataset_stats(), indent=2))


if __name__ == "__main__":
    main()
