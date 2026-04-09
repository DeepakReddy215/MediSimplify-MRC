from __future__ import annotations

import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.disease_classifier import classify_diseases
from models.ner_model import extract_entities
from models.pipeline import run_pipeline
from models.simplifier import simplify
from utils.data_loader import load_disease_dataset, load_ner_dataset, load_simplification_dataset
from utils.evaluator import (
    evaluate_disease_classification,
    evaluate_ner,
    evaluate_simplification,
)


EVAL_PATH = ROOT_DIR / "data" / "eval_results.json"
SYNTH_PATH = ROOT_DIR / "data" / "synthetic" / "prescriptions_500.json"


def _is_cuda_available() -> bool:
    try:
        torch = importlib.import_module("torch")
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _use_limited_eval(use_cuda: bool) -> bool:
    if use_cuda:
        return False
    return os.getenv("FULL_EVAL", "0").strip().lower() not in {"1", "true", "yes"}


def _update_eval_results(key: str, metrics: dict[str, Any]) -> None:
    def _to_json_compatible(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _to_json_compatible(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_json_compatible(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    if EVAL_PATH.exists():
        payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload[key] = _to_json_compatible(metrics)
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _evaluate_ner(max_samples: int | None = None) -> dict[str, Any]:
    rows = load_ner_dataset("test")
    if max_samples is not None:
        rows = rows[:max_samples]

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []

    for row in tqdm(rows, desc="Evaluating NER"):
        tokens = row.get("tokens", [])
        gold = row.get("labels", [])
        pred = extract_entities(tokens)
        pred_labels = pred.get("labels", ["O"] * len(tokens)) if isinstance(pred, dict) else ["O"] * len(tokens)

        if len(gold) != len(pred_labels):
            min_len = min(len(gold), len(pred_labels))
            gold = gold[:min_len]
            pred_labels = pred_labels[:min_len]

        y_true.append([str(x) for x in gold])
        y_pred.append([str(x) for x in pred_labels])

    return evaluate_ner(y_true, y_pred)


def _evaluate_simplification(max_samples: int | None = None) -> dict[str, Any]:
    rows = load_simplification_dataset("test")
    if max_samples is not None:
        rows = rows[:max_samples]

    sources: list[str] = []
    refs: list[str] = []
    preds: list[str] = []

    for row in tqdm(rows, desc="Evaluating simplification"):
        src = str(row.get("input", ""))
        tgt = str(row.get("target", ""))
        if not src or not tgt:
            continue
        pred = simplify(src, mode="patient")
        sources.append(src)
        refs.append(tgt)
        preds.append(pred)

    return evaluate_simplification(sources=sources, predictions=preds, references=refs)


def _evaluate_disease(max_samples: int | None = None) -> dict[str, Any]:
    disease_data = load_disease_dataset("test")
    if max_samples is not None:
        disease_data = disease_data[:max_samples]

    # Keep disease evaluation short on CPU.
    disease_data = disease_data[:50]

    y_true: list[str] = []
    y_pred: list[str] = []

    prepared: list[tuple[str, str]] = []
    for row in disease_data:
        text = str(row.get("text", "")).strip()
        true_icd = str(row.get("icd10", "UNKNOWN")).strip() or "UNKNOWN"
        if text:
            prepared.append((text, true_icd))

    batch_size = 16
    progress = tqdm(total=len(prepared), desc="Evaluating disease classification")
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        texts = [item[0] for item in batch]
        batch_preds = classify_diseases(texts, use_zero_shot=False, batch_size=batch_size)

        if not isinstance(batch_preds, list):
            batch_preds = [[] for _ in texts]

        for (_, true_icd), preds in zip(batch, batch_preds):
            pred_list = preds if isinstance(preds, list) else []
            pred_icd = str(pred_list[0].get("icd10", "NONE")) if pred_list else "NONE"
            y_true.append(true_icd)
            y_pred.append(pred_icd)

        progress.update(len(batch))

    progress.close()

    return evaluate_disease_classification(y_true=y_true, y_pred=y_pred)


def _evaluate_synthetic() -> dict[str, Any]:
    if not SYNTH_PATH.exists():
        return {"count": 0, "message": "synthetic file missing"}

    samples = json.loads(SYNTH_PATH.read_text(encoding="utf-8"))
    rng = random.Random(42)
    chosen = samples if len(samples) <= 20 else rng.sample(samples, 20)

    sources: list[str] = []
    refs: list[str] = []
    preds: list[str] = []
    disease_jaccards: list[float] = []

    for sample in tqdm(chosen, desc="Evaluating synthetic pipeline samples"):
        raw_text = str(sample.get("raw_text", ""))
        ref_simple = str(sample.get("simplified_text", ""))

        out = run_pipeline(raw_text=raw_text, use_zero_shot=False)
        pred_simple = str(out.get("simplified_text", ""))

        sources.append(raw_text)
        refs.append(ref_simple)
        preds.append(pred_simple)

        true_icd = {str(d.get("icd10", "")) for d in sample.get("diseases", []) if str(d.get("icd10", ""))}
        pred_icd = {str(d.get("icd10", "")) for d in out.get("diseases", []) if str(d.get("icd10", ""))}

        if not true_icd and not pred_icd:
            disease_jaccards.append(1.0)
        else:
            inter = len(true_icd & pred_icd)
            union = max(len(true_icd | pred_icd), 1)
            disease_jaccards.append(inter / union)

    simp_metrics = evaluate_simplification(sources=sources, predictions=preds, references=refs)
    avg_jaccard = sum(disease_jaccards) / len(disease_jaccards) if disease_jaccards else 0.0

    return {
        "count": len(chosen),
        "simplification": simp_metrics,
        "avg_disease_jaccard": round(avg_jaccard, 4),
    }


def _print_report(ner: dict[str, Any], simp: dict[str, Any], disease: dict[str, Any]) -> None:
    print("╔══════════════════════════════════╗")
    print("║   PIPELINE EVALUATION REPORT    ║")
    print("╠══════════════════════════════════╣")
    print(f"║ NER        F1:  {ner.get('f1', 0.0):7.3f}           ║")
    print(f"║            Precision: {ner.get('precision', 0.0):7.3f}     ║")
    print(f"║            Recall:    {ner.get('recall', 0.0):7.3f}     ║")
    print("╠══════════════════════════════════╣")
    print(f"║ Simplification SARI:  {simp.get('sari', 0.0):7.3f}     ║")
    print(f"║            ROUGE-2:   {simp.get('rouge2', 0.0):7.3f}     ║")
    print(
        f"║            FK Grade:  {simp.get('flesch_kincaid_before', 0.0):4.1f} → {simp.get('flesch_kincaid_after', 0.0):4.1f}    ║"
    )
    print("╠══════════════════════════════════╣")
    print(f"║ Disease    Macro F1:  {disease.get('macro_f1', 0.0):7.3f}     ║")
    print(f"║            Accuracy:  {disease.get('accuracy', 0.0):7.3f}     ║")
    print("╚══════════════════════════════════╝")


def main() -> None:
    limited_eval = _use_limited_eval(_is_cuda_available())
    ner_max = 600 if limited_eval else None
    simp_max = 120 if limited_eval else None
    disease_max = 600 if limited_eval else None

    ner_metrics = _evaluate_ner(max_samples=ner_max)
    simp_metrics = _evaluate_simplification(max_samples=simp_max)
    disease_metrics = _evaluate_disease(max_samples=disease_max)
    synthetic_metrics = _evaluate_synthetic()

    report = {
        "ner": ner_metrics,
        "simplification": simp_metrics,
        "disease": disease_metrics,
        "synthetic_20": synthetic_metrics,
    }
    _update_eval_results("end_to_end", report)

    if limited_eval:
        print("[PipelineEval] CPU limited profile enabled (set FULL_EVAL=1 for full evaluation)")
    _print_report(ner_metrics, simp_metrics, disease_metrics)


if __name__ == "__main__":
    main()
