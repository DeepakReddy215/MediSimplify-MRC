from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"


def evaluate_ner(y_true: list[list[str]], y_pred: list[list[str]]) -> dict[str, Any]:
    import importlib

    seqeval_metrics = importlib.import_module("seqeval.metrics")
    seq_classification_report = getattr(seqeval_metrics, "classification_report")
    seq_f1_score = getattr(seqeval_metrics, "f1_score")
    seq_precision_score = getattr(seqeval_metrics, "precision_score")
    seq_recall_score = getattr(seqeval_metrics, "recall_score")

    report = seq_classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "precision": float(seq_precision_score(y_true, y_pred)),
        "recall": float(seq_recall_score(y_true, y_pred)),
        "f1": float(seq_f1_score(y_true, y_pred)),
        "per_entity": report,
    }


def evaluate_simplification(
    sources: list[str],
    predictions: list[str],
    references: list[str],
) -> dict[str, Any]:
    import importlib

    evaluate = importlib.import_module("evaluate")
    textstat = importlib.import_module("textstat")

    if not (len(sources) == len(predictions) == len(references)):
        raise ValueError("sources, predictions, references must have same length")

    try:
        sari_metric = evaluate.load("sari")
    except (ImportError, Exception) as e:
        print(f"[WARN] SARI metric unavailable: {e}. Skipping SARI.")
        sari_score = 0.0
    else:
        sari = sari_metric.compute(
            sources=sources,
            predictions=predictions,
            references=[[ref] for ref in references],
        )
        sari_score = float(sari.get("sari", 0.0))

    rouge_metric = evaluate.load("rouge")

    rouge = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    before_grades = [textstat.flesch_kincaid_grade(text) for text in sources if text.strip()]
    after_grades = [textstat.flesch_kincaid_grade(text) for text in predictions if text.strip()]

    avg_before = sum(before_grades) / len(before_grades) if before_grades else 0.0
    avg_after = sum(after_grades) / len(after_grades) if after_grades else 0.0

    return {
        "sari": sari_score,
        "rouge1": float(rouge.get("rouge1", 0.0)),
        "rouge2": float(rouge.get("rouge2", 0.0)),
        "rougeL": float(rouge.get("rougeL", 0.0)),
        "flesch_kincaid_before": round(avg_before, 3),
        "flesch_kincaid_after": round(avg_after, 3),
        "grade_improvement": round(avg_before - avg_after, 3),
    }


def _icd_chapter(code: str) -> str:
    code = (code or "").strip().upper()
    if not code or code in {"NONE", "UNKNOWN"}:
        return code or "UNKNOWN"
    first = code[0]
    if first.isalpha():
        return first
    return "OTHER"


def evaluate_disease_classification(
    y_true: list[str],
    y_pred: list[str],
    y_proba: list[list[float]] | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    import importlib

    sklearn_metrics = importlib.import_module("sklearn.metrics")
    sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")
    accuracy_score = getattr(sklearn_metrics, "accuracy_score")
    f1_score = getattr(sklearn_metrics, "f1_score")
    roc_auc_score = getattr(sklearn_metrics, "roc_auc_score")
    label_binarize = getattr(sklearn_preprocessing, "label_binarize")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    results: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc": None,
    }

    if y_proba is not None and len(labels) > 1:
        try:
            y_true_bin = label_binarize(y_true, classes=labels)
            auc = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro")
            results["auc_roc"] = float(auc)
        except Exception:
            results["auc_roc"] = None

    chapter_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for true_code, pred_code in zip(y_true, y_pred):
        t = _icd_chapter(true_code)
        p = _icd_chapter(pred_code)
        chapter_matrix[t][p] += 1

    results["confusion_by_icd_chapter"] = {
        true_ch: dict(pred_counts) for true_ch, pred_counts in chapter_matrix.items()
    }

    return results


def save_eval_results(results: dict[str, Any], output_path: Path | None = None) -> Path:
    target = output_path or (DATA_DIR / "eval_results.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    return target
