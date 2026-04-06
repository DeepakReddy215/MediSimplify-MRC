from __future__ import annotations

import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_loader import load_disease_dataset


MODEL_DIR = ROOT_DIR / "model_cache" / "disease_classifier_finetuned"
EVAL_PATH = ROOT_DIR / "data" / "eval_results.json"

BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
FALLBACK_MODEL = "dmis-lab/biobert-base-cased-v1.2"


def _dynamic(module_name: str):
    return importlib.import_module(module_name)


def _load_transformers_objects():
    transformers = _dynamic("transformers")
    return {
        "AutoTokenizer": getattr(transformers, "AutoTokenizer"),
        "AutoModelForSequenceClassification": getattr(transformers, "AutoModelForSequenceClassification"),
        "TrainingArguments": getattr(transformers, "TrainingArguments"),
        "Trainer": getattr(transformers, "Trainer"),
    }


def _is_cuda_available() -> bool:
    torch = _dynamic("torch")
    return bool(torch.cuda.is_available())


def _use_fast_profile(use_cuda: bool) -> bool:
    if use_cuda:
        return False
    return os.getenv("FULL_TRAIN", "0").strip().lower() not in {"1", "true", "yes"}


def _limit_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return rows[:max_rows]


def _resolve_model_name(AutoTokenizer) -> str:
    for name in (BASE_MODEL, FALLBACK_MODEL):
        try:
            AutoTokenizer.from_pretrained(name)
            return name
        except Exception:
            continue
    raise RuntimeError("Unable to load base or fallback tokenizer for disease classifier.")


def _group_multilabel(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in tqdm(rows, desc="Grouping disease rows"):
        text = str(row.get("text", "")).strip()
        disease = str(row.get("disease", "")).strip()
        if not text or not disease:
            continue
        grouped[text].add(disease)

    out = []
    for text, diseases in grouped.items():
        out.append({"text": text, "diseases": sorted(diseases)})
    return out


def _build_label_maps(grouped_rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({d for row in grouped_rows for d in row["diseases"]})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def _vectorize_rows(rows: list[dict[str, Any]], label2id: dict[str, int]) -> list[dict[str, Any]]:
    dim = len(label2id)
    out: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Vectorizing labels"):
        vector = [0.0] * dim
        for disease in row["diseases"]:
            idx = label2id[disease]
            vector[idx] = 1.0
        out.append({"text": row["text"], "labels": vector})
    return out


def _build_hf_datasets(train_rows, val_rows, test_rows):
    datasets_mod = _dynamic("datasets")
    Dataset = getattr(datasets_mod, "Dataset")
    return {
        "train": Dataset.from_list(train_rows),
        "val": Dataset.from_list(val_rows),
        "test": Dataset.from_list(test_rows),
    }


def _tokenize(dataset, tokenizer):
    def tokenize_batch(examples):
        encoded = tokenizer(
            examples["text"],
            max_length=256,
            truncation=True,
            padding="max_length",
        )
        encoded["labels"] = examples["labels"]
        return encoded

    return dataset.map(tokenize_batch, batched=True)


def _compute_metrics(eval_pred):
    import numpy as np
    from scipy.special import expit
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    probs = expit(logits)

    preds = (probs >= 0.3).astype(int)

    zero_rows = preds.sum(axis=1) == 0
    if zero_rows.any():
        top1 = np.argmax(probs[zero_rows], axis=1)
        for i, row_idx in enumerate(np.where(zero_rows)[0]):
            preds[row_idx, top1[i]] = 1

    labels = labels.astype(int)

    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))

    return {"accuracy": accuracy, "macro_f1": macro_f1, "micro_f1": micro_f1}


def _update_eval_results(key: str, metrics: dict[str, Any]) -> None:
    if EVAL_PATH.exists():
        payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload[key] = metrics
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    transformers = _load_transformers_objects()
    AutoTokenizer = transformers["AutoTokenizer"]
    AutoModelForSequenceClassification = transformers["AutoModelForSequenceClassification"]
    TrainingArguments = transformers["TrainingArguments"]
    Trainer = transformers["Trainer"]

    train_raw = _group_multilabel(load_disease_dataset("train"))
    val_raw = _group_multilabel(load_disease_dataset("val"))
    test_raw = _group_multilabel(load_disease_dataset("test"))

    use_cuda = _is_cuda_available()
    fast_profile = _use_fast_profile(use_cuda)

    if fast_profile:
        train_raw = _limit_rows(train_raw, 2000)
        val_raw = _limit_rows(val_raw, 400)
        test_raw = _limit_rows(test_raw, 400)

    if not train_raw or not val_raw or not test_raw:
        raise ValueError("Disease train/val/test splits are empty. Run dataset setup first.")

    label2id, id2label = _build_label_maps(train_raw + val_raw + test_raw)

    train_rows = _vectorize_rows(train_raw, label2id)
    val_rows = _vectorize_rows(val_raw, label2id)
    test_rows = _vectorize_rows(test_raw, label2id)

    model_name = _resolve_model_name(AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={idx: label for idx, label in id2label.items()},
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    hf_data = _build_hf_datasets(train_rows, val_rows, test_rows)
    train_ds = _tokenize(hf_data["train"], tokenizer)
    val_ds = _tokenize(hf_data["val"], tokenizer)
    test_ds = _tokenize(hf_data["test"], tokenizer)

    batch_size = 8 if use_cuda else 2
    num_epochs = 5 if not fast_profile else 1
    max_steps = -1 if not fast_profile else 250
    warmup_steps = 1000 if not fast_profile else 30

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_cuda,
        logging_steps=20 if fast_profile else 50,
        report_to="none",
        save_total_limit=2,
    )

    torch = _dynamic("torch")

    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=_compute_metrics,
    )

    checkpoints = list(MODEL_DIR.glob("checkpoint-*")) if MODEL_DIR.exists() else []
    if checkpoints:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    (MODEL_DIR / "label2id.json").write_text(json.dumps(label2id, indent=2, ensure_ascii=True), encoding="utf-8")

    test_pred = trainer.predict(test_ds)
    test_metrics = _compute_metrics((test_pred.predictions, test_pred.label_ids))

    _update_eval_results("disease_classifier", test_metrics)

    if fast_profile:
        print("[DiseaseClassifier] CPU fast profile enabled (set FULL_TRAIN=1 for full training)")
    print("✓ [Disease Classifier] training complete. Saved to model_cache/disease_classifier_finetuned")
    print("✓ Metrics saved to data/eval_results.json")


if __name__ == "__main__":
    main()
