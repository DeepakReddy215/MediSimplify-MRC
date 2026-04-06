from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data_loader import load_ner_dataset


MODEL_DIR = ROOT_DIR / "model_cache" / "ner_finetuned"
EVAL_PATH = ROOT_DIR / "data" / "eval_results.json"

BASE_MODEL = "d4data/biomedical-ner-all"
FALLBACK_MODEL = "dmis-lab/biobert-base-cased-v1.2"

LABEL_LIST = [
    "O",
    "B-DRUG", "I-DRUG",
    "B-DISEASE", "I-DISEASE",
    "B-DOSAGE", "I-DOSAGE",
    "B-DURATION", "I-DURATION",
    "B-TEST", "I-TEST",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-ANATOMY", "I-ANATOMY",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def _dynamic(module_name: str):
    return importlib.import_module(module_name)


def _load_transformers_objects():
    transformers = _dynamic("transformers")
    return {
        "AutoTokenizer": getattr(transformers, "AutoTokenizer"),
        "AutoModelForTokenClassification": getattr(transformers, "AutoModelForTokenClassification"),
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
    raise RuntimeError("Unable to load base or fallback tokenizer for NER.")


def _normalize_label(label: str) -> str:
    if label in LABEL2ID:
        return label

    up = (label or "O").upper()
    if up == "O":
        return "O"

    if up.startswith("B-") or up.startswith("I-"):
        prefix, ent = up.split("-", 1)
    else:
        prefix, ent = "B", up

    if "CHEM" in ent or "DRUG" in ent or "MED" in ent:
        mapped = "DRUG"
    elif "DISEASE" in ent or "DIS" in ent:
        mapped = "DISEASE"
    elif "DOSAGE" in ent or "DOSE" in ent:
        mapped = "DOSAGE"
    elif "DURATION" in ent:
        mapped = "DURATION"
    elif "TEST" in ent or "LAB" in ent or "ASSAY" in ent:
        mapped = "TEST"
    elif "SYMPTOM" in ent or "SIGN" in ent:
        mapped = "SYMPTOM"
    elif "ANATOM" in ent or "CELL" in ent or "TISSUE" in ent or "GENE" in ent or "DNA" in ent or "RNA" in ent or "PROTEIN" in ent:
        mapped = "ANATOMY"
    else:
        return "O"

    candidate = f"{prefix}-{mapped}"
    return candidate if candidate in LABEL2ID else "O"


def _prepare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepped: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Preparing NER rows"):
        tokens = [str(t) for t in row.get("tokens", [])]
        labels = [_normalize_label(str(l)) for l in row.get("labels", [])]
        if not tokens or len(tokens) != len(labels):
            continue
        prepped.append({"tokens": tokens, "labels": labels})
    return prepped


def _build_hf_datasets(train_rows, val_rows, test_rows):
    datasets_mod = _dynamic("datasets")
    Dataset = getattr(datasets_mod, "Dataset")
    return {
        "train": Dataset.from_list(train_rows),
        "val": Dataset.from_list(val_rows),
        "test": Dataset.from_list(test_rows),
    }


def _tokenize_dataset(dataset, tokenizer):
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        aligned_labels = []
        for i, word_labels in enumerate(examples["labels"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev_word = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word:
                    label = _normalize_label(word_labels[word_idx])
                    label_ids.append(LABEL2ID[label])
                else:
                    label_ids.append(-100)
                prev_word = word_idx
            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    return dataset.map(tokenize_and_align, batched=True)


def _compute_metrics(eval_pred):
    seqeval = _dynamic("seqeval.metrics")
    np = _dynamic("numpy")

    predictions, labels = eval_pred
    pred_ids = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(pred_ids, labels):
        seq_true = []
        seq_pred = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_true.append(ID2LABEL[int(l)])
            seq_pred.append(ID2LABEL[int(p)])
        true_labels.append(seq_true)
        pred_labels.append(seq_pred)

    precision = float(getattr(seqeval, "precision_score")(true_labels, pred_labels))
    recall = float(getattr(seqeval, "recall_score")(true_labels, pred_labels))
    f1 = float(getattr(seqeval, "f1_score")(true_labels, pred_labels))
    return {"precision": precision, "recall": recall, "f1": f1}


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
    AutoModelForTokenClassification = transformers["AutoModelForTokenClassification"]
    TrainingArguments = transformers["TrainingArguments"]
    Trainer = transformers["Trainer"]

    train_rows = _prepare_rows(load_ner_dataset("train"))
    val_rows = _prepare_rows(load_ner_dataset("val"))
    test_rows = _prepare_rows(load_ner_dataset("test"))

    use_cuda = _is_cuda_available()
    fast_profile = _use_fast_profile(use_cuda)

    if fast_profile:
        train_rows = _limit_rows(train_rows, 2000)
        val_rows = _limit_rows(val_rows, 400)
        test_rows = _limit_rows(test_rows, 400)

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("NER train/val/test splits are empty. Run dataset setup first.")

    model_name = _resolve_model_name(AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    hf_data = _build_hf_datasets(train_rows, val_rows, test_rows)
    train_ds = _tokenize_dataset(hf_data["train"], tokenizer)
    val_ds = _tokenize_dataset(hf_data["val"], tokenizer)
    test_ds = _tokenize_dataset(hf_data["test"], tokenizer)

    batch_size = 8 if use_cuda else 2
    num_epochs = 5 if not fast_profile else 1
    max_steps = -1 if not fast_profile else 300
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
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=use_cuda,
        logging_steps=20 if fast_profile else 50,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
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

    test_pred = trainer.predict(test_ds)
    test_metrics = _compute_metrics((test_pred.predictions, test_pred.label_ids))

    _update_eval_results("ner", test_metrics)

    if fast_profile:
        print("[NER] CPU fast profile enabled (set FULL_TRAIN=1 for full training)")
    print("✓ [BioBERT NER] training complete. Saved to model_cache/ner_finetuned")
    print("✓ Metrics saved to data/eval_results.json")


if __name__ == "__main__":
    main()
