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

from utils.data_loader import load_simplification_dataset
from utils.evaluator import evaluate_simplification


MODEL_DIR = ROOT_DIR / "model_cache" / "t5_finetuned"
EVAL_PATH = ROOT_DIR / "data" / "eval_results.json"

BASE_MODEL = "google/flan-t5-base"
FALLBACK_MODEL = "t5-base"


def _dynamic(module_name: str):
    return importlib.import_module(module_name)


def _load_transformers_objects():
    transformers = _dynamic("transformers")
    return {
        "AutoTokenizer": getattr(transformers, "AutoTokenizer"),
        "AutoModelForSeq2SeqLM": getattr(transformers, "AutoModelForSeq2SeqLM"),
        "DataCollatorForSeq2Seq": getattr(transformers, "DataCollatorForSeq2Seq"),
        "Seq2SeqTrainer": getattr(transformers, "Seq2SeqTrainer"),
        "Seq2SeqTrainingArguments": getattr(transformers, "Seq2SeqTrainingArguments"),
    }


def _is_cuda_available() -> bool:
    torch = _dynamic("torch")
    return bool(torch.cuda.is_available())


def _use_fast_profile(use_cuda: bool) -> bool:
    if use_cuda:
        return False
    return os.getenv("FULL_TRAIN", "0").strip().lower() not in {"1", "true", "yes"}


def _limit_rows(rows: list[dict[str, str]], max_rows: int) -> list[dict[str, str]]:
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
    raise RuntimeError("Unable to load base or fallback tokenizer for simplifier.")


def _prepare_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for row in tqdm(rows, desc="Preparing simplification rows"):
        src = str(row.get("input", "")).strip()
        tgt = str(row.get("target", "")).strip()
        if not src or not tgt:
            continue
        cleaned.append({"input": src, "target": tgt})
    return cleaned


def _build_hf_datasets(train_rows, val_rows, test_rows):
    datasets_mod = _dynamic("datasets")
    Dataset = getattr(datasets_mod, "Dataset")
    return {
        "train": Dataset.from_list(train_rows),
        "val": Dataset.from_list(val_rows),
        "test": Dataset.from_list(test_rows),
    }


def _tokenize(dataset, tokenizer):
    pad_id = int(tokenizer.pad_token_id)

    def tokenize_batch(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=256,
            truncation=True,
            padding="max_length",
        )

        processed_labels = []
        for seq in labels["input_ids"]:
            processed_labels.append([token if token != pad_id else -100 for token in seq])

        inputs["labels"] = processed_labels
        return inputs

    return dataset.map(tokenize_batch, batched=True)


def _decode_predictions_and_labels(predictions, labels, tokenizer):
    np = _dynamic("numpy")

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if getattr(predictions, "ndim", 0) == 3:
        predictions = np.argmax(predictions, axis=-1)

    # Clamp to valid vocab range before decoding
    vocab_size = tokenizer.vocab_size
    predictions = [[max(0, min(int(t), vocab_size - 1)) for t in seq] for seq in predictions]

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_texts = [t.strip() for t in pred_texts]
    label_texts = [t.strip() for t in label_texts]
    return pred_texts, label_texts


def _compute_metrics_factory(tokenizer):
    evaluate_mod = _dynamic("evaluate")
    textstat = _dynamic("textstat")
    rouge = evaluate_mod.load("rouge")

    def _compute(eval_pred):
        predictions, labels = eval_pred
        pred_texts, label_texts = _decode_predictions_and_labels(predictions, labels, tokenizer)

        rouge_scores = rouge.compute(predictions=pred_texts, references=label_texts, use_stemmer=True)
        grades = [textstat.flesch_kincaid_grade(t) for t in pred_texts if t]
        avg_fk = sum(grades) / len(grades) if grades else 0.0

        return {
            "rouge1": float(rouge_scores.get("rouge1", 0.0)),
            "rouge2": float(rouge_scores.get("rouge2", 0.0)),
            "rougeL": float(rouge_scores.get("rougeL", 0.0)),
            "avg_fk_grade": float(avg_fk),
        }

    return _compute


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
    AutoModelForSeq2SeqLM = transformers["AutoModelForSeq2SeqLM"]
    DataCollatorForSeq2Seq = transformers["DataCollatorForSeq2Seq"]
    Seq2SeqTrainer = transformers["Seq2SeqTrainer"]
    Seq2SeqTrainingArguments = transformers["Seq2SeqTrainingArguments"]

    train_rows = _prepare_rows(load_simplification_dataset("train"))
    val_rows = _prepare_rows(load_simplification_dataset("val"))
    test_rows = _prepare_rows(load_simplification_dataset("test"))

    use_cuda = _is_cuda_available()
    fast_profile = _use_fast_profile(use_cuda)

    if fast_profile:
        train_rows = _limit_rows(train_rows, 1200)
        val_rows = _limit_rows(val_rows, 200)
        test_rows = _limit_rows(test_rows, 200)

    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Simplification train/val/test splits are empty. Run dataset setup first.")

    model_name = _resolve_model_name(AutoTokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    hf_data = _build_hf_datasets(train_rows, val_rows, test_rows)
    train_ds = _tokenize(hf_data["train"], tokenizer)
    val_ds = _tokenize(hf_data["val"], tokenizer)
    test_ds = _tokenize(hf_data["test"], tokenizer)

    batch_size = 2 if use_cuda else 1
    num_epochs = 5 if not fast_profile else 1
    max_steps = -1 if not fast_profile else 120
    warmup_steps = 100 if not fast_profile else 20

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=3e-4,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=256,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rouge2",
        greater_is_better=True,
        fp16=use_cuda,
        logging_steps=20 if fast_profile else 50,
        report_to="none",
        save_total_limit=2,
    )

    compute_metrics = _compute_metrics_factory(tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    checkpoints = list(MODEL_DIR.glob("checkpoint-*")) if MODEL_DIR.exists() else []
    if checkpoints:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    test_output = trainer.predict(test_ds)
    trainer_metrics = compute_metrics((test_output.predictions, test_output.label_ids))

    pred_texts, _ = _decode_predictions_and_labels(test_output.predictions, test_output.label_ids, tokenizer)
    sources = [row["input"] for row in test_rows]
    references = [row["target"] for row in test_rows]
    sari_bundle = evaluate_simplification(sources=sources, predictions=pred_texts, references=references)

    _update_eval_results(
        "simplification",
        {
            "trainer_metrics": trainer_metrics,
            "sari_bundle": sari_bundle,
        },
    )

    if fast_profile:
        print("[Simplifier] CPU fast profile enabled (set FULL_TRAIN=1 for full training)")
    print("✓ [T5 Simplifier] training complete. Saved to model_cache/t5_finetuned")
    print("✓ Metrics saved to data/eval_results.json")


if __name__ == "__main__":
    main()
