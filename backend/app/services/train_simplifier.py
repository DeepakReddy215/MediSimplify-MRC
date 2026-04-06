import argparse
import csv
import json
import os
from pathlib import Path


def _load_pairs(data_path: Path) -> list[dict[str, str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    pairs: list[dict[str, str]] = []

    if data_path.suffix.lower() == ".jsonl":
        for raw in data_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            source = str(row.get("source", "")).strip()
            target = str(row.get("target", "")).strip()
            audience = str(row.get("audience", "patient")).strip().lower()
            if source and target:
                pairs.append(
                    {
                        "source": source,
                        "target": target,
                        "audience": audience if audience in {"patient", "caregiver"} else "patient",
                    }
                )
    elif data_path.suffix.lower() == ".csv":
        with data_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = str(row.get("source", "")).strip()
                target = str(row.get("target", "")).strip()
                audience = str(row.get("audience", "patient")).strip().lower()
                if source and target:
                    pairs.append(
                        {
                            "source": source,
                            "target": target,
                            "audience": audience if audience in {"patient", "caregiver"} else "patient",
                        }
                    )
    else:
        raise ValueError("Training data must be .jsonl or .csv")

    return _dedupe_pairs(pairs)


def _load_pairs_from_mongodb(
    mongo_uri: str,
    mongo_db: str,
    mongo_collection: str,
    mongo_limit: int,
    include_private: bool,
    include_generated: bool,
    include_caregiver: bool,
    min_clarity_rating: int,
    min_accuracy_rating: int,
) -> list[dict[str, str]]:
    try:
        from pymongo import MongoClient
    except Exception as exc:
        raise RuntimeError(
            "PyMongo is required for MongoDB training source. Install backend/requirements.txt first."
        ) from exc

    client = MongoClient(mongo_uri)
    try:
        collection = client[mongo_db][mongo_collection]
        query: dict = {
            "source_text": {"$exists": True, "$ne": ""},
        }
        if not include_private:
            query["is_shared"] = True

        cursor = (
            collection
            .find(
                query,
                {
                    "source_text": 1,
                    "target_text": 1,
                    "caregiver_target_text": 1,
                    "corrected_target_text": 1,
                    "clarity_rating": 1,
                    "accuracy_rating": 1,
                    "source_type": 1,
                    "source": 1,
                    "target": 1,
                },
            )
            .sort("created_at", -1)
            .limit(mongo_limit)
        )

        pairs: list[dict[str, str]] = []
        for doc in cursor:
            source = str(doc.get("source_text") or doc.get("source") or "").strip()
            if not source:
                continue

            if include_generated:
                patient_target = str(doc.get("target_text") or doc.get("target") or "").strip()
                if patient_target:
                    pairs.append(
                        {
                            "source": source,
                            "target": patient_target,
                            "audience": "patient",
                        }
                    )

                caregiver_target = str(doc.get("caregiver_target_text") or "").strip()
                if include_caregiver and caregiver_target:
                    pairs.append(
                        {
                            "source": source,
                            "target": caregiver_target,
                            "audience": "caregiver",
                        }
                    )

            corrected_target = str(doc.get("corrected_target_text") or "").strip()
            source_type = str(doc.get("source_type") or "")
            if not corrected_target and source_type == "user_feedback":
                corrected_target = str(doc.get("target_text") or "").strip()

            clarity = doc.get("clarity_rating")
            accuracy = doc.get("accuracy_rating")
            clarity_ok = clarity is None or int(clarity) >= min_clarity_rating
            accuracy_ok = accuracy is None or int(accuracy) >= min_accuracy_rating

            if corrected_target and clarity_ok and accuracy_ok:
                pairs.append(
                    {
                        "source": source,
                        "target": corrected_target,
                        "audience": "patient",
                    }
                )

        return _dedupe_pairs(pairs)
    finally:
        client.close()


def _dedupe_pairs(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        key = (pair["source"], pair["target"], pair.get("audience", "patient"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pair)

    return deduped


def _train(
    pairs: list[dict[str, str]],
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_input_tokens: int,
    max_target_tokens: int,
) -> None:
    try:
        import importlib

        torch = importlib.import_module("torch")
        torch_utils_data = importlib.import_module("torch.utils.data")
        Dataset = getattr(torch_utils_data, "Dataset")
        transformers = importlib.import_module("transformers")
        AutoModelForSeq2SeqLM = getattr(transformers, "AutoModelForSeq2SeqLM")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")
        DataCollatorForSeq2Seq = getattr(transformers, "DataCollatorForSeq2Seq")
        Seq2SeqTrainer = getattr(transformers, "Seq2SeqTrainer")
        Seq2SeqTrainingArguments = getattr(transformers, "Seq2SeqTrainingArguments")
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install backend/requirements-training.txt "
            "in a Python 3.11 environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    split_idx = max(1, int(len(pairs) * 0.9))
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:] if len(pairs) > 1 else pairs[:1]

    class SimplificationDataset(Dataset):
        def __init__(self, rows: list[dict[str, str]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            row = self.rows[index]
            audience = row.get("audience", "patient")
            if audience == "caregiver":
                prompt = (
                    "Rewrite this medical text for a caregiver supporting a patient at home. "
                    "Keep treatment details accurate and include practical monitoring notes.\n"
                    f"Medical text: {row['source']}\n"
                    "Caregiver explanation:"
                )
            else:
                prompt = (
                    "Simplify the following medical text for a patient with no medical background. "
                    "Keep treatment details accurate.\n"
                    f"Medical text: {row['source']}\n"
                    "Simple explanation:"
                )

            model_inputs = tokenizer(
                prompt,
                truncation=True,
                max_length=max_input_tokens,
            )
            labels = tokenizer(
                text_target=row["target"],
                truncation=True,
                max_length=max_target_tokens,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    train_dataset = SimplificationDataset(train_pairs)
    eval_dataset = SimplificationDataset(eval_pairs)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_eval = len(eval_pairs) > 0
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        gradient_accumulation_steps=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="epoch" if use_eval else "no",
        save_strategy="epoch" if use_eval else "no",
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        logging_steps=20,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if use_eval else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune simplifier model for this backend")
    parser.add_argument("--source", choices=["mongodb", "file", "both"], default="mongodb")
    parser.add_argument("--data", help="Path to .jsonl or .csv with source,target columns")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URL", "mongodb://localhost:27017"))
    parser.add_argument("--mongo-db", default=os.getenv("DATABASE_NAME", "medical_report_simplifier"))
    parser.add_argument("--mongo-collection", default="training_samples")
    parser.add_argument("--mongo-limit", type=int, default=5000)
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--include-generated", action="store_true")
    parser.add_argument("--include-caregiver", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-clarity-rating", type=int, default=4)
    parser.add_argument("--min-accuracy-rating", type=int, default=4)
    parser.add_argument("--base-model", default="google/flan-t5-small")
    parser.add_argument("--output-dir", default="./model_cache/simplifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-input-tokens", type=int, default=384)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_pairs: list[dict[str, str]] = []

    if args.source in {"mongodb", "both"}:
        mongo_pairs = _load_pairs_from_mongodb(
            mongo_uri=args.mongo_uri,
            mongo_db=args.mongo_db,
            mongo_collection=args.mongo_collection,
            mongo_limit=args.mongo_limit,
            include_private=args.include_private,
            include_generated=args.include_generated,
            include_caregiver=args.include_caregiver,
            min_clarity_rating=args.min_clarity_rating,
            min_accuracy_rating=args.min_accuracy_rating,
        )
        all_pairs.extend(mongo_pairs)
        print(f"Loaded {len(mongo_pairs)} curated pairs from MongoDB")

    if args.source in {"file", "both"}:
        if not args.data:
            raise ValueError("--data is required when --source is file or both")
        data_path = Path(args.data)
        file_pairs = _load_pairs(data_path)
        all_pairs.extend(file_pairs)
        print(f"Loaded {len(file_pairs)} pairs from file")

    pairs = _dedupe_pairs(all_pairs)
    print(f"Using {len(pairs)} unique training pairs")

    if len(pairs) < 20:
        raise ValueError(
            "Need at least 20 training pairs. Add more corrected feedback, or include generated pairs with --include-generated."
        )

    _train(
        pairs=pairs,
        base_model=args.base_model,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_input_tokens=args.max_input_tokens,
        max_target_tokens=args.max_target_tokens,
    )

    print("Training complete.")
    print(f"Set SIMPLIFIER_MODEL_PATH={output_dir.resolve()} in backend/.env")


if __name__ == "__main__":
    main()
