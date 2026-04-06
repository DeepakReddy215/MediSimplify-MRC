from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _run(script: str, extra_env: dict[str, str] | None = None) -> None:
    module_name = script.replace("/", ".").replace(".py", "")
    command = [sys.executable, "-m", module_name]
    print("Running:", " ".join(command))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(command, check=True, cwd=str(ROOT_DIR), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all model training + evaluation steps")
    parser.add_argument("--skip-ner", action="store_true", help="skip NER training")
    parser.add_argument("--skip-simplifier", action="store_true", help="skip T5 simplifier training")
    parser.add_argument("--skip-classifier", action="store_true", help="skip disease classifier training")
    parser.add_argument("--eval-only", action="store_true", help="skip all training and run evaluation only")
    parser.add_argument("--full-train", action="store_true", help="disable CPU fast profile for training")
    parser.add_argument("--full-eval", action="store_true", help="disable CPU limited profile for evaluation")
    args = parser.parse_args()

    profile_env: dict[str, str] = {}
    if args.full_train:
        profile_env["FULL_TRAIN"] = "1"
    if args.full_eval:
        profile_env["FULL_EVAL"] = "1"

    if not args.eval_only:
        if not args.skip_ner:
            _run("training/train_ner.py", profile_env)
        if not args.skip_simplifier:
            _run("training/train_simplifier.py", profile_env)
        if not args.skip_classifier:
            _run("training/train_disease_classifier.py", profile_env)

    _run("training/evaluate_pipeline.py", profile_env)


if __name__ == "__main__":
    main()
