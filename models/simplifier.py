from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_T5_DIR = ROOT_DIR / "model_cache" / "t5_finetuned"
ALT_LOCAL_T5_DIR = ROOT_DIR / "model_cache" / "simplifier"
BASE_T5_MODEL = "google/flan-t5-base"
FALLBACK_T5_MODEL = "t5-base"

_GENERATOR = None
_USE_RULE_FALLBACK = False


def _load_t5():
    global _GENERATOR, _USE_RULE_FALLBACK
    if _GENERATOR is not None or _USE_RULE_FALLBACK:
        return _GENERATOR

    transformers = importlib.import_module("transformers")
    pipeline_fn = getattr(transformers, "pipeline")
    AutoModelForSeq2SeqLM = getattr(transformers, "AutoModelForSeq2SeqLM")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")

    candidates = [
        str(LOCAL_T5_DIR),
        str(ALT_LOCAL_T5_DIR),
    ]
    allow_remote_fallback = os.getenv("ALLOW_REMOTE_T5_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
    if allow_remote_fallback:
        candidates.extend([BASE_T5_MODEL, FALLBACK_T5_MODEL])
    for model_name in candidates:
        if model_name == str(LOCAL_T5_DIR) and not LOCAL_T5_DIR.exists():
            continue
        if model_name == str(ALT_LOCAL_T5_DIR) and not ALT_LOCAL_T5_DIR.exists():
            continue
        try:
            _GENERATOR = pipeline_fn("text2text-generation", model=model_name)
            return _GENERATOR
        except Exception:
            # Retry with explicit tokenizer/model to handle local model directories.
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                _GENERATOR = pipeline_fn("text2text-generation", model=model, tokenizer=tokenizer)
                return _GENERATOR
            except Exception:
                continue

    _USE_RULE_FALLBACK = True
    return None


def _rule_based_simplify(text: str, mode: str = "patient") -> str:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return "No readable text was found in this report."

    # Keep fallback deterministic and concise so evaluation can continue.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    short_text = " ".join(sentences[:3]) if sentences else cleaned[:600]

    if mode == "caregiver":
        return (
            "Caregiver summary: "
            + short_text
            + " Monitor symptoms, medication timings, and follow up with the doctor for any worsening signs."
        )

    return "Simple summary: " + short_text


def _build_prompt(text: str, mode: str, reading_level: str, style: str) -> str:
    mode = (mode or "patient").lower()
    level = (reading_level or "standard").lower()
    style = (style or "paragraph").lower()

    audience = "caregiver" if mode == "caregiver" else "patient"

    level_hint = {
        "basic": "Use very simple words and short sentences.",
        "standard": "Use clear plain language and keep medical meaning accurate.",
        "bullet": "Return as bullet checklist with short action points.",
    }.get(level, "Use clear plain language and keep medical meaning accurate.")

    style_hint = "Return in bullet checklist format." if style == "bullet" else "Return as a concise paragraph."

    if audience == "caregiver":
        prefix = (
            "Rewrite this medical text for a caregiver with practical context including medications, monitoring, and follow-up questions."
        )
    else:
        prefix = "Simplify the following medical text for a patient:"

    return f"{prefix} {level_hint} {style_hint}\nMedical text: {text}\nAnswer:"


def simplify(text: str, mode: str = "patient", reading_level: str = "standard", style: str = "paragraph") -> str:
    generator = _load_t5()
    if generator is None:
        return _rule_based_simplify(text, mode=mode)

    prompt = _build_prompt(text, mode=mode, reading_level=reading_level, style=style)
    try:
        out = generator(prompt, max_new_tokens=220, do_sample=False)
        result = str(out[0]["generated_text"]).strip()
        if result:
            return result
    except Exception:
        pass

    return _rule_based_simplify(text, mode=mode)


@dataclass
class T5Simplifier:
    model_name: str = BASE_T5_MODEL

    def simplify(self, text: str, mode: str = "patient") -> str:
        return simplify(text, mode=mode)
