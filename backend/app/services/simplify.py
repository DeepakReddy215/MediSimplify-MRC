import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

settings = get_settings()

MEDICAL_TERM_MAP = {
    "hypertension": "high blood pressure",
    "myocardial infarction": "heart attack",
    "edema": "swelling",
    "dyspnea": "shortness of breath",
    "tachycardia": "fast heart rate",
    "bradycardia": "slow heart rate",
    "analgesic": "pain relief medicine",
    "benign": "not cancerous",
    "malignant": "cancerous",
    "lesion": "damaged tissue area",
    "inflammation": "body swelling and irritation",
}

IMPORTANT_TERMS = sorted(MEDICAL_TERM_MAP.keys())

PLAIN_LANGUAGE_REPLACEMENTS = {
    "no evidence of": "does not show",
    "within normal limits": "looks normal",
    "unremarkable": "normal",
    "consistent with": "matches",
    "suggestive of": "may indicate",
    "cannot be excluded": "cannot be ruled out",
    "clinical correlation": "your doctor should compare this with symptoms",
    "follow-up recommended": "a follow-up check is recommended",
    "mild": "small",
    "moderate": "medium",
    "severe": "serious",
}


def _model_candidates() -> list[str]:
    candidates: list[str] = []
    configured_path = (settings.simplifier_model_path or "").strip()
    if configured_path and Path(configured_path).exists():
        candidates.append(configured_path)

    configured_base = (settings.simplifier_base_model or "").strip()
    if configured_base:
        candidates.append(configured_base)

    if "google/flan-t5-small" not in candidates:
        candidates.append("google/flan-t5-small")

    return list(dict.fromkeys(candidates))


@lru_cache(maxsize=1)
def get_summarizer():
    try:
        import importlib

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        AutoModelForSeq2SeqLM = getattr(transformers, "AutoModelForSeq2SeqLM")
        AutoTokenizer = getattr(transformers, "AutoTokenizer")
    except Exception:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    for model_name in _model_candidates():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
            model.to(device)
            model.eval()
            return tokenizer, model, device
        except Exception:
            continue

    return None


def simplify_text(text: str) -> str:
    patient_text, _ = simplify_dual_output(text)
    return patient_text


def simplify_dual_output(text: str) -> tuple[str, str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        fallback = "No readable text was found in this document."
        return fallback, fallback

    summarizer = get_summarizer()
    patient_text = _simplify_for_audience(cleaned, "patient", summarizer)
    caregiver_text = _simplify_for_audience(cleaned, "caregiver", summarizer)
    return patient_text, caregiver_text


def _build_prompt(truncated: str, audience: str) -> str:
    terms = extract_important_terms(truncated)
    glossary = ""
    if terms:
        mapped = "; ".join(f"{term} means {MEDICAL_TERM_MAP[term]}" for term in terms[:8])
        glossary = f" Keep these meanings in mind: {mapped}."

    if audience == "caregiver":
        return (
            "Rewrite this medical text for a caregiver supporting a patient at home. "
            "Use clear language with practical details: what it means, what to monitor, and what to ask the doctor next. "
            "Do not invent new diagnoses."
            f"{glossary}\nMedical text:\n{truncated}\nCaregiver explanation:"
        )

    return (
        "Rewrite this medical text for a patient in plain, clear English. "
        "Keep instructions medically accurate and concise."
        f"{glossary}\nMedical text:\n{truncated}\nSimple explanation:"
    )


def _fallback_for_audience(cleaned: str, audience: str) -> str:
    fallback = _rule_based_simplify(cleaned)
    if audience == "caregiver":
        return (
            "Caregiver-focused explanation:\n"
            f"{fallback}\n"
            "Track symptom changes and share this summary during clinician follow-up."
        )

    return (
        "Here is a simpler explanation:\n"
        f"{fallback}\n"
        "Please consult your doctor for medical decisions."
    )


def _simplify_for_audience(cleaned: str, audience: str, summarizer) -> str:
    truncated = cleaned[: settings.simplifier_prompt_max_chars]
    prompt = _build_prompt(truncated, audience)

    if summarizer:
        try:
            tokenizer, model, device = summarizer
            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            output_ids = model.generate(
                **encoded,
                max_new_tokens=settings.simplifier_max_new_tokens,
                min_new_tokens=settings.simplifier_min_new_tokens,
                num_beams=settings.simplifier_num_beams,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if not result.strip():
                raise RuntimeError("Model returned empty simplification")
            candidate = _replace_medical_terms(_post_process_generated(result))
            if not _is_too_similar(truncated, candidate):
                return candidate
        except Exception:
            pass

    return _fallback_for_audience(cleaned, audience)


def _is_too_similar(source: str, candidate: str) -> bool:
    source_norm = _normalize_for_similarity(source)
    candidate_norm = _normalize_for_similarity(candidate)
    if not source_norm or not candidate_norm:
        return False

    ratio = SequenceMatcher(None, source_norm, candidate_norm).ratio()
    # If the generated text is basically the same as source, force fallback rewrite.
    return ratio >= 0.86


def _normalize_for_similarity(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text.lower())).strip()


def _rule_based_simplify(text: str) -> str:
    simplified = _replace_medical_terms(text)

    for phrase, plain in PLAIN_LANGUAGE_REPLACEMENTS.items():
        simplified = re.sub(rf"\b{re.escape(phrase)}\b", plain, simplified, flags=re.IGNORECASE)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", simplified) if s.strip()]
    if not sentences:
        return "No readable text was found in this document."

    key_points = []
    for sentence in sentences[:4]:
        cleaned_sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
        key_points.append(f"- {cleaned_sentence}")

    return "Main points in simple language:\n" + "\n".join(key_points)


def extract_important_terms(text: str) -> list[str]:
    lowered = text.lower()
    return [term for term in IMPORTANT_TERMS if term in lowered]


def _replace_medical_terms(text: str) -> str:
    simplified = text
    for term, meaning in MEDICAL_TERM_MAP.items():
        pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
        simplified = pattern.sub(lambda m: f"{m.group(0)} ({meaning})", simplified)
    return simplified


def _post_process_generated(text: str) -> str:
    cleaned = " ".join(text.split())
    for marker in ("Medical text:", "Simple explanation:", "Caregiver explanation:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)

    return " ".join(deduped).strip() or cleaned
