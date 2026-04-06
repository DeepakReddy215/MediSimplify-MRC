import re
from typing import NamedTuple


class _SentenceSpan(NamedTuple):
    text: str
    start: int
    end: int


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with", "your", "you",
}


def build_grounded_points(source_text: str, simplified_text: str, max_points: int = 6) -> list[dict]:
    source = source_text or ""
    simplified = simplified_text or ""
    if not source.strip() or not simplified.strip():
        return []

    source_sentences = _split_sentences_with_spans(source)
    simplified_sentences = [s.strip() for s in _split_sentences(simplified) if s.strip()]

    grounded: list[dict] = []
    used_source_ranges: set[tuple[int, int]] = set()

    for statement in simplified_sentences:
        best = _best_source_match(statement, source_sentences)
        if not best:
            continue

        sentence_span, confidence = best
        span_key = (sentence_span.start, sentence_span.end)
        if span_key in used_source_ranges:
            continue

        used_source_ranges.add(span_key)
        grounded.append(
            {
                "statement": statement,
                "evidence_text": sentence_span.text,
                "evidence_start": sentence_span.start,
                "evidence_end": sentence_span.end,
                "confidence": round(confidence, 2),
            }
        )

        if len(grounded) >= max_points:
            break

    return grounded


def _split_sentences_with_spans(text: str) -> list[_SentenceSpan]:
    spans: list[_SentenceSpan] = []
    for match in re.finditer(r"[^.!?\n]+[.!?]?", text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        spans.append(_SentenceSpan(sentence, match.start(), match.end()))
    return spans


def _split_sentences(text: str) -> list[str]:
    return re.findall(r"[^.!?\n]+[.!?]?", text)


def _best_source_match(statement: str, source_sentences: list[_SentenceSpan]) -> tuple[_SentenceSpan, float] | None:
    statement_tokens = _tokenize(statement)
    if not statement_tokens:
        return None

    best_sentence: _SentenceSpan | None = None
    best_score = 0.0

    for source in source_sentences:
        source_tokens = _tokenize(source.text)
        if not source_tokens:
            continue

        overlap = statement_tokens & source_tokens
        if not overlap:
            continue

        precision = len(overlap) / max(len(statement_tokens), 1)
        recall = len(overlap) / max(len(source_tokens), 1)
        score = 0.7 * precision + 0.3 * recall

        if score > best_score:
            best_score = score
            best_sentence = source

    if not best_sentence or best_score < 0.2:
        return None

    return best_sentence, min(best_score, 0.99)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", text.lower())
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 2}
