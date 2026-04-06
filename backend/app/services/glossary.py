from datetime import datetime, timezone
import re
from beanie import PydanticObjectId

from app.models.medical_term_memory import MedicalTermMemory
from app.services.simplify import MEDICAL_TERM_MAP


def _snippet_for_term(text: str, term: str) -> str:
    pattern = re.compile(rf"[^.\n]*\b{re.escape(term)}\b[^.\n]*", flags=re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    return " ".join(match.group(0).split())[:220]


def build_glossary_entries(source_text: str, important_terms: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for term in important_terms:
        definition = MEDICAL_TERM_MAP.get(term.lower())
        if not definition:
            continue
        entries.append(
            {
                "term": term,
                "plain_meaning": definition,
                "source_snippet": _snippet_for_term(source_text, term),
            }
        )
    return entries


async def update_term_memory(entries: list[dict[str, str]]) -> None:
    if not entries:
        return

    now = datetime.now(timezone.utc)
    for entry in entries:
        normalized_term = entry["term"].strip().lower()
        if not normalized_term:
            continue

        existing = await MedicalTermMemory.find_one(MedicalTermMemory.term == normalized_term)
        if existing:
            existing.seen_count += 1
            existing.last_seen_at = now
            existing.definition = entry["plain_meaning"]
            await existing.save()
            continue

        await MedicalTermMemory(
            term=normalized_term,
            definition=entry["plain_meaning"],
            seen_count=1,
            last_seen_at=now,
        ).insert()
