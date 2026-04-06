from __future__ import annotations

from typing import Iterable


def select_uncertain_samples(scored_rows: Iterable[dict], top_k: int = 50) -> list[dict]:
    rows = list(scored_rows)
    rows.sort(key=lambda r: float(r.get("uncertainty", 0.0)), reverse=True)
    return rows[:top_k]


def build_feedback_training_rows(rows: Iterable[dict]) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        correction = str(row.get("suggested_correction", "")).strip()
        if not correction:
            continue
        output.append(
            {
                "source": row.get("source_text", ""),
                "target": correction,
                "audience": row.get("audience", "patient"),
                "quality": "feedback_corrected",
            }
        )
    return output
