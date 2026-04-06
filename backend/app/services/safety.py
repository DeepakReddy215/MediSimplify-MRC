import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyRule:
    code: str
    title: str
    severity: str
    recommendation: str
    pattern: str


SAFETY_RULES: list[SafetyRule] = [
    SafetyRule(
        code="urgent_chest_pain",
        title="Possible cardiac emergency signs",
        severity="urgent",
        recommendation="Seek emergency care immediately, especially if chest pain is severe or persistent.",
        pattern=r"\b(chest pain|chest tightness|heart attack|myocardial infarction)\b",
    ),
    SafetyRule(
        code="urgent_stroke_signs",
        title="Possible stroke warning signs",
        severity="urgent",
        recommendation="Call emergency services immediately if weakness, slurred speech, or facial droop is present.",
        pattern=r"\b(stroke|slurred speech|facial droop|one-sided weakness|hemiparesis)\b",
    ),
    SafetyRule(
        code="urgent_breathing_distress",
        title="Breathing distress indicators",
        severity="urgent",
        recommendation="Seek urgent care if breathing difficulty is worsening or severe.",
        pattern=r"\b(shortness of breath|severe dyspnea|respiratory distress|cannot breathe)\b",
    ),
    SafetyRule(
        code="urgent_anaphylaxis",
        title="Possible severe allergic reaction",
        severity="urgent",
        recommendation="Use emergency allergy treatment if prescribed and seek immediate emergency care.",
        pattern=r"\b(anaphylaxis|throat swelling|tongue swelling|lip swelling)\b",
    ),
    SafetyRule(
        code="warning_sepsis_fever",
        title="Infection risk red flags",
        severity="warning",
        recommendation="Contact a clinician promptly if fever is high or associated with confusion or low blood pressure.",
        pattern=r"\b(sepsis|high fever|fever with chills|confusion|hypotension)\b",
    ),
    SafetyRule(
        code="warning_bleeding",
        title="Potential serious bleeding",
        severity="warning",
        recommendation="Seek urgent medical advice if bleeding is heavy, recurrent, or accompanied by dizziness.",
        pattern=r"\b(internal bleeding|blood in stool|vomiting blood|hemorrhage|heavy bleeding)\b",
    ),
]


def detect_clinical_safety_alerts(text: str) -> list[dict[str, str]]:
    normalized = " ".join((text or "").split()).lower()
    if not normalized:
        return []

    alerts: list[dict[str, str]] = []
    for rule in SAFETY_RULES:
        match = re.search(rule.pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue

        alerts.append(
            {
                "code": rule.code,
                "title": rule.title,
                "severity": rule.severity,
                "recommendation": rule.recommendation,
                "matched_text": match.group(0),
            }
        )

    return alerts
