from datetime import datetime
from pydantic import BaseModel, ConfigDict


class SafetyAlert(BaseModel):
    code: str
    title: str
    severity: str
    recommendation: str
    matched_text: str


class GroundedPoint(BaseModel):
    statement: str
    evidence_text: str
    evidence_start: int
    evidence_end: int
    confidence: float


class GlossaryEntry(BaseModel):
    term: str
    plain_meaning: str
    source_snippet: str


class ReportFeedbackRequest(BaseModel):
    clarity_rating: int
    accuracy_rating: int
    corrected_text: str = ""
    comment: str = ""


class ReportFeedbackResponse(BaseModel):
    message: str


class ReportResponse(BaseModel):
    id: str
    user_id: str
    file_name: str
    extracted_text: str
    simplified_text: str
    caregiver_text: str
    important_terms: list[str]
    glossary_entries: list[GlossaryEntry]
    safety_alerts: list[SafetyAlert]
    grounded_points: list[GroundedPoint]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UploadResponse(BaseModel):
    saved: bool
    report: ReportResponse | None
    file_name: str
    extracted_text: str
    simplified_text: str
    caregiver_text: str
    important_terms: list[str]
    glossary_entries: list[GlossaryEntry]
    safety_alerts: list[SafetyAlert]
    grounded_points: list[GroundedPoint]
    created_at: datetime


class SimplifyTextRequest(BaseModel):
    text: str
