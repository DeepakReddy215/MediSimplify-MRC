from datetime import datetime, timezone
from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field


class Report(Document):
    user_id: Indexed(PydanticObjectId)
    file_name: str
    extracted_text: str
    simplified_text: str
    caregiver_text: str = ""
    important_terms: list[str] = Field(default_factory=list)
    glossary_entries: list[dict[str, str]] = Field(default_factory=list)
    safety_alerts: list[dict[str, str]] = Field(default_factory=list)
    grounded_points: list[dict] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "reports"
