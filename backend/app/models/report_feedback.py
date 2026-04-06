from datetime import datetime, timezone

from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field


class ReportFeedback(Document):
    user_id: Indexed(PydanticObjectId)
    report_id: Indexed(PydanticObjectId)
    clarity_rating: int = Field(ge=1, le=5)
    accuracy_rating: int = Field(ge=1, le=5)
    corrected_text: str = ""
    comment: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "report_feedback"
