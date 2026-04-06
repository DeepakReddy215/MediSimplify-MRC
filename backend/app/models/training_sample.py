from datetime import datetime, timezone

from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field


class TrainingSample(Document):
    user_id: Indexed(PydanticObjectId)
    report_id: Indexed(PydanticObjectId, unique=True)
    source_text: str
    target_text: str
    caregiver_target_text: str = ""
    corrected_target_text: str = ""
    clarity_rating: int | None = None
    accuracy_rating: int | None = None
    sample_quality: str = "generated"
    source_type: str = "report"
    is_shared: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "training_samples"
