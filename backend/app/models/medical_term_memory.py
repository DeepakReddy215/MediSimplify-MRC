from datetime import datetime, timezone
from beanie import Document
from pydantic import Field


class MedicalTermMemory(Document):
    term: str
    definition: str
    seen_count: int = 0
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "medical_term_memory"
