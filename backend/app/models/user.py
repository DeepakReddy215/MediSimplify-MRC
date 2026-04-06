from datetime import datetime, timezone
from beanie import Document, Indexed
from pydantic import EmailStr, Field


class User(Document):
    email: Indexed(EmailStr, unique=True)
    full_name: str
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "users"
