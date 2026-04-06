from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import get_settings

settings = get_settings()

client: AsyncIOMotorClient = None


async def init_db():
    global client
    client = AsyncIOMotorClient(settings.mongodb_url)
    
    from app.models.user import User
    from app.models.report import Report
    from app.models.training_sample import TrainingSample
    from app.models.report_feedback import ReportFeedback
    from app.models.medical_term_memory import MedicalTermMemory
    
    await init_beanie(
        database=client[settings.database_name],
        document_models=[User, Report, TrainingSample, ReportFeedback, MedicalTermMemory]
    )


async def close_db():
    global client
    if client:
        client.close()
