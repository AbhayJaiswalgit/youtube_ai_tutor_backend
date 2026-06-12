from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class Database:
    client: AsyncIOMotorClient = None

db = Database()

async def connect_to_mongo():
    """Establish database connection."""
    db.client = AsyncIOMotorClient(settings.MONGODB_URL)
    print("✅ Connected to MongoDB")

async def close_mongo_connection():
    """Close database connection."""
    if db.client:
        db.client.close()
        print("🛑 Closed MongoDB connection")

def get_database():
    """Dependency to get the database instance."""
    return db.client[settings.DATABASE_NAME]