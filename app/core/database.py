# from motor.motor_asyncio import AsyncIOMotorClient
# from app.core.config import settings
# from app.utils.logger import logger

# class Database:
#     client: AsyncIOMotorClient = None

# db = Database()

# async def connect_to_mongo():
#     """Establish database connection."""
#     db.client = AsyncIOMotorClient(settings.MONGODB_URL)
#     logger.info("Connected to MongoDB")
#     await _initialize_indexes()

# async def close_mongo_connection():
#     """Close database connection."""
#     if db.client:
#         db.client.close()
#         logger.info("Closed MongoDB connection")

# def get_database():
#     """Dependency to get the database instance."""
#     return db.client[settings.DATABASE_NAME]


# async def _initialize_indexes():
#     database = db.client[settings.DATABASE_NAME]
#     try:
#         await database["users"].create_index("email")
#         await database["videos"].create_index("youtube_id")
#         await database["chat_sessions"].create_index([("user_id", 1), ("video_id", 1)])
#         await database["messages"].create_index("session_id")
#         await database["notes"].create_index([("user_id", 1), ("video_id", 1)])
#         await database["quizzes"].create_index([("user_id", 1), ("video_id", 1)])
#         logger.info("MongoDB indexes ensured.")
#     except Exception:
#         logger.exception("Failed to ensure MongoDB indexes.")

import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.utils.logger import logger

class Database:
    client: AsyncIOMotorClient = None

db = Database()

async def connect_to_mongo():
    """Establish database connection."""
    # Add tlsCAFile=certifi.where() to force correct SSL routing
    db.client = AsyncIOMotorClient(settings.MONGODB_URL, tlsCAFile=certifi.where())
    logger.info("Connected to MongoDB")
    await _initialize_indexes()

async def close_mongo_connection():
    """Close database connection."""
    if db.client:
        db.client.close()
        logger.info("Closed MongoDB connection")

def get_database():
    """Dependency to get the database instance."""
    return db.client[settings.DATABASE_NAME]

async def _initialize_indexes():
    database = db.client[settings.DATABASE_NAME]
    try:
        await database["users"].create_index("email")
        await database["videos"].create_index("youtube_id")
        await database["chat_sessions"].create_index([("user_id", 1), ("video_id", 1)])
        await database["messages"].create_index("session_id")
        await database["notes"].create_index([("user_id", 1), ("video_id", 1)])
        await database["quizzes"].create_index([("user_id", 1), ("video_id", 1)])
        logger.info("MongoDB indexes ensured.")
    except Exception:
        logger.exception("Failed to ensure MongoDB indexes.")