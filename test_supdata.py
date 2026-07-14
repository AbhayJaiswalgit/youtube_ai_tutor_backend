from app.core.config import settings
import os
from app.utils.logger import logger

logger.info("Settings PINECONE_API_KEY set: %s", bool(settings.PINECONE_API_KEY))
logger.info("os.getenv PINECONE_API_KEY set: %s", bool(os.getenv("PINECONE_API_KEY")))