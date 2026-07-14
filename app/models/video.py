from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone

# Base properties shared across different operations
class VideoBase(BaseModel):
    youtube_id: str
    title: str
    url: str
    duration: int
    thumbnail: Optional[str] = None

# Schema for creating a new video
class VideoCreate(VideoBase):
    pass

# Schema for how it is stored in MongoDB and returned to the client
class VideoInDB(VideoBase):
    id: Optional[str] = Field(default=None, alias="_id")
    processing_status: str = Field(default="pending") # pending, completed, failed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Generated artifacts
    video_summary: Optional[str] = None
    section_summaries: Optional[list] = None
    transcript_chunks: Optional[list] = None

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat()}
