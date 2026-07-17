from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    youtube_id: str = Field(..., alias="video_id")
    session_id: Optional[str] = None  # If None, it's a new chat session
    message: str

    model_config = ConfigDict(populate_by_name=True)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
