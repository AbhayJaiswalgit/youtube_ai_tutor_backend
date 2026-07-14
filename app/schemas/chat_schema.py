from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    video_id: str
    session_id: Optional[str] = None  # If None, it's a new chat session
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
