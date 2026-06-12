from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    video_id: str
    session_id: Optional[str] = None  # If None, it's a new chat session
    message: str

class Citation(BaseModel):
    text_snippet: str
    start_time: float
    end_time: float

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Citation] = []