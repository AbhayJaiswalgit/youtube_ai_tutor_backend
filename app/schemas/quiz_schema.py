from pydantic import BaseModel, Field
from typing import List

class QuizRequest(BaseModel):
    video_id: str
    difficulty: str = Field(default="medium", description="easy, medium, or hard")
    question_count: int = Field(default=5, le=10)

class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(..., min_items=4, max_items=4)
    correct_answer: str
    explanation: str

class QuizResponse(BaseModel):
    video_id: str
    difficulty: str
    questions: List[QuizQuestion]