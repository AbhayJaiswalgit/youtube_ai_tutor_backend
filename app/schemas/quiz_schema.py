from typing import List

from pydantic import BaseModel, ConfigDict, Field


class QuizRequest(BaseModel):
    youtube_id: str = Field(..., alias="video_id")
    difficulty: str = Field(default="medium", description="easy, medium, or hard")
    question_count: int = Field(default=5, le=10)

    model_config = ConfigDict(populate_by_name=True)


class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(..., min_items=4, max_items=4)
    correct_answer: str
    explanation: str


class QuizResponse(BaseModel):
    youtube_id: str = Field(..., alias="video_id")
    difficulty: str
    questions: List[QuizQuestion]

    model_config = ConfigDict(populate_by_name=True)