from pydantic import BaseModel, Field
from typing import List

class NotesRequest(BaseModel):
    video_id: str
    note_type: str = Field(
        default="detailed_summary", 
        description="Options: short_summary, detailed_summary, interview_notes, revision_notes"
    )

class NoteSection(BaseModel):
    heading: str
    bullet_points: List[str]

class NotesResponse(BaseModel):
    video_id: str
    note_type: str
    content: List[NoteSection]