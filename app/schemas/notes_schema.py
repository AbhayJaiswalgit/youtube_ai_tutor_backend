from typing import List

from pydantic import BaseModel, ConfigDict, Field


class NotesRequest(BaseModel):
    youtube_id: str = Field(..., alias="video_id")
    note_type: str = Field(
        default="detailed_summary",
        description="Options: short_summary, detailed_summary, interview_notes, revision_notes",
    )

    model_config = ConfigDict(populate_by_name=True)


class NoteSection(BaseModel):
    heading: str
    bullet_points: List[str]


class NotesResponse(BaseModel):
    youtube_id: str = Field(..., alias="video_id")
    note_type: str
    content: List[NoteSection]

    model_config = ConfigDict(populate_by_name=True)