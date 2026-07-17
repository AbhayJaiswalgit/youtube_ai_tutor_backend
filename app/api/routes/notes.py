from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone

from app.schemas.notes_schema import NotesRequest, NotesResponse
from app.services.notes_service import NotesService
from app.core.database import get_database
from app.api.dependencies import get_current_user
from bson import ObjectId
from app.utils.logger import logger

router = APIRouter()
notes_service = NotesService()

@router.post("/generate", response_model=NotesResponse)
async def generate_notes(
    request: NotesRequest, 
    db = Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    try:
        # 1. Fetch the pre-generated summaries from MongoDB
        video_doc = await db["videos"].find_one({"youtube_id": request.youtube_id})
        if not video_doc or not video_doc.get("section_summaries"):
            raise HTTPException(status_code=400, detail="Video summaries not ready. Please wait for processing to finish.")

        # 2. Generate structured notes via AI
        content = await notes_service.generate_notes(
            video_id=request.youtube_id,
            note_type=request.note_type,
            section_summaries=video_doc.get("section_summaries"),
            video_summary=video_doc.get("video_summary", "")
        )
        
        # 3. Save to MongoDB (Leave this exactly as you have it)
        notes_document = {
            "user_id": current_user["_id"],
            "youtube_id": request.youtube_id,
            "note_type": request.note_type,
            "content": content,
            "created_at": datetime.now(timezone.utc)
        }
        await db["notes"].insert_one(notes_document)
        
        return NotesResponse(
            video_id=request.youtube_id,
            note_type=request.note_type,
            content=content
        )
        
    except Exception:
        logger.exception("Error generating notes for video: %s", request.youtube_id)
        raise HTTPException(status_code=500, detail="Internal server error")
    



@router.get("/", response_model=list)
async def get_user_notes(
    db = Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """Retrieve all notes grouped by video with sequence naming."""
    notes_cursor = db["notes"].find({"user_id": current_user["_id"]}).sort("created_at", 1)
    notes = await notes_cursor.to_list(length=500)
    
    grouped = {}
    for note in notes:
        note["_id"] = str(note["_id"])
        vid = note.get("youtube_id") or note.get("video_id")
        
        if vid not in grouped:
            video = await db["videos"].find_one({"youtube_id": vid})
            title = video.get("title", "Unknown Video") if video else "Unknown Video"
            grouped[vid] = {
                "video_id": vid,
                "video_title": title,
                "created_at": note["created_at"], # Used to sort chronologically
                "items": []
            }
        grouped[vid]["items"].append(note)

    # Convert dictionary to a chronologically sorted list
    grouped_list = list(grouped.values())
    grouped_list.sort(key=lambda x: x["created_at"])
    
    # Assign sequence numbers based on chronological creation
    final_result = []
    for idx, group in enumerate(grouped_list):
        seq_no = idx + 1
        group["display_name"] = f"{seq_no}_{group['video_title']}"
        final_result.append(group)
        
    # Reverse so the newest video group is at the top of the Library
    final_result.reverse()
    return final_result


@router.delete("/{note_id}")
async def delete_note(
    note_id: str,
    db = Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """Deletes a specific note entry ensuring the user owns it."""
    result = await db["notes"].delete_one({
        "_id": ObjectId(note_id), 
        "user_id": current_user["_id"]
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found or you are not authorized to delete it.")
        
    return {"message": "Note deleted successfully"}