from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone

from app.schemas.quiz_schema import QuizRequest, QuizResponse
from app.services.quiz_service import QuizService
from app.core.database import get_database
from app.api.dependencies import get_current_user
from bson import ObjectId

router = APIRouter()
quiz_service = QuizService()

@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(
    request: QuizRequest, 
    db = Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    try:
        # 1. Fetch the pre-generated summaries from MongoDB
        video_doc = await db["videos"].find_one({"youtube_id": request.video_id})
        if not video_doc or not video_doc.get("section_summaries"):
            raise HTTPException(status_code=400, detail="Video summaries not ready. Please wait for processing to finish.")

        # 2. Generate the questions via AI
        questions = quiz_service.generate_quiz(
            video_id=request.video_id,
            difficulty=request.difficulty,
            count=request.question_count,
            section_summaries=video_doc.get("section_summaries")
        )
        
        # 3. Save to MongoDB (Leave this exactly as you have it)
        quiz_document = {
            "user_id": current_user["_id"],
            "video_id": request.video_id,
            "difficulty": request.difficulty,
            "questions": questions,
            "created_at": datetime.now(timezone.utc)
        }
        await db["quizzes"].insert_one(quiz_document)
        
        return QuizResponse(
            video_id=request.video_id,
            difficulty=request.difficulty,
            questions=questions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/", response_model=list)
async def get_user_quizzes(
    db = Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """Retrieve all quizzes grouped by video with sequence naming."""
    quizzes_cursor = db["quizzes"].find({"user_id": current_user["_id"]}).sort("created_at", 1)
    quizzes = await quizzes_cursor.to_list(length=500)
    
    grouped = {}
    for quiz in quizzes:
        quiz["_id"] = str(quiz["_id"])
        vid = quiz["video_id"]
        
        if vid not in grouped:
            video = await db["videos"].find_one({"youtube_id": vid})
            title = video.get("title", "Unknown Video") if video else "Unknown Video"
            grouped[vid] = {
                "video_id": vid,
                "video_title": title,
                "created_at": quiz["created_at"],
                "items": []
            }
        grouped[vid]["items"].append(quiz)

    grouped_list = list(grouped.values())
    grouped_list.sort(key=lambda x: x["created_at"])
    
    final_result = []
    for idx, group in enumerate(grouped_list):
        seq_no = idx + 1
        group["display_name"] = f"{seq_no}_{group['video_title']}"
        final_result.append(group)
        
    final_result.reverse()
    return final_result

@router.delete("/{quiz_id}")
async def delete_quiz(
    quiz_id: str,
    db = Depends(get_database),
    current_user: dict = Depends(get_current_user)
):
    """Deletes a specific quiz entry ensuring the user owns it."""
    result = await db["quizzes"].delete_one({
        "_id": ObjectId(quiz_id), 
        "user_id": current_user["_id"]
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Quiz not found or you are not authorized to delete it.")
        
    return {"message": "Quiz deleted successfully"}