import re
import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from app.schemas.video_schema import VideoProcessRequest
from app.models.video import VideoInDB
from app.core.database import get_database
from app.services.youtube import YouTubeService
from app.services.vector_store import VectorStoreService
from app.services.summary_service import SummaryService
from app.utils.logger import logger

router = APIRouter()

def extract_youtube_id(url: str) -> str:
    """Helper function to extract the 11-character YouTube ID."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", str(url))
    return match.group(1) if match else None

# async def mock_ai_pipeline(video_id: str, db):
#     """
#     The actual background worker fetching real data.
#     """
#     print(f"⚙️ [BACKGROUND] Starting AI pipeline for video: {video_id}")
#     collection = db["videos"]
    
#     # 1. Fetch the real transcript
#     transcript_data = YouTubeService.fetch_transcript(video_id)
    
#     if not transcript_data:
#         # If it fails, we need to update the database so the frontend knows it failed.
#         await collection.update_one(
#             {"youtube_id": video_id},
#             {"$set": {"processing_status": "failed"}}
#         )
#         print(f"🛑 [BACKGROUND] Pipeline failed: No transcript available.")
#         return

#     # 2. Later, we will chunk this data and send it to FAISS.
#     # For now, let's just print a snippet to prove it worked!
#     print(f"📝 [BACKGROUND] Snippet of transcript: {transcript_data[0]['text']}")
    
#     # 3. Update the database status to completed
#     await collection.update_one(
#         {"youtube_id": video_id},
#         {"$set": {"processing_status": "completed"}}
#     )
#     print(f"✅ [BACKGROUND] Finished processing video: {video_id}")


async def process_video_pipeline(video_id: str, db):
    """
    The actual background worker fetching real data and embedding it.
    """
    logger.info("Starting background AI pipeline for video: %s", video_id)
    collection = db["videos"]
    
    # 1. Fetch the real transcript
    transcript_data = YouTubeService.fetch_transcript(video_id)
    
    if not transcript_data:
        await collection.update_one(
            {"youtube_id": video_id},
            {"$set": {"processing_status": "failed"}}
        )
        logger.warning("Background pipeline failed: no transcript available for %s", video_id)
        return

    # 2. Initialize our Vector Store Service and embed the data!
    try:
        vector_service = VectorStoreService()
        vector_result = vector_service.process_and_store(video_id, transcript_data)

        # 3. Hierarchical summarization from raw transcript segments.
        summary_service = SummaryService()
        summaries = await summary_service.generate_hierarchical_summary(transcript_data, video_id)
        
        # 3. Update the database status to completed
        await collection.update_one(
            {"youtube_id": video_id},
            {"$set": {
                "processing_status": "completed",
                "video_summary": summaries["video_summary"],
                "section_summaries": summaries["section_summaries"],
                "transcript_chunks": vector_result["transcript_chunks"],
            }}
        )
        logger.info("Background processing completed for video: %s", video_id)
        
    except Exception:
        logger.exception("Background processing failed for video: %s", video_id)
        await collection.update_one(
            {"youtube_id": video_id},
            {"$set": {"processing_status": "failed"}}
        )

@router.post("/process", response_model=VideoInDB)
async def process_video(
    request: VideoProcessRequest, 
    background_tasks: BackgroundTasks,
    db = Depends(get_database)
):
    video_id = extract_youtube_id(request.url)
    logger.info("Video process request received for URL: %s", request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    collection = db["videos"]
    
    # 1. Check if video already exists (Cache Hit)
    existing_video = await collection.find_one({"youtube_id": video_id})
    if existing_video:
        existing_video["_id"] = str(existing_video["_id"])
        return existing_video

    # 2. Fetch real metadata asynchronously so we don't block the API
    metadata = await asyncio.to_thread(YouTubeService.get_video_metadata, video_id)

    # 3. Create a pending database entry with REAL data
    new_video = VideoInDB(
        youtube_id=video_id,
        title=metadata["title"], 
        url=str(request.url),
        duration=metadata["duration"],
        thumbnail=metadata["thumbnail"],
        processing_status="pending"
    )
    
    # Insert into MongoDB
    result = await collection.insert_one(new_video.model_dump(by_alias=True, exclude={"id"}))
    new_video.id = str(result.inserted_id)

    # 4. Trigger the background AI Pipeline
    background_tasks.add_task(process_video_pipeline, video_id, db)

    return new_video


from fastapi import Path

# Add this below your existing /process endpoint
@router.get("/{video_id}", response_model=VideoInDB)
async def get_video_status(video_id: str = Path(...), db = Depends(get_database)):
    """Allows the frontend to poll for the background processing status."""
    logger.info("Video status requested for video_id: %s", video_id)
    video = await db["videos"].find_one({"youtube_id": video_id})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video["_id"] = str(video["_id"])
    return video
