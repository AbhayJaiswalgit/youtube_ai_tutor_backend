from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_current_user
from app.core.database import get_database
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.services.query_intent import QueryIntentRouter
from app.utils.logger import logger


router = APIRouter()
chat_service = ChatService()
intent_router = QueryIntentRouter()


async def _get_or_create_session(session_collection, request, current_user):
    session_id = request.session_id
    if not session_id:
        return await _create_session(session_collection, request, current_user)

    try:
        session = await session_collection.find_one({"_id": ObjectId(session_id)})
    except Exception:
        return await _create_session(session_collection, request, current_user)

    if not session:
        return await _create_session(session_collection, request, current_user)

    if session.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")

    if session.get("video_id") != request.video_id:
        logger.info("Stale session_id for a different video (%s vs %s). Creating a new session.", session.get("video_id"), request.video_id)
        return await _create_session(session_collection, request, current_user)

    return str(session["_id"])


async def _create_session(session_collection, request, current_user):
    new_session = await session_collection.insert_one(
        {
            "video_id": request.video_id,
            "user_id": current_user["_id"],
            "created_at": datetime.now(timezone.utc),
        }
    )
    session_id = str(new_session.inserted_id)
    logger.info("Created new chat session %s for video: %s", session_id, request.video_id)
    return session_id


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    session_collection = db["chat_sessions"]
    message_collection = db["messages"]

    session_id = await _get_or_create_session(session_collection, request, current_user)
    logger.info("Chat request received for video %s session_id=%s user=%s", request.video_id, session_id, current_user.get("email", "unknown"))

    past_messages = (
        await message_collection.find({"session_id": session_id})
        .sort("created_at", 1)
        .to_list(length=100)
    )
    history = [{"sender": message["sender"], "content": message["content"]} for message in past_messages]

    await message_collection.insert_one(
        {
            "session_id": session_id,
            "sender": "user",
            "content": request.message,
            "created_at": datetime.now(timezone.utc),
        }
    )

    intent = intent_router.classify(request.message)
    logger.debug("Classified intent for session %s: %s", session_id, intent.as_dict())

    video_doc = await db["videos"].find_one({"youtube_id": request.video_id})
    try:
        ai_response = chat_service.answer_query(
            video_doc=video_doc,
            query=intent.clean_query or request.message,
            chat_history=history,
            intent=intent,
        )
    except Exception:
        logger.exception("Error while computing chat answer for video %s", request.video_id)
        raise HTTPException(status_code=500, detail="Internal server error")

    await message_collection.insert_one(
        {
            "session_id": session_id,
            "sender": "ai",
            "content": ai_response["answer"],
            "created_at": datetime.now(timezone.utc),
        }
    )

    return ChatResponse(session_id=session_id, answer=ai_response["answer"])


@router.get("/sessions", response_model=list)
async def get_user_chat_sessions(
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Retrieve all chat sessions with user-friendly sequence naming."""
    sessions_cursor = db["chat_sessions"].find({"user_id": current_user["_id"]}).sort("created_at", 1)
    sessions = await sessions_cursor.to_list(length=500)

    formatted_sessions = []

    for idx, session in enumerate(sessions):
        session["_id"] = str(session["_id"])
        video = await db["videos"].find_one({"youtube_id": session["video_id"]})
        title = video.get("title", "Unknown Video") if video else "Unknown Video"
        session["chat_name"] = f"{idx + 1}_{title}"
        formatted_sessions.append(session)

    formatted_sessions.reverse()
    return formatted_sessions


@router.get("/sessions/{session_id}/messages", response_model=list)
async def get_chat_history(
    session_id: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Retrieve all messages for a specific chat session."""
    session = await db["chat_sessions"].find_one({"_id": ObjectId(session_id)})
    if not session or session.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to view this chat")

    messages_cursor = db["messages"].find({"session_id": session_id}).sort("created_at", 1)
    messages = await messages_cursor.to_list(length=500)

    for msg in messages:
        msg["_id"] = str(msg["_id"])

    return messages


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db=Depends(get_database),
    current_user: dict = Depends(get_current_user),
):
    """Deletes a chat session and all its associated messages."""
    session = await db["chat_sessions"].find_one({"_id": ObjectId(session_id)})
    if not session or session.get("user_id") != current_user["_id"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this chat")

    await db["messages"].delete_many({"session_id": session_id})
    await db["chat_sessions"].delete_one({"_id": ObjectId(session_id)})

    return {"message": "Chat session deleted successfully"}
