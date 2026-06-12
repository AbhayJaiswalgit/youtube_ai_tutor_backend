from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.database import connect_to_mongo, close_mongo_connection
from app.api.routes import video,chat,auth,quiz,notes # <--- NEW IMPORT
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_to_mongo()
    yield
    await close_mongo_connection()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for the YouTube AI Tutor Platform",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Your Vite Web App
        "http://localhost:3000",
        "chrome-extension://*"    # Allows your future Chrome Extension
    ],
    allow_credentials=True,
    allow_methods=["*"], # <--- This explicitly fixes the OPTIONS 405 error!
    allow_headers=["*"],
)

# <--- NEW ROUTER INCLUSION --->
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(video.router, prefix="/api/video", tags=["Video Processing"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat & Tutor"])
app.include_router(quiz.router, prefix="/api/quiz", tags=["Content Generation"])
app.include_router(notes.router, prefix="/api/notes", tags=["Content Generation"])

@app.get("/")
async def root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME} API", "status": "active"}