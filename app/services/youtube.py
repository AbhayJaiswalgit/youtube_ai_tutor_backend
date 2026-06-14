# import sys
# import subprocess
# from pathlib import Path
# from typing import List, Dict, Optional
# import whisper
# from youtube_transcript_api import YouTubeTranscriptApi
# import yt_dlp

# class YouTubeService:
#     """
#     Service class to handle all YouTube-related data fetching.
#     Implements a fallback to Whisper STT if transcripts are disabled.
#     """
    
#     whisper_model = None

#     @classmethod
#     def get_whisper_model(cls):
#         """Lazy load the Whisper model only when we actually need it."""
#         if cls.whisper_model is None:
#             print("⏳ [WHISPER] Loading 'small' model into memory (this takes a moment)...")
#             cls.whisper_model = whisper.load_model("small")
#         return cls.whisper_model

#     @staticmethod
#     def fetch_transcript(video_id: str) -> Optional[List[Dict]]:
#         """Attempt to fetch via YouTube API first, fallback to Whisper if it fails."""
#         print(f"🔍 [YOUTUBE SERVICE] Attempting standard fetch for {video_id}...")
        
#         # Instantiate the object as required by your library version
#         ytt = YouTubeTranscriptApi()
        
#         try:
#             transcript_list = ytt.list(video_id)
#             fetched_data = None
            
#             # 1. Try to find native English transcripts first
#             for t in transcript_list:
#                 if t.language_code.startswith("en"):
#                     fetched = t.fetch()
#                     fetched_data = fetched.to_raw_data()
#                     break
            
#             # 2. If no native English exists, find the first translatable one
#             if not fetched_data:
#                 for t in transcript_list:
#                     if t.is_translatable:
#                         print("⚠️ [YOUTUBE SERVICE] No native English found. Translating...")
#                         fetched = t.translate("en").fetch()
#                         fetched_data = fetched.to_raw_data()
#                         break
            
#             if not fetched_data:
#                 raise Exception("No English or translatable transcripts available.")
            
#             # 3. Format it for our MongoDB / FAISS database
#             formatted_transcript = []
#             for item in fetched_data:
#                 start_time = float(item['start'])
#                 duration = float(item['duration'])
#                 formatted_transcript.append({
#                     "text": item['text'].replace("\n", " "),
#                     "start_time": start_time,
#                     "end_time": round(start_time + duration, 2)
#                 })
                
#             print(f"✅ [YOUTUBE SERVICE] Success! Fetched {len(formatted_transcript)} chunks natively.")
#             return formatted_transcript

#         except Exception as e:
#             print(f"⚠️ [YOUTUBE SERVICE] Native fetch failed: {e}")
#             print(f"🔄 [YOUTUBE SERVICE] Initiating Whisper STT Fallback Protocol...")
#             return YouTubeService.fallback_to_whisper(video_id)
        
#     @staticmethod
#     def get_video_metadata(video_id: str) -> dict:
#         """Lightweight fetch of video title, duration, and thumbnail without downloading media."""
#         print(f"🔍 [YOUTUBE SERVICE] Fetching metadata for {video_id}...")
#         url = f"https://www.youtube.com/watch?v={video_id}"
#         ydl_opts = {
#             'quiet': True,
#             'extract_flat': True, # Skips deep extraction, making it nearly instant
#         }
#         try:
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 info = ydl.extract_info(url, download=False)
#                 return {
#                     "title": info.get('title', 'Unknown Title'),
#                     "duration": info.get('duration', 0),
#                     "thumbnail": info.get('thumbnails', [{}])[0].get('url', '') if info.get('thumbnails') else None
#                 }
#         except Exception as e:
#             print(f"⚠️ [YOUTUBE SERVICE] Metadata fetch failed: {e}")
#             return {"title": "Unknown Title", "duration": 0, "thumbnail": None}

#     @staticmethod
#     def fallback_to_whisper(video_id: str) -> Optional[List[Dict]]:
#         """Downloads audio and uses local AI to transcribe it."""
#         video_url = f"https://www.youtube.com/watch?v={video_id}"
#         audio_path = Path(f"{video_id}.webm")

#         try:
#             print(f"📥 [WHISPER] Downloading audio via yt-dlp...")
#             subprocess.run(
#                 [
#                     sys.executable, "-m", "yt_dlp", "-f", "bestaudio", 
#                     "-o", str(audio_path), video_url
#                 ],
#                 check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#             )
            
#             print(f"🧠 [WHISPER] Transcribing audio... (This might take a few minutes)")
#             model = YouTubeService.get_whisper_model()
            
#             # fp16=False removes that CPU warning you were seeing!
#             result = model.transcribe(str(audio_path), fp16=False)
            
#             formatted_transcript = []
#             for segment in result["segments"]:
#                 formatted_transcript.append({
#                     "text": segment["text"].strip(),
#                     "start_time": round(segment["start"], 2),
#                     "end_time": round(segment["end"], 2)
#                 })
            
#             print(f"✅ [WHISPER] Transcription complete! Generated {len(formatted_transcript)} chunks.")
            
#             if audio_path.exists():
#                 audio_path.unlink()
                
#             return formatted_transcript
            
#         except Exception as e:
#             print(f"❌ [WHISPER] Fallback totally failed: {e}")
#             if audio_path.exists():
#                 audio_path.unlink()
#             return None

import os
import httpx
from typing import List, Dict, Optional

class YouTubeService:
    """
    Service class to handle all YouTube-related data fetching.
    Uses Supadata API to completely bypass Render IP blocks.
    Strictly text-based: No audio downloading or Whisper AI fallback.
    """

    @staticmethod
    def fetch_transcript(video_id: str) -> Optional[List[Dict]]:
        print(f"🔍 [YOUTUBE SERVICE] Attempting Supadata API fetch for {video_id}...")
        
        api_key = os.environ.get("SUPADATA_KEY")
        if not api_key:
            print("❌ [YOUTUBE SERVICE] SUPADATA_KEY environment variable is missing.")
            return None

        try:
            # We explicitly omit "text=true" so Supadata returns an array of timestamped segments.
            # This is CRITICAL so your Vector Database has start and end times for the AI to use!
            with httpx.Client(timeout=30) as client:
                r = client.get(
                    "https://api.supadata.ai/v1/youtube/transcript",
                    params={"videoId": video_id}, 
                    headers={"x-api-key": api_key}
                )

            if r.status_code != 200:
                print(f"⚠️ [YOUTUBE SERVICE] Supadata API returned HTTP {r.status_code}: {r.text}")
                return None

            data = r.json()
            content = data.get("content")

            # If Supadata returns an empty transcript or video has no captions
            if not content:
                print("⚠️ [YOUTUBE SERVICE] No captions available for this video.")
                return None

            formatted_transcript = []
            
            # Parse the Supadata segment array into our RAG chunk format
            if isinstance(content, list):
                for item in content:
                    start_time = float(item.get("offset", item.get("start", 0.0)))
                    duration = float(item.get("duration", 0.0))
                    
                    formatted_transcript.append({
                        "text": item.get("text", "").replace("\n", " ").strip(),
                        "start_time": round(start_time, 2),
                        "end_time": round(start_time + duration, 2)
                    })
            
            # Fallback just in case the API forces a string response
            elif isinstance(content, str):
                formatted_transcript.append({
                    "text": content.replace("\n", " ").strip(),
                    "start_time": 0.0,
                    "end_time": 0.0
                })

            if not formatted_transcript:
                return None

            print(f"✅ [YOUTUBE SERVICE] Success! Fetched {len(formatted_transcript)} chunks.")
            return formatted_transcript

        except Exception as e:
            print(f"❌ [YOUTUBE SERVICE] Transcript fetch crashed: {e}")
            return None

    @staticmethod
    def get_video_metadata(video_id: str) -> dict:
        """Lightweight fetch of video title and thumbnail using YouTube's official oEmbed API (Never blocked)."""
        print(f"🔍 [YOUTUBE SERVICE] Fetching metadata for {video_id}...")
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        
        try:
            with httpx.Client(timeout=10) as client:
                res = client.get(oembed_url)
            
            if res.status_code == 200:
                data = res.json()
                return {
                    "title": data.get("title", "Unknown Title"),
                    "duration": 0, # Duration omitted as it's not strictly needed for RAG
                    "thumbnail": data.get("thumbnail_url", f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
                }
        except Exception as e:
            print(f"⚠️ [YOUTUBE SERVICE] Metadata fetch failed: {e}")
            
        return {"title": "Unknown Title", "duration": 0, "thumbnail": None}