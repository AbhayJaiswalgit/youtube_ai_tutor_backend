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
import requests
from typing import List, Dict, Optional

class YouTubeService:
    """
    Service class to handle all YouTube-related data fetching.
    Uses a custom Google Apps Script microservice to bypass Render IP blocks 
    using Google's own trusted infrastructure.
    """

    @staticmethod
    def fetch_transcript(video_id: str) -> Optional[List[Dict]]:
        print(f"🔍 [YOUTUBE SERVICE] Attempting GAS Proxy fetch for {video_id}...")

        # Pull credentials from environment variables
        apps_script_url = os.environ.get("APPS_SCRIPT_URL")
        apps_script_token = os.environ.get("APPS_SCRIPT_TOKEN")

        if not apps_script_url or not apps_script_token:
            print("❌ [YOUTUBE SERVICE] Missing APPS_SCRIPT_URL or APPS_SCRIPT_TOKEN in env variables.")
            return None

        try:
            # Call our Google Apps Script Web App
            response = requests.get(
                apps_script_url,
                params={
                    "videoId": video_id,
                    "lang": "en",
                    "token": apps_script_token
                },
                timeout=20 # Give GAS time to scrape the page
            )

            if response.status_code != 200:
                print(f"⚠️ [YOUTUBE SERVICE] Proxy returned HTTP {response.status_code}")
                return None

            data = response.json()

            if "error" in data:
                print(f"⚠️ [YOUTUBE SERVICE] Proxy error: {data['error']}")
                return None

            segments = data.get("segments", [])
            if not segments:
                print("⚠️ [YOUTUBE SERVICE] Proxy returned empty segments.")
                return None

            # Format it exactly how your Vector Store and MongoDB expect it
            formatted_transcript = []
            for seg in segments:
                start_time = float(seg["start"])
                duration = float(seg["duration"])
                
                formatted_transcript.append({
                    "text": seg["text"],
                    "start_time": round(start_time, 2),
                    "end_time": round(start_time + duration, 2) # Translate duration to end_time
                })

            print(f"✅ [YOUTUBE SERVICE] Success via Proxy! Extracted {len(formatted_transcript)} chunks.")
            return formatted_transcript

        except requests.exceptions.RequestException as e:
            print(f"❌ [YOUTUBE SERVICE] Proxy fetch failed: {e}")
            return None

    @staticmethod
    def get_video_metadata(video_id: str) -> dict:
        """Lightweight fetch of video title and thumbnail using YouTube's official oEmbed API (Never blocked)."""
        print(f"🔍 [YOUTUBE SERVICE] Fetching metadata for {video_id}...")
        
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        
        try:
            res = requests.get(oembed_url, timeout=10)
            if res.status_code == 200:
                data = res.json()
                return {
                    "title": data.get("title", "Unknown Title"),
                    "duration": 0, # oEmbed doesn't provide duration, but our RAG chunking doesn't require it
                    "thumbnail": data.get("thumbnail_url", f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
                }
        except Exception as e:
            print(f"⚠️ [YOUTUBE SERVICE] Metadata fetch failed: {e}")
            
        return {"title": "Unknown Title", "duration": 0, "thumbnail": None}