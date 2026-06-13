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
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional
import yt_dlp

class YouTubeService:
    """
    Service class to handle all YouTube-related data fetching.
    Uses yt-dlp to bypass Cloud IP bans using cookies, extracting JSON3 transcripts.
    """
    
    # Path to the cookies file in the root backend directory
    COOKIE_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "youtube_cookies.txt"))

    @staticmethod
    def fetch_transcript(video_id: str) -> Optional[List[Dict]]:
        """Fetch transcripts entirely via yt-dlp to bypass datacenter IP blocks."""
        print(f"🔍 [YOUTUBE SERVICE] Attempting yt-dlp transcript fetch for {video_id}...")
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Temporary base filename for the subtitle download
        sub_filename_base = f"{video_id}_subs"
        
        ydl_opts = {
            'quiet': True,
            'skip_download': True,              # NEVER download the video/audio (prevents FFmpeg crash)
            'writesubtitles': True,             # Get manually created subtitles
            'writeautomaticsub': True,          # Fallback to auto-generated subtitles
            'subtitleslangs': ['en.*', 'en'],   # Target all English variants
            'subtitlesformat': 'json3',         # Download as easy-to-parse JSON
            'outtmpl': sub_filename_base,       # Output name (yt-dlp appends .en.json3)
        }
        
        if os.path.exists(YouTubeService.COOKIE_FILE_PATH):
            print("🍪 [YOUTUBE SERVICE] Injecting Cookies into yt-dlp...")
            ydl_opts['cookiefile'] = YouTubeService.COOKIE_FILE_PATH
        else:
            print("⚠️ [YOUTUBE SERVICE] No cookies found. Render may block this request.")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
                
            # yt-dlp creates a file like: EpZb7mnMHwQ_subs.en.json3
            # We use glob to find it because the exact language code suffix might vary slightly
            downloaded_files = glob.glob(f"{sub_filename_base}*.json3")
            
            if not downloaded_files:
                print("⚠️ [YOUTUBE SERVICE] yt-dlp succeeded but no English subtitle file was generated.")
                return None
                
            target_file = Path(downloaded_files[0])
            
            with open(target_file, 'r', encoding='utf-8') as f:
                sub_data = json.load(f)
                
            formatted_transcript = []
            
            # Extract data from YouTube's native JSON3 format
            for event in sub_data.get('events', []):
                if 'segs' not in event:
                    continue
                    
                text = "".join(seg.get('utf8', '') for seg in event['segs']).strip()
                if not text or text == '\n':
                    continue
                    
                start_time = event.get('tStartMs', 0) / 1000.0
                duration = event.get('dDurationMs', 0) / 1000.0
                
                formatted_transcript.append({
                    "text": text.replace("\n", " "),
                    "start_time": round(start_time, 2),
                    "end_time": round(start_time + duration, 2)
                })
                
            # Immediately delete the subtitle file so we don't fill up Render's disk
            target_file.unlink()
            
            print(f"✅ [YOUTUBE SERVICE] Success! Extracted {len(formatted_transcript)} chunks natively.")
            return formatted_transcript

        except Exception as e:
            print(f"❌ [YOUTUBE SERVICE] Transcript fetch failed: {e}")
            # Ensure cleanup if the script crashed midway
            for p in glob.glob(f"{sub_filename_base}*"):
                try:
                    os.remove(p)
                except Exception:
                    pass
            return None

    @staticmethod
    def get_video_metadata(video_id: str) -> dict:
        """Lightweight fetch of video title, duration, and thumbnail."""
        print(f"🔍 [YOUTUBE SERVICE] Fetching metadata for {video_id}...")
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True, 
        }
        
        if os.path.exists(YouTubeService.COOKIE_FILE_PATH):
            ydl_opts['cookiefile'] = YouTubeService.COOKIE_FILE_PATH

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get('title', 'Unknown Title'),
                    "duration": info.get('duration', 0),
                    "thumbnail": info.get('thumbnails', [{}])[0].get('url', '') if info.get('thumbnails') else None
                }
        except Exception as e:
            print(f"⚠️ [YOUTUBE SERVICE] Metadata fetch failed: {e}")
            return {"title": "Unknown Title", "duration": 0, "thumbnail": None}