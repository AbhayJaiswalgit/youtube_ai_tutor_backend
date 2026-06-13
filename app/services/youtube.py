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

import requests
import re
from typing import List, Dict, Optional

class YouTubeService:
    """
    Service class to handle all YouTube-related data fetching.
    Uses the Piped API network (free, open-source YouTube proxies) 
    to completely bypass Cloud/Datacenter IP bans without needing cookies or JS Runtimes.
    """
    
    # A list of reliable public Piped instances to ensure high availability
    PIPED_INSTANCES = [
        "https://pipedapi.kavin.rocks",
        "https://pipedapi.syncpundit.io",
        "https://pipedapi.adminforge.de",
        "https://api.piped.projectsegfau.lt"
    ]

    @staticmethod
    def fetch_transcript(video_id: str) -> Optional[List[Dict]]:
        print(f"🔍 [YOUTUBE SERVICE] Attempting Piped API fetch for {video_id}...")
        
        for instance in YouTubeService.PIPED_INSTANCES:
            try:
                # 1. Fetch the video streams data
                res = requests.get(f"{instance}/streams/{video_id}", timeout=15)
                if res.status_code != 200:
                    continue
                
                data = res.json()
                subtitles = data.get("subtitles", [])
                
                if not subtitles:
                    print(f"⚠️ [YOUTUBE SERVICE] {instance} reported no subtitles available.")
                    return None
                
                # 2. Find the English subtitle track (prioritize manual over auto-generated if possible)
                en_subs = [s for s in subtitles if s.get("name", "").startswith("English")]
                if not en_subs:
                    # Fallback to the very first subtitle track if no English is found
                    target_sub = subtitles[0]
                else:
                    # Try to get non-auto-generated first
                    manual_en = next((s for s in en_subs if not s.get("autoGenerated")), None)
                    target_sub = manual_en if manual_en else en_subs[0]
                
                sub_url = target_sub.get("url")
                
                # 3. Download the actual VTT subtitle file
                sub_res = requests.get(sub_url, timeout=15)
                if sub_res.status_code != 200:
                    continue
                
                # 4. Parse the VTT text into our chunked dictionary format
                formatted_transcript = YouTubeService._parse_vtt(sub_res.text)
                
                if formatted_transcript:
                    print(f"✅ [YOUTUBE SERVICE] Success via {instance}! Extracted {len(formatted_transcript)} chunks.")
                    return formatted_transcript
                    
            except Exception as e:
                print(f"⚠️ [YOUTUBE SERVICE] Instance {instance} failed: {e}")
                continue
                
        print("❌ [YOUTUBE SERVICE] All Piped instances failed to fetch the transcript.")
        return None

    @staticmethod
    def get_video_metadata(video_id: str) -> dict:
        """Lightweight fetch of video title, duration, and thumbnail via Piped API."""
        print(f"🔍 [YOUTUBE SERVICE] Fetching metadata for {video_id}...")
        
        for instance in YouTubeService.PIPED_INSTANCES:
            try:
                res = requests.get(f"{instance}/streams/{video_id}", timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    return {
                        "title": data.get("title", "Unknown Title"),
                        "duration": data.get("duration", 0),
                        "thumbnail": data.get("thumbnailUrl", "")
                    }
            except Exception:
                continue
                
        return {"title": "Unknown Title", "duration": 0, "thumbnail": None}

    @staticmethod
    def _parse_vtt(vtt_text: str) -> List[Dict]:
        """Parses a standard WebVTT subtitle file into formatted chunks."""
        formatted = []
        # Matches VTT timestamps (e.g., 00:00:05.123 --> 00:00:07.456 OR 05:12.345 --> 05:14.000)
        pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})')
        
        def time_to_sec(t_str):
            parts = t_str.split(':')
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return 0.0

        blocks = vtt_text.strip().split('\n\n')
        for block in blocks:
            lines = block.strip().split('\n')
            start_sec, end_sec = 0.0, 0.0
            text_lines = []
            
            for line in lines:
                match = pattern.search(line)
                if match:
                    start_sec = time_to_sec(match.group(1))
                    end_sec = time_to_sec(match.group(2))
                elif '-->' not in line and line.upper() != 'WEBVTT' and not re.match(r'^\d+$', line):
                    # Clean up HTML-like formatting tags (e.g., <c.color>, <i>, <00:00:01>)
                    clean_line = re.sub(r'<[^>]+>', '', line).strip()
                    if clean_line:
                        text_lines.append(clean_line)
            
            if text_lines and end_sec > 0:
                # VTT files often repeat words for scrolling effect. Remove exact duplicate lines natively.
                text = " ".join(dict.fromkeys(text_lines))
                formatted.append({
                    "text": text,
                    "start_time": round(start_sec, 2),
                    "end_time": round(end_sec, 2)
                })
                
        return formatted