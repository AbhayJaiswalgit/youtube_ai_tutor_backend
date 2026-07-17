# from typing import Dict, List, Optional

# from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pinecone import Pinecone

# from app.core.config import settings
# from app.utils.logger import logger


# class VectorStoreService:
#     """Builds flat, sequential transcript chunks and uploads them to Pinecone."""

#     def __init__(self):
#         self.embeddings = HuggingFaceEndpointEmbeddings(
#             model="sentence-transformers/all-MiniLM-L6-v2",
#             task="feature-extraction",
#             huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
#         )
#         self.index_name = settings.PINECONE_INDEX_NAME
#         self.chunk_size = 900
#         self.chunk_overlap = 180
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             add_start_index=True,
#             separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
#         )

#     def process_and_store(self, video_id: str, formatted_transcript: list) -> Dict[str, list]:
#         """Create sequential chunks, embed them, and return Mongo-ready payloads."""
#         logger.info(
#             "Building flat sequential chunk index for video %s from %d transcript segments",
#             video_id,
#             len(formatted_transcript or []),
#         )

#         transcript_chunks = self._build_transcript_chunks(video_id, formatted_transcript)
#         child_documents = self._build_child_documents(video_id, transcript_chunks)
#         logger.info("Built %d sequential chunks for video: %s", len(child_documents), video_id)

#         if not child_documents:
#             raise ValueError("No transcript chunks were available to embed.")

#         pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
#         pinecone_index = pinecone_client.Index(self.index_name)
#         vector_store = PineconeVectorStore(index=pinecone_index, embedding=self.embeddings)
#         vector_store.add_documents(child_documents)

#         logger.info("Uploaded %d sequential chunks to Pinecone for video: %s", len(child_documents), video_id)
#         return {"transcript_chunks": transcript_chunks}

#     def _build_transcript_chunks(self, video_id: str, transcript: list) -> List[Dict]:
#         segments = self._normalize_transcript_segments(transcript)
#         if not segments:
#             logger.warning("No usable transcript segments found for video: %s", video_id)
#             return []

#         full_text, offsets = self._join_segments_with_offsets(segments)
#         split_documents = self.text_splitter.create_documents([full_text])

#         transcript_chunks = []
#         for chunk_index, split_doc in enumerate(split_documents):
#             text = " ".join((split_doc.page_content or "").split())
#             if not text:
#                 continue

#             start_offset = split_doc.metadata.get("start_index")
#             if start_offset is None:
#                 start_offset = full_text.find(split_doc.page_content)
#             end_offset = int(start_offset) + len(split_doc.page_content)

#             start_time = self._time_for_offset(offsets, int(start_offset), prefer_end=False)
#             end_time = self._time_for_offset(offsets, end_offset, prefer_end=True)

#             transcript_chunks.append(
#                 {
#                     "text": text,
#                     "chunk_index": len(transcript_chunks),
#                     "start_time": round(start_time, 2),
#                     "end_time": round(end_time, 2),
#                 }
#             )

#         logger.info(
#             "Created %d flat transcript chunks for video %s covering %.2fs-%.2fs",
#             len(transcript_chunks),
#             video_id,
#             transcript_chunks[0]["start_time"] if transcript_chunks else 0.0,
#             transcript_chunks[-1]["end_time"] if transcript_chunks else 0.0,
#         )
#         return transcript_chunks

#     def _normalize_transcript_segments(self, transcript: list) -> List[Dict]:
#         normalized = []
#         for segment in transcript or []:
#             text = " ".join((segment.get("text") or "").split())
#             if not text:
#                 continue

#             start_time = self._safe_float(segment.get("start_time"), default=0.0)
#             end_time = self._safe_float(segment.get("end_time"), default=start_time)
#             if end_time < start_time:
#                 end_time = start_time

#             normalized.append(
#                 {
#                     "text": text,
#                     "start_time": start_time,
#                     "end_time": end_time,
#                 }
#             )

#         return sorted(normalized, key=lambda item: item["start_time"])

#     def _join_segments_with_offsets(self, segments: List[Dict]) -> tuple[str, List[Dict]]:
#         full_text_parts = []
#         offsets = []
#         cursor = 0

#         for segment in segments:
#             if full_text_parts:
#                 full_text_parts.append(" ")
#                 cursor += 1

#             start_offset = cursor
#             full_text_parts.append(segment["text"])
#             cursor += len(segment["text"])
#             end_offset = cursor

#             offsets.append(
#                 {
#                     "start_offset": start_offset,
#                     "end_offset": end_offset,
#                     "start_time": segment["start_time"],
#                     "end_time": segment["end_time"],
#                 }
#             )

#         return "".join(full_text_parts), offsets

#     def _time_for_offset(self, offsets: List[Dict], offset: int, prefer_end: bool) -> float:
#         if not offsets:
#             return 0.0

#         bounded_offset = max(offsets[0]["start_offset"], min(offset, offsets[-1]["end_offset"]))
#         for item in offsets:
#             if item["start_offset"] <= bounded_offset <= item["end_offset"]:
#                 return float(item["end_time"] if prefer_end else item["start_time"])

#         return float(offsets[-1]["end_time"] if prefer_end else offsets[0]["start_time"])

#     def _build_child_documents(self, video_id: str, transcript_chunks: List[Dict]) -> List[Document]:
#         documents = []
#         for chunk in transcript_chunks:
#             metadata = {
#                 "video_id": video_id,
#                 "chunk_index": int(chunk["chunk_index"]),
#                 "start_time": float(chunk["start_time"]),
#                 "end_time": float(chunk["end_time"]),
#             }
#             documents.append(Document(page_content=chunk["text"], metadata=metadata))
#         return documents

#     def _safe_float(self, value, default: Optional[float] = None) -> float:
#         try:
#             return float(value)
#         except (TypeError, ValueError):
#             return float(default or 0.0)


import asyncio
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from app.core.config import settings
from app.utils.logger import logger


class VectorStoreService:
    """Builds flat, sequential transcript chunks and uploads them to Pinecone."""

    def __init__(self):
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        )
        self.index_name = settings.PINECONE_INDEX_NAME
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=180,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

    async def process_and_store(self, video_id: str, formatted_transcript: list) -> Dict[str, list]:
        """Create sequential chunks, embed them, and return Mongo-ready payloads."""
        logger.info(
            "Building flat sequential chunk index for video %s from %d transcript segments",
            video_id,
            len(formatted_transcript or []),
        )

        transcript_chunks = self._build_transcript_chunks(formatted_transcript)
        
        if not transcript_chunks:
            raise ValueError("No transcript chunks were available to embed.")

        documents = [
            Document(
                page_content=chunk["text"], 
                metadata={
                    "video_id": video_id,
                    "chunk_index": chunk["chunk_index"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                }
            )
            for chunk in transcript_chunks
        ]

        def _upload_documents() -> None:
            pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
            pinecone_index = pinecone_client.Index(self.index_name)
            vector_store = PineconeVectorStore(index=pinecone_index, embedding=self.embeddings)
            vector_store.add_documents(documents)

        await asyncio.to_thread(_upload_documents)

        logger.info("Uploaded %d sequential chunks to Pinecone for video: %s", len(documents), video_id)
        return {"transcript_chunks": transcript_chunks}

    def _build_transcript_chunks(self, transcript: list) -> List[Dict]:
        """Normalizes, flattens, chunks, and maps text back to timestamps."""
        
        # 1. Normalize and Sort Segments
        segments = []
        for seg in transcript or []:
            text = " ".join((seg.get("text") or "").split())
            if not text:
                continue
            
            # Isolated parsing prevents cascading data loss if one field is malformed
            start = self._safe_float(seg.get("start_time"), 0.0)
            end = self._safe_float(seg.get("end_time"), start)
                
            segments.append({"text": text, "start_time": start, "end_time": max(start, end)})
            
        segments.sort(key=lambda x: x["start_time"])
        if not segments:
            return []

        # 2. Join into a single string and build an offset mapping
        full_text_parts = []
        offsets = []
        cursor = 0

        for seg in segments:
            if full_text_parts:
                full_text_parts.append(" ")
                cursor += 1
                
            full_text_parts.append(seg["text"])
            offsets.append({
                "start": cursor,
                "end": cursor + len(seg["text"]),
                "start_time": seg["start_time"],
                "end_time": seg["end_time"]
            })
            cursor += len(seg["text"])

        full_text = "".join(full_text_parts)

        # 3. Chunk the unified text
        split_docs = self.text_splitter.create_documents([full_text])

        # 4. Map chunks back to dynamic real-world timestamps
        chunks = []
        for doc in split_docs:
            text = " ".join((doc.page_content or "").split())
            if not text:
                continue

            start_char = doc.metadata.get("start_index")

            if start_char is None:
                start_char = full_text.find(doc.page_content)

            if start_char < 0:
                start_char = 0

            end_char = start_char + len(doc.page_content)

            chunks.append({
                "text": text,
                "chunk_index": len(chunks),
                "start_time": round(self._resolve_time(offsets, start_char, prefer_end=False), 2),
                "end_time": round(self._resolve_time(offsets, end_char, prefer_end=True), 2),
            })

        return chunks

    def _resolve_time(self, offsets: List[Dict], char_index: int, prefer_end: bool) -> float:
        """Finds the precise timestamp for a specific character index."""
        if not offsets:
            return 0.0

        # Bound the index to prevent out-of-range lookups
        bounded_index = max(offsets[0]["start"], min(char_index, offsets[-1]["end"]))
        
        for offset in offsets:
            if offset["start"] <= bounded_index <= offset["end"]:
                return float(offset["end_time"] if prefer_end else offset["start_time"])

        return float(offsets[-1]["end_time"] if prefer_end else offsets[0]["start_time"])

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely converts a value to float without failing the entire segment."""
        try:
            return float(value if value is not None else default)
        except (TypeError, ValueError):
            return float(default)