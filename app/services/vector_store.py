import os
from typing import Dict, List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings

class VectorStoreService:
    """Builds a per-video parent-child index and uploads to Pinecone."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        )
        self.index_name = settings.PINECONE_INDEX_NAME
        
        # Chunking constraints
        self.parent_target_seconds = 180
        self.parent_max_seconds = 300
        self.parent_max_chars = 5500
        self.child_target_chars = 750
        self.child_overlap_chars = 140

    def process_and_store(self, video_id: str, formatted_transcript: list) -> Dict[str, list]:
        """Create parent sections, embed searchable child chunks, and save to Pinecone."""
        print(f"☁️ [VECTOR STORE] Building parent-child index for video: {video_id}")
        
        parent_sections = self._build_parent_sections(video_id, formatted_transcript)
        child_documents = self._build_child_documents(parent_sections)

        if not child_documents:
            raise ValueError("No transcript chunks were available to embed.")

        # 1. Upload the small chunks to Pinecone!
        PineconeVectorStore.from_documents(
            documents=child_documents,
            embedding=self.embeddings,
            index_name=self.index_name,
            pinecone_api_key=settings.PINECONE_API_KEY
        )
        print(f"✅ [VECTOR STORE] Uploaded {len(child_documents)} child chunks to Pinecone.")

        # 2. Format the parent sections so video.py can save them to MongoDB
        public_parent_sections = [
            {
                "parent_id": parent["parent_id"],
                "section_index": parent["section_index"],
                "start_time": parent["start_time"],
                "end_time": parent["end_time"],
                "text": parent["text"],
            }
            for parent in parent_sections
        ]

        # Return the parents so they can be saved safely in MongoDB (No local files!)
        return {"parent_sections": public_parent_sections}

    # ---------------------------------------------------------
    # HELPER METHODS (Unchanged from your logic)
    # ---------------------------------------------------------
    def _build_parent_sections(self, video_id: str, transcript: list) -> List[Dict]:
        parent_sections = []
        current_chunks = []
        current_chars = 0
        for chunk in transcript:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            current_chunks.append(chunk)
            current_chars += len(text)
            start_time = float(current_chunks[0].get("start_time", 0.0))
            end_time = float(chunk.get("end_time", start_time))
            duration = end_time - start_time
            should_flush = (
                duration >= self.parent_target_seconds or 
                duration >= self.parent_max_seconds or 
                current_chars >= self.parent_max_chars
            )
            if should_flush:
                parent_sections.append(self._make_parent_section(video_id, len(parent_sections), current_chunks))
                current_chunks = []
                current_chars = 0
                
        if current_chunks:
            parent_sections.append(self._make_parent_section(video_id, len(parent_sections), current_chunks))
        return parent_sections

    def _make_parent_section(self, video_id: str, index: int, chunks: list) -> Dict:
        start_time = float(chunks[0].get("start_time", 0.0))
        end_time = float(chunks[-1].get("end_time", start_time))
        text = " ".join(chunk.get("text", "").strip() for chunk in chunks if chunk.get("text"))
        return {
            "parent_id": f"{video_id}:parent:{index}",
            "section_index": index + 1,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "text": text,
            "chunks": chunks,
        }

    def _build_child_documents(self, parent_sections: List[Dict]) -> List[Document]:
        child_documents = []
        for parent in parent_sections:
            child_chunks = self._split_parent_chunks(parent["chunks"])
            for child_index, child in enumerate(child_chunks):
                metadata = {
                    "video_id": parent["parent_id"].split(":parent:")[0],
                    "parent_id": parent["parent_id"],
                    "section_index": parent["section_index"],
                    "chunk_index": child_index,
                    "start_time": child["start_time"],
                    "end_time": child["end_time"],
                }
                child_documents.append(Document(page_content=child["text"], metadata=metadata))
        return child_documents

    def _split_parent_chunks(self, transcript_chunks: list) -> List[Dict]:
        children = []
        current_chunks = []
        current_chars = 0
        for chunk in transcript_chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            current_chunks.append(chunk)
            current_chars += len(text)
            if current_chars >= self.child_target_chars:
                children.append(self._make_child_chunk(current_chunks))
                current_chunks = self._overlap_tail(current_chunks)
                current_chars = sum(len(item.get("text", "")) for item in current_chunks)
                
        if current_chunks:
            child = self._make_child_chunk(current_chunks)
            if not children or child["text"] != children[-1]["text"]:
                children.append(child)
        return children

    def _overlap_tail(self, chunks: list) -> list:
        tail = []
        char_count = 0
        for chunk in reversed(chunks):
            tail.insert(0, chunk)
            char_count += len(chunk.get("text", ""))
            if char_count >= self.child_overlap_chars:
                break
        return tail

    def _make_child_chunk(self, chunks: list) -> Dict:
        start_time = float(chunks[0].get("start_time", 0.0))
        end_time = float(chunks[-1].get("end_time", start_time))
        text = " ".join(chunk.get("text", "").strip() for chunk in chunks if chunk.get("text"))
        return {
            "text": text,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
        }