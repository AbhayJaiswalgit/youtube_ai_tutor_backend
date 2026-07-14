from typing import List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import settings
from app.services.query_intent import QueryIntent, QueryIntentResult
from app.utils.logger import logger


class ChatService:
    def __init__(self):
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        )
        self.vector_store = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            pinecone_api_key=settings.PINECONE_API_KEY,
        )
        self.llm = self._get_llm(temperature=0.2)
        self.max_retrieved_chunks = 6
        self.neighbor_window = 1
        self.max_context_chars = 9000
        self.max_history_messages = 8

    def _get_llm(self, temperature: float = 0):
        if not settings.GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in environment variables.")
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
            max_retries=2,
        )

    def get_summary_answer(self, video_doc: dict, query: str, intent: QueryIntentResult) -> Optional[dict]:
        """Answer broad, non-temporal requests from stored summaries."""
        if not video_doc or intent.intent != QueryIntent.GLOBAL or intent.is_temporal:
            return None

        video_summary = video_doc.get("video_summary")
        section_summaries = sorted(video_doc.get("section_summaries") or [], key=lambda s: s.get("section_index", 0))
        if not video_summary and not section_summaries:
            return None

        logger.info("GLOBAL summary route selected for video %s query=%s", video_doc.get("youtube_id"), query)
        if video_summary:
            context = "Full video summary:\n" + video_summary
            if section_summaries:
                section_text = "\n\n".join(
                    f"Section {s.get('section_index', '?')}:\n{s.get('summary', '')[:700]}"
                    for s in section_summaries[:6]
                )
                context += "\n\nSupporting section summaries:\n" + section_text
        else:
            context = "Chronological section summaries:\n\n"
            context += "\n\n".join(
                f"Section {s.get('section_index', '?')}:\n{s.get('summary', '')[:700]}"
                for s in section_summaries[:8]
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You answer broad questions about a YouTube video using only the stored summary information provided. "
                    "Use the full video summary as the primary source. If it contains the answer, respond directly in the same language as the user's question. "
                    "Use supporting section summaries only when needed. Do not add outside facts. "
                    "If the answer is not contained in the provided summary material, say exactly: \"I don't know based on this video.\"",
                ),
                ("system", "Video context:\n{context}"),
                ("human", "{input}"),
            ]
        )

        logger.info("GLOBAL prompt context size=%d for video %s", len(context), video_doc.get("youtube_id"))
        result = (prompt | self.llm).invoke({"context": context, "input": query})
        return {"answer": result.content}

    def get_temporal_answer(
        self,
        video_doc: dict,
        query: str,
        chat_history: list,
        intent: QueryIntentResult,
    ) -> dict:
        """Route temporal queries through direct Mongo chunks or filtered vector search."""
        if not video_doc:
            logger.warning("Temporal query received with missing video document: %s", query)
            return self._unknown_answer()

        bounds = self._resolve_temporal_bounds(intent, video_doc)
        if bounds == (None, None):
            logger.warning("Temporal intent had no usable bounds; falling back to standard RAG for query=%s", query)
            return self.get_answer(video_doc, intent.clean_query or query, chat_history)

        if intent.intent == QueryIntent.GLOBAL:
            chunks = self._fetch_chunks_by_time(video_doc, bounds)
            logger.info(
                "Pure temporal route selected for video %s bounds=%s chunks=%d",
                video_doc.get("youtube_id"),
                bounds,
                len(chunks),
            )
            if not chunks:
                return self._unknown_answer()
            return self._answer_from_chunks(video_doc, query, chat_history, chunks)

        logger.info(
            "Hybrid temporal semantic route selected for video %s bounds=%s query=%s",
            video_doc.get("youtube_id"),
            bounds,
            intent.clean_query,
        )
        return self.get_answer(video_doc, intent.clean_query or query, chat_history, time_bounds=bounds)

    def get_answer(
        self,
        video_doc: dict,
        query: str,
        chat_history: list,
        time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> dict:
        """Answer factual questions with vector hits expanded by chronological neighbors."""
        logger.info("RAG_QUERY route selected for query: %s", query)

        if not video_doc:
            logger.warning("RAG query received with missing video document for query: %s", query)
            return self._unknown_answer()

        video_id = video_doc.get("youtube_id")
        search_query = self._standalone_query(query, chat_history)
        search_filter = self._build_search_filter(video_id, time_bounds)

        logger.info("Retrieving RAG documents for video %s with filter=%s", video_id, search_filter)
        child_docs = self._retrieve_child_documents(self.vector_store, search_query, search_filter)
        if not child_docs:
            logger.warning("No vector retrieval results for video %s query=%s filter=%s", video_id, query, search_filter)
            return self._unknown_answer()

        matched_indices = self._extract_chunk_indices(child_docs[: self.max_retrieved_chunks])
        chunks = self._neighbor_window_chunks(video_doc, matched_indices, time_bounds)
        if not chunks:
            logger.warning(
                "No flat Mongo chunks found for video %s indices=%s; falling back to vector payload text",
                video_id,
                matched_indices,
            )
            chunks = self._chunks_from_docs(child_docs[: self.max_retrieved_chunks], time_bounds)

        if not chunks:
            return self._unknown_answer()

        logger.info(
            "Expanded %d vector matches into %d chronological context chunks for video %s",
            len(matched_indices),
            len(chunks),
            video_id,
        )
        return self._answer_from_chunks(video_doc, query, chat_history, chunks)

    def _retrieve_child_documents(self, vector_store: PineconeVectorStore, query: str, search_filter: dict) -> list:
        try:
            return vector_store.max_marginal_relevance_search(
                query,
                k=12,
                fetch_k=24,
                filter=search_filter,
            )
        except Exception as exc:
            logger.warning("MMR retrieval failed: %s; falling back to similarity search", exc)
            return vector_store.similarity_search(query, k=12, filter=search_filter)

    def _build_search_filter(self, video_id: str, time_bounds: Optional[Tuple[Optional[float], Optional[float]]]) -> dict:
        search_filter = {"video_id": video_id}
        if not time_bounds:
            return search_filter

        start_time, end_time = time_bounds
        if start_time is not None:
            search_filter["start_time"] = {"$gte": float(start_time)}
        if end_time is not None:
            search_filter["end_time"] = {"$lte": float(end_time)}
        return search_filter

    def _extract_chunk_indices(self, child_docs: list) -> List[int]:
        indices = []
        for doc in child_docs:
            try:
                indices.append(int(doc.metadata.get("chunk_index")))
            except (TypeError, ValueError, AttributeError):
                continue
        return indices

    def _neighbor_window_chunks(
        self,
        video_doc: dict,
        matched_indices: List[int],
        time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> List[dict]:
        if not matched_indices:
            return []

        target_indices = set()
        for index in matched_indices:
            for offset in range(-self.neighbor_window, self.neighbor_window + 1):
                neighbor_index = index + offset
                if neighbor_index >= 0:
                    target_indices.add(neighbor_index)

        return self._fetch_chunks_by_indices(video_doc, target_indices, time_bounds)

    def _fetch_chunks_by_indices(
        self,
        video_doc: dict,
        indices: set[int],
        time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> List[dict]:
        chunks = self._flat_transcript_chunks(video_doc)
        if not chunks:
            return []

        selected = []
        for chunk in chunks:
            try:
                chunk_index = int(chunk.get("chunk_index"))
            except (TypeError, ValueError):
                continue
            if chunk_index not in indices:
                continue
            if time_bounds and not self._chunk_within_bounds(chunk, time_bounds):
                continue
            selected.append(self._normalize_chunk(chunk))

        return self._dedupe_and_sort_chunks(selected)

    def _fetch_chunks_by_time(self, video_doc: dict, time_bounds: Tuple[Optional[float], Optional[float]]) -> List[dict]:
        chunks = self._flat_transcript_chunks(video_doc)
        if not chunks:
            logger.warning("Video %s does not have transcript_chunks for temporal lookup", video_doc.get("youtube_id"))
            return []

        selected = [self._normalize_chunk(chunk) for chunk in chunks if self._chunk_within_bounds(chunk, time_bounds)]
        return self._dedupe_and_sort_chunks(selected)

    def _flat_transcript_chunks(self, video_doc: dict) -> list:
        chunks = video_doc.get("transcript_chunks") or []
        if chunks:
            return chunks
        logger.warning("Video %s missing flat transcript_chunks; legacy document cannot use neighbor lookup", video_doc.get("youtube_id"))
        return []

    def _chunk_within_bounds(self, chunk: dict, time_bounds: Tuple[Optional[float], Optional[float]]) -> bool:
        start_bound, end_bound = time_bounds
        start_time = self._safe_float(chunk.get("start_time"), 0.0)
        end_time = self._safe_float(chunk.get("end_time"), start_time)
        if start_bound is not None and start_time < start_bound:
            return False
        if end_bound is not None and end_time > end_bound:
            return False
        return True

    def _chunks_from_docs(
        self,
        docs: list,
        time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> List[dict]:
        chunks = []
        for fallback_index, doc in enumerate(docs):
            metadata = getattr(doc, "metadata", {}) or {}
            chunk = {
                "text": getattr(doc, "page_content", "") or "",
                "chunk_index": metadata.get("chunk_index", fallback_index),
                "start_time": metadata.get("start_time", 0.0),
                "end_time": metadata.get("end_time", metadata.get("start_time", 0.0)),
            }
            if time_bounds and not self._chunk_within_bounds(chunk, time_bounds):
                continue
            chunks.append(self._normalize_chunk(chunk))
        return self._dedupe_and_sort_chunks(chunks)

    def _dedupe_and_sort_chunks(self, chunks: List[dict]) -> List[dict]:
        deduped = {}
        for chunk in chunks:
            deduped[int(chunk["chunk_index"])] = chunk
        return [deduped[index] for index in sorted(deduped)]

    def _normalize_chunk(self, chunk: dict) -> dict:
        start_time = self._safe_float(chunk.get("start_time"), 0.0)
        return {
            "text": " ".join((chunk.get("text") or "").split()),
            "chunk_index": int(self._safe_float(chunk.get("chunk_index"), 0)),
            "start_time": start_time,
            "end_time": self._safe_float(chunk.get("end_time"), start_time),
        }

    def _answer_from_chunks(self, video_doc: dict, query: str, chat_history: list, chunks: List[dict]) -> dict:
        context = self._context_from_chunks(chunks)
        if not context:
            return self._unknown_answer()

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI Tutor for a YouTube video. Answer using only the provided transcript context. "
                    "Do not use outside knowledge or generic textbook definitions. If the answer is not supported by the provided context, say exactly: \"I don't know based on this video.\" "
                    "Answer in the same language as the user's question. Do not add timing references.\n\n"
                    "Transcript context:\n{context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        logger.info(
            "LLM answer context size=%d chunks=%d video=%s",
            len(context),
            len(chunks),
            video_doc.get("youtube_id"),
        )
        result = (qa_prompt | self.llm).invoke(
            {
                "context": context,
                "chat_history": self._format_history(chat_history),
                "input": query,
            }
        )
        return {"answer": result.content}

    def _context_from_chunks(self, chunks: List[dict]) -> str:
        blocks = []
        used_chars = 0
        for chunk in chunks:
            text = chunk.get("text") or ""
            if not text:
                continue

            remaining_chars = self.max_context_chars - used_chars
            if remaining_chars <= 0:
                break

            trimmed_text = text[:remaining_chars]
            used_chars += len(trimmed_text)
            blocks.append(f"[Transcript chunk {chunk.get('chunk_index')}]\n{trimmed_text}")

        return "\n\n".join(blocks)

    def _resolve_temporal_bounds(
        self,
        intent: QueryIntentResult,
        video_doc: dict,
    ) -> Tuple[Optional[float], Optional[float]]:
        start_time = intent.start_time_seconds
        end_time = intent.end_time_seconds
        duration = self._safe_float(video_doc.get("duration"), 0.0)

        if start_time is not None and start_time < 0:
            if duration <= 0:
                logger.warning("Relative temporal query needs duration, but video %s has no duration", video_doc.get("youtube_id"))
                return (None, None)
            start_time = max(0.0, duration + start_time)
            end_time = duration if end_time is None else min(duration, end_time)

        if start_time is None and end_time is None:
            return (None, None)
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = self._last_chunk_end(video_doc) or duration or None
        if end_time is not None and end_time < start_time:
            start_time, end_time = end_time, start_time

        return (start_time, end_time)

    def _last_chunk_end(self, video_doc: dict) -> Optional[float]:
        chunks = self._flat_transcript_chunks(video_doc)
        if not chunks:
            return None
        return max(self._safe_float(chunk.get("end_time"), 0.0) for chunk in chunks)

    def _standalone_query(self, query: str, chat_history: list) -> str:
        meaningful_history = self._meaningful_history(chat_history, limit=self.max_history_messages)
        if not meaningful_history:
            return query
        if not self._needs_rewrite(query):
            return query

        history_text = "\n".join(
            f"{msg.get('sender', 'user')}: {self._trim_history_content(msg.get('content', ''))}"
            for msg in meaningful_history
        )
        prompt = (
            "Rewrite the latest user question as a standalone search query for a "
            "video transcript. Do not answer it. Return only the rewritten query.\n\n"
            f"Chat history:\n{history_text}\n\nLatest question: {query}"
        )
        response = self.llm.invoke(prompt)
        rewritten = response.content.strip()
        if rewritten and rewritten != query:
            logger.info("Rewrote query for RAG retrieval: %s -> %s", query, rewritten)
        return rewritten or query

    def _needs_rewrite(self, query: str) -> bool:
        normalized = (query or "").strip().lower()
        if not normalized:
            return True

        follow_up_exact = {
            "why",
            "why?",
            "explain more",
            "explain that",
            "what about that",
            "what about this",
            "what about",
            "tell me more",
            "can you elaborate",
            "elaborate",
            "and then?",
        }
        follow_up_prefixes = [
            "why is it",
            "why is that",
            "why is this",
            "why does it",
            "why does that",
            "why does this",
            "why are",
            "why am",
            "what about",
            "tell me more",
            "can you elaborate",
            "explain more",
            "explain that",
            "and then",
        ]

        if normalized in follow_up_exact or any(normalized.startswith(prefix + " ") for prefix in follow_up_prefixes):
            return True

        words = normalized.split()
        if len(words) <= 2 and words:
            if words[0] in {"why", "what", "how", "who", "when", "where"}:
                return True
            if words[0] in {"that", "this", "it", "they", "them"}:
                return True

        return False

    def _format_history(self, chat_history: list) -> list:
        formatted_history = []
        for msg in self._meaningful_history(chat_history, limit=6):
            content = self._trim_history_content(msg.get("content", ""), max_chars=700)
            if msg.get("sender") == "user":
                formatted_history.append(HumanMessage(content=content))
            else:
                formatted_history.append(AIMessage(content=content))
        return formatted_history

    def _meaningful_history(self, chat_history: list, limit: int) -> list:
        meaningful = []
        for msg in reversed(chat_history or []):
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if msg.get("sender") != "user" and self._is_verbose_assistant_output(content):
                continue
            meaningful.append(msg)
            if len(meaningful) >= limit:
                break

        return list(reversed(meaningful))

    def _is_verbose_assistant_output(self, content: str) -> bool:
        normalized = content[:600].lower()
        summary_markers = [
            "comprehensive summary",
            "chapter-wise notes",
            "study notes",
            "quiz",
            "questions:",
            "here are the chapter",
            "here is the comprehensive summary",
        ]
        return len(content.split()) > 180 or any(marker in normalized for marker in summary_markers)

    def _trim_history_content(self, content: str, max_chars: int = 500) -> str:
        clean_content = " ".join((content or "").split())
        if len(clean_content) <= max_chars:
            return clean_content
        return clean_content[:max_chars].rstrip() + "..."

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _unknown_answer(self) -> dict:
        return {"answer": "I don't know based on this video."}
