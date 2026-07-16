# from typing import Any, List, Optional, Tuple

# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEndpointEmbeddings
# from langchain_pinecone import PineconeVectorStore

# from app.core.config import settings
# from app.services.query_intent import QueryIntent, QueryIntentResult
# from app.utils.logger import logger


# class ChatService:
#     FOLLOW_UP_EXACT = {
#         "why",
#         "why?",
#         "explain more",
#         "explain that",
#         "what about that",
#         "what about this",
#         "what about",
#         "tell me more",
#         "can you elaborate",
#         "elaborate",
#         "and then?",
#     }
#     FOLLOW_UP_PREFIXES = [
#         "why is it",
#         "why is that",
#         "why is this",
#         "why does it",
#         "why does that",
#         "why does this",
#         "why are",
#         "why am",
#         "what about",
#         "tell me more",
#         "can you elaborate",
#         "explain more",
#         "explain that",
#         "and then",
#     ]
#     FOLLOW_UP_QUESTION_STARTERS = {"why", "what", "how", "who", "when", "where"}
#     FOLLOW_UP_REFERENCE_STARTERS = {"that", "this", "it", "they", "them"}

#     def __init__(self):
#         self.embeddings = HuggingFaceEndpointEmbeddings(
#             model="sentence-transformers/all-MiniLM-L6-v2",
#             task="feature-extraction",
#             huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
#         )
#         self.vector_store = PineconeVectorStore(
#             index_name=settings.PINECONE_INDEX_NAME,
#             embedding=self.embeddings,
#             pinecone_api_key=settings.PINECONE_API_KEY,
#         )
#         self.llm = self._get_llm(temperature=0.2)
#         self.max_retrieved_chunks = 6
#         self.neighbor_window = 1
#         self.max_context_chars = 9000
#         self.max_history_messages = 8

#     def _get_llm(self, temperature: float = 0):
#         if not settings.GROQ_API_KEY:
#             raise ValueError("Missing GROQ_API_KEY in environment variables.")
#         return ChatGroq(
#             model="llama-3.1-8b-instant",
#             temperature=temperature,
#             api_key=settings.GROQ_API_KEY,
#             max_retries=2,
#         )

#     def answer_query(
#         self,
#         video_doc: Optional[dict],
#         query: str,
#         chat_history: list,
#         intent: QueryIntentResult,
#         time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
#     ) -> dict:
#         """Generate an answer by dispatching the classified query to the right context strategy."""
#         if not video_doc:
#             logger.warning("Chat request received without a video document for query=%s", query)
#             return self._unknown_answer()

#         if intent.intent == QueryIntent.GLOBAL and not intent.is_temporal:
#             answer = self._answer_global_summary(video_doc, query, chat_history)
#             if answer:
#                 return answer

#         if intent.is_temporal and intent.intent == QueryIntent.GLOBAL:
#             answer = self._answer_temporal_transcript(video_doc, query, chat_history, intent)
#             if answer:
#                 return answer

#         return self._answer_rag(video_doc, query, chat_history, intent, time_bounds=time_bounds)

#     def _answer_global_summary(self, video_doc: dict, query: str, chat_history: list) -> Optional[dict]:
#         summary_context = self._build_summary_context(video_doc)
#         if not summary_context:
#             return None

#         logger.info("GLOBAL summary route selected for video %s query=%s", video_doc.get("youtube_id"), query)
#         return self._answer_with_context(
#             video_doc=video_doc,
#             query=query,
#             chat_history=chat_history,
#             context=summary_context,
#             context_kind="summary",
#         )

#     def _answer_temporal_transcript(
#         self,
#         video_doc: dict,
#         query: str,
#         chat_history: list,
#         intent: QueryIntentResult,
#     ) -> Optional[dict]:
#         bounds = self._resolve_temporal_bounds(intent, video_doc)
#         if bounds == (None, None):
#             logger.warning("Temporal intent had no usable bounds; falling back to transcript RAG for query=%s", query)
#             return None

#         logger.info(
#             "Temporal transcript route selected for video %s bounds=%s query=%s",
#             video_doc.get("youtube_id"),
#             bounds,
#             query,
#         )
#         chunks = self._resolve_transcript_chunks(video_doc, query, chat_history, time_bounds=bounds)
#         if not chunks:
#             return self._unknown_answer()

#         return self._answer_with_chunks(video_doc, query, chat_history, chunks)

#     def _answer_rag(
#         self,
#         video_doc: dict,
#         query: str,
#         chat_history: list,
#         intent: QueryIntentResult,
#         time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
#     ) -> dict:
#         if time_bounds is None and intent.is_temporal:
#             time_bounds = self._resolve_temporal_bounds(intent, video_doc)

#         logger.info("RAG route selected for video %s query=%s", video_doc.get("youtube_id"), query)
#         chunks = self._resolve_transcript_chunks(video_doc, query, chat_history, time_bounds=time_bounds)
#         if not chunks:
#             return self._unknown_answer()

#         return self._answer_with_chunks(video_doc, query, chat_history, chunks)

#     def _build_summary_context(self, video_doc: dict) -> Optional[str]:
#         video_summary = video_doc.get("video_summary")
#         section_summaries = sorted(video_doc.get("section_summaries") or [], key=lambda s: s.get("section_index", 0))
#         if not video_summary and not section_summaries:
#             return None

#         if video_summary:
#             context = "Full video summary:\n" + video_summary
#             if section_summaries:
#                 context += "\n\nSupporting section summaries:\n" + "\n\n".join(
#                     f"Section {item.get('section_index', '?')}:\n{(item.get('summary') or '')[:700]}"
#                     # for item in section_summaries[:6]
#                     for item in section_summaries[:10]
#                 )
#             return context

#         return "Chronological section summaries:\n\n" + "\n\n".join(
#             f"Section {item.get('section_index', '?')}:\n{(item.get('summary') or '')[:700]}"
#             # for item in section_summaries[:8]
#             for item in section_summaries[:20]
#         )

#     def _resolve_transcript_chunks(
#         self,
#         video_doc: dict,
#         query: str,
#         chat_history: list,
#         time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
#     ) -> List[dict]:
#         video_id = video_doc.get("youtube_id")
#         search_query = self._rewrite_query(query, chat_history)
#         search_filter = self._build_search_filter(video_id, time_bounds)
#         logger.info("Retrieving vector results for video %s with filter=%s", video_id, search_filter)

#         child_docs = self._retrieve_child_documents(self.vector_store, search_query, search_filter)
#         if not child_docs:
#             logger.warning("No vector retrieval results for video %s query=%s filter=%s", video_id, query, search_filter)
#             return []

#         matched_indices = self._chunk_indices(child_docs[: self.max_retrieved_chunks])
#         chunks = self._select_transcript_chunks(video_doc, matched_indices, time_bounds)
#         if chunks:
#             logger.info(
#                 "Expanded %d vector matches into %d chronological context chunks for video %s",
#                 len(matched_indices),
#                 len(chunks),
#                 video_id,
#             )
#             return chunks

#         logger.warning(
#             "No flat transcript chunks found for video %s indices=%s; falling back to vector payload text",
#             video_id,
#             matched_indices,
#         )
#         return self._select_vector_chunks(child_docs[: self.max_retrieved_chunks], time_bounds)

#     def _retrieve_child_documents(self, vector_store: Any, query: str, search_filter: dict) -> list:
#         try:
#             return vector_store.max_marginal_relevance_search(query, k=12, fetch_k=24, filter=search_filter)
#         except Exception as exc:
#             logger.warning("MMR retrieval failed: %s; falling back to similarity search", exc)
#             return vector_store.similarity_search(query, k=12, filter=search_filter)

#     def _build_search_filter(self, video_id: str, time_bounds: Optional[Tuple[Optional[float], Optional[float]]]) -> dict:
#         search_filter: dict[str, Any] = {"video_id": video_id}
#         if not time_bounds:
#             return search_filter

#         start_time, end_time = time_bounds
#         if start_time is not None:
#             search_filter["start_time"] = {"$gte": float(start_time)}
#         if end_time is not None:
#             search_filter["end_time"] = {"$lte": float(end_time)}
#         return search_filter

#     def _chunk_indices(self, child_docs: list) -> List[int]:
#         indices: List[int] = []
#         for doc in child_docs:
#             try:
#                 indices.append(int(doc.metadata.get("chunk_index")))
#             except (TypeError, ValueError, AttributeError):
#                 continue
#         return indices

#     def _select_transcript_chunks(
#         self,
#         video_doc: dict,
#         matched_indices: Optional[List[int]] = None,
#         time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
#     ) -> List[dict]:
#         chunks = video_doc.get("transcript_chunks") or []
#         if not chunks:
#             logger.warning("Video %s missing flat transcript_chunks; falling back to vector payload text", video_doc.get("youtube_id"))
#             return []

#         selected = [self._normalize_chunk(chunk) for chunk in chunks]
#         if matched_indices:
#             target_indices = self._neighbor_indices(matched_indices)
#             selected = [chunk for chunk in selected if int(chunk["chunk_index"]) in target_indices]

#         if time_bounds:
#             selected = [chunk for chunk in selected if self._chunk_within_bounds(chunk, time_bounds)]

#         return self._dedupe_chunks(selected)

#     def _neighbor_indices(self, matched_indices: List[int]) -> set[int]:
#         return {
#             index + offset
#             for index in matched_indices
#             for offset in range(-self.neighbor_window, self.neighbor_window + 1)
#             if index + offset >= 0
#         }

#     def _select_vector_chunks(
#         self,
#         docs: list,
#         time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
#     ) -> List[dict]:
#         selected = []
#         for fallback_index, doc in enumerate(docs):
#             metadata = getattr(doc, "metadata", {}) or {}
#             chunk = {
#                 "text": getattr(doc, "page_content", "") or "",
#                 "chunk_index": metadata.get("chunk_index", fallback_index),
#                 "start_time": metadata.get("start_time", 0.0),
#                 "end_time": metadata.get("end_time", metadata.get("start_time", 0.0)),
#             }
#             if time_bounds and not self._chunk_within_bounds(chunk, time_bounds):
#                 continue
#             selected.append(self._normalize_chunk(chunk))
#         return self._dedupe_chunks(selected)

#     def _chunk_within_bounds(self, chunk: dict, time_bounds: Tuple[Optional[float], Optional[float]]) -> bool:
#         start_bound, end_bound = time_bounds
#         start_time = self._safe_float(chunk.get("start_time"), 0.0)
#         end_time = self._safe_float(chunk.get("end_time"), start_time)
#         if start_bound is not None and start_time < start_bound:
#             return False
#         if end_bound is not None and end_time > end_bound:
#             return False
#         return True

#     def _dedupe_chunks(self, chunks: List[dict]) -> List[dict]:
#         deduped = {}
#         for chunk in chunks:
#             deduped[int(chunk["chunk_index"])] = chunk
#         return [deduped[index] for index in sorted(deduped)]

#     def _normalize_chunk(self, chunk: dict) -> dict:
#         start_time = self._safe_float(chunk.get("start_time"), 0.0)
#         return {
#             "text": " ".join((chunk.get("text") or "").split()),
#             "chunk_index": int(self._safe_float(chunk.get("chunk_index"), 0)),
#             "start_time": start_time,
#             "end_time": self._safe_float(chunk.get("end_time"), start_time),
#         }

#     def _answer_with_chunks(
#         self,
#         video_doc: dict,
#         query: str,
#         chat_history: list,
#         chunks: List[dict],
#     ) -> dict:
#         return self._answer_with_context(
#             video_doc=video_doc,
#             query=query,
#             chat_history=chat_history,
#             context=self._build_context_from_chunks(chunks),
#             context_kind="transcript",
#         )

#     def _answer_with_context(
#         self,
#         video_doc: dict,
#         query: str,
#         chat_history: list,
#         context: str,
#         context_kind: str,
#     ) -> dict:
#         if not context:
#             return self._unknown_answer()

#         system_instruction = (
#             "You answer broad questions about a YouTube video using only the stored summary information provided. "
#             "Use the full video summary as the primary source. If it contains the answer, respond directly in the same language as the user's question. "
#             "Use supporting section summaries only when needed. Do not add outside facts. "
#             "If the answer is not contained in the provided summary material, say exactly: \"I don't know based on this video.\""
#             if context_kind == "summary"
#             else "You are an AI Tutor for a YouTube video. Answer using only the provided transcript context. "
#             "Do not use outside knowledge or generic textbook definitions. If the answer is not supported by the provided context, say exactly: \"I don't know based on this video.\" "
#             "Answer in the same language as the user's question. Do not add timing references."
#         )

#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", system_instruction),
#                 ("system", f"{context_kind.capitalize()} context:\n{{context}}"),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )

#         logger.info(
#             "LLM answer context size=%d context_kind=%s video=%s",
#             len(context),
#             context_kind,
#             video_doc.get("youtube_id"),
#         )
#         result = (prompt | self.llm).invoke(
#             {
#                 "context": context,
#                 "chat_history": self._prepare_history(chat_history),
#                 "input": query,
#             }
#         )
#         return {"answer": result.content}

#     def _build_context_from_chunks(self, chunks: List[dict]) -> str:
#         blocks = []
#         used_chars = 0
#         for chunk in chunks:
#             text = (chunk.get("text") or "").strip()
#             if not text:
#                 continue

#             remaining_chars = self.max_context_chars - used_chars
#             if remaining_chars <= 0:
#                 break

#             trimmed_text = text[:remaining_chars]
#             used_chars += len(trimmed_text)
#             blocks.append(f"[Transcript chunk {chunk.get('chunk_index')}]\n{trimmed_text}")

#         return "\n\n".join(blocks)

#     def _resolve_temporal_bounds(
#         self,
#         intent: QueryIntentResult,
#         video_doc: dict,
#     ) -> Tuple[Optional[float], Optional[float]]:
#         start_time = intent.start_time_seconds
#         end_time = intent.end_time_seconds
#         duration = self._safe_float(video_doc.get("duration"), 0.0)

#         if start_time is not None and start_time < 0:
#             if duration <= 0:
#                 logger.warning("Relative temporal query needs duration, but video %s has no duration", video_doc.get("youtube_id"))
#                 return (None, None)
#             start_time = max(0.0, duration + start_time)
#             end_time = duration if end_time is None else min(duration, end_time)

#         if start_time is None and end_time is None:
#             return (None, None)
#         if start_time is None:
#             start_time = 0.0
#         if end_time is None:
#             end_time = self._last_chunk_end(video_doc) or duration or None
#         if end_time is not None and end_time < start_time:
#             start_time, end_time = end_time, start_time

#         return (start_time, end_time)

#     def _last_chunk_end(self, video_doc: dict) -> Optional[float]:
#         chunks = video_doc.get("transcript_chunks") or []
#         if not chunks:
#             return None
#         return max(self._safe_float(chunk.get("end_time"), 0.0) for chunk in chunks)

#     def _rewrite_query(self, query: str, chat_history: list) -> str:
#         if not self._needs_rewrite(query):
#             return query

#         history_messages = self._meaningful_history(chat_history, limit=self.max_history_messages)
#         if not history_messages:
#             return query

#         history_text = "\n".join(
#             f"{sender}: {self._trim_text(content)}"
#             for sender, content in history_messages
#         )
#         prompt = (
#             "Rewrite the latest user question as a standalone search query for a video transcript. "
#             "Do not answer it. Return only the rewritten query.\n\n"
#             f"Chat history:\n{history_text}\n\nLatest question: {query}"
#         )
#         response = self.llm.invoke(prompt)
#         rewritten = self._clean_text(response.content)
#         if rewritten and rewritten != query:
#             logger.info("Rewrote query for RAG retrieval: %s -> %s", query, rewritten)
#         return rewritten or query

#     def _needs_rewrite(self, query: str) -> bool:
#         normalized = (query or "").strip().lower()
#         if not normalized:
#             return True

#         has_follow_up_prefix = any(normalized.startswith(prefix + " ") for prefix in self.FOLLOW_UP_PREFIXES)
#         if normalized in self.FOLLOW_UP_EXACT or has_follow_up_prefix:
#             return True

#         words = normalized.split()
#         if len(words) <= 2 and words:
#             if words[0] in self.FOLLOW_UP_QUESTION_STARTERS:
#                 return True
#             if words[0] in self.FOLLOW_UP_REFERENCE_STARTERS:
#                 return True

#         return False

#     def _prepare_history(self, chat_history: list) -> list:
#         formatted = []
#         for sender, content in self._meaningful_history(chat_history, limit=6):
#             content = self._trim_text(content, max_chars=700)
#             if sender == "user":
#                 formatted.append(HumanMessage(content=content))
#             else:
#                 formatted.append(AIMessage(content=content))
#         return formatted

#     def _meaningful_history(self, chat_history: list, limit: int) -> list:
#         meaningful = []
#         for message in reversed(chat_history or []):
#             content = self._clean_text(message.get("content", ""))
#             if not content:
#                 continue
#             if message.get("sender") != "user" and self._is_verbose_assistant_output(content):
#                 continue
#             meaningful.append((message.get("sender", "user"), content))
#             if len(meaningful) >= limit:
#                 break
#         return list(reversed(meaningful))

#     def _is_verbose_assistant_output(self, content: str) -> bool:
#         normalized = content[:600].lower()
#         summary_markers = [
#             "comprehensive summary",
#             "chapter-wise notes",
#             "study notes",
#             "quiz",
#             "questions:",
#             "here are the chapter",
#             "here is the comprehensive summary",
#         ]
#         return len(content.split()) > 180 or any(marker in normalized for marker in summary_markers)

#     def _clean_text(self, content: Any) -> str:
#         return " ".join((str(content or "")).split())

#     def _trim_text(self, content: str, max_chars: int = 500) -> str:
#         clean_content = self._clean_text(content)
#         if len(clean_content) <= max_chars:
#             return clean_content
#         return clean_content[:max_chars].rstrip() + "..."

#     def _safe_float(self, value: Any, default: float = 0.0) -> float:
#         try:
#             return float(value)
#         except (TypeError, ValueError):
#             return float(default)

#     def _unknown_answer(self) -> dict:
#         return {"answer": "I don't know based on this video."}


from typing import Any, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import settings
from app.services.query_intent import QueryIntent, QueryIntentResult
from app.utils.logger import logger


# System prompts pulled out as named constants instead of an inline ternary,
# so `_generate_answer` reads as "build prompt -> call LLM" without a wall of text in the middle.
SUMMARY_SYSTEM_PROMPT = (
    "You answer broad questions about a YouTube video using only the stored summary information provided. "
    "Use the full video summary as the primary source. If it contains the answer, respond directly in the "
    "same language as the user's question. Use supporting section summaries only when needed. "
    "Do not add outside facts. If the answer is not contained in the provided summary material, say exactly: "
    "\"I don't know based on this video.\""
)

TRANSCRIPT_SYSTEM_PROMPT = (
    "You are an AI Tutor for a YouTube video. Answer using only the provided transcript context. "
    "Do not use outside knowledge or generic textbook definitions. If the answer is not supported by the "
    "provided context, say exactly: \"I don't know based on this video.\" Answer in the same language as the "
    "user's question. Do not add timing references."
)

# Assistant messages matching these markers (or long ones) are generated study material,
# not natural conversation - including them in chat history just wastes context budget.
VERBOSE_ASSISTANT_MARKERS = (
    "comprehensive summary",
    "chapter-wise notes",
    "study notes",
    "quiz",
    "questions:",
    "here are the chapter",
    "here is the comprehensive summary",
)


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
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
            max_retries=2,
        )

    # ------------------------------------------------------------------
    # Entry point / routing
    # ------------------------------------------------------------------

    def answer_query(
        self,
        video_doc: Optional[dict],
        query: str,
        chat_history: list,
        intent: QueryIntentResult,
        time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> dict:
        """Generate an answer by dispatching the classified query to the right context strategy.

        There are two real strategies:
          - GLOBAL, non-temporal questions try the stored video summary first.
          - Everything else (including temporal questions) goes through transcript RAG,
            optionally filtered to a time window.
        If the summary route has nothing to work with, we fall through to RAG rather
        than returning "I don't know" prematurely.
        """
        if not video_doc:
            logger.warning("Chat request received without a video document for query=%s", query)
            return self._unknown_answer()

        if intent.intent == QueryIntent.GLOBAL and not intent.is_temporal:
            summary_context = self._build_summary_context(video_doc)
            if summary_context:
                logger.info(
                    "GLOBAL summary route selected for video %s query=%s",
                    video_doc.get("youtube_id"),
                    query,
                )
                return self._generate_answer(video_doc, query, chat_history, summary_context, kind="summary")
            # No summary stored for this video - fall through to RAG below.

        if time_bounds is None and intent.is_temporal:
            time_bounds = self._resolve_temporal_bounds(intent, video_doc)
            if time_bounds == (None, None):
                logger.warning(
                    "Temporal intent had no usable bounds; retrieving without a time filter for query=%s",
                    query,
                )
                time_bounds = None

        logger.info(
            "RAG route selected for video %s query=%s bounds=%s",
            video_doc.get("youtube_id"),
            query,
            time_bounds,
        )
        chunks = self.retrieve_context(video_doc, query, chat_history, time_bounds)
        if not chunks:
            return self._unknown_answer()

        context = self._build_context_from_chunks(chunks)
        return self._generate_answer(video_doc, query, chat_history, context, kind="transcript")

    # ------------------------------------------------------------------
    # Summary route
    # ------------------------------------------------------------------

    def _build_summary_context(self, video_doc: dict) -> Optional[str]:
        video_summary = video_doc.get("video_summary")
        section_summaries = sorted(video_doc.get("section_summaries") or [], key=lambda s: s.get("section_index", 0))
        if not video_summary and not section_summaries:
            return None

        if video_summary:
            context = "Full video summary:\n" + video_summary
            if section_summaries:
                context += "\n\nSupporting section summaries:\n" + "\n\n".join(
                    f"Section {item.get('section_index', '?')}:\n{(item.get('summary') or '')[:700]}"
                    for item in section_summaries[:10]
                )
            return context

        return "Chronological section summaries:\n\n" + "\n\n".join(
            f"Section {item.get('section_index', '?')}:\n{(item.get('summary') or '')[:700]}"
            for item in section_summaries[:20]
        )

    # ------------------------------------------------------------------
    # RAG route: one linear pipeline instead of six scattered helpers
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        video_doc: dict,
        query: str,
        chat_history: list,
        time_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ) -> List[dict]:
        """Turn a user query into ordered, deduplicated transcript chunks.

        Steps: rewrite query for standalone search -> build a Pinecone metadata filter
        -> vector search -> expand matches to neighboring chunks for continuity ->
        hydrate matched indices against the flat transcript (falling back to the raw
        vector payload text if the video has no flat transcript_chunks) -> filter to
        the time window if any -> dedupe by chunk index.
        """
        video_id = video_doc.get("youtube_id")
        search_query = self._rewrite_query(query, chat_history)

        search_filter: dict[str, Any] = {"video_id": video_id}
        if time_bounds:
            start_time, end_time = time_bounds
            if start_time is not None:
                search_filter["start_time"] = {"$gte": float(start_time)}
            if end_time is not None:
                search_filter["end_time"] = {"$lte": float(end_time)}

        logger.info("Retrieving vector results for video %s with filter=%s", video_id, search_filter)
        child_docs = self._search_vector_store(search_query, search_filter)
        if not child_docs:
            logger.warning("No vector retrieval results for video %s query=%s filter=%s", video_id, query, search_filter)
            return []
        child_docs = child_docs[: self.max_retrieved_chunks]

        matched_indices = []
        for doc in child_docs:
            try:
                matched_indices.append(int(doc.metadata.get("chunk_index")))
            except (TypeError, ValueError, AttributeError):
                continue

        target_indices = {
            index + offset
            for index in matched_indices
            for offset in range(-self.neighbor_window, self.neighbor_window + 1)
            if index + offset >= 0
        }

        transcript_chunks = video_doc.get("transcript_chunks") or []
        if transcript_chunks:
            candidates = [self._normalize_chunk(chunk) for chunk in transcript_chunks]
            if target_indices:
                candidates = [c for c in candidates if c["chunk_index"] in target_indices]
        else:
            logger.warning(
                "Video %s missing flat transcript_chunks; using vector payload text instead",
                video_id,
            )
            candidates = []
            for fallback_index, doc in enumerate(child_docs):
                metadata = getattr(doc, "metadata", {}) or {}
                candidates.append(
                    self._normalize_chunk(
                        {
                            "text": getattr(doc, "page_content", "") or "",
                            "chunk_index": metadata.get("chunk_index", fallback_index),
                            "start_time": metadata.get("start_time", 0.0),
                            "end_time": metadata.get("end_time", metadata.get("start_time", 0.0)),
                        }
                    )
                )

        if time_bounds:
            candidates = [c for c in candidates if self._chunk_within_bounds(c, time_bounds)]

        chunks = self._dedupe_chunks(candidates)
        logger.info(
            "Retrieved %d context chunks for video %s from %d vector matches",
            len(chunks),
            video_id,
            len(matched_indices),
        )
        return chunks

    def _search_vector_store(self, query: str, search_filter: dict) -> list:
        try:
            return self.vector_store.max_marginal_relevance_search(query, k=12, fetch_k=24, filter=search_filter)
        except Exception as exc:
            logger.warning("MMR retrieval failed: %s; falling back to similarity search", exc)
            return self.vector_store.similarity_search(query, k=12, filter=search_filter)

    def _rewrite_query(self, query: str, chat_history: list) -> str:
        """Ask the LLM to rewrite the query as standalone if it depends on prior context.

        Replaces a heuristic rule engine (follow-up phrase lists, prefix matching, word
        counts) with a single model call: it's cheap, generalizes across languages and
        phrasings, and needs zero maintenance as new follow-up patterns show up.
        """
        history = self._recent_history(chat_history, limit=self.max_history_messages)
        if not history:
            return query

        history_text = "\n".join(f"{sender}: {self._trim_text(text, max_chars=500)}" for sender, text in history)
        prompt = (
            "You are given a conversation and the latest user message about a video.\n"
            "If the latest message already stands on its own as a search query, return it unchanged.\n"
            "Otherwise, rewrite it into a standalone search query using context from the conversation.\n"
            "Return only the rewritten query, nothing else.\n\n"
            f"Conversation:\n{history_text}\n\nLatest message: {query}"
        )
        response = self.llm.invoke(prompt)
        rewritten = self._clean_text(response.content)
        if rewritten and rewritten != query:
            logger.info("Rewrote query for RAG retrieval: %s -> %s", query, rewritten)
        return rewritten or query

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def _generate_answer(
        self,
        video_doc: dict,
        query: str,
        chat_history: list,
        context: str,
        kind: str,
    ) -> dict:
        if not context:
            return self._unknown_answer()

        system_instruction = SUMMARY_SYSTEM_PROMPT if kind == "summary" else TRANSCRIPT_SYSTEM_PROMPT
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_instruction),
                ("system", f"{kind.capitalize()} context:\n{{context}}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        logger.info("LLM answer context size=%d kind=%s video=%s", len(context), kind, video_doc.get("youtube_id"))
        result = (prompt | self.llm).invoke(
            {
                "context": context,
                "chat_history": self._prepare_history(chat_history),
                "input": query,
            }
        )
        return {"answer": result.content}

    def _build_context_from_chunks(self, chunks: List[dict]) -> str:
        blocks = []
        used_chars = 0
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue

            remaining_chars = self.max_context_chars - used_chars
            if remaining_chars <= 0:
                break

            trimmed_text = text[:remaining_chars]
            used_chars += len(trimmed_text)
            blocks.append(f"[Transcript chunk {chunk.get('chunk_index')}]\n{trimmed_text}")

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def _recent_history(self, chat_history: list, limit: int) -> List[Tuple[str, str]]:
        """Last `limit` non-empty, non-verbose messages as (sender, cleaned_text), oldest first.

        Shared by `_rewrite_query` (needs raw text) and `_prepare_history` (needs
        LangChain messages) so the "what counts as real conversation" rule lives in
        exactly one place instead of being split across separate filter/format methods.
        """
        recent: List[Tuple[str, str]] = []
        for message in reversed(chat_history or []):
            text = self._clean_text(message.get("content", ""))
            if not text:
                continue

            sender = message.get("sender", "user")
            if sender != "user":
                is_verbose = len(text.split()) > 180 or any(
                    marker in text[:600].lower() for marker in VERBOSE_ASSISTANT_MARKERS
                )
                if is_verbose:
                    continue

            recent.append((sender, text))
            if len(recent) >= limit:
                break
        return list(reversed(recent))

    def _prepare_history(self, chat_history: list) -> list:
        formatted = []
        for sender, text in self._recent_history(chat_history, limit=6):
            trimmed = self._trim_text(text, max_chars=700)
            formatted.append(HumanMessage(content=trimmed) if sender == "user" else AIMessage(content=trimmed))
        return formatted

    # ------------------------------------------------------------------
    # Temporal bounds
    # ------------------------------------------------------------------

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
        chunks = video_doc.get("transcript_chunks") or []
        if not chunks:
            return None
        return max(self._safe_float(chunk.get("end_time"), 0.0) for chunk in chunks)

    # ------------------------------------------------------------------
    # Small shared utilities (pure, reused across the pipeline above)
    # ------------------------------------------------------------------

    def _normalize_chunk(self, chunk: dict) -> dict:
        start_time = self._safe_float(chunk.get("start_time"), 0.0)
        return {
            "text": " ".join((chunk.get("text") or "").split()),
            "chunk_index": int(self._safe_float(chunk.get("chunk_index"), 0)),
            "start_time": start_time,
            "end_time": self._safe_float(chunk.get("end_time"), start_time),
        }

    def _dedupe_chunks(self, chunks: List[dict]) -> List[dict]:
        deduped = {int(chunk["chunk_index"]): chunk for chunk in chunks}
        return [deduped[index] for index in sorted(deduped)]

    def _chunk_within_bounds(self, chunk: dict, time_bounds: Tuple[Optional[float], Optional[float]]) -> bool:
        start_bound, end_bound = time_bounds
        start_time = self._safe_float(chunk.get("start_time"), 0.0)
        end_time = self._safe_float(chunk.get("end_time"), start_time)
        if start_bound is not None and start_time < start_bound:
            return False
        if end_bound is not None and end_time > end_bound:
            return False
        return True

    def _clean_text(self, content: Any) -> str:
        return " ".join((str(content or "")).split())

    def _trim_text(self, content: str, max_chars: int = 500) -> str:
        clean_content = self._clean_text(content)
        if len(clean_content) <= max_chars:
            return clean_content
        return clean_content[:max_chars].rstrip() + "..."

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _unknown_answer(self) -> dict:
        return {"answer": "I don't know based on this video."}