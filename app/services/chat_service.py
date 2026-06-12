import json
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import settings
from app.services.query_intent import QueryIntent


class ChatService:
    def __init__(self):
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
        )
        # 1. Initialize Pinecone directly (No more local folder paths!)
        self.vector_store = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=self.embeddings,
            pinecone_api_key=settings.PINECONE_API_KEY
        )
        self.llm = self._get_llm(temperature=0.2)
        self.max_parent_sections = 4
        self.max_context_chars = 12000

    def _get_llm(self, temperature: float = 0):
        if not settings.GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in environment variables.")
        return ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
            max_retries=2,
        )

    def get_summary_answer(self, video_doc: dict, query: str, intent: QueryIntent) -> Optional[dict]:
        """Answer broad requests from stored summaries without vector search."""
        if not video_doc:
            return None

        video_summary = video_doc.get("video_summary")
        section_summaries = video_doc.get("section_summaries") or []

        if intent == QueryIntent.SUMMARY and video_summary:
            if self._is_plain_summary_request(query):
                return {
                    "answer": f"Here is the comprehensive summary of the video:\n\n{video_summary}",
                    "citations": self._summary_citations(section_summaries),
                }

            context = self._summary_context(video_summary, section_summaries)
            answer = self._invoke_grounded_summary(query, context)
            return {"answer": answer, "citations": self._summary_citations(section_summaries)}

        if intent == QueryIntent.SECTION_SUMMARY and section_summaries:
            if self._is_plain_section_request(query):
                answer = self._format_section_summaries(section_summaries)
            else:
                context = self._summary_context(video_summary or "", section_summaries)
                answer = self._invoke_grounded_summary(query, context)

            return {"answer": answer, "citations": self._summary_citations(section_summaries)}

        return None

    def get_answer(self, video_doc: dict, query: str, chat_history: list) -> dict:
        """Answer factual questions with small-to-big retrieval."""
        print(f"[CHAT SERVICE] Thinking about: '{query}'")

        video_id=video_doc.get("youtube_id")

        parents = video_doc.get("parent_sections", [])
        formatted_history = self._format_history(chat_history)
        search_query = self._standalone_query(query, chat_history)

        child_docs = self._retrieve_child_documents(self.vector_store, video_id, search_query)
        if not child_docs:
            return {
                "answer": "I don't know based on this video.",
                "citations": [],
            }

        if parents:
            context_blocks, citations = self._expand_to_parent_context(child_docs, parents)
            if not context_blocks:
                context_blocks, citations = self._legacy_child_context(child_docs)
        else:
            context_blocks, citations = self._legacy_child_context(child_docs)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI Tutor for a YouTube video. Answer using only the "
                    "provided video context. If the answer is not supported by the "
                    "context, say: \"I don't know based on this video.\" Explain clearly "
                    "and cite the relevant section timing when helpful.\n\n"
                    "Video context:\n{context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = qa_prompt | self.llm
        result = chain.invoke(
            {
                "context": "\n\n".join(context_blocks),
                "chat_history": formatted_history,
                "input": query,
            }
        )

        return {"answer": result.content, "citations": citations}

    

    

    def _retrieve_child_documents(self, vector_store: PineconeVectorStore, video_id: str, query: str) -> list:
        search_filter = {"video_id": video_id}
        try:
            return vector_store.max_marginal_relevance_search(
                query,
                k=12,
                fetch_k=24,
                filter=search_filter,
            )
        except Exception as exc:
            print(f"[CHAT SERVICE] MMR retrieval failed, falling back to similarity search: {exc}")
            return vector_store.similarity_search(query, k=12, filter=search_filter)

    def _expand_to_parent_context(self, child_docs: list, parents: List[dict]) -> Tuple[List[str], list]:
        parent_lookup = {parent["parent_id"]: parent for parent in parents}
        selected_parents = []
        seen_parent_ids = set()

        for doc in child_docs:
            parent_id = doc.metadata.get("parent_id")
            if not parent_id or parent_id in seen_parent_ids:
                continue
            parent = parent_lookup.get(parent_id)
            if not parent:
                continue

            selected_parents.append(parent)
            seen_parent_ids.add(parent_id)
            if len(selected_parents) >= self.max_parent_sections:
                break

        selected_parents.sort(key=lambda item: item.get("start_time", 0.0))
        return self._parent_context_blocks(selected_parents)

    def _parent_context_blocks(self, parents: List[dict]) -> Tuple[List[str], list]:
        context_blocks = []
        citations = []
        used_chars = 0

        for parent in parents:
            text = parent.get("text", "")
            remaining_chars = self.max_context_chars - used_chars
            if remaining_chars <= 0:
                break

            trimmed_text = text[:remaining_chars]
            used_chars += len(trimmed_text)
            context_blocks.append(
                "[Section {section} | {start:.2f}s-{end:.2f}s]\n{text}".format(
                    section=parent.get("section_index", "?"),
                    start=float(parent.get("start_time", 0.0)),
                    end=float(parent.get("end_time", 0.0)),
                    text=trimmed_text,
                )
            )
            citations.append(
                {
                    "text_snippet": self._snippet(text),
                    "start_time": float(parent.get("start_time", 0.0)),
                    "end_time": float(parent.get("end_time", 0.0)),
                }
            )

        return context_blocks, citations

    def _legacy_child_context(self, child_docs: list) -> Tuple[List[str], list]:
        context_blocks = []
        citations = []
        used_chars = 0

        for doc in child_docs[:5]:
            text = doc.page_content
            remaining_chars = self.max_context_chars - used_chars
            if remaining_chars <= 0:
                break

            trimmed_text = text[:remaining_chars]
            used_chars += len(trimmed_text)
            start_time = float(doc.metadata.get("start_time", 0.0))
            end_time = float(doc.metadata.get("end_time", 0.0))
            context_blocks.append(
                f"[Transcript excerpt | {start_time:.2f}s-{end_time:.2f}s]\n{trimmed_text}"
            )
            citations.append(
                {
                    "text_snippet": self._snippet(text),
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

        return context_blocks, citations

    def _standalone_query(self, query: str, chat_history: list) -> str:
        if not chat_history:
            return query

        history_text = "\n".join(
            f"{msg.get('sender', 'user')}: {msg.get('content', '')}"
            for msg in chat_history[-8:]
        )
        prompt = (
            "Rewrite the latest user question as a standalone search query for a "
            "video transcript. Do not answer it. Return only the rewritten query.\n\n"
            f"Chat history:\n{history_text}\n\nLatest question: {query}"
        )
        response = self.llm.invoke(prompt)
        rewritten = response.content.strip()
        return rewritten or query

    def _format_history(self, chat_history: list) -> list:
        formatted_history = []
        for msg in chat_history:
            if msg.get("sender") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content", "")))
            else:
                formatted_history.append(AIMessage(content=msg.get("content", "")))
        return formatted_history

    def _is_plain_summary_request(self, query: str) -> bool:
        normalized = query.strip().lower()
        return normalized in {
            "summary",
            "summarize",
            "summarise",
            "summarize this video",
            "summarise this video",
            "what is discussed in the video?",
            "what is discussed in the video",
            "what is this video about?",
            "what is this video about",
        }

    def _is_plain_section_request(self, query: str) -> bool:
        normalized = query.strip().lower()
        return any(
            keyword in normalized
            for keyword in ["chapter", "chapters", "timeline", "timestamp", "notes"]
        )

    def _summary_context(self, video_summary: str, section_summaries: list) -> str:
        section_text = "\n\n".join(
            "[Section {section} | {start:.2f}s-{end:.2f}s]\n{summary}".format(
                section=section.get("section_index", "?"),
                start=float(section.get("start_time", 0.0)),
                end=float(section.get("end_time", 0.0)),
                summary=section.get("summary", ""),
            )
            for section in section_summaries
        )
        return f"Overall summary:\n{video_summary}\n\nSection summaries:\n{section_text}"

    def _invoke_grounded_summary(self, query: str, context: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You answer questions about a YouTube video using only the "
                    "stored video summary and section summaries below. Do not add "
                    "outside facts. If the summaries do not contain the answer, say "
                    "\"I don't know based on this video.\"\n\n{context}",
                ),
                ("human", "{input}"),
            ]
        )
        result = (prompt | self.llm).invoke({"context": context, "input": query})
        return result.content

    def _format_section_summaries(self, section_summaries: list) -> str:
        sections_text = "\n\n".join(
            "**Section {section} ({start:.2f}s-{end:.2f}s):**\n{summary}".format(
                section=section.get("section_index", "?"),
                start=float(section.get("start_time", 0.0)),
                end=float(section.get("end_time", 0.0)),
                summary=section.get("summary", ""),
            )
            for section in section_summaries
        )
        return f"Here are the chapter-wise notes for this video:\n\n{sections_text}"

    def _summary_citations(self, section_summaries: list) -> list:
        citations = []
        for section in section_summaries[:12]:
            summary = section.get("summary", "")
            citations.append(
                {
                    "text_snippet": self._snippet(summary),
                    "start_time": float(section.get("start_time", 0.0)),
                    "end_time": float(section.get("end_time", 0.0)),
                }
            )
        return citations

    def _snippet(self, text: str, length: int = 160) -> str:
        clean_text = " ".join((text or "").split())
        if len(clean_text) <= length:
            return clean_text
        return clean_text[:length].rstrip() + "..."
