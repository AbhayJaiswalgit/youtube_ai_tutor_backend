import re
from enum import Enum
from typing import Optional

from langchain_groq import ChatGroq

from app.core.config import settings


class QueryIntent(str, Enum):
    SUMMARY = "SUMMARY"
    SECTION_SUMMARY = "SECTION_SUMMARY"
    FACTUAL_QA = "FACTUAL_QA"


class QueryIntentRouter:
    """Fast intent routing for video chat queries."""

    SUMMARY_PATTERNS = [
        r"\bwhat\s+(is|was)\s+(discussed|covered|talked about)\b",
        r"\bwhat'?s\s+this\s+video\s+about\b",
        r"\bsummar(y|ize|ise)\b",
        r"\boverview\b",
        r"\brecap\b",
        r"\bmain\s+(points|ideas|takeaways)\b",
        r"\bkey\s+(points|ideas|takeaways)\b",
        r"\bin\s+short\b",
    ]
    SECTION_PATTERNS = [
        r"\bchapter(s)?\b",
        r"\bsection(s)?\b",
        r"\btimeline\b",
        r"\btime\s*stamps?\b",
        r"\bdetailed\s+(notes|summary|breakdown)\b",
        r"\bstudy\s+notes\b",
        r"\bnotes\b",
        r"\bbreak\s*(it|this)?\s*down\b",
    ]

    def __init__(self):
        self.llm: Optional[ChatGroq] = None

    def classify(self, query: str) -> QueryIntent:
        heuristic_intent = self._classify_with_heuristics(query)
        if heuristic_intent:
            return heuristic_intent

        return self._classify_with_llm(query)

    def _classify_with_heuristics(self, query: str) -> Optional[QueryIntent]:
        normalized = query.strip().lower()
        if not normalized:
            return QueryIntent.FACTUAL_QA

        if any(re.search(pattern, normalized) for pattern in self.SECTION_PATTERNS):
            return QueryIntent.SECTION_SUMMARY

        if any(re.search(pattern, normalized) for pattern in self.SUMMARY_PATTERNS):
            return QueryIntent.SUMMARY

        if normalized.startswith(("who ", "what ", "when ", "where ", "why ", "how ")):
            if len(normalized.split()) <= 14:
                return QueryIntent.FACTUAL_QA

        return None

    def _classify_with_llm(self, query: str) -> QueryIntent:
        if self.llm is None:
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=settings.GROQ_API_KEY,
                max_retries=2,
            )

        prompt = (
            "Classify the user message into exactly one category:\n"
            "SUMMARY: asks for an overview, recap, main points, or what the whole video discusses.\n"
            "SECTION_SUMMARY: asks for chapters, notes, timestamps, timeline, or detailed breakdown.\n"
            "FACTUAL_QA: asks a specific factual question.\n\n"
            "Return ONLY SUMMARY, SECTION_SUMMARY, or FACTUAL_QA.\n\n"
            f"Message: {query}"
        )
        intent = self.llm.invoke(prompt).content.strip().upper()

        if "SECTION" in intent:
            return QueryIntent.SECTION_SUMMARY
        if "SUMMARY" in intent:
            return QueryIntent.SUMMARY
        return QueryIntent.FACTUAL_QA
