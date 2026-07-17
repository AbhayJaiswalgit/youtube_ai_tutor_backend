import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.core.config import settings
from app.utils.logger import logger

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - runtime guard for minimal environments
    ChatGroq = None


class QueryIntent(str, Enum):
    GLOBAL = "GLOBAL"
    RAG_QUERY = "RAG_QUERY"


@dataclass
class QueryIntentResult:
    intent: QueryIntent
    is_temporal: bool
    start_time_seconds: Optional[float]
    end_time_seconds: Optional[float]
    clean_query: str

    def as_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "is_temporal": self.is_temporal,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "clean_query": self.clean_query,
        }


class IntentClassification(BaseModel):
    intent: Literal["GLOBAL", "RAG_QUERY"]
    is_temporal: bool = Field(default=False)
    start_time_seconds: Optional[float] = Field(default=None)
    end_time_seconds: Optional[float] = Field(default=None)
    clean_query: str = Field(default="")


class QueryIntentRouter:
    """Classifies chat queries and extracts temporal bounds."""

    def __init__(self):
        self.llm: Optional[Any] =None

    async def classify(self, query: str) -> QueryIntentResult:
        normalized = " ".join((query or "").strip().split())
        if not normalized:
            return QueryIntentResult(QueryIntent.RAG_QUERY, False, None, None, "")

        try:
            result = await self._classify_with_llm(normalized)
        except Exception:
            logger.exception("LLM query intent extraction failed; falling back to a safe result")
            result = QueryIntentResult(QueryIntent.RAG_QUERY, False, None, None, normalized)

        logger.debug("Query intent: %s", result.as_dict())
        return result

    async def _classify_with_llm(self, query: str) -> QueryIntentResult:
        if self.llm is None:
            if ChatGroq is None:
                raise RuntimeError("langchain_groq is not available")
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=settings.GROQ_API_KEY,
                max_retries=2,
            ).with_structured_output(IntentClassification)

        prompt = self._build_prompt(query)
        response = await self.llm.ainvoke(prompt)
        payload = self._coerce_payload(response, fallback_query=query)
        return self._sanitize_result(payload, fallback_query=query)

    def _build_prompt(self, query: str) -> str:
        return f"""You are a precise intent router for a YouTube transcript assistant.

Classify the user's message into one of two intents:
- GLOBAL: whole-video summaries, recaps, overviews, notes, chapters, timelines, key points, main ideas, study notes, or section summaries.
- RAG_QUERY: specific factual questions, definitions, explanations, comparisons, or questions about a specific concept.

Extract any temporal references into start_time_seconds and end_time_seconds. Remove only the temporal phrases from clean_query while preserving the semantic meaning.

Examples:
- Input: 'summarize the main ideas between 5 and 10 minutes'
  Output: {{"intent": "GLOBAL", "is_temporal": true, "start_time_seconds": 300, "end_time_seconds": 600, "clean_query": "summarize the main ideas"}}
- Input: 'what is the definition of entropy'
  Output: {{"intent": "RAG_QUERY", "is_temporal": false, "start_time_seconds": null, "end_time_seconds": null, "clean_query": "what is the definition of entropy"}}
- Input: 'explain the concept at 10:35'
  Output: {{"intent": "RAG_QUERY", "is_temporal": true, "start_time_seconds": 635, "end_time_seconds": 695, "clean_query": "explain the concept"}}
- Input: 'give me the first 10 minutes'
  Output: {{"intent": "GLOBAL", "is_temporal": true, "start_time_seconds": 0, "end_time_seconds": 600, "clean_query": "give me"}}
- Input: 'what happened near the end'
  Output: {{"intent": "RAG_QUERY", "is_temporal": true, "start_time_seconds": null, "end_time_seconds": null, "clean_query": "what happened"}}

Return a valid structured object only. Do not add commentary.

User message: {query}"""

    def _coerce_payload(self, response: Any, fallback_query: str) -> dict:
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"intent": "RAG_QUERY", "is_temporal": False, "clean_query": fallback_query}
            if isinstance(content, dict):
                return content
        raise ValueError("Unsupported LLM response payload")

    def _sanitize_result(self, payload: dict, fallback_query: str) -> QueryIntentResult:
        intent_value = str(payload.get("intent") or "RAG_QUERY").upper()
        intent = QueryIntent.GLOBAL if intent_value == QueryIntent.GLOBAL.value else QueryIntent.RAG_QUERY

        start_time = self._coerce_float(payload.get("start_time_seconds"))
        end_time = self._coerce_float(payload.get("end_time_seconds"))
        if start_time is not None and end_time is not None and start_time > end_time:
            start_time, end_time = end_time, start_time

        clean_query = str(payload.get("clean_query") or "").strip()
        if not clean_query:
            clean_query = fallback_query
        is_temporal = bool(payload.get("is_temporal")) or start_time is not None or end_time is not None

        return QueryIntentResult(
            intent=intent,
            is_temporal=is_temporal,
            start_time_seconds=start_time,
            end_time_seconds=end_time,
            clean_query=clean_query or fallback_query,
        )

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None



# import json
# import re
# from dataclasses import dataclass
# from enum import Enum
# from typing import Optional

# from langchain_groq import ChatGroq

# from app.core.config import settings
# from app.utils.logger import logger


# class QueryIntent(str, Enum):
#     GLOBAL = "GLOBAL"
#     RAG_QUERY = "RAG_QUERY"


# @dataclass
# class QueryIntentResult:
#     intent: QueryIntent
#     is_temporal: bool
#     start_time_seconds: Optional[float]
#     end_time_seconds: Optional[float]
#     clean_query: str

#     def as_dict(self) -> dict:
#         return {
#             "intent": self.intent.value,
#             "is_temporal": self.is_temporal,
#             "start_time_seconds": self.start_time_seconds,
#             "end_time_seconds": self.end_time_seconds,
#             "clean_query": self.clean_query,
#         }


# class QueryIntentRouter:
#     """Classifies chat queries and extracts temporal bounds."""

#     GLOBAL_PATTERNS = [
#         r"\bwhat\s+(is|was)\s+(discussed|covered|talked about)\b",
#         r"\bwhat'?s\s+this\s+video\s+about\b",
#         r"\bsummar(y|ize|ise)\b",
#         r"\boverview\b",
#         r"\brecap\b",
#         r"\bmain\s+(points|ideas|takeaways)\b",
#         r"\bkey\s+(points|ideas|takeaways)\b",
#         r"\bin\s+short\b",
#         r"\bchapter(s)?\b",
#         r"\btimeline\b",
#         r"\btime\s*stamps?\b",
#         r"\bdetailed\s+(notes|summary|breakdown)\b",
#         r"\bstudy\s+notes\b",
#         r"\bnotes\b",
#         r"\bbreak\s*(it|this)?\s*down\b",
#         r"\bwhat\s+happened\b",
#     ]

#     PURE_TEMPORAL_WORDS = {
#         "summarize",
#         "summarise",
#         "summary",
#         "explain",
#         "describe",
#         "tell",
#         "me",
#         "what",
#         "happened",
#         "occurs",
#         "occurred",
#         "between",
#         "during",
#         "there",
#         "video",
#         "part",
#         "segment",
#         "section",
#         "range",
#         "this",
#         "that",
#         "the",
#         "in",
#         "of",
#         "about",
#     }

#     def __init__(self):
#         self.llm: Optional[ChatGroq] = None

#     def classify(self, query: str) -> QueryIntentResult:
#         heuristic_result = self._classify_with_heuristics(query)
#         if heuristic_result:
#             logger.debug("Heuristic query intent: %s", heuristic_result.as_dict())
#             return heuristic_result

#         result = self._classify_with_llm(query)
#         logger.debug("LLM query intent: %s", result.as_dict())
#         return result

#     def _classify_with_heuristics(self, query: str) -> Optional[QueryIntentResult]:
#         normalized = " ".join((query or "").strip().split())
#         if not normalized:
#             return QueryIntentResult(QueryIntent.RAG_QUERY, False, None, None, "")

#         start_time, end_time = self._extract_temporal_bounds(normalized)
#         is_temporal = start_time is not None or end_time is not None
#         clean_query = self._clean_temporal_phrases(normalized) if is_temporal else normalized
#         intent = self._determine_intent(clean_query, normalized, is_temporal)

#         if is_temporal or intent == QueryIntent.GLOBAL:
#             return QueryIntentResult(
#                 intent=intent,
#                 is_temporal=is_temporal,
#                 start_time_seconds=start_time,
#                 end_time_seconds=end_time,
#                 clean_query=clean_query or normalized,
#             )

#         if normalized.lower().startswith(("who ", "what ", "when ", "where ", "why ", "how ")):
#             if len(normalized.split()) <= 14:
#                 return QueryIntentResult(QueryIntent.RAG_QUERY, False, None, None, normalized)

#         return None

#     def _classify_with_llm(self, query: str) -> QueryIntentResult:
#         if self.llm is None:
#             self.llm = ChatGroq(
#                 model="llama-3.1-8b-instant",
#                 temperature=0,
#                 api_key=settings.GROQ_API_KEY,
#                 max_retries=2,
#             )

#         prompt = (
#             "Classify the user message for a YouTube transcript chat system. "
#             "Return one strict JSON object only, with this exact schema:\n"
#             '{"intent":"GLOBAL | RAG_QUERY","is_temporal":boolean,'
#             '"start_time_seconds":float_or_null,"end_time_seconds":float_or_null,'
#             '"clean_query":"string"}\n\n'
#             "Rules:\n"
#             "- GLOBAL means whole-video summaries, broad notes, chapters, timelines, or structural summaries of a time range.\n"
#             "- RAG_QUERY means a specific semantic/factual question.\n"
#             "- is_temporal is true when the user references a timestamp or time range.\n"
#             "- For 'at N minutes', use a 60 second window centered on that mark.\n"
#             "- For 'last N minutes', set start_time_seconds to -N seconds and end_time_seconds to null.\n"
#             "- clean_query removes timing words but keeps the user's semantic concept.\n\n"
#             f"Message: {query}"
#         )

#         try:
#             raw = self.llm.invoke(prompt).content.strip()
#             payload = self._extract_json(raw)
#             intent_value = str(payload.get("intent", "RAG_QUERY")).upper()
#             intent = QueryIntent.GLOBAL if intent_value == QueryIntent.GLOBAL.value else QueryIntent.RAG_QUERY
#             start_time = self._nullable_float(payload.get("start_time_seconds"))
#             end_time = self._nullable_float(payload.get("end_time_seconds"))
#             clean_query = str(payload.get("clean_query") or query).strip()
#             is_temporal = bool(payload.get("is_temporal")) or start_time is not None or end_time is not None
#             return QueryIntentResult(intent, is_temporal, start_time, end_time, clean_query or query)
#         except Exception:
#             logger.exception("LLM query intent extraction failed; falling back to RAG_QUERY")
#             return QueryIntentResult(QueryIntent.RAG_QUERY, False, None, None, query)

#     def _extract_temporal_bounds(self, query: str) -> tuple[Optional[float], Optional[float]]:
#         lower = query.lower()
#         time_pattern = r"(\d{1,2}(?::\d{2}){1,2}|\d+(?:\.\d+)?\s*-?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?))"

#         range_match = re.search(rf"\b(?:between|from)\s+{time_pattern}\s+(?:and|to|-)\s+{time_pattern}", lower)
#         if range_match:
#             start = self._parse_time_token(range_match.group(1))
#             end = self._parse_time_token(range_match.group(2))
#             if start is not None and end is not None:
#                 return (min(start, end), max(start, end))

#         first_match = re.search(r"\b(?:first|initial|opening)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)", lower)
#         if first_match:
#             end = self._duration_seconds(first_match.group(1), first_match.group(2))
#             return (0.0, end)

#         last_match = re.search(r"\blast\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)", lower)
#         if last_match:
#             duration = self._duration_seconds(last_match.group(1), last_match.group(2))
#             return (-duration, None)

#         around_match = re.search(rf"\b(?:at|around|near|by)\s+(?:the\s+)?{time_pattern}(?:\s+mark)?", lower)
#         if around_match:
#             point = self._parse_time_token(around_match.group(1))
#             if point is not None:
#                 return (max(0.0, point - 30.0), point + 30.0)

#         bare_timestamp = re.search(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b", lower)
#         if bare_timestamp:
#             point = self._parse_time_token(bare_timestamp.group(1))
#             if point is not None:
#                 return (max(0.0, point - 30.0), point + 30.0)

#         minute_mark = re.search(r"\b(\d+(?:\.\d+)?)\s*-?\s*(?:minute|min)\s+mark\b", lower)
#         if minute_mark:
#             point = float(minute_mark.group(1)) * 60.0
#             return (max(0.0, point - 30.0), point + 30.0)

#         return (None, None)

#     def _parse_time_token(self, token: str) -> Optional[float]:
#         token = (token or "").strip().lower()
#         if not token:
#             return None

#         if ":" in token:
#             parts = [float(part) for part in token.split(":")]
#             if len(parts) == 2:
#                 return parts[0] * 60 + parts[1]
#             if len(parts) == 3:
#                 return parts[0] * 3600 + parts[1] * 60 + parts[2]
#             return None

#         match = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?)", token)
#         if not match:
#             return None
#         return self._duration_seconds(match.group(1), match.group(2))

#     def _duration_seconds(self, value: str, unit: str) -> float:
#         amount = float(value)
#         normalized_unit = unit.lower()
#         if normalized_unit.startswith(("hour", "hr")):
#             return amount * 3600.0
#         if normalized_unit.startswith(("minute", "min")):
#             return amount * 60.0
#         return amount

#     def _clean_temporal_phrases(self, query: str) -> str:
#         cleaned = query
#         patterns = [
#             r"\b(?:between|from)\s+\d{1,2}(?::\d{2}){1,2}\s+(?:and|to|-)\s+\d{1,2}(?::\d{2}){1,2}",
#             r"\b(?:between|from)\s+\d+(?:\.\d+)?\s*-?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\s+(?:and|to|-)\s+\d+(?:\.\d+)?\s*-?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)",
#             r"\b(?:first|initial|opening|last)\s+\d+(?:\.\d+)?\s*-?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)",
#             r"\b(?:at|around|near|by)\s+(?:the\s+)?\d{1,2}(?::\d{2}){1,2}(?:\s+mark)?",
#             r"\b(?:at|around|near|by)\s+(?:the\s+)?\d+(?:\.\d+)?\s*-?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)(?:\s+mark)?",
#             r"\b\d+(?:\.\d+)?\s*-?\s*(?:minute|min)\s+mark\b",
#         ]
#         for pattern in patterns:
#             cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

#         cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?")
#         while True:
#             next_cleaned = re.sub(
#                 r"\b(?:in|during|for|of|at|around|near|by|the)\s*$",
#                 "",
#                 cleaned,
#                 flags=re.IGNORECASE,
#             ).strip(" ,.?")
#             if next_cleaned == cleaned:
#                 break
#             cleaned = next_cleaned
#         return cleaned

#     def _determine_intent(self, clean_query: str, original_query: str, is_temporal: bool) -> QueryIntent:
#         lower_clean = (clean_query or "").lower()
#         lower_original = (original_query or "").lower()

#         if any(re.search(pattern, lower_clean) for pattern in self.GLOBAL_PATTERNS):
#             return QueryIntent.GLOBAL
#         if any(re.search(pattern, lower_original) for pattern in self.GLOBAL_PATTERNS) and not self._has_semantic_focus(lower_clean):
#             return QueryIntent.GLOBAL
#         if is_temporal and not self._has_semantic_focus(lower_clean):
#             return QueryIntent.GLOBAL
#         return QueryIntent.RAG_QUERY

#     def _has_semantic_focus(self, clean_query: str) -> bool:
#         words = re.findall(r"[a-zA-Z0-9_+#.-]+", clean_query.lower())
#         meaningful_words = [word for word in words if word not in self.PURE_TEMPORAL_WORDS]
#         return len(meaningful_words) >= 2

#     def _extract_json(self, raw: str) -> dict:
#         match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
#         if not match:
#             raise ValueError("No JSON object found in classifier output")
#         return json.loads(match.group(0))

#     def _nullable_float(self, value) -> Optional[float]:
#         if value is None:
#             return None
#         try:
#             return float(value)
#         except (TypeError, ValueError):
#             return None
