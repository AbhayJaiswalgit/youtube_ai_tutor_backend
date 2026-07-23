# app/services/summary_service.py
import asyncio
import time

from langchain_groq import ChatGroq

from app.core.config import settings
from app.utils.logger import logger
from langchain_google_genai import ChatGoogleGenerativeAI


class SummaryService:
    def __init__(self):
        # self.llm = ChatGroq(
        #     # model="llama-3.1-8b-instant",
        #     model="meta-llama/llama-4-scout-17b-16e-instruct",
        #     temperature=0.2,  # Low temp for factual summaries
        #     api_key=settings.GROQ_API_KEY,
        #     max_retries=2,
        # )

        self.llm = ChatGoogleGenerativeAI(
                            model="gemini-3.1-flash-lite", 
                            temperature=0.2,
                            api_key=settings.GEMINI_API_KEY,
                            max_retries=2,
                        )
        # Keep this aligned with VectorStoreService.
        self.section_chunk_size = 200
        self.section_concurrency = 1
        self.token_budget_per_minute = 200000
        self._token_lock = asyncio.Lock()
        self._rate_window_started = time.monotonic()
        self._rate_window_tokens = 0

    async def generate_hierarchical_summary(self, transcript_data: list, video_id: str = "") -> dict:
        is_canonical_sections = bool(transcript_data and isinstance(transcript_data[0], dict) and transcript_data[0].get("section_id"))
        if is_canonical_sections:
            sections = transcript_data
            logger.info(
                "Generating hierarchical summaries from %d canonical sections for video: %s",
                len(sections),
                video_id,
            )
        else:
            logger.info("Processing %d transcript chunks for hierarchical summary for video: %s", len(transcript_data), video_id)
            sections = [
                transcript_data[i:i + self.section_chunk_size]
                for i in range(0, len(transcript_data), self.section_chunk_size)
            ]

        semaphore = asyncio.Semaphore(self.section_concurrency)
        section_summaries = await asyncio.gather(
            *[
                self._summarize_section(idx, section, semaphore, video_id)
                for idx, section in enumerate(sections)
            ]
        )

        logger.info("Generated %d section summaries for video: %s", len(section_summaries), video_id)

        all_summaries_text = "\n\n".join(
            [f"Section {s['section_index']}: {s['summary']}" for s in section_summaries]
        )
        final_prompt = (
            "Based only on these chronological section summaries, write a coherent "
            "summary of the entire video. Cover the major topics in order, preserve "
            "important terminology, and avoid outside information.\n\n"
            f"{all_summaries_text}"
        )
        await self._wait_for_token_budget(final_prompt)
        final_response = await self.llm.ainvoke(final_prompt)

        logger.info("Hierarchical summarization complete for video: %s", video_id)

        raw_content = final_response.content

        # Check if Gemini returned a list of blocks instead of a plain string
        if isinstance(raw_content, list):
            # Extract the text from the first dictionary in the list
            text_content = raw_content[0].get("text", "") if isinstance(raw_content[0], dict) else str(raw_content[0])
        else:
            # Fallback for when the model returns a plain string
            text_content = raw_content

        return {
            "section_summaries": section_summaries,
            "video_summary": text_content.strip()
        }

    async def _summarize_section(
        self,
        idx: int,
        section,
        semaphore: asyncio.Semaphore,
        video_id: str,
    ) -> dict:
        async with semaphore:
            if isinstance(section, dict):
                text = section.get("text", "")
                section_id = section.get("section_id") or f"{video_id}:section:{idx}"
                section_index = section.get("section_index", idx + 1)
                start_time = section.get("start_time", 0.0)
                end_time = section.get("end_time", start_time)
            else:
                text = " ".join([chunk["text"] for chunk in section])
                section_id = f"{video_id}:section:{idx}" if video_id else f"section:{idx}"
                section_index = idx + 1
                start_time = section[0]["start_time"]
                end_time = section[-1]["end_time"]

            prompt = (
                "Summarize this chronological section of a video transcript for later "
                "question answering. Keep the summary factual, concise, and focused "
                "(about 70-110 words). Include main concepts, important terminology, "
                "named entities, and conclusions that appear in this section. Do not "
                "add outside information.\n\n"
                f"Transcript section:\n{text}"
            )

            await self._wait_for_token_budget(prompt)
            response = await self.llm.ainvoke(prompt)

            raw_content = response.content

            # Check if Gemini returned a list of blocks instead of a plain string
            if isinstance(raw_content, list):
                # Extract the text from the first dictionary in the list
                text_content = raw_content[0].get("text", "") if isinstance(raw_content[0], dict) else str(raw_content[0])
            else:
                # Fallback for when the model returns a plain string
                text_content = raw_content

            return {
                "section_index": section_index,
                "section_id": section_id,
                "start_time": start_time,
                "end_time": end_time,
                "summary": text_content.strip()
            }

    async def _wait_for_token_budget(self, prompt: str) -> None:
        estimated_tokens = min(self._estimate_tokens(prompt), self.token_budget_per_minute)

        while True:
            async with self._token_lock:
                now = time.monotonic()
                elapsed = now - self._rate_window_started
                if elapsed >= 60:
                    self._rate_window_started = now
                    self._rate_window_tokens = 0
                    elapsed = 0

                if self._rate_window_tokens + estimated_tokens <= self.token_budget_per_minute:
                    self._rate_window_tokens += estimated_tokens
                    return

                print("===============sleeping===========")
                sleep_for = max(1, 60 - elapsed)

            await asyncio.sleep(sleep_for)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4 + 300)
