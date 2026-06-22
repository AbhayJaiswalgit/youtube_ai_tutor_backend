# app/services/summary_service.py
from langchain_groq import ChatGroq
from app.core.config import settings
import asyncio

class SummaryService:
    def __init__(self):
        self.llm = ChatGroq(
            #model="llama-3.1-8b-instant",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2, # Low temp for factual summaries
            api_key=settings.GROQ_API_KEY,
            max_retries=2,
        )

    async def generate_hierarchical_summary(self, transcript_data: list) -> dict:
        print(f"📄 [SUMMARY SERVICE] Processing {len(transcript_data)} chunks hierarchically...")
        
        # 1. Group chunks into sections (15 chunks each)
        chunk_size = 15
        sections = [transcript_data[i:i + chunk_size] for i in range(0, len(transcript_data), chunk_size)]
        
        section_summaries = []
        
        # 2. Generate Section Summaries
        for idx, section in enumerate(sections):
            text = " ".join([chunk["text"] for chunk in section])
            prompt = (
                "Summarize this chronological section of a video transcript for later "
                "question answering. Keep the summary factual, concise, and specific. "
                "Include the main concepts, examples, named entities, and conclusions "
                "that appear in this section. Do not add outside information.\n\n"
                f"Transcript section:\n{text}"
            )
            
            # Use ainvoke for async background processing
            response = await self.llm.ainvoke(prompt)
            
            section_summaries.append({
                "section_index": idx + 1,
                "start_time": section[0]["start_time"],
                "end_time": section[-1]["end_time"],
                "summary": response.content.strip()
            })
            
        print(f"📄 [SUMMARY SERVICE] Generated {len(section_summaries)} section summaries.")

        # 3. Generate Final Video Summary
        all_summaries_text = "\n\n".join([f"Section {s['section_index']}: {s['summary']}" for s in section_summaries])
        final_prompt = (
            "Based only on these chronological section summaries, write a coherent "
            "summary of the entire video. Cover the major topics in order, preserve "
            "important terminology, and avoid outside information.\n\n"
            f"{all_summaries_text}"
        )
        final_response = await self.llm.ainvoke(final_prompt)
        
        print("✅ [SUMMARY SERVICE] Hierarchical summarization complete.")
        
        return {
            "section_summaries": section_summaries,
            "video_summary": final_response.content.strip()
        }
