from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import settings
from app.schemas.notes_schema import NoteSection
from app.utils.logger import logger

class NotesService:
    def __init__(self):
        # Removed FAISS & HuggingFace completely. 
        # Lower temperature for strict factual notes.
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            api_key=settings.GROQ_API_KEY,
            max_retries=2,
        )

    # Note the new parameters: section_summaries and video_summary
    async def generate_notes(self, video_id: str, note_type: str, section_summaries: list, video_summary: str) -> list:
        logger.info("Generating notes of type %s for video: %s", note_type, video_id)

        # 1. Build a clean, chronological context from our MongoDB summaries
        if not section_summaries:
            raise Exception("No summaries found in database. Please re-process the video.")

        context_blocks = [f"Overall Video Summary: {video_summary}\n"]
        for s in section_summaries:
            context_blocks.append(f"Chapter {s['section_index']} ({s['start_time']}s): {s['summary']}")
        
        context = "\n".join(context_blocks)

        # 2. Setup Parser and Prompt
        parser = JsonOutputParser(pydantic_object=NoteSection)

        prompt = PromptTemplate(
            template="""You are an expert AI tutor creating highly structured study notes from a video's chronological summaries.
            
            Based on the following chapter summaries, generate comprehensive notes formatted as a {note_type}. 
            Organize the information logically with clear, descriptive headings and concise bullet points.
            
            Context:
            {context}
            
            Format instructions:
            {format_instructions}
            
            Return ONLY a JSON array of sections. Do not hallucinate outside information.
            """,
            input_variables=["note_type", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # 3. Execute
        chain = prompt | self.llm | parser
        result = await chain.ainvoke({
            "note_type": note_type,
            "context": context
        })
        
        logger.info("Notes generated successfully for video: %s", video_id)
        
        if isinstance(result, dict):
            return [result]
        return result