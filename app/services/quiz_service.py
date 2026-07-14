from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import settings
from app.schemas.quiz_schema import QuizQuestion
from app.utils.logger import logger

class QuizService:
    def __init__(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
            api_key=settings.GROQ_API_KEY,
            max_retries=2,
        )

    def generate_quiz(self, video_id: str, difficulty: str, count: int, section_summaries: list) -> list:
        logger.info("Generating %d %s quiz questions for video: %s", count, difficulty, video_id)

        if not section_summaries:
            raise Exception("No summaries found in database. Please re-process the video.")

        # Build chronological context
        context = "\n".join([f"Chapter {s['section_index']}: {s['summary']}" for s in section_summaries])

        parser = JsonOutputParser(pydantic_object=QuizQuestion)

        prompt = PromptTemplate(
            template="""You are an expert AI tutor creating a quiz based on chronological video summaries.
            
            Based on the following context, generate exactly {count} multiple-choice questions at a {difficulty} difficulty level.
            The wrong options (distractors) must be plausible but definitively incorrect based on the context.
            
            Context:
            {context}
            
            Format instructions:
            {format_instructions}
            
            Return ONLY a JSON array of the questions. Do not include markdown like ```json.
            """,
            input_variables=["count", "difficulty", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.llm | parser
        result = chain.invoke({
            "count": count,
            "difficulty": difficulty,
            "context": context
        })
        
        logger.info("Quiz generated successfully for video: %s", video_id)
        
        if isinstance(result, dict):
            return [result]
        return result