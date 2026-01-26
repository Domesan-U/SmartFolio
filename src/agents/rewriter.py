from src.dto.state_dto import StateSchema

from src.llm_model import Llm
from src.pydantic_models import RewriterModel
from langchain_core.output_parsers import JsonOutputParser
from src.prompt import prompts
from src.utils import convert_ai_response_to_valid_json
import json
from langfuse_config import langfuse_client
from typing import List

class RewriterAgent:
    
    def __init__(self, user_question: str, user_previous_questions: List[str]):
        self.user_question = user_question
        self.user_previous_questions = user_previous_questions
        self.llm = Llm()
    
    async def run_agent(self):
        if self.user_question is None :
            return 
        
        prompt = langfuse_client.get_prompt("rewriter_prompt").compile(
            user_question=self.user_question,
            user_previous_questions=self.user_previous_questions,
            format_instruction=JsonOutputParser(pydantic_object=RewriterModel).get_format_instructions()
        )
        response = await self.llm.invoke_llm(prompt)  
        response = response.content
        try:
            final_content = convert_ai_response_to_valid_json(response)
            data = json.loads(final_content.strip())
            model_output = RewriterModel(**data)
            return {
                'rewritten_query': model_output.rewritten_query
            }
        except Exception as e:
            print(f"Failed to parse JSON output In rewriter agent: {e}")
            return {
                'rewritten_query': self.user_question
            }
        