from src.dto.state_dto import StateSchema

from src.llm_model import Llm
from src.pydantic_models import GuardrailModel
from langchain_core.output_parsers import JsonOutputParser
from src.prompt import prompts
from src.utils import convert_ai_response_to_valid_json
import json
from langfuse_config import langfuse_client

class GuardrailAgent:
    
    def __init__(self, user_question: str):
        self.user_question = user_question
        self.llm = Llm()
    
    async def run_agent(self):
        if self.user_question is None :
            return 
        
        prompt = langfuse_client.get_prompt("guardrail_prompt").compile(
            user_question=self.user_question,
            format_instruction=JsonOutputParser(pydantic_object=GuardrailModel).get_format_instructions()
        )
        response = await self.llm.invoke_llm(prompt)  
        response = response.content
        try:
            final_content = convert_ai_response_to_valid_json(response)
            data = json.loads(final_content.strip())
            model_output = GuardrailModel(**data)
            return {
                'is_safe_query': model_output.is_relavant_query,
                'rewritten_query': model_output.rewritten_query,
                'is_attempt_to_jailbreak': model_output.is_attempt_to_jailbreak,
                'reason': model_output.reason
            }
        except Exception as e:
            print(f"Failed to parse JSON output: {e}")
            return {
                'is_safe_query': False,
                'reason': "Failed to parse JSON output"
            }
        