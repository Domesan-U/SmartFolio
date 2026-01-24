from src.dto.state_dto import StateSchema

from src.llm_model import Llm
from src.pydantic_models import DemolisherModel
from langchain_core.output_parsers import JsonOutputParser
from src.prompt import prompts
from src.utils import convert_ai_response_to_valid_json
import json
from typing import List
from langchain_core.documents import Document

class Demolisher:
    def __init__(self, user_question: str, retrieved_docs: List[str]):
        self.user_question = user_question
        self.llm = Llm()
        self.retrieved_docs = retrieved_docs
    
    async def run_agent(self):
        if self.user_question is None :
            return 
        prompt = prompts['demolisher_prompt'].format(
            user_question=self.user_question,
            retrieved_docs=self.retrieved_docs,
            format_instruction=JsonOutputParser(pydantic_object=DemolisherModel).get_format_instructions()
        )
        response = await self.llm.invoke_llm(prompt)
        print("First response of demolisther ",response)
        response = response.content
        print("Resposne of demolisher model ",response)
        try:
            final_content = convert_ai_response_to_valid_json(response)
            data = json.loads(final_content.strip())
            model_output = DemolisherModel(**data)
            return {
                'relevant_docs': model_output.relevant_docs,
        }
        except Exception as e:
            print(f"Failed to parse JSON output: {e}")
            return {
                'relevant_docs': []
            }