from src.dto.state_dto import StateSchema

from src.llm_model import Llm
from src.pydantic_models import DomainFilterModel
from langchain_core.output_parsers import JsonOutputParser
from src.prompt import prompts
from src.utils import convert_ai_response_to_valid_json
import json
from typing import List
from langchain_core.documents import Document
from langfuse import Langfuse
from langfuse_config import langfuse_client

class DomainSpecificFilter:
    def __init__(self, user_question: str, retrieved_docs: List[str]):
        self.user_question = user_question
        self.llm = Llm()
        self.retrieved_docs = retrieved_docs
    
    async def run_agent(self):
        if self.user_question is None :
            return 
        prompt = langfuse_client.get_prompt("domain_specific_filter_prompt").compile(
            user_question=self.user_question,
            retrieved_docs=self.retrieved_docs,
            format_instruction=JsonOutputParser(pydantic_object=DomainFilterModel).get_format_instructions()
        )
        response = await self.llm.invoke_llm(prompt)
        # from rich import print
        # print("Query for domain filter ",prompt)
        # print("First response of domain specific filter ",response)
        response = response.content
        try:
            final_content = convert_ai_response_to_valid_json(response)
            data = json.loads(final_content.strip())
            model_output = DomainFilterModel(**data)
            return {
                'is_question_porfolio_related': model_output.is_question_porfolio_related,
        }
        except Exception as e:
            print(f"Failed to parse JSON output in Domain specific filter: {e}")
            return {
                'is_question_porfolio_related': True
            }