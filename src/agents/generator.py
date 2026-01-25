from src.dto.state_dto import StateSchema

from src.llm_model import Llm
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from src.dto.state_dto import ModelResponse
from src.utils import convert_ai_response_to_valid_json
import json
from langfuse_config import langfuse_client

class Generator:
    def __init__(self, user_question: str, retrieved_docs: List[Document]):
        self.user_question = user_question
        self.llm = Llm()
        self.retrieved_docs = retrieved_docs
    
    async def run_agent(self):
        if self.user_question is None :
            return 
        prompt = langfuse_client.get_prompt("generator_prompt").compile(
            user_question=self.user_question,
            retrieved_docs=self.retrieved_docs,
            format_instruction = JsonOutputParser(pydantic_object=ModelResponse).get_format_instructions()
        )
        
        response = await self.llm.invoke_llm(prompt)
        response = response.content
        print("The early generator response ",response)
        try:
            final_content = convert_ai_response_to_valid_json(response)
            data = json.loads(final_content.strip())
            model_output = ModelResponse(**data)
            return model_output
        except Exception as e:
            print(f"Failed to parse JSON output: {e}")
            return {
                'output': ModelResponse(text_content="Failed to parse JSON output", has_ui_render_component="NONE")
            }