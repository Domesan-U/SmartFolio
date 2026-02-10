from src.dto.state_dto import StateSchema

from src.llm_model import Llm
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from src.dto.state_dto import ModelResponse
from src.utils import convert_ai_response_to_valid_json, check_question_existence, write_answer_to_existing_question
import json
from langfuse_config import langfuse_client as langfuse

class Generator:
    def __init__(self, user_question: str, retrieved_docs: List[Document], past_conversation: List[str]):
        self.user_question = user_question
        self.past_conversation = past_conversation
        self.llm = Llm()
        self.retrieved_docs = retrieved_docs
    
    async def run_agent(self):
        if self.user_question is None:
            return

        with langfuse.start_as_current_observation(
            as_type="span",
            name="process-request",
            input=self.user_question
        ) as span:

            faq_question = check_question_existence(self.user_question)
            faq_answer = None

            # -----------------
            # FAQ PATH
            # -----------------
            if faq_question != {}:
                if faq_question.get("answer", "") != "":
                    faq_answer = faq_question.get("answer", "")

            if faq_answer is not None:
                # trace FAQ answer
                observability_answer = faq_answer
                observability_answer['is_answer_from_faq'] = True
                span.update(output=observability_answer)
                return faq_answer

            # -----------------
            # GENERATION PATH
            # -----------------
            prompt = langfuse.get_prompt("generator_prompt").compile(
                user_question=self.user_question,
                past_conversation=self.past_conversation,
                retrieved_docs=self.retrieved_docs,
                format_instruction=JsonOutputParser(
                    pydantic_object=ModelResponse
                ).get_format_instructions()
            )

            response = await self.llm.invoke_llm(prompt)
            response = response.content

            try:
                final_content = convert_ai_response_to_valid_json(response)
                data = json.loads(final_content.strip())
                model_output = ModelResponse(**data)

                write_answer_to_existing_question(
                    self.user_question,
                    model_output
                )

                # trace generated answer
                span.update(output=model_output.model_dump())

                return model_output

            except Exception as e:
                error_output = ModelResponse(
                    text_content="Failed to parse JSON output",
                    has_ui_render_component="NONE"
                )

                # trace error output
                span.update(
                    output=error_output.model_dump(),
                    metadata={"error": str(e)}
                )

                return error_output
