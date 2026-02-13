from src.llm_model import Llm
from langfuse_config import langfuse_client as langfuse
from typing import List
from langchain_core.documents import Document
import os

class ToolCallerAgent:
    def __init__(self, user_question: str, retrieved_docs: List[Document]):
      self.user_question = user_question
      self.retrieved_docs = retrieved_docs
    
    async def run_agent(self):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="tool_caller",
            input=self.user_question
        ) as span:
            llm = Llm(bind_tool=True, api_key=os.getenv("GROQ_API_KEY_2"))
            prompt = langfuse.get_prompt("tool_caller_prompt").compile(
                user_question=self.user_question,
            retrieved_docs=self.retrieved_docs,
            )
            try:
                res = await llm.invoke_llm(prompt)
                span.update(output=res)
                return res
            except Exception as e:
                span.update(output=f"Error: {e}")
                return {"error": str(e)}