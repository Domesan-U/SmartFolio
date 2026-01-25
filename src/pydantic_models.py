from pydantic import BaseModel
from typing import List

class GuardrailModel(BaseModel):
    is_relavant_query: bool
    rewritten_query: str
    is_attempt_to_jailbreak: bool
    reason: str

class DemolisherModel(BaseModel):
    relevant_docs: List[str]
