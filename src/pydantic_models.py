from pydantic import BaseModel
from typing import List

class GuardrailModel(BaseModel):
    is_relavant_query: bool
    reason: str

class DemolisherModel(BaseModel):
    relevant_docs: List[str]
