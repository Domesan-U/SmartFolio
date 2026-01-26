from pydantic import BaseModel

from typing import List


class GuardrailModel(BaseModel):   
    is_attempt_to_jailbreak: bool
    reason: str


class RewriterModel(BaseModel):

    rewritten_query: str

class DomainFilterModel(BaseModel):
    is_question_porfolio_related: bool
    reason: str

