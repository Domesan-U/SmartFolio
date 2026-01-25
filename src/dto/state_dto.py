from pydantic import BaseModel
from typing import Literal, Optional, List, Union
from enum import Enum

class TimelineEvent(BaseModel):
    year: str
    title: str
    desc: Optional[str] = ""

class TimelineData(BaseModel):
    events: List[TimelineEvent]

class SkillData(BaseModel):
    skills: List[str]

class ProjectData(BaseModel):
    name: str
    description: str
    url: Optional[str] = None

class ModelResponse(BaseModel):
    text_content: str
    has_ui_render_component: Literal["TIMELINE", "SKILLCARD", "PROJECTCARD", "NONE"]
    ui_component: Optional[Union[TimelineData, SkillData, List[ProjectData]]] = None

class StateSchema(BaseModel):
    user_question: str
    is_question_porfolio_related: Optional[bool] = None
    is_attempt_to_jailbreak: Optional[bool] = None
    retrieved_docs: list = []
    output: Optional[ModelResponse] = None

