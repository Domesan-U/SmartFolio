# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from main import run_agent, get_questions_history
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

class QueryRequest(BaseModel):
    user_question: str
    user_previous_questions: List[str]

@app.post("/ask")
async def ask_agent(payload: QueryRequest):
    try:
        result = await run_agent(payload.user_question, payload.user_previous_questions)
        return result
    except Exception as e:
        return {
            "error": "Oops! Something went wrong. Try again or come back later "
        }

@app.post("/get_questions_history")
def get_questions():
    return get_questions_history()