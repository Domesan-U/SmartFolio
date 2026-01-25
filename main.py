from langgraph.graph import StateGraph,START, END
import json
from rich import print
from src.agents.guardrail import GuardrailAgent
from src.dto.state_dto import StateSchema
from src.agents.retriever import retriever
from dotenv import load_dotenv
import random
from src.agents.generator import Generator
from src.utils import DEFAULT_RESPONSE, DATA_SHORTAGE_RESPONSE, JAILBREAK_ATTEMPT_RESPONSE, store_question, get_all_questions
from src.agents.demolisher import Demolisher
from src.dto.state_dto import ModelResponse
from src.utils import send_failure_mail


load_dotenv(override=True)
async def guardrail_agent(state: StateSchema):  
    guardrail_agent = GuardrailAgent(state.user_question)
    guardrail_response = await guardrail_agent.run_agent()
    print("Guarrailagent response ",guardrail_response)
    return {
        'user_question': guardrail_response['rewritten_query'],
        'is_attempt_to_jailbreak': guardrail_response['is_attempt_to_jailbreak'],
        'is_question_porfolio_related': guardrail_response['is_safe_query'],
        'reason': guardrail_response['reason']
    }

def check_if_question_is_related_to_portfolio(state: StateSchema):
    if state.is_question_porfolio_related:
        print("Question is related to portfolio")
        return "retriever_agent"
    else:
        print("Question is not related to portfolio")
        return "default_response"

def default_response(state: StateSchema):
    if state.is_attempt_to_jailbreak:
        return {
            'output': ModelResponse(text_content=random.choice(JAILBREAK_ATTEMPT_RESPONSE), has_ui_render_component="NONE")
        }
    return {
        'output': ModelResponse(text_content=random.choice(DEFAULT_RESPONSE), has_ui_render_component="NONE")
    }
    
def retriever_agent(state: StateSchema):
    ret = retriever(state.user_question)
    print("Retriever",ret)
    return {
        'retrieved_docs': ret
    }

async def demolisher_agent(state: StateSchema):
    stored_docs = [doc.page_content for doc in state.retrieved_docs]
    demolisher_agent = Demolisher(state.user_question, stored_docs)
    # demolisher_response = await demolisher_agent.run_agent()
    # print("Demolisher response ",demolisher_response)
    return {
        'retrieved_docs': state.retrieved_docs
    }

def check_if_retrieved_docs_are_empty(state: StateSchema):
    if len(state.retrieved_docs) == 0:
        print("Retrieved docs are empty")
        return "data_shortage_response"
    else:
        print("Retrieved docs are not empty")
        return "generator_agent"

def data_shortage_response(state: StateSchema):
    send_failure_mail("The following question was not able to generate a response due to data shortage: " + state.user_question)
    return {
        'output': ModelResponse(text_content=random.choice(DATA_SHORTAGE_RESPONSE), has_ui_render_component="NONE")
    } 

async def generator_agent(state: StateSchema):
    generator = Generator(state.user_question, state.retrieved_docs)
    store_question(state.user_question)
    generator_response = await generator.run_agent()
    print("Generator response ",generator_response)
    return {
        'output': generator_response
    }
    

    
graph=StateGraph(StateSchema)

graph.add_node("guardrail_agent",guardrail_agent)
graph.add_node("default_response",default_response)
graph.add_node("retriever_agent",retriever_agent)
graph.add_node("demolisher_agent",demolisher_agent)
graph.add_node("generator_agent",generator_agent)
graph.add_node("check_if_retrieved_docs_are_empty",check_if_retrieved_docs_are_empty)
graph.add_node("data_shortage_response",data_shortage_response)


graph.add_edge(START, "guardrail_agent")
graph.add_conditional_edges(
    "guardrail_agent",
    check_if_question_is_related_to_portfolio
)

graph.add_edge("retriever_agent", "demolisher_agent")
graph.add_conditional_edges(
    "demolisher_agent",
    check_if_retrieved_docs_are_empty
)


graph.add_edge("generator_agent", END)
graph.add_edge("default_response", END)
graph.add_edge("data_shortage_response", END)



app = graph.compile()

async def run_agent(user_question):
    model_resposne = await app.ainvoke({"user_question": user_question})
    return model_resposne

def get_questions_history():
    return {"data":get_all_questions()}
