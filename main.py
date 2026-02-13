from langgraph.graph import StateGraph,START, END
import json
import asyncio
from rich import print
from src.agents.guardrail import GuardrailAgent
from src.dto.state_dto import StateSchema
from src.agents.retriever import RetrieverAgent
from src.agents.rewriter import RewriterAgent
from dotenv import load_dotenv
import random
from rich import print
from src.agents.generator import Generator
from src.utils import DEFAULT_RESPONSE, DATA_SHORTAGE_RESPONSE, JAILBREAK_ATTEMPT_RESPONSE, store_question, get_all_questions
from src.dto.state_dto import ModelResponse
from src.agents.tool_caller import ToolCallerAgent
from src.agents.domain_filter import DomainSpecificFilter


load_dotenv(override=True)
async def guardrail_agent(state: StateSchema):  
    guardrail_agent = GuardrailAgent(state.user_question)
    guardrail_response = await guardrail_agent.run_agent()
    print("Guarrailagent response ",guardrail_response)
    return {
        'is_attempt_to_jailbreak': guardrail_response['is_attempt_to_jailbreak'],
        'reason': guardrail_response['reason']
    }

async def query_rewriter(state: StateSchema):
    rewriter_agent = RewriterAgent(
        state.user_question,
        state.user_previous_questions)
    rewriter_response = await rewriter_agent.run_agent()
    print("Rewriter response ",rewriter_response)
    return {
        'rewritten_query': rewriter_response['rewritten_query']
    }

    
def router_to_generator(state: StateSchema):
    if (state.is_attempt_to_jailbreak or 
        state.retrieved_docs is None or 
        state.retrieved_docs is [] or
        state.is_question_porfolio_related is False
        ):
        print("Question is related to portfolio")
        return "default_response"
    else:
        print("Question is not related to portfolio")
        return "generator_agent"

def default_response(state: StateSchema):
    if state.is_attempt_to_jailbreak:
        return {
            'output': ModelResponse(text_content=random.choice(JAILBREAK_ATTEMPT_RESPONSE), has_ui_render_component="NONE")
        }
    if not state.is_question_porfolio_related:
        return {
            'output': ModelResponse(text_content=random.choice(DEFAULT_RESPONSE), has_ui_render_component="NONE")
        }
    return {
        'output': ModelResponse(text_content=random.choice(DATA_SHORTAGE_RESPONSE), has_ui_render_component="NONE")
    }
    
async def retriever_agent(state: StateSchema):
    retriever_agent = RetrieverAgent(state.user_question)
    docs = await retriever_agent.run_agent()
    return {
        'retrieved_docs': docs
    }

async def domain_specific_filter(state: StateSchema):
    if(state.is_attempt_to_jailbreak):
        return {
            'retrieved_docs': []
        }
    domain_specfic_filter = DomainSpecificFilter(state.user_question, state.retrieved_docs)
    domain_specfic_filter_response = await domain_specfic_filter.run_agent()
    return {
        'is_question_porfolio_related': domain_specfic_filter_response['is_question_porfolio_related']
    }

async def tool_caller(state: StateSchema):
    # We use gemini for tool calling
    print("Tool calleer Invoked ")
    tool_caller_agent = ToolCallerAgent(state.user_question, state.retrieved_docs)
    asyncio.create_task(tool_caller_agent.run_agent())
    return state

async def generator_agent(state: StateSchema):
    if state.user_previous_questions is None or state.user_previous_questions == []:
        store_question(state.user_question)
        state.user_previous_questions = []
    generator = Generator(state.user_question, state.retrieved_docs, state.user_previous_questions)
    generator_response = await generator.run_agent()
    return {
        'output': generator_response
    }
    

    
graph=StateGraph(StateSchema)

# graph.add_node("guardrail_agent",guardrail_agent)
# graph.add_node("default_response",default_response)
# graph.add_node("query_rewriter",query_rewriter)
graph.add_node("retriever_agent",retriever_agent)
# graph.add_node("domain_specific_filter",domain_specific_filter)
graph.add_node("generator_agent",generator_agent)
graph.add_node("tool_caller",tool_caller)


# graph.add_edge(START, "guardrail_agent")
# graph.add_edge(START, "query_rewriter")
# graph.add_edge("query_rewriter", "retriever_agent")
# graph.add_edge("retriever_agent", "domain_specific_filter")
# graph.add_edge("guardrail_agent","domain_specific_filter")

# graph.add_conditional_edges(
#     "domain_specific_filter",
#     router_to_generator
# )

# graph.add_edge("generator_agent", END)
# graph.add_edge("default_response", END)

graph.add_edge(START, "retriever_agent")
graph.add_edge("retriever_agent", "generator_agent")
graph.add_edge("retriever_agent", "tool_caller")
graph.add_edge("generator_agent", END)
graph.add_edge("tool_caller", END)



app = graph.compile()

async def run_agent(user_question, user_previous_questions):
    if(user_previous_questions is not None and 
    user_previous_questions != [] and
    type(user_previous_questions) is list and
    len(user_previous_questions)>3):
        user_previous_questions = user_previous_questions[-3:]
    model_resposne = await app.ainvoke({
        "user_question": user_question,
        "user_previous_questions": user_previous_questions
        })
    return model_resposne

def get_questions_history():
    return {"data":get_all_questions()}
