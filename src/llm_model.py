from langgraph.prebuilt import create_react_agent
from rich import print
from langchain_groq import ChatGroq
import os
from langchain_community.embeddings import JinaEmbeddings
from src.utils import send_mail_tool
from src.utils import send_mail
class Llm:
    def __init__(self, output_model = None, bind_tool = False, api_key=None):
        self.bind_tool = bind_tool
        self.api_key = api_key
        self.llm = self.initialize_llm(output_model)
        
    def initialize_llm(self, output_model = None):
        llm = ChatGroq(
            model_name="qwen/qwen3-32b",
            groq_api_key = self.api_key if self.api_key else os.getenv("GROQ_API_KEY"),
            temperature=0.7,
            reasoning_format = 'hidden'
        )
        if(self.bind_tool):
            llm = llm.bind_tools([send_mail_tool])
        # llm = ChatHuggingFace(llm = llm)
        
        if(output_model):
            return llm.with_structured_output(schema=output_model)
        
        return llm
    
    async def invoke_llm(self, prompt):
        res = await self.llm.ainvoke(prompt)
        if(len(res.tool_calls) > 0):
            try:
                send_mail(**res.tool_calls[0]['args'])
            except Exception as e:
                print(f"Failed to send email: {e}")
        return res


class EmbeddingModel:
    def __init__(self):
        self.embedding_model = self.initialize_embedding_model()
    
    def initialize_embedding_model(self):
        embedding_model = JinaEmbeddings(
            jina_api_key=os.getenv("JINA_API_KEY"), 
            model_name="jina-embeddings-v2-base-en"
        )
        # embedding_model = HuggingFaceEndpointEmbeddings(
        #     huggingfacehub_api_token=os.getenv("HF_KEY_1"),
        #     model="sentence-transformers/all-mpnet-base-v2",
        #     # api_url="https://router.huggingface.co/hf-inference/models/all-mpnet-base-v2/pipeline/feature-extraction"
        # )
        return embedding_model
