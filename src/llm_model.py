from langgraph.prebuilt import create_react_agent
from rich import print
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings

class Llm:
    def __init__(self, output_model = None):
        self.llm = self.initialize_llm(output_model)
        
    def initialize_llm(self, output_model = None):
        llm = HuggingFaceEndpoint(
            repo_id= os.getenv("MODEL")
        )
        llm = ChatHuggingFace(llm = llm)
        
        if(output_model):
            return llm.with_structured_output(schema=output_model)
        
        return llm
    
    async def invoke_llm(self, prompt):
        return await self.llm.ainvoke(prompt)


class EmbeddingModel:
    def __init__(self):
        self.embedding_model = self.initialize_embedding_model()
    
    def initialize_embedding_model(self):
        import requests
        import os

        # Your specific model setup
        API_URL = "https://router.huggingface.co/hf-inference/models/all-mpnet-base-v2"
        headers = {"Authorization": f"Bearer {os.getenv('HF_KEY_1')}"}

        # Simple payload
        payload = {"inputs": "Test sentence for embedding."}

        response = requests.post(API_URL, headers=headers, json=payload)

        print(f"Status Code: {response.status_code}")
        print("--- Raw Response Content ---")
        print(response.text)  # <--- This will likely show HTML or a "Model Loading" error
        embedding_model = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.getenv("HF_KEY_1"),
            model="sentence-transformers/all-mpnet-base-v2",
            # api_url="https://router.huggingface.co/hf-inference/models/all-mpnet-base-v2/pipeline/feature-extraction"
        )
        return embedding_model
