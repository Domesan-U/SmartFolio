from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, TypedDict
import json
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from src.llm_model import EmbeddingModel
from src.vector_db import VectorDb



def split_text(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80,
    )
    chunks = text_splitter.split_documents([doc])
    return chunks


def retriever(query):
    if os.path.exists(os.getenv("PORTFOLIO_DATA_PATH")):
        with open(os.getenv("PORTFOLIO_DATA_PATH"), 'r', encoding='utf-8') as json_file:
            doc= json.load(json_file)
            doc = Document(
                page_content=str(doc),
                metadata={}
            )
            chunks = split_text(doc)
            persist_dir = "my_personal_data"
            collection_name = "portfolio"
            if not os.path.exists(os.path.join(persist_dir, collection_name)):
                embedding_model = EmbeddingModel().embedding_model
                vector_db = VectorDb(persist_dir,collection_name,embedding_model)
                vector_db.initialize_vector_db_from_documents(chunks)
                ret = vector_db.get_retriever(k=5)
                return ret.invoke(query)
            else:
                embedding_model = EmbeddingModel().embedding_model
                vector_db = VectorDb(persist_dir,collection_name,embedding_model)
                vector_db.initialize_vector_db_from_existing_db()
                ret = vector_db.get_retriever(k=5)
                return ret.invoke(query)
    else:
        print("File not found")

