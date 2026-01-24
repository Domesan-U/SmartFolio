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
        chunk_size=500,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents([doc])
    return chunks


def retriever(query):
    # 1. Setup Configuration
    index_name = os.getenv("PINECONE_INDEX_NAME")  # Your Pinecone Index Name
    data_path = os.getenv("PORTFOLIO_DATA_PATH")
    
    # Initialize your helper classes
    embedding_model = EmbeddingModel().embedding_model
    vector_db_wrapper = VectorDb(None, index_name, embedding_model)

    # 2. CHECK: Is the Pinecone Index empty?
    # We use the native client to check stats quickly
    stats = vector_db_wrapper.get_index_stats()
    
    # 3. LOGIC: Upload if empty, otherwise connect
    if stats.total_vector_count == 0:
        print(f"Index '{index_name}' is empty (0 vectors). Starting upload...")
        
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as json_file:
                doc_data = json.load(json_file)
                
                # Create Document & Split
                doc = Document(
                    page_content=str(doc_data),
                    metadata={"source": "portfolio"}
                )
                chunks = split_text(doc) # Assuming split_text is defined elsewhere
                
                # Upload using your wrapper
                vector_db_wrapper.initialize_vector_db_from_documents(chunks)
                print("Upload complete.")
        else:
            print("Error: Index is empty but data file was not found.")
            return []
            
    else:
        print(f"Index '{index_name}' has {stats.total_vector_count} vectors. Connecting...")
        # Just connect to existing data
        vector_db_wrapper.initialize_vector_db_from_existing_db()

    # 4. RETRIEVE: Get relevant docs
    ret = vector_db_wrapper.get_retriever(k=5)
    return ret.invoke(query)