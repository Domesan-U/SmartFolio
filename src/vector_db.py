from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone

class VectorDb:
    def __init__(self, persist_directory, collection_name, embedding_model):
        # NOTE: persist_directory is unused in Pinecone (it lives in the cloud),
        # but kept here to maintain compatibility with your existing calls.
        self.persist_directory = persist_directory 
        
        # We treat your 'collection_name' as the Pinecone 'index_name'
        self.index_name = collection_name 
        self.embedding_model = embedding_model
        self.vector_db = None
    
    def get_index_stats(self):
        pc = Pinecone(api_key="pcsk_6Q4q3y_6oF4H8hkQ32jSSWGFh1WfKHHs9esyaNnR2N1FVkBUiPdq4HFDbWgbfjTdQTFeAr")
        # pc.get_
        index = pc.Index(self.index_name)
        stats = index.describe_index_stats()
        return stats
    
    def initialize_vector_db_from_documents(self, chunks):
        # This uploads the documents to your Pinecone Cloud Index
        
        self.vector_db = PineconeVectorStore.from_documents(
            documents=chunks, 
            embedding=self.embedding_model, 
            index_name=self.index_name,
        )
        return self.vector_db
    
    def initialize_vector_db_from_existing_db(self):
        # This connects to the existing Pinecone Cloud Index
        self.vector_db = PineconeVectorStore(
            embedding=self.embedding_model, 
            index_name=self.index_name,
        )
        return self.vector_db
    
    def get_retriever(self, k=3):
        return self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.75, "k": k}
            )