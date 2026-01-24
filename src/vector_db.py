from langchain_community.vectorstores import Chroma

class VectorDb:
    def __init__(self,persist_directory,collection_name,embedding_model):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_db = None
        
    
    def initialize_vector_db_from_documents(self, chunks):
        db = Chroma.from_documents(
                embedding=self.embedding_model, 
                documents=chunks, 
                persist_directory=self.persist_directory, 
                collection_name=self.collection_name
        )
        self.vector_db = db
        return db
    
    def initialize_vector_db_from_existing_db(self):
        db = Chroma(
            embedding=self.embedding_model, 
            persist_directory=self.persist_directory, 
            collection_name=self.collection_name
        )
        self.vector_db = db
        return db
    
    def get_retriever(self,k=3):
        return self.vector_db.as_retriever(search_kwargs={"k": k})