import chromadb
from chromadb.config import Settings

class VectorStoreService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./documents/vector_db",
            settings=Settings(anonymized_telemetry=False)
        )
    
    def create(self):
        """Create and return vector store"""
        collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        return collection
