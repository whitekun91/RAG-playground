import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

class VectorStoreService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./documents/vector_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def create(self):
        """Create and return vector store"""
        # Use LangChain ChromaDB wrapper without embedding function
        vector_store = Chroma(
            client=self.client,
            collection_name="rag_documents",
            collection_metadata={"hnsw:space": "cosine"},
            embedding_function=None  # Disable embedding function
        )
        return vector_store
