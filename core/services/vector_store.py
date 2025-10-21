import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from setting import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_PATH,
)


class VectorStoreService:
    def __init__(self, db_path: str = VECTOR_DB_PATH, embedding_model_path: str = EMBEDDING_MODEL_PATH):
        self.db_path = db_path
        self.embedding_model_path = embedding_model_path

    def setup_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create(self):
        embeddings = self.setup_embeddings()
        os.makedirs(self.db_path, exist_ok=True)
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=embeddings,
            collection_name="pdf_db"
        )


