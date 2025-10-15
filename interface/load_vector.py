import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from setting import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_PATH,
)


def setup_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )


def setup_vector_store():
    embeddings = setup_embeddings()
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name="pdf_db"
    )
    return vector_store
