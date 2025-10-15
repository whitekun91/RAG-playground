from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from setting import RERANKER_MODEL_PATH
import gc
import torch


def get_reranked_documents(retriever, question):
    model_reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_PATH)
    compressor = CrossEncoderReranker(model=model_reranker, top_n=4)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    docs = compression_retriever.invoke(question)
    del model_reranker, compressor, compression_retriever
    gc.collect()
    torch.cuda.empty_cache()
    return docs
