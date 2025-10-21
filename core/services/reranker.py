import gc
import torch
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from setting import RERANKER_MODEL_PATH


class RerankService:
    def __init__(self, model_path: str = RERANKER_MODEL_PATH):
        self.model_path = model_path

    def rerank(self, retriever, question: str):
        model_reranker = HuggingFaceCrossEncoder(model_name=self.model_path)
        compressor = CrossEncoderReranker(model=model_reranker, top_n=4)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        try:
            return compression_retriever.invoke(question)
        finally:
            del model_reranker, compressor, compression_retriever
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


