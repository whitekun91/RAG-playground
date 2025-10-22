from sentence_transformers import CrossEncoder

class RerankService:
    def __init__(self):
        self.model = CrossEncoder("core/models/embeddings/bge-reranker-v2-m3")
    
    def rerank(self, query, documents, top_k=5):
        """Rerank documents based on query relevance"""
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort by scores and return top_k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
