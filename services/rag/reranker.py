from sentence_transformers import CrossEncoder

class RerankService:
    def __init__(self):
        self.model = CrossEncoder("core/models/embeddings/bge-reranker-v2-m3")
    
    def rerank(self, query, documents, top_k=5):
        """Rerank documents based on query relevance"""
        # Extract document contents (handle Document objects or strings)
        doc_contents = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                doc_contents.append(doc.page_content)
            elif isinstance(doc, str):
                doc_contents.append(doc)
            else:
                # Convert other types to string
                doc_contents.append(str(doc))
        
        pairs = [(query, content) for content in doc_contents]
        scores = self.model.predict(pairs)
        
        # Sort by scores and return top_k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
