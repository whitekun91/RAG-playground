"""
RAG Composer - RAG 입력을 구성하는 클래스
"""
from typing import List, Any
from langchain_core.runnables import RunnableLambda


class RAGComposer:
    """RAG 입력을 구성하는 클래스"""
    
    def compose_inputs(self, reranked_docs) -> RunnableLambda:
        """리랭크된 문서들을 RAG 입력으로 구성"""
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        return RunnableLambda(format_docs)
