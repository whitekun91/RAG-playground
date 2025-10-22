"""
RAG Composer - 문서들을 RAG 입력으로 구성하는 모듈
"""
from typing import List, Dict, Any
from langchain_core.documents import Document


class RAGComposer:
    """RAG 입력을 구성하는 클래스"""
    
    def __init__(self):
        pass
    
    def compose_inputs(self, documents: List[Document]) -> Dict[str, Any]:
        """
        문서들을 RAG 입력으로 구성
        
        Args:
            documents: 검색된 문서들
            
        Returns:
            RAG 입력 딕셔너리
        """
        if not documents:
            return {"context": "", "sources": []}
        
        # 문서 내용을 결합
        context_parts = []
        sources = []
        
        for doc in documents:
            if hasattr(doc, 'page_content') and doc.page_content:
                context_parts.append(doc.page_content)
            
            # 소스 정보 수집
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'Unknown')
                if source not in sources:
                    sources.append(source)
        
        context = "\n\n".join(context_parts)
        
        return {
            "context": context,
            "sources": sources
        }
