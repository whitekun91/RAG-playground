"""
RAG Composer - Compose documents into RAG input format
"""
from typing import List, Dict, Any
from langchain_core.documents import Document


class RAGComposer:
    """Class for composing RAG inputs"""
    
    def __init__(self):
        pass
    
    def compose_inputs(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Compose documents into RAG input format
        
        Args:
            documents: Retrieved documents
            
        Returns:
            RAG input dictionary
        """
        if not documents:
            return {"context": "", "sources": []}
        
        # Combine document contents
        context_parts = []
        sources = []
        
        for doc in documents:
            # Check if it's a Document object
            if hasattr(doc, 'page_content'):
                if doc.page_content:
                    context_parts.append(doc.page_content)
            elif isinstance(doc, str):
                # Use string as-is
                context_parts.append(doc)
            
            # Collect source information
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'Unknown')
                if source not in sources:
                    sources.append(source)
        
        context = "\n\n".join(context_parts)
        
        return {
            "context": context,
            "sources": sources if sources else []
        }
