from typing import List, Dict, Any

from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.services import ImageLinkBuilder, RAGComposer


class QueryService:
    def __init__(self, db_classifier, vector_store, prompt, call_llm, reranker):
        self.db_classifier = db_classifier
        self.vector_store = vector_store
        self.prompt = prompt
        self.call_llm = call_llm
        self.reranker = reranker
        self._composer = RAGComposer()
        self._image_links = ImageLinkBuilder()

    def classify_db_type(self, question_text: str) -> str:
        return self.db_classifier(question_text)

    def retrieve_and_rerank(self, question_text: str):
        retriever = self.vector_store.as_retriever()
        return self.reranker.rerank(retriever, question_text)

    def generate_answer(self, question_text: str, reranked_docs) -> str:
        inputs = self._composer.compose_inputs(reranked_docs)
        rag_chain = (
            inputs
            | self.prompt
            | RunnableLambda(lambda x: self.call_llm(x.to_string()))
            | StrOutputParser()
        )
        return rag_chain.invoke(question_text)

    def collect_image_urls(self, reranked_docs) -> List[str]:
        image_refs: List[str] = []
        for doc in reranked_docs:
            try:
                import json
                img_list = json.loads(doc.metadata.get("image_refs", "[]"))
                image_refs.extend(img_list)
            except Exception:
                continue
        return self._image_links.build(image_refs)

    def handle(self, question_text: str) -> Dict[str, Any]:
        db_type = self.classify_db_type(question_text)
        print(db_type)
        reranked_docs = self.retrieve_and_rerank(question_text)
        answer_text = self.generate_answer(question_text, reranked_docs)
        image_urls = self.collect_image_urls(reranked_docs) if db_type == "pdf" else []
        return {
            "db_type": db_type,
            "answer_text": answer_text,
            "image_urls": image_urls or None,
        }


