from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

class ChainService:
    def __init__(self):
        pass
    
    def create_db_chain(self, prompt, call_llm):
        """Create database selection chain"""
        return (
            prompt
            | RunnableLambda(lambda x: call_llm(x.to_string()))
            | StrOutputParser()
        )
