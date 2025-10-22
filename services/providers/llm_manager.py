from ..llm_core import call_llm

class LLMManager:
    def __init__(self):
        pass
    
    def call_llm(self, prompt: str, provider: str = None):
        return call_llm(prompt, provider)

def create_llm_manager():
    return LLMManager()
