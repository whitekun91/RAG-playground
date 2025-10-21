from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser


class ChainService:
    """
    프롬프트 기반 체인 생성을 담당하는 서비스.
    """

    def create_db_chain(self, db_select_prompt, call_llm):
        return (
            db_select_prompt
            | RunnableLambda(lambda pv: pv.to_string())
            | RunnableLambda(lambda s: call_llm(s))
            | StrOutputParser()
        )

    def create_image_chain(self, image_detect_prompt, call_llm):
        return (
            image_detect_prompt
            | RunnableLambda(lambda x: call_llm(x.to_string()))
            | StrOutputParser()
        )


