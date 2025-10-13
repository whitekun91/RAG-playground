# 파일명: create_chain.py
from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser


def create_db_chain(db_select_prompt, call_vllm):
    db_selector_chain = (
            db_select_prompt
            | RunnableLambda(lambda x: call_vllm(x.to_string()))
            | StrOutputParser()
    )
    return db_selector_chain


def create_image_chain(image_detect_prompt, call_vllm):
    image_request_chain = (
            image_detect_prompt
            | RunnableLambda(lambda x: call_vllm(x.to_string()))
            | StrOutputParser()
    )
    return image_request_chain
