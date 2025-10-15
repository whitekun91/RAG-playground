from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser


def create_db_chain(db_select_prompt, call_llm):
    db_selector_chain = (
            db_select_prompt
            | RunnableLambda(lambda pv: pv.to_string())
            | RunnableLambda(lambda s: call_llm(s))
            | StrOutputParser()
    )
    return db_selector_chain

# def create_image_chain(image_detect_prompt, call_llm):
#     image_request_chain = (
#             image_detect_prompt
#             | RunnableLambda(lambda x: call_llm(x.to_string()))
#             | StrOutputParser()
#     )
#     return image_request_chain
