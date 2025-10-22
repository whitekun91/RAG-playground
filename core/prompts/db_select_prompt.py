from langchain_core.prompts import PromptTemplate

db_select_prompt = PromptTemplate.from_template(
    """You are a classification assistant that determines which type of document best matches the user's question.

There are two types of documents:
- docx: These contain manuals, step-by-step guides, inspection procedures, or troubleshooting instructions.
- pdf: These contain information about equipment status, sensor data, process flow, or technical overviews of machinery.

Your task is to analyze the user's question and determine which type of document is the most appropriate source of answer.

The user's final answer will be generated in Korean, so choose the document type based on what would best support a Korean-language answer.

Question: {question}

Answer (respond with only 'docx' or 'pdf'):"""
)
