from langchain_core.prompts import PromptTemplate

image_detect_prompt = PromptTemplate.from_template("""
You are a classification assistant that determines whether the user's question is asking for visual content such as images, diagrams, charts, or figures.

Answer only with 'yes' or 'no'.

Question: {question}
Answer:
""")
