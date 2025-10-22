from langchain_core.prompts import PromptTemplate


origin_prompt = PromptTemplate.from_template(
    """Use the following context to answer the question in Korean.

If the context or metadata contains any image references (e.g., `image_refs` or image URLs),
do not include any image URLs, file paths, or image links in your answer.
You may refer to the existence of related images, such as saying "There are relevant images available" or "Please refer to the images below," but never display the actual URL, file path, or any text like [이미지 보기: ...].

If you don't know the answer, say you don't know.
Cite the source only if relevant context is provided.

# Context:
{context}

# Question:
{question}

# Answer:"""
)
