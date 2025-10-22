def is_image_request(image_request_chain, question: str) -> bool:
    try:
        response = image_request_chain.invoke({"question": question})
        return response.strip().lower() == "yes"
    except Exception:
        return False
