def build_db_classifier(db_selector_chain):
    """Build database classifier function"""
    def classify_question_to_db_type(question_text: str) -> str:
        """Classify question to determine which database to use"""
        result = db_selector_chain.invoke(question_text)
        # Simple classification logic - can be enhanced
        if "pdf" in result.lower() or "document" in result.lower():
            return "pdf"
        elif "image" in result.lower() or "picture" in result.lower():
            return "image"
        else:
            return "general"
    
    return classify_question_to_db_type
