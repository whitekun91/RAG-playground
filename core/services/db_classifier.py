def classify_question_to_db_type(db_selector_chain, question: str) -> str:
    raw = db_selector_chain.invoke({"question": question})
    db_type = raw.strip().lower()
    return db_type if db_type in ["docx", "pdf"] else "docx"


def build_db_classifier(db_selector_chain):
    def _classifier(question_text: str) -> str:
        return classify_question_to_db_type(db_selector_chain, question_text)
    return _classifier


