from kiwipiepy import Kiwi

def split_korean_sentences(text):
    kiwi = Kiwi()
    return [s.text for s in kiwi.split_into_sents(text)]