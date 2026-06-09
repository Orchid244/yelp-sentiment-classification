import re

from config import LABEL2ID


def map_rating_to_label(rating: int) -> str:
    rating = int(rating)
    if rating in [1, 2]:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"



def map_label_to_id(label_name: str) -> int:
    return LABEL2ID[label_name]



def clean_text_for_baseline(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s'.,!?-]", " ", text)
    return text.strip().lower()



def clean_text_for_transformer(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()