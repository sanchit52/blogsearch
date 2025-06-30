
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

tokenizer = AutoTokenizer.from_pretrained("models/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained("models/bert-tiny")

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def is_personal_blog(text: str, threshold=0.6) -> bool:
    scores = pipeline(text[:512])[0]  # slice to fit max length
    # Expecting two labels: LABEL_0 (non-personal), LABEL_1 (personal)
    for item in scores:
        if item['label'] in ("LABEL_1", "1") and item['score'] >= threshold:
            return True
    return False

if __name__ == "__main__":
    # Quick manual test
    samples = [
        "I started my blog to share my personal growth journey...",
        "Welcome to Acme Corp’s services page."
    ]
    for s in samples:
        print(s, "→", is_personal_blog(s))
