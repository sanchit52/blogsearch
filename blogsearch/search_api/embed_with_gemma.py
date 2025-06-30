from transformers import AutoTokenizer, AutoModel
import torch

model_name = "google/gemma-1.1-1b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example blog text
texts = ["How to optimize search engine embeddings for blog ranking"]

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get hidden states
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token output

print(embeddings.shape)
