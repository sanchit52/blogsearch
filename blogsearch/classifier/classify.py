import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
from tqdm import tqdm

# Load tokenizer and base model with LoRA adapter
base_model = "distilbert-base-uncased"
adapter_path = "./distilbert_lora_adapter"
output_file = "../data/processed/final/personal_blogs_fixed.jsonl"
input_file = "../scraping/scraped_blogs.jsonl"

tokenizer = AutoTokenizer.from_pretrained(base_model)
base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    return pred  # 1 = personal, 0 = non-personal

# Process the input file
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="Classifying"):
        entry = json.loads(line)
        text = f"{entry.get('title', '')}\n{entry.get('content', '')}"
        if classify(text) == 1:
            outfile.write(json.dumps(entry) + "\n")

print(f"Finished. Personal blogs saved to: {output_file}")
