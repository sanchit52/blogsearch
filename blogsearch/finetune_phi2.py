import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# === CONFIG ===
DATA_DIR = Path("data/processed/final")
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "phi-2/finetuned"
BATCH_SIZE = 4
NUM_EPOCHS = 3
MAX_LENGTH = 512

# === Load and process JSONL data ===
def load_jsonl(file_path, label):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [{"text": json.loads(line)["content"], "label": label} for line in f if line.strip()]

all_data = []

for file in DATA_DIR.glob("*.jsonl"):
    if "non_personal" in file.name:
        label = 0
    elif "personal" in file.name:
        label = 1
    else:
        continue
    all_data.extend(load_jsonl(file, label))

dataset = Dataset.from_list(all_data)

# === Tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Phi-2 has no pad token

def tokenize(batch):
    tokenized = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    tokenized["labels"] = batch["label"]
    return tokenized


dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1)
tokenized = dataset.map(tokenize, batched=True)

# === Load and prepare model ===
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    device_map="auto",
    load_in_4bit=True
)
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust as needed
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# === Training ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=NUM_EPOCHS,
    logging_dir="./logs",
    logging_steps=20,
    fp16=True,
    save_total_limit=1,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)
