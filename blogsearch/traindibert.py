import os
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

# Load datasets from the JSONL files
data_files = {
    "train": "./data/processed/final/train.jsonl",
    "validation": "./data/processed/final/val.jsonl"
}
dataset = load_dataset("json", data_files=data_files)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Apply LoRA adapters via PEFT
peft_config = LoraConfig(
    task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"], bias="none"
)
model = get_peft_model(base_model, peft_config)

# Tokenization function: combine title and content, truncate to max 512 tokens
def tokenize_fn(batch):
    texts = [ (t or "") + " " + (c or "") for t, c in zip(batch["title"], batch["content"]) ]
    return tokenizer(texts, truncation=True, padding=False)

# Apply tokenization to the dataset
tokenized = dataset.map(tokenize_fn, batched=True)
tokenized = tokenized.remove_columns(["title", "url", "content"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer)

# Compute metrics: weighted F1 and accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}

# Training arguments with early stopping and best-model saving
training_args = TrainingArguments(
    output_dir="./classifier/distilbert_lora_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./classifier/logs",
    logging_steps=50,
    report_to="none"  # or "tensorboard" to enable TensorBoard logging
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train the model
trainer.train()

# After training, save the best LoRA adapter model
os.makedirs("./classifier/distilbert_lora_adapter/", exist_ok=True)
model.save_pretrained("./classifier/distilbert_lora_adapter/")
print("LoRA adapter saved to blogsearch/classifier/distilbert_lora_adapter/")
