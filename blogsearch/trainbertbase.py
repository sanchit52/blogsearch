import os
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

# 1. Load the dataset
data_files = {
    "train": "./data/processed/final/train.jsonl",
    "validation": "./data/processed/final/val.jsonl"
}
dataset = load_dataset("json", data_files=data_files)

# 2. Load tokenizer and base BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. Configure LoRA for BERT
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "key", "value", "dense"]  # For BERT
)
model = get_peft_model(base_model, peft_config)

# 4. Tokenization: combine title and content
def tokenize_fn(batch):
    texts = [ (t or "") + " " + (c or "") for t, c in zip(batch["title"], batch["content"]) ]
    return tokenizer(texts, truncation=True, padding=False)

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized = tokenized.remove_columns(["title", "url", "content"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

# 5. Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# 6. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1": f1_score(labels, preds, average="weighted"),
        "accuracy": accuracy_score(labels, preds)
    }

# 7. Training arguments (optimized for 3.7 GB GPU)
training_args = TrainingArguments(
    output_dir="./classifier/bertbase_lora_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./classifier/logs",
    logging_strategy="epoch",
    logging_steps=10,
    report_to="none",
    fp16=True  # Enable mixed precision (if supported)
)

# 8. Trainer
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

# 9. Train
trainer.train()

# 10. Save final LoRA adapter
os.makedirs("./classifier/bertbase_lora_adapter/", exist_ok=True)
model.save_pretrained("./classifier/bertbase_lora_adapter/")
print("LoRA adapter saved to ./classifier/bertbase_lora_adapter/")

# 11. Optional: visualize training vs eval loss
try:
    import json
    import matplotlib.pyplot as plt
    with open("./classifier/bertbase_lora_output/trainer_state.json") as f:
        logs = json.load(f)["log_history"]
    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
    plt.plot(train_loss, label="Train Loss")
    plt.plot(eval_loss, label="Eval Loss")
    plt.legend()
    plt.title("Train vs Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
except Exception as e:
    print("Plotting skipped:", e)
