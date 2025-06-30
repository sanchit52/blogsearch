import json
import os
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType

# ===== Custom Trainer to prevent .to() on 4-bit model =====
class NoMoveTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        # Skip moving quantized model; assume device_map handled it
        return model

# === CONFIGURATION ===
DATA_DIR        = Path("data/processed/final")
MODEL_NAME      = "microsoft/phi-1_5"
OUTPUT_DIR      = "phi-1.5/finetuned_lora_gpu_auto"
BATCH_SIZE      = 2
NUM_EPOCHS      = 3
MAX_LENGTH      = 256
LEARNING_RATE   = 5e-5
MIN_TEXT_LENGTH = 50
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === DATA LOADING ===
def load_jsonl(fp, label):
    items = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                txt = j["content"].strip()
                if len(txt) >= MIN_TEXT_LENGTH:
                    items.append({"text": txt, "label": label})
            except:
                continue
    return items

print("Loading data...")
all_data = []
for fn in DATA_DIR.glob("*.jsonl"):
    lbl = 0 if "non_personal" in fn.name.lower() else 1
    all_data += load_jsonl(fn, lbl)
print("Distribution:", Counter(d["label"] for d in all_data))
dataset = Dataset.from_list(all_data)

# === TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    tok = tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
    )
    tok["labels"] = examples["label"]
    return tok

# === SPLIT & TOKENIZE ===
ds = dataset.shuffle(42)
sp = ds.train_test_split(test_size=0.15, seed=42)
tv = sp["train"].train_test_split(test_size=0.1, seed=42)

tok_ds = {
    "train":      tv["train"].map(tokenize_fn, batched=True, remove_columns=["text"]),
    "validation": tv["test"].map(tokenize_fn, batched=True, remove_columns=["text"]),
    "test":       sp["test"].map(tokenize_fn, batched=True, remove_columns=["text"]),
}

# === LOAD QUANTIZED MODEL WITH CPU LOADING + PATCH ===
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
print("Loading 4-bit quantized model onto CPU first...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_cfg,
    device_map=None,
    num_labels=2,
    trust_remote_code=True,
)

# === MOVE MANUALLY TO GPU IF AVAILABLE ===
if torch.cuda.is_available():
    print("✓ Manually moving model to GPU")
    model = model.cuda()

# === APPLY LORA ===
peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# === METRICS ===
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    pr, rc, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

# === TRAINING ARGUMENTS ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    remove_unused_columns=True,
    fp16=True,
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# === TRAIN & EVAL ===
print("Starting LoRA fine-tuning...")
trainer.train()
print("\nTest evaluation:", trainer.evaluate(tok_ds["test"]))

# === SAVE ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! Model & tokenizer saved to", OUTPUT_DIR)
