import json
from sklearn.model_selection import train_test_split

# Load and label data from JSONL files
personal_file = "data/processed/final/personal_blogs_fixed.jsonl"
nonpersonal_file = "data/processed/final/non_personal_blogs_fixed.jsonl"
personal_data = []
nonpersonal_data = []
with open(personal_file, 'r') as f:
    for line in f:
        entry = json.loads(line)
        entry["label"] = 1  # Personal blog
        personal_data.append(entry)
with open(nonpersonal_file, 'r') as f:
    for line in f:
        entry = json.loads(line)
        entry["label"] = 0  # Non-personal blog
        nonpersonal_data.append(entry)

# Combine and shuffle with stratified split
all_data = personal_data + nonpersonal_data
labels = [e["label"] for e in all_data]
train_data, val_data, _, _ = train_test_split(
    all_data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Save splits back to JSONL
with open("data/processed/final/train.jsonl", "w") as fout:
    for entry in train_data:
        fout.write(json.dumps(entry) + "\n")
with open("data/processed/final/val.jsonl", "w") as fout:
    for entry in val_data:
        fout.write(json.dumps(entry) + "\n")
