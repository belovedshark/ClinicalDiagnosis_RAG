from datasets import load_dataset
import json

DATASET_NAME = "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B"
OUTPUT_FILE = "openmed_train.jsonl"
MAX_SAMPLES = 2000   # start small, increase later

print("Loading OpenMed dataset (streaming)...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

def format_example(example):
    """
    Convert OpenMed 'messages' into a single training text.
    """
    messages = example["messages"]

    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if role == "user":
            text += f"USER: {content}\n"
        elif role == "assistant":
            text += f"ASSISTANT: {content}\n"

    return {"text": text.strip()}

print("Writing formatted samples...")

count = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for example in dataset:
        formatted = format_example(example)
        f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
        count += 1

        if count >= MAX_SAMPLES:
            break

print(f"✅ Saved {count} samples to {OUTPUT_FILE}")
