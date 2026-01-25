from datasets import load_dataset
import json

KEYWORDS = [
    "malaria", "dengue", "tuberculosis", "leishmaniasis",
    "schistosomiasis", "chikungunya", "zika",
    "yellow fever", "filariasis", "trypanosomiasis",
    "melioidosis", "scrub typhus", "leprosy"
]

def contains_kw(text):
    return any(k in text.lower() for k in KEYWORDS)

dataset = load_dataset(
    "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B",
    split="train",
    streaming=True
)

OUTPUT = "dataset_instruction_response.jsonl"
MAX = 20000
count = 0

with open(OUTPUT, "w", encoding="utf-8") as f:
    for ex in dataset:
        messages = ex.get("messages", [])
        if not messages:
            continue

        full_text = " ".join(m.get("content", "") for m in messages)
        if not contains_kw(full_text):
            continue

        user = []
        assistant = []

        for m in messages:
            if m["role"] == "user":
                user.append(m["content"])
            elif m["role"] == "assistant":
                assistant.append(m["content"])

        if not user or not assistant:
            continue

        f.write(json.dumps({
            "instruction": "\n".join(user).strip(),
            "response": assistant[-1].strip()
        }, ensure_ascii=False) + "\n")

        count += 1
        if count >= MAX:
            break

print(f"✅ Saved {count} instruction-response samples")
