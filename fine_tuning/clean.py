import json
import re

MIN_TOKENS = 20

fixed = []

with open("medical_reasoning_tropical_sft.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)

        response = re.sub(
            r"<think>.*?</think>",
            "",
            ex["response"],
            flags=re.DOTALL
        ).strip()

        if len(response.split()) < MIN_TOKENS:
            continue  # DROP bad samples

        ex["response"] = response
        fixed.append(ex)

print("Kept samples:", len(fixed))

with open("medical_reasoning_tropical_sft_clean.jsonl", "w", encoding="utf-8") as f:
    for ex in fixed:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
