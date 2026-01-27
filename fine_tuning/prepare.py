import json
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================

PUBMED_OUTPUT_CSV = "pubmedqa_tropical_filtered.csv"
PUBMED_OUTPUT_JSONL = "pubmedqa_tropical_filtered.jsonl"
REASONING_OUTPUT_JSONL = "medical_reasoning_tropical_sft.jsonl"

MAX_REASONING_SAMPLES = 800

TROPICAL_DISEASES = [
    "filoviral",
    "crusted scabies",
    "malaria",
    "dengue",
    "chikungunya",
    "zika",
    "leishmaniasis",
    "schistosomiasis",
    "trypanosomiasis",
    "filariasis",
    "tuberculosis",
    "scrub typhus",
    "melioidosis",
    "leprosy",
    "yellow fever",
    "plasmodium",
    "onchocerciasis",
    "strongyloidiasis",
    "amebiasis",
    "ascariasis",
    "hookworm",
    "ebola"
]

# ==============================
# PART 1: PUBMEDQA
# ==============================

print("\n=== Loading PubMedQA ===")

dataset = load_dataset(
    "qiaojin/PubMedQA",
    "pqa_artificial",
    split="train"
)

print("Total records:", len(dataset))


def is_tropical(example):
    text = (
        example["question"] + " " +
        " ".join(example["context"]["contexts"]) + " " +
        example["long_answer"]
    ).lower()

    return any(d in text for d in TROPICAL_DISEASES)


print("Filtering tropical disease records...")
tropical_dataset = dataset.filter(is_tropical)

print("Tropical-related records:", len(tropical_dataset))


# ---- count diseases
disease_counter = Counter()

for ex in tropical_dataset:
    text = (
        ex["question"] + " " +
        " ".join(ex["context"]["contexts"]) + " " +
        ex["long_answer"]
    ).lower()

    for d in TROPICAL_DISEASES:
        if d in text:
            disease_counter[d] += 1

print("\n=== TROPICAL DISEASE COUNTS ===")
for d, c in disease_counter.most_common():
    print(f"{d}: {c}")

# ---- save
tropical_dataset.to_csv(PUBMED_OUTPUT_CSV)
tropical_dataset.to_json(
    PUBMED_OUTPUT_JSONL,
    orient="records",
    lines=True
)

print(f"\n✅ Saved:")
print(f"- {PUBMED_OUTPUT_CSV}")
print(f"- {PUBMED_OUTPUT_JSONL}")

# ==============================
# PART 2: MEDICAL REASONING SFT
# ==============================

print("\n=== Loading Medical Reasoning Dataset (streaming) ===")

dataset_reasoning = load_dataset(
    "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B",
    split="train",
    streaming=True
)

TROPICAL_KEYWORDS = [
    "malaria",
    "dengue",
    "tuberculosis",
    "leishmaniasis",
    "schistosomiasis",
    "chikungunya",
    "zika",
    "yellow fever",
    "filariasis",
    "trypanosomiasis",
    "melioidosis",
    "scrub typhus",
    "leprosy"
]


def process_example(example):
    messages = example.get("messages", [])

    full_text = " ".join(
        m.get("content", "").lower()
        for m in messages
    )

    if not any(k in full_text for k in TROPICAL_KEYWORDS):
        return None

    user_parts = []
    assistant_parts = []

    for m in messages:
        role = m.get("role")
        content = m.get("content", "").strip()

        if not content:
            continue

        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    if not user_parts or not assistant_parts:
        return None

    return {
        "instruction": "\n".join(user_parts),
        "response": assistant_parts[-1]
    }


print("Processing medical reasoning samples...")

count = 0

with open(REASONING_OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for example in tqdm(dataset_reasoning, total=MAX_REASONING_SAMPLES):
        processed = process_example(example)

        if processed is None:
            continue

        f.write(json.dumps(processed, ensure_ascii=False) + "\n")
        count += 1

        if count >= MAX_REASONING_SAMPLES:
            break

print(f"\n✅ Saved {count} samples to {REASONING_OUTPUT_JSONL}")

# ---- preview
print("\n=== SAMPLE OUTPUT ===")
with open(REASONING_OUTPUT_JSONL, encoding="utf-8") as f:
    for _ in range(2):
        print(json.loads(next(f)))
        print("=" * 80)
