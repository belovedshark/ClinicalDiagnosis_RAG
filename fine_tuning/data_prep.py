"""Data preparation for Gemma fine-tuning.

Converts pubmedqa_tropical_filtered.csv to Hugging Face Dataset
with Gemma chat format.
"""

import ast
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .config import DATA_PATH, BASE_MODEL, TRAIN_TEST_SPLIT, RANDOM_SEED


def parse_context(context_str: str) -> str:
    """Parse the context dictionary string into readable text.

    The context column contains a string representation of a dict with
    'contexts' (array of text) and 'labels' (array of section names).
    """
    try:
        # The context is stored as a string repr of a dict
        # Convert numpy arrays to lists for safer eval
        context_str = context_str.replace("array(", "[").replace(", dtype=object)", "]")
        context_dict = ast.literal_eval(context_str)

        contexts = context_dict.get("contexts", [])
        labels = context_dict.get("labels", [])

        # Combine labels with their text
        parts = []
        for i, text in enumerate(contexts):
            label = labels[i] if i < len(labels) else "TEXT"
            parts.append(f"{label}: {text}")

        return "\n\n".join(parts)
    except Exception as e:
        # If parsing fails, return as-is
        return str(context_str)


def format_chat_message(row: dict) -> dict:
    """Format a single row into Gemma chat format.

    Returns dict with 'text' field containing the formatted conversation.
    """
    context = parse_context(row["context"])
    question = row["question"]
    answer = row["long_answer"]

    # Gemma 3 chat format
    text = f"""<bos><start_of_turn>user
Context: {context}

Question: {question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn><eos>"""

    return {"text": text}


def load_and_prepare_data() -> DatasetDict:
    """Load CSV and convert to train/validation DatasetDict."""
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} examples")

    # Convert to HF Dataset
    dataset = Dataset.from_pandas(df)

    # Format each example
    print("Formatting examples to chat format...")
    dataset = dataset.map(format_chat_message, remove_columns=dataset.column_names)

    # Split into train/validation
    print(f"Splitting data: {1-TRAIN_TEST_SPLIT:.0%} train, {TRAIN_TEST_SPLIT:.0%} validation")
    split = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT, seed=RANDOM_SEED)

    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    print(f"Train examples: {len(dataset_dict['train'])}")
    print(f"Validation examples: {len(dataset_dict['validation'])}")

    return dataset_dict


def verify_tokenization(dataset: DatasetDict, max_length: int = 1024) -> dict:
    """Verify dataset tokenizes properly and check sequence lengths."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lengths = []
    truncated = 0

    for example in dataset["train"]:
        tokens = tokenizer(example["text"], truncation=False)
        length = len(tokens["input_ids"])
        lengths.append(length)
        if length > max_length:
            truncated += 1

    stats = {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "truncated_count": truncated,
        "truncated_pct": truncated / len(lengths) * 100,
    }

    print(f"\nTokenization stats:")
    print(f"  Min length: {stats['min_length']}")
    print(f"  Max length: {stats['max_length']}")
    print(f"  Avg length: {stats['avg_length']:.0f}")
    print(f"  Truncated (>{max_length}): {stats['truncated_count']} ({stats['truncated_pct']:.1f}%)")

    return stats


if __name__ == "__main__":
    dataset = load_and_prepare_data()

    # Show a sample
    print("\n" + "="*50)
    print("Sample formatted example:")
    print("="*50)
    print(dataset["train"][0]["text"][:500] + "...")

    # Verify tokenization
    print("\n" + "="*50)
    verify_tokenization(dataset)
