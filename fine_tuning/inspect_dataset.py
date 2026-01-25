from datasets import load_dataset

dataset = load_dataset(
    "OpenMed/Medical-Reasoning-SFT-GPT-OSS-120B",
    split="train",
    streaming=True
)

# Print the first example
for example in dataset:
    print(example)
    break
