from datasets import load_dataset
from transformers import AutoTokenizer

# Load data
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:3]")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("=== RAW DATA ===")
for i, example in enumerate(dataset):
    print(f"\nExample {i+1}:")
    print(f"Instruction: {example['instruction']}")
    print(f"Context: {example['context']}")
    print(f"Response: {example['response']}")

print("\n=== FORMATTED DATA ===")
def format_example(example):
    if example['context']:
        return f"Human: {example['instruction']}\n\nContext: {example['context']}\n\nAssistant: {example['response']}<|endoftext|>"
    else:
        return f"Human: {example['instruction']}\n\nAssistant: {example['response']}<|endoftext|>"

for i, example in enumerate(dataset):
    formatted = format_example(example)
    print(f"\nFormatted {i+1}:")
    print(repr(formatted))
