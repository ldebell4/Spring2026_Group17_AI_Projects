"""
Project 8 - Dataset Preprocessing Script
Dataset: MikePfunk28/resume-training-dataset
Target format: JSONL prompt/response pairs for Ollama LoRA fine-tuning

Usage:
    pip install datasets
    python preprocess_dataset.py
    
Output:
    training_data.jsonl  - formatted training set
    test_data.jsonl      - formatted test set (for Step 5 evaluation)
"""

import json
from datasets import load_dataset

# ── 1. Load dataset ────────────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset("MikePfunk28/resume-training-dataset")
print(f"Dataset loaded: {ds}")

# ── 2. Inspect raw structure (helps with debugging) ────────────────────────────
print("\nFirst raw example:")
print(ds["train"][0])
print("\nColumn names:", ds["train"].column_names)

# ── 3. Conversion function ─────────────────────────────────────────────────────
def convert_to_prompt_response(example):
    """
    The dataset is in conversational format with a 'messages' field
    containing system / user / assistant turns.
    
    We extract:
        prompt   = the user message
        response = the assistant message
    
    If the dataset uses different field names (e.g. 'instruction'/'output'),
    the fallback block below handles that automatically.
    """
    try:
        # Case 1: messages array format (system / user / assistant turns)
        if "messages" in example:
            messages = example["messages"]
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
            if user_msg and asst_msg:
                return {"prompt": user_msg.strip(), "response": asst_msg.strip()}

        # Case 2: flat instruction / output fields
        if "instruction" in example and "output" in example:
            return {
                "prompt": example["instruction"].strip(),
                "response": example["output"].strip()
            }

        # Case 3: prompt / response already exist
        if "prompt" in example and "response" in example:
            return {
                "prompt": example["prompt"].strip(),
                "response": example["response"].strip()
            }

    except Exception as e:
        print(f"Skipping malformed example: {e}")

    return None  # skip this example


# ── 4. Process and filter ──────────────────────────────────────────────────────
def process_split(split, name):
    results = []
    skipped = 0

    for example in split:
        converted = convert_to_prompt_response(example)
        if converted:
            # Basic quality filter: skip very short responses
            if len(converted["response"]) > 20:
                results.append(converted)
            else:
                skipped += 1
        else:
            skipped += 1

    print(f"\n{name}: {len(results)} valid examples, {skipped} skipped")
    return results


train_data = process_split(ds["train"], "Train")

# If the dataset has a test split use it, otherwise carve 10% off train
if "test" in ds:
    test_data = process_split(ds["test"], "Test")
else:
    split_idx = int(len(train_data) * 0.9)
    test_data = train_data[split_idx:]
    train_data = train_data[:split_idx]
    print(f"No test split found — carved off {len(test_data)} examples for test set")


# ── 5. Write JSONL files ───────────────────────────────────────────────────────
def write_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples → {filename}")


write_jsonl(train_data, "training_data.jsonl")
write_jsonl(test_data, "test_data.jsonl")


# ── 6. Sanity check — print first 2 examples ──────────────────────────────────
print("\n── Sample output (first 2 training examples) ──")
with open("training_data.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        item = json.loads(line)
        print(f"\nExample {i+1}:")
        print(f"  PROMPT:   {item['prompt'][:120]}...")
        print(f"  RESPONSE: {item['response'][:120]}...")

print("\nDone! Hand training_data.jsonl and test_data.jsonl to your partner.")
print("She'll use training_data.jsonl for LoRA fine-tuning in Colab.")
print("You'll both use test_data.jsonl for Step 5 evaluation prompts.")
