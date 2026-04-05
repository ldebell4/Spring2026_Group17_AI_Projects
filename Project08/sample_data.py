import json

# Sample training data
with open("training_data.jsonl", "r") as f:
    lines = f.readlines()

with open("training_data_sample.jsonl", "w") as f:
    f.writelines(lines[:1500])

# Sample test data
with open("test_data.jsonl", "r") as f:
    test_lines = f.readlines()

with open("test_data_sample.jsonl", "w") as f:
    f.writelines(test_lines[:100])

print("Done!")
print(f"Training sample: 1500 examples")
print(f"Test sample: 100 examples")