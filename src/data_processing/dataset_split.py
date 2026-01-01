import jsonlines
import random

base_dataset_dir = "/home/rngo/code/intel-gpu-fine-tune/dataset"
with jsonlines.open(f"{base_dataset_dir}/dataset.jsonl", "r") as j:
  data = list(j)

random.shuffle(data)

num_examples = len(data)

eval_split_idx = int(0.05 * num_examples)
eval_data = data[:eval_split_idx]

data = data[eval_split_idx:]

# test data should have the same number of examples as eval
test_split_idx = len(eval_data)
test_data = data[:test_split_idx]

# the rest is training data
train_data = data[test_split_idx:]

with jsonlines.open(f"{base_dataset_dir}/train_data.jsonl", "w") as j:
  j.write_all(train_data)

with jsonlines.open(f"{base_dataset_dir}/eval_data.jsonl", "w") as j:
  j.write_all(eval_data)

with jsonlines.open(f"{base_dataset_dir}/test_data.jsonl", "w") as j:
  j.write_all(test_data)
