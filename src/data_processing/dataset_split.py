import jsonlines
import random

with jsonlines.open("data_filtered.jsonl", "r") as j:
    data = list(j)

random.shuffle(data)

num_examples = len(data)

eval_split_idx = int(0.15 * num_examples)
eval_data = data[:eval_split_idx]

data = data[eval_split_idx:]

test_split_idx = int(0.15 * num_examples)
test_data = data[:test_split_idx]

train_data = data[test_split_idx:]

with jsonlines.open("train_data.jsonl", "w") as j:
    j.write_all(train_data)

with jsonlines.open("eval_data.jsonl", "w") as j:
    j.write_all(eval_data)

with jsonlines.open("test_data.jsonl", "w") as j:
    j.write_all(test_data)

