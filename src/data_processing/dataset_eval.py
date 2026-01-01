import openai
import os
import json
import jsonlines
from utils.clean_message import clean_message

base_dataset_directory = "/home/rngo/code/intel-gpu-fine-tune/dataset"

def create_client() -> openai.Client:
  base_api_url = os.environ.get("OPENAI_BASE_URL")
  api_key = os.environ.get("OPENAI_API_KEY")

  return openai.Client(
    base_url=base_api_url,
    api_key=api_key
  )

def eval_example(example: dict, client: openai.Client, attempt = 0):
  if attempt >= 3:
    print(f"Couldn't evaluate example for: {example}")

    return {"result": "low", "reason": "Could not evaluate example."}

  prompt = r"""ROLE:
You are a dataset evaluator.

We are fine-tuning a small language model to be able to take invalid JSON and product a valid version of the JSON.

This includes the following:
- fixing all keys and values to adhere to JSON spec. this includes adding quotes and fixing values in various ways such as adding proper escape characters.
- prettifying the incoming JSON payload

TASK:
We have synthetically generated the dataset. What you need to do now is to compare the invalid JSON with the fixed JSON and see whether the fixed JSON maintains the accuracy and fidelity of key-value pairs. This means that keys must not be changed in their names and values must not be changed unless it is to add proper escape sequences or the value is a semantic substitue for null/undefined values.

Notes:
- If invalid JSON contains Infinity and fixed contains "Infinity" (wrapped with quotes) then that is intentional.
- NaN must be wrapped in quotes: "NaN"
- Be mindful about keys in the invalid JSON payload with leading or trailing spaces, that may be intentional, and should left alone.
- Consecutive commas in arrays for invalid JSON should be removed in the fixed. Not replaced with null or undefined.

We want to only keep the data for training if you deem them to be high quality.

OUTPUT:
Output a JSON which contain the evaluation result of the data example. If the quality is low, then please provide reasoning. If the quality is high, then just state the quality meets standards for the reasoning. 

DO NOT include anything else other than the JSON representation of your evaluation!

{
  "result": "<high|low>",
  "reason": "<1-2 sentences describing why the quality is low>"
}


EXAMPLE 1:
Invalid JSON:
{"hello": "world", b: "hahahaa
blah blah
"}

Fixed JSON:
{
  "hello": "world",
  "b": "hahahaa\nblah blah\n"
}

Output:
{
  "result": "high",
  "reason": "The data example meets quality standards."
}

EXAMPLE 2:
Invalid JSON:
{
  "message": "He said, "Hello!"",
  "path": "C:\Users\John" 
}

Fixed JSON:
{
  "message": "He said, \"Hello!\"",
  "path": "C:\\Users\\Jon"
}

Output:
{
  "result": "low",
  "reason": "The original value for path had the sub-string \"John\", while the fixed result contains \"Jon\". This will result in the trained model emitting incorrect results."
}
"""

  user_prompt = f"""Invalid JSON:
{example["invalid_json"]}

Fixed JSON:
{example["fixed_json"]}
"""

  messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": user_prompt}
  ]

  response = client.chat.completions.create(
    model="gpt-5.2",
    messages=messages,
    reasoning_effort="medium"
  )

  assistant_response = response.choices[0].message.content

  try:
    assistant_response = clean_message(assistant_response)  
    result = json.loads(assistant_response)

    return result
  except:
    print(f"Couldn't evaluate the example. Retrying... Attempt: {attempt + 1}")

    return eval_example(example, client, attempt + 1)


if __name__ == "__main__":
  with jsonlines.open(f"{base_dataset_directory}/dataset.jsonl", "r") as j:
    dataset = list(j)

  print(f"Got {len(dataset)} training examples.")
  print("Filtering away duplicates first...")
  # go through entire dataset and get rid of duplicates
  seen = set()
  unique_dataset = []
  for example in dataset:
    # Create a consistent string representation for hashing
    example_hash = json.dumps(example, sort_keys=True)
    if example_hash not in seen:
      seen.add(example_hash)
      unique_dataset.append(example)
  
  dataset = unique_dataset
  print(f"Dataset examples after deduplication: {len(dataset)}")

  client = create_client()

  filtered = []
  for example in dataset:
    result = eval_example(example, client)

    print(result)

    if result["result"] == "high":
      filtered.append(example)
    elif result["result"] == "low":
      print("\nLOW QUALITY ALERT!!!\n") 
      print(example)
      print()
      print(result)
      print()

  with jsonlines.open(f"{base_dataset_directory}/dataset_filtered.jsonl", "w") as j:
    j.write_all(filtered)

  print(f"Original number of examples: {len(dataset)}")
  print(f"After filtering: {len(filtered)}")
