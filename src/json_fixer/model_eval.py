import json
import openai
import jsonlines
from utils.clean_message import clean_message
from openai.types.chat import ChatCompletion

test_dataset_file = "/home/rngo/code/intel-gpu-fine-tune/src/data_processing/test_data.jsonl"

base_api_url = "http://192.168.1.36:8000/v1"
api_key = "none"
model = "Qwen3-0.6B-finetuned"

client: openai.Client = openai.Client(
    base_url=base_api_url,
    api_key=api_key
)

with jsonlines.open(test_dataset_file, "r") as j:
    data = list(j)

score = 0
for example in data:
    input = example["invalid_json"]

    user_prompt = f"/no_think only output JSON. fix this JSON: {input}" if model == "Qwen3-0.6B" else f"fix this JSON: {input}"

    response: ChatCompletion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.01
    )

    assistant_message = response.choices[0].message.content
    try:
        assistant_message = clean_message(assistant_message)
        assistant_message_deserialized = json.loads(assistant_message)
        assistant_message_prettified = json.dumps(assistant_message_deserialized, indent=2)

        ground_truth = json.dumps(json.loads(example["fixed_json"]), indent=2)

        if assistant_message_prettified == ground_truth:
            score += 1
        else:
            print(f"{assistant_message_prettified} did not match ground truth: {ground_truth}")
    except:
        print(f"{assistant_message} did not match ground truth")

print(f"Final score for test set: {float(1.0 * score) / len(data)}")
