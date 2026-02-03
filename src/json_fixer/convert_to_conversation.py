from utils.json_pretty import prettify_json


def convert_to_conversation(example):
  input_json = example["invalid_json"]
  fixed_json = example["fixed_json"]

  messages = [
    {
      "role": "system",
      "content": f"You are an expert JSON parser, formatter and fixer. Only output JSON wrapped in markdown code fences."
    },
    {
      "role": "user",
      "content": f"Fix this JSON:\n{input_json}"
    },
    {
      "role": "assistant",
      "content": f"""```json
{prettify_json(fixed_json)}
```
"""
    }
  ]

  return {"conversations": messages}
